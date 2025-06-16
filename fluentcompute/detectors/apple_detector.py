import asyncio
import subprocess
import re
import platform
import xml.etree.ElementTree as ET
from typing import List, Optional
from datetime import datetime

from fluentcompute.models import GPUInfo, TelemetryData, VendorType, PerformanceTier, ComputeCapability
from .base import HardwareDetector
from fluentcompute.utils.logging_config import logger

class AppleDetector(HardwareDetector):
    async def detect_hardware(self) -> List[GPUInfo]:
        gpus = []
        if platform.system() == "Darwin":
            gpus.extend(await self._detect_apple_silicon_macos())
        return gpus

    async def _detect_apple_silicon_macos(self) -> List[GPUInfo]:
        gpus_info_list = []
        try:
            cmd = ['system_profiler', 'SPDisplaysDataType', '-xml']
            process = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                output_xml = stdout.decode(errors='ignore')
                root = ET.fromstring(output_xml)
                
                # system_profiler XML structure: plist -> array -> dict -> key (_items) -> array (list of devices)
                # Each device is a dict.
                data_array = root.find("array")
                if data_array is None or not len(data_array): return []
                
                # Look for the "_items" key which contains the list of display controllers
                items_node = None
                for top_dict_child in data_array[0]: # data_array[0] should be the main dict
                    if top_dict_child.tag == 'key' and top_dict_child.text == '_items':
                        items_node = top_dict_child.getnext() # The next sibling is the array of items
                        break
                
                if items_node is None or items_node.tag != 'array':
                    logger.warning("Could not find '_items' array in system_profiler SPDisplaysDataType XML.")
                    return []

                gpu_idx_counter = 0
                for item_dict in items_node.findall('dict'):
                    chip_model_text, vendor_text, vram_text, metal_family_text = None, None, None, None
                    # Iterate through key/string pairs in the item_dict
                    children = list(item_dict) # Get all children (key, string, key, string ...)
                    for i in range(0, len(children), 2): # Step by 2 (key, then value)
                        key_node = children[i]
                        val_node = children[i+1] if (i+1) < len(children) else None

                        if key_node.tag == 'key' and val_node is not None and val_node.tag == 'string':
                            key_text = key_node.text
                            if key_text == 'sppci_model': chip_model_text = val_node.text
                            elif key_text == 'spdisplays_vendor': vendor_text = val_node.text
                            elif key_text == 'spdisplays_vram_shared': vram_text = val_node.text # Prefer shared for M-series
                            elif key_text == 'spdisplays_vram' and not vram_text: vram_text = val_node.text # Fallback
                            elif key_text == 'spdisplays_metal_family': metal_family_text = val_node.text

                    if chip_model_text and vendor_text and "apple" in vendor_text.lower():
                        raw_gpu_name = chip_model_text.strip()
                        
                        core_match = re.search(r'\((\d+)[-\s]core GPU\)', raw_gpu_name)
                        apple_cores = int(core_match.group(1)) if core_match else self._get_apple_gpu_cores(raw_gpu_name)
                        
                        gpu_name = re.sub(r'\s*\(\d+[-\s]core GPU\)', '', raw_gpu_name).strip()

                        memory_total_mb = 0
                        if vram_text:
                            mem_match = re.search(r'(\d+)\s*(GB|MB)', vram_text, re.IGNORECASE)
                            if mem_match:
                                val = int(mem_match.group(1))
                                memory_total_mb = val * 1024 if mem_match.group(2).upper() == "GB" else val
                        
                        if memory_total_mb == 0: 
                             try:
                                ram_proc_cmd = ["sysctl", "hw.memsize"]
                                ram_process = await asyncio.create_subprocess_exec(*ram_proc_cmd, stdout=subprocess.PIPE)
                                ram_out_bytes, _ = await ram_process.communicate()
                                ram_out_str = ram_out_bytes.decode().strip()
                                memory_total_mb = int(ram_out_str.split(":")[1].strip()) // (1024*1024)
                             except Exception as e_sysctl: 
                                logger.debug(f"Failed to get system RAM for Apple GPU via sysctl: {e_sysctl}")
                                pass
                        
                        metal_version = ""
                        if metal_family_text:
                             metal_version = metal_family_text.replace("Metal API Family ", "").replace("Metal Family ", "") # Handle both "Metal API Family..." and "Metal Family..."

                        gpu_info = GPUInfo(
                            name=gpu_name, vendor=VendorType.APPLE, gpu_index=gpu_idx_counter,
                            uuid=f"apple-{gpu_name.replace(' ', '-')}-{apple_cores or 'X'}core-{gpu_idx_counter}",
                            pci_bus_id="N/A (Apple Silicon)",
                            memory_total=memory_total_mb, memory_free=0, # Unified, 'free' is dynamic
                            memory_bandwidth=self._get_apple_memory_bandwidth(gpu_name),
                            compute_capability=f"Metal Family {metal_version}" if metal_version else "Metal",
                            apple_gpu_cores=apple_cores,
                            performance_tier=self._determine_apple_performance_tier(gpu_name),
                            driver_version=platform.mac_ver()[0], 
                            metal_version=metal_version,
                            supported_apis=[ComputeCapability.METAL, ComputeCapability.VULKAN], 
                        )
                        gpus_info_list.append(gpu_info)
                        gpu_idx_counter += 1
            else:
                 logger.debug(f"system_profiler for Apple GPU failed. Stderr: {stderr.decode(errors='ignore').strip()}")
        except FileNotFoundError: logger.info("system_profiler not found. Apple GPU detection not possible.")
        except ET.ParseError as e_xml: logger.error(f"❌ Failed to parse XML from system_profiler for Apple GPU: {e_xml}")
        except Exception as e: logger.error(f"❌ Apple GPU detection failed: {e}", exc_info=True)
        return gpus_info_list

    def _get_apple_gpu_cores(self, gpu_name_str: str) -> Optional[int]:
        name = gpu_name_str.lower()
        # More specific matches first
        if "m3 max chip with 16-core cpu and 40-core gpu" in name or "m3 max 40c gpu" in name: return 40
        if "m3 max chip with 14-core cpu and 30-core gpu" in name or "m3 max 30c gpu" in name: return 30
        if "m3 pro chip with 12-core cpu and 18-core gpu" in name or "m3 pro 18c gpu" in name: return 18
        if "m3 pro chip with 11-core cpu and 14-core gpu" in name or "m3 pro 14c gpu" in name: return 14
        if "m3 chip with 8-core cpu and 10-core gpu" in name or "m3 10c gpu" in name: return 10
        if "m3 chip with 8-core cpu and 8-core gpu" in name or "m3 8c gpu" in name: return 8

        if "m2 ultra" in name: return 76 # Can also be 60
        if "m2 max" in name: return 38 # Can also be 30
        if "m2 pro" in name: return 19 # Can also be 16
        if "m2" in name: return 10 # Can also be 8
        
        if "m1 ultra" in name: return 64 # Can also be 48
        if "m1 max" in name: return 32 # Can also be 24
        if "m1 pro" in name: return 16 # Can also be 14
        if "m1" in name: return 8 # Can also be 7
        return None # Could not infer from name

    def _get_apple_memory_bandwidth(self, gpu_name_str: str) -> float: # GB/s
        name = gpu_name_str.lower()
        if "m3 max" in name: return 400.0 
        if "m3 pro" in name: return 150.0
        if "m3" in name: return 100.0

        if "m2 ultra" in name: return 800.0
        if "m2 max" in name: return 400.0
        if "m2 pro" in name: return 200.0
        if "m2" in name: return 100.0

        if "m1 ultra" in name: return 800.0
        if "m1 max" in name: return 400.0
        if "m1 pro" in name: return 200.0
        if "m1" in name: return 68.25 # LPDDR4X
        return 0.0

    def _determine_apple_performance_tier(self, gpu_name_str: str) -> PerformanceTier:
        name = gpu_name_str.lower()
        if "ultra" in name: return PerformanceTier.ENTERPRISE 
        if "max" in name: return PerformanceTier.PROFESSIONAL
        if "pro" in name: return PerformanceTier.ENTHUSIAST
        # Base M-series
        if "m1" in name and not any(x in name for x in ["pro", "max", "ultra"]): return PerformanceTier.MAINSTREAM 
        if "m2" in name and not any(x in name for x in ["pro", "max", "ultra"]): return PerformanceTier.MAINSTREAM
        if "m3" in name and not any(x in name for x in ["pro", "max", "ultra"]): return PerformanceTier.MAINSTREAM
        return PerformanceTier.UNKNOWN

    def get_telemetry(self, gpu: GPUInfo) -> TelemetryData:
        # Powermetrics requires sudo and complex parsing.
        # iostat can give some disk/cpu but not GPU specific utilization easily.
        # `top -l 1 -s 0 -n 0` can give CPU, but again not GPU.
        # As of now, macOS doesn't have a simple, non-sudo user-level command for detailed GPU telemetry like nvidia-smi.
        logger.debug(f"Apple GPU telemetry for {gpu.name} via command line tools is very limited / requires sudo, and not fully implemented.")
        
        # We can try to get CPU usage as a rough proxy for system load
        gpu_util = 0.0
        try:
            # Get overall CPU idle percentage, then derive usage.
            # `top -l 1 -s 0 | grep CPUsage` is one way but messy. psutil is better.
            cpu_percent = psutil.cpu_percent(interval=None) # Overall system CPU usage
            gpu_util = cpu_percent # Very rough approximation.
        except Exception:
            pass # Ignore if psutil fails here.

        return TelemetryData(
            timestamp=datetime.now(), 
            gpu_utilization=gpu_util, # This is CPU util, not GPU.
            memory_utilization=0.0, # Unified memory, utilization hard to get easily
            temperature=float(gpu.temperature) if gpu.temperature else 0.0, 
            power_draw=0.0 # No easy way to get this.
        )
