# fluentcompute/detectors/apple_detector.py
import asyncio
import subprocess
import re
import platform
import xml.etree.ElementTree as ET
import json # For loading specs
from pathlib import Path # For loading specs
from typing import List, Optional, Any, Dict
from datetime import datetime
import psutil # For telemetry fallback

from fluentcompute.models import GPUInfo, TelemetryData, VendorType, PerformanceTier, ComputeCapability
from .base import HardwareDetector
from fluentcompute.utils.logging_config import logger

class AppleDetector(HardwareDetector):
    _SPEC_FILE_NAME = "apple_specs.json"

    def __init__(self):
        self.specs: Dict[str, Any] = self._load_specs()

    def _load_specs(self) -> Dict[str, Any]:
        try:
            spec_path = Path(__file__).resolve().parent.parent / "data" / "specs" / self._SPEC_FILE_NAME
            if not spec_path.exists():
                logger.warning(f"Apple spec file not found at {spec_path}. Using empty specs.")
                return {}
            with open(spec_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load Apple specs from {self._SPEC_FILE_NAME}: {e}", exc_info=True)
            return {}

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
                data_array = root.find("array")
                if data_array is None or not len(data_array): return []
                
                items_node = None
                for top_dict_child in data_array[0]:
                    if top_dict_child.tag == 'key' and top_dict_child.text == '_items':
                        items_node = top_dict_child.getnext()
                        break
                
                if items_node is None or items_node.tag != 'array':
                    logger.warning("Could not find '_items' array in system_profiler XML.")
                    return []

                gpu_idx_counter = 0
                for item_dict in items_node.findall('dict'):
                    chip_model_text, vendor_text, vram_text, metal_family_text = None, None, None, None
                    children = list(item_dict)
                    for i in range(0, len(children), 2):
                        key_node, val_node = children[i], children[i+1] if (i+1) < len(children) else None
                        if key_node.tag == 'key' and val_node is not None and val_node.tag == 'string':
                            key_text = key_node.text
                            if key_text == 'sppci_model': chip_model_text = val_node.text
                            elif key_text == 'spdisplays_vendor': vendor_text = val_node.text
                            elif key_text == 'spdisplays_vram_shared': vram_text = val_node.text
                            elif key_text == 'spdisplays_vram' and not vram_text: vram_text = val_node.text
                            elif key_text == 'spdisplays_metal_family': metal_family_text = val_node.text

                    if chip_model_text and vendor_text and "apple" in vendor_text.lower():
                        raw_gpu_name = chip_model_text.strip()
                        apple_cores = self._get_apple_gpu_cores(raw_gpu_name)
                        gpu_name = re.sub(r'\s*\(\d+[-\s]core GPU\)', '', raw_gpu_name).strip()

                        memory_total_mb = 0
                        if vram_text:
                            mem_match = re.search(r'(\d+)\s*(GB|MB)', vram_text, re.IGNORECASE)
                            if mem_match:
                                memory_total_mb = int(mem_match.group(1)) * (1024 if mem_match.group(2).upper() == "GB" else 1)
                        
                        if memory_total_mb == 0: 
                             try:
                                ram_proc_cmd = ["sysctl", "hw.memsize"]
                                ram_process = await asyncio.create_subprocess_exec(*ram_proc_cmd, stdout=subprocess.PIPE)
                                ram_out_bytes, _ = await ram_process.communicate()
                                memory_total_mb = int(ram_out_bytes.decode().strip().split(":")[1].strip()) // (1024*1024)
                             except Exception as e_sysctl: logger.debug(f"Failed to get sys RAM: {e_sysctl}")

                        metal_version = metal_family_text.replace("Metal API Family ", "").replace("Metal Family ", "") if metal_family_text else ""

                        gpu_info = GPUInfo(
                            name=gpu_name, vendor=VendorType.APPLE, gpu_index=gpu_idx_counter,
                            uuid=f"apple-{gpu_name.replace(' ', '-')}-{apple_cores or 'X'}core-{gpu_idx_counter}",
                            pci_bus_id="N/A (Apple Silicon)",
                            memory_total=memory_total_mb, memory_free=0, 
                            memory_bandwidth=self._get_apple_memory_bandwidth(gpu_name),
                            compute_capability=f"Metal Family {metal_version}" if metal_version else "Metal",
                            apple_gpu_cores=apple_cores,
                            performance_tier=self._determine_apple_performance_tier(gpu_name),
                            driver_version=platform.mac_ver()[0], metal_version=metal_version,
                            supported_apis=[ComputeCapability.METAL, ComputeCapability.VULKAN], 
                        )
                        gpus_info_list.append(gpu_info)
                        gpu_idx_counter += 1
            else:
                 logger.debug(f"system_profiler failed. Stderr: {stderr.decode(errors='ignore').strip()}")
        except FileNotFoundError: logger.info("system_profiler not found.")
        except ET.ParseError as e_xml: logger.error(f"❌ Failed to parse system_profiler XML: {e_xml}")
        except Exception as e: logger.error(f"❌ Apple GPU detection failed: {e}", exc_info=True)
        return gpus_info_list

    def _get_apple_gpu_cores(self, gpu_name_str: str) -> Optional[int]:
        name = gpu_name_str.lower()
        core_map = self.specs.get("gpu_cores_map", {})
        for key, cores in core_map.items():
            if key in name:
                return cores
        return None

    def _get_apple_memory_bandwidth(self, gpu_name_str: str) -> float: # GB/s
        name = gpu_name_str.lower()
        bw_map = self.specs.get("memory_bandwidth_map", {})
        for key, bw in bw_map.items():
            if key in name: # "m3 max" in "Apple M3 Max"
                return bw
        return 0.0

    def _determine_apple_performance_tier(self, gpu_name_str: str) -> PerformanceTier:
        name = gpu_name_str.lower()
        if "ultra" in name: return PerformanceTier.ENTERPRISE 
        if "max" in name: return PerformanceTier.PROFESSIONAL
        if "pro" in name: return PerformanceTier.ENTHUSIAST
        if "m1" in name and not any(x in name for x in ["pro", "max", "ultra"]): return PerformanceTier.MAINSTREAM 
        if "m2" in name and not any(x in name for x in ["pro", "max", "ultra"]): return PerformanceTier.MAINSTREAM
        if "m3" in name and not any(x in name for x in ["pro", "max", "ultra"]): return PerformanceTier.MAINSTREAM
        return PerformanceTier.UNKNOWN

    def get_telemetry(self, gpu: GPUInfo) -> TelemetryData:
        logger.debug(f"Apple GPU telemetry for {gpu.name} is limited.")
        gpu_util = 0.0
        try:
            gpu_util = psutil.cpu_percent(interval=None) # Approximation via CPU load
        except Exception: pass

        return TelemetryData(
            timestamp=datetime.now(), gpu_utilization=gpu_util, memory_utilization=0.0,
            temperature=float(gpu.temperature or 0.0), power_draw=0.0 # Added or 0.0
        )
