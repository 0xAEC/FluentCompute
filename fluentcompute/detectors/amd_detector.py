import asyncio
import subprocess
import re
import platform
from typing import List, Optional
from datetime import datetime

from fluentcompute.models import GPUInfo, TelemetryData, VendorType, PerformanceTier, ComputeCapability
from .base import HardwareDetector
from fluentcompute.utils.logging_config import logger

class AMDDetector(HardwareDetector):
    async def detect_hardware(self) -> List[GPUInfo]:
        gpus = []
        rocm_smi_path = "rocm-smi" # Could be configured or found in PATH

        try:
            process = await asyncio.create_subprocess_exec(rocm_smi_path, '-d', '0', '--showid',
                                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await process.communicate()
            if process.returncode == 0 :
                logger.info("ROCm SMI detected. Attempting AMD GPU detection via rocm-smi.")
                gpus.extend(await self._detect_with_rocm_smi(rocm_smi_path))
            else:
                stderr_msg = stderr.decode(errors='ignore').strip()
                logger.info(f"rocm-smi not fully functional (or no AMD GPUs for it to manage). Fallback initiated. Error: {stderr_msg}")
                if platform.system() == "Linux":
                    gpus.extend(await self._detect_with_lspci_amd())
        except FileNotFoundError:
            logger.info(f"{rocm_smi_path} not found. Falling back for AMD GPU detection.")
            if platform.system() == "Linux":
                gpus.extend(await self._detect_with_lspci_amd())
        except Exception as e:
            logger.error(f"Error during initial AMD rocm-smi check: {e}")
            if platform.system() == "Linux":
                gpus.extend(await self._detect_with_lspci_amd())
        
        if not gpus and platform.system() == "Windows":
            logger.info("AMD detection on Windows is currently basic and relies on future wmic/dxdiag integration or AMD ADL SDK.")
            # Placeholder for potential WMIC-based AMD detection if added
        
        # Add ROCm version if detected (simplistic for now)
        if gpus and any(gpu.vendor == VendorType.AMD for gpu in gpus):
             rocm_ver = self._get_rocm_version(rocm_smi_path)
             for gpu in gpus:
                 if gpu.vendor == VendorType.AMD and not gpu.rocm_version:
                     gpu.rocm_version = rocm_ver if rocm_ver else ""
        return gpus

    def _get_rocm_version(self, rocm_smi_path: str) -> Optional[str]:
        try:
            # rocm-smi --version might give SMI version, not ROCm stack version easily.
            # Check /opt/rocm/.info/version-utils or rocm_agent_enumerator
            # For now, let's assume rocm-smi --showdriverversion provides a useful ROCm version string
            proc = subprocess.run([rocm_smi_path, '--showdriverversion'], capture_output=True, text=True, timeout=5, check=True)
            output = proc.stdout.strip()
            # Example "Driver version: 5.7.31050" or similar; needs robust parsing.
            # Here, we'll try to extract a version-like string.
            match = re.search(r"ROCm version: ([\d\.]+)|Driver version: ([\d\.]+)", output, re.IGNORECASE)
            if match:
                return match.group(1) or match.group(2)
        except Exception as e:
            logger.debug(f"Could not determine ROCm version via {rocm_smi_path}: {e}")
        # Alternative for Linux:
        if platform.system() == "Linux":
            try:
                with open("/opt/rocm/.info/version", "r") as f: # Common location
                    return f.read().strip()
            except FileNotFoundError:
                logger.debug("/opt/rocm/.info/version not found.")
            except Exception as e:
                logger.debug(f"Error reading ROCm version file: {e}")
        return None

    async def _detect_with_rocm_smi(self, rocm_smi_path: str) -> List[GPUInfo]:
        gpus_info = []
        try:
            # Attempt to get JSON output for all devices in one go if rocm-smi supports it
            json_proc = await asyncio.create_subprocess_exec(rocm_smi_path, '--showallinfo', '--json',
                                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_json, stderr_json = await json_proc.communicate()

            if json_proc.returncode == 0:
                try:
                    all_device_data = json.loads(stdout_json.decode())
                    idx_counter = 0
                    for card_key, data in all_device_data.items():
                        if not card_key.startswith("card"): continue # Process only GPU entries

                        name = data.get("Card series", data.get("Card model", "AMD GPU"))
                        uuid_val = data.get("Unique ID", f"amd-rocm-{idx_counter}")
                        pci_bus_id_val = data.get("PCI Bus", "unknown")
                        
                        mem_total_str = data.get("VRAM Total Memory (B)")
                        mem_used_str = data.get("VRAM Total Used Memory (B)")
                        mem_total_mb = int(mem_total_str) // (1024**2) if mem_total_str and mem_total_str.isdigit() else 0
                        mem_used_mb = int(mem_used_str) // (1024**2) if mem_used_str and mem_used_str.isdigit() else 0
                        mem_free_mb = mem_total_mb - mem_used_mb

                        power_limit_str = data.get("Max Graphics Package Power (W)") # Needs parsing from string like "250.0 W"
                        power_limit_val = 0
                        if power_limit_str:
                            match_power = re.search(r'([\d\.]+)\s*W', power_limit_str)
                            if match_power: power_limit_val = int(float(match_power.group(1)))
                        
                        temp_str = data.get("Temperature (Sensor edge) (C)")
                        temp_val = 0
                        if temp_str:
                            match_temp = re.search(r'([\d\.]+)', temp_str)
                            if match_temp: temp_val = int(float(match_temp.group(1)))
                        
                        max_sclk_str = data.get("Max Graphics Clock (MHz)") # needs parsing from something like "Current sclk: 700Mhz" - json output might be better
                        max_sclk_val = 0
                        if max_sclk_str and isinstance(max_sclk_str, str): # Handle if it's already a number or a string.
                           sclk_m = re.search(r'(\d+)', max_sclk_str)
                           if sclk_m: max_sclk_val = int(sclk_m.group(1))
                        elif isinstance(max_sclk_str, (int, float)):
                           max_sclk_val = int(max_sclk_str)
                           
                        gfx_arch = self._get_amd_gfx_arch(name) # Or parse from rocm-smi if it provides GFX version

                        gpu = GPUInfo(
                            name=name, vendor=VendorType.AMD, gpu_index=idx_counter, uuid=uuid_val, pci_bus_id=pci_bus_id_val,
                            memory_total=mem_total_mb, memory_free=mem_free_mb,
                            memory_bandwidth=self._get_amd_memory_bandwidth(name),
                            compute_capability=gfx_arch,
                            rocm_cus=self._get_amd_cus(name),
                            boost_clock=max_sclk_val, # Assuming max sclk is boost
                            performance_tier=self._determine_amd_performance_tier(name),
                            power_limit=power_limit_val,
                            temperature=temp_val,
                            driver_version=data.get("Driver version",""), # JSON might have a top-level driver version
                            supported_apis=[ComputeCapability.ROCM, ComputeCapability.OPENCL, ComputeCapability.VULKAN],
                        )
                        gpus_info.append(gpu)
                        idx_counter +=1
                    if gpus_info: return gpus_info # Successfully parsed JSON
                except json.JSONDecodeError as jde:
                    logger.warning(f"rocm-smi JSON output parsing failed: {jde}. Falling back to iterative query.")
                except Exception as e: # Catch other parsing errors
                    logger.warning(f"Unexpected error parsing rocm-smi JSON: {e}. Falling back.")


            # Fallback to iterative calls if JSON fails or isn't comprehensive enough
            # (Simplified; real implementation should be more robust with iterative calls as in original gendata.py)
            logger.info("ROCm-SMI JSON detection incomplete or failed, full iterative parsing for AMD is complex and omitted for this refactor pass, lspci might pick up some info.")
            # Original iterative detection logic from gendata.py could be placed here as a fallback
            # but it was quite complex and prone to rocm-smi output format changes.
            # For this refactor, if JSON fails, we'll rely on lspci or assume no further info from rocm-smi.

        except Exception as e:
            logger.error(f"❌ rocm-smi detection failed: {e}")
        return gpus_info

    async def _detect_with_lspci_amd(self) -> List[GPUInfo]:
        gpus = []
        try:
            process = await asyncio.create_subprocess_shell(
                "lspci -nnk | grep -Ei 'VGA compatible controller.*AMD/ATI|Display controller.*AMD/ATI|3D controller.*AMD/ATI'",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                executable='/bin/bash' if platform.system() == "Linux" else None # ensure shell context if using pipe
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                output = stdout.decode(errors='ignore').strip()
                gpu_idx = 0
                for line_block_match in re.finditer(r"^(\S+)\s+(VGA compatible controller|Display controller|3D controller).*?\[(1002):([0-9a-fA-F]{4})\]", output, re.MULTILINE):
                    pci_bus_id = line_block_match.group(1)
                    device_id_hex = line_block_match.group(4) # Keep as hex string
                    
                    full_line = line_block_match.group(0) # The matched line itself
                    name_match = re.search(r"AMD/ATI\]\s*([^\[\(]+)", full_line) # More careful name extraction
                    gpu_name_from_lspci = name_match.group(1).strip() if name_match else f"AMD Radeon {device_id_hex}"
                    
                    # Try to find Kernel driver in use in the block related to this PCI ID
                    # This requires parsing multi-line lspci -nnk output blocks
                    # Simplified for now:
                    driver_ver = "unknown"

                    gpu = GPUInfo(
                        name=gpu_name_from_lspci, vendor=VendorType.AMD, gpu_index=gpu_idx,
                        uuid=f"amd-pci-{pci_bus_id}-{device_id_hex}", pci_bus_id=pci_bus_id,
                        memory_total=self._get_amd_memory_estimate(gpu_name_from_lspci, device_id_hex), memory_free=0,
                        memory_bandwidth=self._get_amd_memory_bandwidth(gpu_name_from_lspci, device_id_hex),
                        compute_capability=self._get_amd_gfx_arch(gpu_name_from_lspci, device_id_hex),
                        rocm_cus=self._get_amd_cus(gpu_name_from_lspci, device_id_hex),
                        performance_tier=self._determine_amd_performance_tier(gpu_name_from_lspci),
                        driver_version=driver_ver, 
                        supported_apis=[ComputeCapability.OPENCL, ComputeCapability.VULKAN]
                    )
                    gpus.append(gpu)
                    gpu_idx += 1
            else:
                 logger.debug(f"lspci command for AMD failed or no AMD GPUs found. stderr: {stderr.decode(errors='ignore').strip()}")
        except FileNotFoundError:
            logger.info("lspci not found. AMD GPU detection via lspci not possible on this system.")
        except Exception as e:
            logger.error(f"❌ AMD lspci detection failed: {e}")
        return gpus

    def _get_amd_memory_bandwidth(self, gpu_name: str, device_id: Optional[str] = None) -> float:
        bandwidth_map = {
            'RX 7900 XTX': 960.0, 'RX 7900 XT': 800.0, 'RX 7800 XT': 624.0, 'RX 7700 XT': 432.0,
            'RX 6950 XT': 576.0, 'RX 6900 XT': 512.0, 'RX 6800 XT': 512.0, 'RX 6800': 512.0, 'RX 6700 XT': 384.0, 'RX 6600 XT': 256.0,
            'MI300X': 5200.0, 'MI250X': 3200.0, 'MI210': 1600.0, 'MI100': 1228.0,
            'Radeon VII': 1024.0, 'Vega 64': 483.8, 'Vega 56': 409.6
        }
        for model, bw in bandwidth_map.items():
            if model in gpu_name: return bw
        return 0.0
    
    def _get_amd_memory_estimate(self, gpu_name: str, device_id: Optional[str] = None) -> int:
        mem_map = { 
            'RX 7900 XTX': 24 * 1024, 'RX 7900 XT': 20 * 1024, 'RX 7800 XT': 16 * 1024, 'RX 7700 XT': 12*1024,
            'RX 6950 XT': 16 * 1024, 'RX 6900 XT': 16 * 1024, 'RX 6800 XT': 16 * 1024, 'RX 6800': 16 * 1024, 'RX 6700 XT': 12 * 1024, 'RX 6600 XT': 8 * 1024,
            'MI300X': 192 * 1024, 'MI250X': 128 * 1024, 'MI100': 32*1024,
            'Radeon VII': 16*1024, 'Vega 64': 8*1024, 'Vega 56': 8*1024
        }
        for model, mem in mem_map.items():
            if model in gpu_name: return mem
        if "Radeon Pro W" in gpu_name: return 16 * 1024 
        return 0 

    def _get_amd_gfx_arch(self, gpu_name: str, device_id: Optional[str] = None) -> str:
        name_lower = gpu_name.lower()
        if "mi300" in name_lower or (device_id and device_id in ["740f", "740c"]): return "gfx94x" # CDNA3
        if "mi2" in name_lower or (device_id and device_id in ["738c", "7460", "7461"]): return "gfx90a" # CDNA2 (MI2xx)
        if "mi100" in name_lower or (device_id and device_id == "7388"): return "gfx908" # CDNA1

        if any(s in name_lower for s in ["rx 7900", "rx 7800", "rx 7700", "navi31", "navi32", "navi33"]) or \
           (device_id and device_id.startswith("74")): # Navi 3x, RDNA3
            if device_id in ["744c"]: return "gfx1100" # Navi 31
            if device_id in ["7445"]: return "gfx1101" # Navi 32
            if device_id in ["746a"]: return "gfx1102" # Navi 33
            return "gfx110x" # RDNA3 general
        if any(s in name_lower for s in ["rx 69", "rx 68", "rx 67", "rx 66", "navi21", "navi22", "navi23"]) or \
           (device_id and device_id.startswith("73")): # Navi 2x, RDNA2
            if device_id in ["73bf", "73a5"]: return "gfx1030" # Navi 21
            if device_id in ["73df"]: return "gfx1031" # Navi 22
            if device_id in ["73ef"]: return "gfx1032" # Navi 23
            return "gfx103x" # RDNA2 general
        if any(s in name_lower for s in ["rx 5700", "rx 5600", "rx 5500", "navi10", "navi12", "navi14"]) or \
           (device_id and device_id.startswith("731") or device_id.startswith("734")): # Navi 1x, RDNA1
            if device_id in ["731f"]: return "gfx1010" # Navi 10
            # Add more for Navi 12, 14
            return "gfx101x" # RDNA1 general
        if "radeon vii" in name_lower or "vega 20" in name_lower or (device_id and device_id == "66a0"): return "gfx906" # Vega20
        if any(s in name_lower for s in ["vega 64", "vega 56"]) or (device_id and device_id == "687f"): return "gfx900" # Vega10
        if any(s in name_lower for s in ["rx 580", "rx 570", "polaris"]) or (device_id and (device_id.startswith("67df") or device_id.startswith("67ef"))): return "gfx803" # Polaris

        return "unknown"

    def _get_amd_cus(self, gpu_name: str, device_id: Optional[str] = None) -> Optional[int]:
        cu_map = {
            'RX 7900 XTX': 96, 'RX 7900 XT': 84, 'RX 7800 XT': 60, 'RX 7700 XT': 54,
            'RX 6950 XT': 80, 'RX 6900 XT': 80, 'RX 6800 XT': 72, 'RX 6800': 60, 'RX 6700 XT': 40, 'RX 6600 XT': 32,
            'MI300X': 304, 'MI250X': 220, 'MI100': 120,
            'Radeon VII': 60, 'Vega 64': 64, 'Vega 56': 56,
            'RX 580': 36
        }
        for model, cus in cu_map.items():
            if model in gpu_name: return cus
        return None

    def _determine_amd_performance_tier(self, gpu_name: str) -> PerformanceTier:
        name_upper = gpu_name.upper()
        if any(card in name_upper for card in ['MI300', 'MI250', 'MI210', 'MI100', 'W7900', 'W6800', 'RADEON PRO WX 9100', 'RADEON INSTINCT']): return PerformanceTier.ENTERPRISE
        if any(card in name_upper for card in ['RX 7900 XTX', 'RX 6950 XT', 'RX 6900 XT', 'W7800', 'RADEON VII', 'VEGA 64']): return PerformanceTier.PROFESSIONAL
        if any(card in name_upper for card in ['RX 7900 XT', 'RX 6800 XT', 'RX 7800 XT', 'RX 5700 XT']): return PerformanceTier.ENTHUSIAST
        if any(card in name_upper for card in ['RX 6750 XT', 'RX 6700 XT', 'RX 7700 XT', 'RX 6800', 'VEGA 56']): return PerformanceTier.MAINSTREAM
        if any(card in name_upper for card in ['RX 6600 XT', 'RX 6600', 'RX 7600', 'RX 580', 'RX 570', 'RX 5500 XT']): return PerformanceTier.BUDGET
        if "RADEON GRAPHICS" in name_upper and not any(pro_tier_indicator in name_upper for pro_tier_indicator in ["XTX", "XT", "MI", "PRO W"]): return PerformanceTier.INTEGRATED
        return PerformanceTier.UNKNOWN

    def get_telemetry(self, gpu: GPUInfo) -> TelemetryData:
        rocm_smi_path = "rocm-smi"
        try:
            # Query relevant metrics using JSON if possible for robustness
            cmd = [rocm_smi_path, '-d', str(gpu.gpu_index), 
                   '--showuse', '--showtemp', '--showpower', '--showclocks', '--showfan', '--json']
            result_proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # This needs to be sync for the current signature of get_telemetry
            # To run async command synchronously (for now, ideally get_telemetry itself would be async):
            # This is a workaround; in a true async design, HardwareManager's telemetry loop would await this.
            # For simplicity, using blocking subprocess.run if this part needs to stay sync:
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3)
            data_all = json.loads(result.stdout)
            
            # The JSON structure for rocm-smi can be like: {"cardX": {metrics...}}
            # We need to find the data for the specific card_index.
            card_key = f"card{gpu.gpu_index}"
            data = data_all.get(card_key, {})

            gpu_util = float(data.get("GPU use (%)", 0.0))
            
            vram_total_b_str = data.get("VRAM Total Memory (B)")
            vram_used_b_str = data.get("VRAM Total Used Memory (B)")
            mem_util = 0.0
            if vram_total_b_str and vram_used_b_str and vram_total_b_str.isdigit() and vram_used_b_str.isdigit():
                vram_total_b, vram_used_b = int(vram_total_b_str), int(vram_used_b_str)
                if vram_total_b > 0: mem_util = (vram_used_b / vram_total_b) * 100

            temp_str = data.get("Temperature (Sensor edge) (C)", str(gpu.temperature))
            temp_val = float(re.search(r'([\d\.]+)', temp_str).group(1)) if re.search(r'([\d\.]+)', temp_str) else gpu.temperature

            power_str = data.get("Average Graphics Package Power (W)", "0.0 W")
            power_val = float(re.search(r'([\d\.]+)', power_str).group(1)) if re.search(r'([\d\.]+)', power_str) else 0.0
            
            fan_str = data.get("Fan Speed (%)", "N/A")
            fan_val = None
            if fan_str != "N/A":
                fan_m = re.search(r'(\d+)', fan_str)
                if fan_m: fan_val = float(fan_m.group(1))

            sclk_str = data.get("Current Graphics Clock (MHz)") # or average, or peak?
            mclk_str = data.get("Current Memory Clock (MHz)")
            sclk_val = int(sclk_str) if sclk_str and sclk_str.isdigit() else None
            mclk_val = int(mclk_str) if mclk_str and mclk_str.isdigit() else None


            return TelemetryData(
                timestamp=datetime.now(), gpu_utilization=gpu_util, memory_utilization=mem_util,
                temperature=temp_val, power_draw=power_val, fan_speed=fan_val,
                clock_graphics=sclk_val, clock_memory=mclk_val
            )

        except FileNotFoundError:
            logger.debug(f"{rocm_smi_path} not found for telemetry.")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.debug(f"rocm-smi telemetry call failed for AMD GPU {gpu.name}: {e}")
        except json.JSONDecodeError as e:
            logger.debug(f"rocm-smi JSON telemetry parsing error for {gpu.name}: {e}")
        except Exception as e:
            logger.error(f"❌ AMD telemetry for {gpu.name} failed: {e}")
        return TelemetryData(datetime.now(), 0, 0, float(gpu.temperature), 0)
