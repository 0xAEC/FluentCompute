# fluentcompute/detectors/amd_detector.py
import asyncio
import subprocess
import re
import platform
import json # For loading specs
from pathlib import Path # For loading specs
from typing import List, Optional, Any, Dict
from datetime import datetime

from fluentcompute.models import GPUInfo, TelemetryData, VendorType, PerformanceTier, ComputeCapability
from .base import HardwareDetector
from fluentcompute.utils.logging_config import logger

class AMDDetector(HardwareDetector):
    _SPEC_FILE_NAME = "amd_specs.json"

    def __init__(self):
        self.specs: Dict[str, Any] = self._load_specs()

    def _load_specs(self) -> Dict[str, Any]:
        try:
            spec_path = Path(__file__).resolve().parent.parent / "data" / "specs" / self._SPEC_FILE_NAME
            if not spec_path.exists():
                logger.warning(f"AMD spec file not found at {spec_path}. Using empty specs.")
                return {}
            with open(spec_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load AMD specs from {self._SPEC_FILE_NAME}: {e}", exc_info=True)
            return {}

    async def detect_hardware(self) -> List[GPUInfo]:
        gpus = []
        rocm_smi_path = "rocm-smi" 

        try:
            process = await asyncio.create_subprocess_exec(rocm_smi_path, '-d', '0', '--showid',
                                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await process.communicate()
            if process.returncode == 0 :
                logger.info("ROCm SMI detected. Attempting AMD GPU detection via rocm-smi.")
                gpus.extend(await self._detect_with_rocm_smi(rocm_smi_path))
            else:
                stderr_msg = stderr.decode(errors='ignore').strip()
                logger.info(f"rocm-smi not fully functional. Fallback initiated. Error: {stderr_msg}")
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
            logger.info("AMD detection on Windows is currently basic.")
            
        if gpus and any(gpu.vendor == VendorType.AMD for gpu in gpus):
             rocm_ver = self._get_rocm_version(rocm_smi_path)
             for gpu in gpus:
                 if gpu.vendor == VendorType.AMD and not gpu.rocm_version:
                     gpu.rocm_version = rocm_ver if rocm_ver else ""
        return gpus

    def _get_rocm_version(self, rocm_smi_path: str) -> Optional[str]:
        try:
            proc = subprocess.run([rocm_smi_path, '--showdriverversion'], capture_output=True, text=True, timeout=5, check=True)
            output = proc.stdout.strip()
            match = re.search(r"ROCm version: ([\d\.]+)|Driver version: ([\d\.]+)", output, re.IGNORECASE)
            if match:
                return match.group(1) or match.group(2)
        except Exception as e:
            logger.debug(f"Could not determine ROCm version via {rocm_smi_path}: {e}")
        if platform.system() == "Linux":
            try:
                with open("/opt/rocm/.info/version", "r") as f:
                    return f.read().strip()
            except FileNotFoundError: logger.debug("/opt/rocm/.info/version not found.")
            except Exception as e: logger.debug(f"Error reading ROCm version file: {e}")
        return None

    async def _detect_with_rocm_smi(self, rocm_smi_path: str) -> List[GPUInfo]:
        gpus_info = []
        try:
            json_proc = await asyncio.create_subprocess_exec(rocm_smi_path, '--showallinfo', '--json',
                                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_json, stderr_json = await json_proc.communicate()

            if json_proc.returncode == 0:
                try:
                    all_device_data = json.loads(stdout_json.decode())
                    idx_counter = 0
                    for card_key, data in all_device_data.items():
                        if not card_key.startswith("card"): continue

                        name = data.get("Card series", data.get("Card model", "AMD GPU"))
                        uuid_val = data.get("Unique ID", f"amd-rocm-{idx_counter}")
                        pci_bus_id_val = data.get("PCI Bus", "unknown")
                        
                        mem_total_str = data.get("VRAM Total Memory (B)")
                        mem_used_str = data.get("VRAM Total Used Memory (B)")
                        mem_total_mb = int(mem_total_str) // (1024**2) if mem_total_str and mem_total_str.isdigit() else 0
                        mem_used_mb = int(mem_used_str) // (1024**2) if mem_used_str and mem_used_str.isdigit() else 0
                        mem_free_mb = mem_total_mb - mem_used_mb

                        power_limit_val = 0
                        power_limit_str = data.get("Max Graphics Package Power (W)")
                        if power_limit_str:
                            match_power = re.search(r'([\d\.]+)\s*W', power_limit_str)
                            if match_power: power_limit_val = int(float(match_power.group(1)))
                        
                        temp_val = 0
                        temp_str = data.get("Temperature (Sensor edge) (C)")
                        if temp_str:
                            match_temp = re.search(r'([\d\.]+)', temp_str)
                            if match_temp: temp_val = int(float(match_temp.group(1)))
                        
                        max_sclk_val = 0
                        max_sclk_str = data.get("Max Graphics Clock (MHz)")
                        if max_sclk_str and isinstance(max_sclk_str, str):
                           sclk_m = re.search(r'(\d+)', max_sclk_str)
                           if sclk_m: max_sclk_val = int(sclk_m.group(1))
                        elif isinstance(max_sclk_str, (int, float)):
                           max_sclk_val = int(max_sclk_str)
                           
                        gfx_arch = self._get_amd_gfx_arch(name) 

                        gpu = GPUInfo(
                            name=name, vendor=VendorType.AMD, gpu_index=idx_counter, uuid=uuid_val, pci_bus_id=pci_bus_id_val,
                            memory_total=mem_total_mb, memory_free=mem_free_mb,
                            memory_bandwidth=self._get_amd_memory_bandwidth(name),
                            compute_capability=gfx_arch,
                            rocm_cus=self._get_amd_cus(name),
                            boost_clock=max_sclk_val,
                            performance_tier=self._determine_amd_performance_tier(name),
                            power_limit=power_limit_val,
                            temperature=temp_val,
                            driver_version=data.get("Driver version",""),
                            supported_apis=[ComputeCapability.ROCM, ComputeCapability.OPENCL, ComputeCapability.VULKAN],
                        )
                        gpus_info.append(gpu)
                        idx_counter +=1
                    if gpus_info: return gpus_info
                except json.JSONDecodeError as jde:
                    logger.warning(f"rocm-smi JSON output parsing failed: {jde}. Falling back.")
                except Exception as e:
                    logger.warning(f"Unexpected error parsing rocm-smi JSON: {e}. Falling back.")
            else:
                 logger.info("ROCm-SMI JSON detection incomplete or failed.")
        except Exception as e:
            logger.error(f"❌ rocm-smi detection failed: {e}")
        return gpus_info

    async def _detect_with_lspci_amd(self) -> List[GPUInfo]:
        gpus = []
        try:
            process = await asyncio.create_subprocess_shell(
                "lspci -nnk | grep -Ei 'VGA compatible controller.*AMD/ATI|Display controller.*AMD/ATI|3D controller.*AMD/ATI'",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                executable='/bin/bash' if platform.system() == "Linux" else None
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                output = stdout.decode(errors='ignore').strip()
                gpu_idx = 0
                for line_block_match in re.finditer(r"^(\S+)\s+(VGA compatible controller|Display controller|3D controller).*?\[(1002):([0-9a-fA-F]{4})\]", output, re.MULTILINE):
                    pci_bus_id = line_block_match.group(1)
                    device_id_hex = line_block_match.group(4).lower()
                    
                    full_line = line_block_match.group(0)
                    name_match = re.search(r"AMD/ATI\]\s*([^\[\(]+)", full_line)
                    gpu_name_from_lspci = name_match.group(1).strip() if name_match else f"AMD Radeon {device_id_hex}"
                    
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
                 logger.debug(f"lspci command for AMD failed. stderr: {stderr.decode(errors='ignore').strip()}")
        except FileNotFoundError:
            logger.info("lspci not found for AMD detection.")
        except Exception as e:
            logger.error(f"❌ AMD lspci detection failed: {e}")
        return gpus

    def _get_spec_value_by_name(self, gpu_name: str, spec_map_key: str, default_val: Any = 0) -> Any:
        spec_map = self.specs.get(spec_map_key, {})
        for model, val in spec_map.items():
            if model in gpu_name:
                return val
        return default_val

    def _get_amd_memory_bandwidth(self, gpu_name: str, device_id: Optional[str] = None) -> float:
        return self._get_spec_value_by_name(gpu_name, "bandwidth_map", 0.0)
    
    def _get_amd_memory_estimate(self, gpu_name: str, device_id: Optional[str] = None) -> int:
        mem_val = self._get_spec_value_by_name(gpu_name, "memory_map", 0)
        if mem_val == 0 and "Radeon Pro W" in gpu_name: # Handle special cases from spec.
            return self.specs.get("memory_map", {}).get("Radeon Pro W",0)
        return mem_val

    def _get_amd_gfx_arch(self, gpu_name: str, device_id: Optional[str] = None) -> str:
        name_lower = gpu_name.lower()
        
        gfx_arch_map_name = self.specs.get("gfx_arch_map_name", {})
        for name_key, arch in gfx_arch_map_name.items():
            if name_key in name_lower:
                return arch
        
        if device_id:
            device_id_lower = device_id.lower()
            gfx_arch_map_id = self.specs.get("gfx_arch_map_id", {})
            if device_id_lower in gfx_arch_map_id:
                return gfx_arch_map_id[device_id_lower]
            
            gfx_arch_id_prefix = self.specs.get("gfx_arch_id_prefix", {})
            for prefix, arch in gfx_arch_id_prefix.items():
                 if device_id_lower.startswith(prefix):
                     return arch
        return "unknown"

    def _get_amd_cus(self, gpu_name: str, device_id: Optional[str] = None) -> Optional[int]:
        return self._get_spec_value_by_name(gpu_name, "cu_map", None)

    def _determine_amd_performance_tier(self, gpu_name: str) -> PerformanceTier:
        name_upper = gpu_name.upper()
        # This logic uses fixed lists, which could also be externalized but is more complex than simple maps.
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
            cmd = [rocm_smi_path, '-d', str(gpu.gpu_index), 
                   '--showuse', '--showtemp', '--showpower', '--showclocks', '--showfan', '--json']
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3)
            data_all = json.loads(result.stdout)
            
            card_key = f"card{gpu.gpu_index}"
            data = data_all.get(card_key, {})

            gpu_util = float(data.get("GPU use (%)", 0.0))
            
            mem_util = 0.0
            vram_total_b_str = data.get("VRAM Total Memory (B)")
            vram_used_b_str = data.get("VRAM Total Used Memory (B)")
            if vram_total_b_str and vram_used_b_str and vram_total_b_str.isdigit() and vram_used_b_str.isdigit():
                vram_total_b, vram_used_b = int(vram_total_b_str), int(vram_used_b_str)
                if vram_total_b > 0: mem_util = (vram_used_b / vram_total_b) * 100

            temp_str = data.get("Temperature (Sensor edge) (C)", str(gpu.temperature or 0.0))
            temp_val = float(re.search(r'([\d\.]+)', temp_str).group(1)) if re.search(r'([\d\.]+)', temp_str) else (gpu.temperature or 0.0)

            power_str = data.get("Average Graphics Package Power (W)", "0.0 W")
            power_val = float(re.search(r'([\d\.]+)', power_str).group(1)) if re.search(r'([\d\.]+)', power_str) else 0.0
            
            fan_val = None
            fan_str = data.get("Fan Speed (%)", "N/A")
            if fan_str != "N/A":
                fan_m = re.search(r'(\d+)', fan_str)
                if fan_m: fan_val = float(fan_m.group(1))

            sclk_str = data.get("Current Graphics Clock (MHz)")
            mclk_str = data.get("Current Memory Clock (MHz)")
            sclk_val = int(sclk_str) if sclk_str and sclk_str.isdigit() else None
            mclk_val = int(mclk_str) if mclk_str and mclk_str.isdigit() else None

            return TelemetryData(
                timestamp=datetime.now(), gpu_utilization=gpu_util, memory_utilization=mem_util,
                temperature=temp_val, power_draw=power_val, fan_speed=fan_val,
                clock_graphics=sclk_val, clock_memory=mclk_val
            )
        except FileNotFoundError: logger.debug(f"{rocm_smi_path} not found for telemetry.")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.debug(f"rocm-smi telemetry call failed for AMD GPU {gpu.name}: {e}")
        except json.JSONDecodeError as e:
            logger.debug(f"rocm-smi JSON telemetry parsing error for {gpu.name}: {e}")
        except Exception as e:
            logger.error(f"❌ AMD telemetry for {gpu.name} failed: {e}")
        return TelemetryData(datetime.now(), 0, 0, float(gpu.temperature or 0.0), 0)
