# fluentcompute/detectors/intel_detector.py
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

class IntelDetector(HardwareDetector):
    _SPEC_FILE_NAME = "intel_specs.json"

    def __init__(self):
        self.specs: Dict[str, Any] = self._load_specs()

    def _load_specs(self) -> Dict[str, Any]:
        try:
            spec_path = Path(__file__).resolve().parent.parent / "data" / "specs" / self._SPEC_FILE_NAME
            if not spec_path.exists():
                logger.warning(f"Intel spec file not found at {spec_path}. Using empty specs.")
                return {}
            with open(spec_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load Intel specs from {self._SPEC_FILE_NAME}: {e}", exc_info=True)
            return {}

    async def detect_hardware(self) -> List[GPUInfo]:
        gpus = []
        if platform.system() == "Linux":
            gpus.extend(await self._detect_with_lspci_intel())
            if gpus:
                await self._enrich_with_oneapi_tools(gpus)
        elif platform.system() == "Windows":
            gpus.extend(await self._detect_with_wmic_windows())
        
        logger.info(f"Intel detector found {len(gpus)} potential devices.")
        return gpus

    async def _detect_with_lspci_intel(self) -> List[GPUInfo]:
        gpus_info_list = []
        try:
            cmd = "lspci -nnk | grep -Ei 'VGA compatible controller.*Intel|Display controller.*Intel|3D controller.*Intel|Processing accelerators.*Intel'"
            process = await asyncio.create_subprocess_shell(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                executable='/bin/bash'
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                output = stdout.decode(errors='ignore').strip()
                gpu_idx_counter = 0
                device_blocks = []
                current_block_lines = []
                for line in output.splitlines():
                    if re.match(r"^\S+:\S+\.\S+", line) and current_block_lines:
                        device_blocks.append("\n".join(current_block_lines))
                        current_block_lines = [line]
                    else:
                        current_block_lines.append(line)
                if current_block_lines: device_blocks.append("\n".join(current_block_lines))
                
                for block in device_blocks:
                    intel_match = re.search(r"Intel Corporation", block)
                    pci_match = re.search(r"^(\S+)\s+(VGA compatible controller|3D controller|Display controller|Processing accelerators) \[.*?\]:\s*(?:Intel(?: Corporation)?)\s*(.*?)\s*\[(8086):(\S+)\]", block, re.MULTILINE | re.IGNORECASE)

                    if not intel_match or not pci_match : continue
                        
                    pci_bus_id = pci_match.group(1)
                    raw_name_from_lspci = pci_match.group(3).strip()
                    device_id = pci_match.group(5).lower()
                    
                    name = self._refine_intel_gpu_name(raw_name_from_lspci, device_id)
                    driver_match = re.search(r"Kernel driver in use:\s*(\S+)", block)
                    driver = driver_match.group(1) if driver_match else "unknown"
                    
                    apis = [ComputeCapability.OPENCL, ComputeCapability.VULKAN]

                    gpu_info = GPUInfo(
                        name=name, vendor=VendorType.INTEL, gpu_index=gpu_idx_counter,
                        uuid=f"intel-pci-{pci_bus_id}-{device_id}", pci_bus_id=pci_bus_id,
                        memory_total=self._get_intel_memory_estimate(name, device_id), memory_free=0,
                        memory_bandwidth=self._get_intel_memory_bandwidth(name, device_id),
                        compute_capability=self._get_intel_gen_arch(name, device_id),
                        intel_eus=self._get_intel_eus(name, device_id),
                        performance_tier=self._determine_intel_performance_tier(name),
                        driver_version=driver, supported_apis=apis 
                    )
                    gpus_info_list.append(gpu_info)
                    gpu_idx_counter += 1
            else:
                 logger.debug(f"lspci for Intel failed. stderr: {stderr.decode(errors='ignore').strip()}")
        except FileNotFoundError: logger.info("lspci not found for Intel detection.")
        except Exception as e: logger.error(f"❌ Intel lspci detection failed: {e}")
        return gpus_info_list
    
    async def _detect_with_wmic_windows(self) -> List[GPUInfo]:
        gpus_info_list = []
        try:
            cmd = 'wmic path Win32_VideoController get Name,AdapterCompatibility,PNPDeviceID,DriverVersion,AdapterRAM /format:csv'
            process = await asyncio.create_subprocess_shell(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                output = stdout.decode('utf-8', errors='ignore').strip()
                lines = output.splitlines()
                if len(lines) > 1:
                    header = lines[0].lower().split(',')
                    try:
                        name_idx, compat_idx, pnp_idx, driver_idx, ram_idx = (
                            header.index("name"), header.index("adaptercompatibility"),
                            header.index("pnpdeviceid"), header.index("driverversion"), header.index("adapterram")
                        )
                    except ValueError:
                        logger.error("WMIC output CSV header missing expected columns.")
                        return []

                    gpu_idx_counter = 0
                    for line_content in lines[1:]:
                        parts = line_content.split(',')
                        if len(parts) < max(name_idx, compat_idx, pnp_idx, driver_idx, ram_idx) + 1: continue

                        adapter_compat = parts[compat_idx].strip()
                        raw_name_from_wmic = parts[name_idx].strip()

                        if "intel" not in adapter_compat.lower() and "intel" not in raw_name_from_wmic.lower(): continue 
                        
                        pnp_id = parts[pnp_idx].strip()
                        driver_ver = parts[driver_idx].strip()
                        adapter_ram_str = parts[ram_idx].strip()

                        dev_id_match = re.search(r"DEV_(\S{4})", pnp_id)
                        device_id = dev_id_match.group(1).lower() if dev_id_match else None
                        name = self._refine_intel_gpu_name(raw_name_from_wmic, device_id)
                        
                        adapter_ram_mb = int(adapter_ram_str) // (1024 * 1024) if adapter_ram_str else 0
                        
                        mem_total = self._get_intel_memory_estimate(name, device_id)
                        if mem_total == 0: mem_total = adapter_ram_mb

                        apis = [ComputeCapability.OPENCL, ComputeCapability.DIRECTML, ComputeCapability.VULKAN]
                        if "oneAPI Level Zero" in driver_ver or "Intel Graphics Driver" in driver_ver :
                            apis.append(ComputeCapability.ONEAPI)
                        
                        gpu_info = GPUInfo(
                            name=name, vendor=VendorType.INTEL, gpu_index=gpu_idx_counter,
                            uuid=f"intel-wmic-{pnp_id.replace('PCI\\\\','').replace('&','_')}",
                            pci_bus_id="unknown",
                            memory_total=mem_total, memory_free=0,
                            memory_bandwidth=self._get_intel_memory_bandwidth(name, device_id),
                            compute_capability=self._get_intel_gen_arch(name, device_id),
                            intel_eus=self._get_intel_eus(name, device_id),
                            performance_tier=self._determine_intel_performance_tier(name),
                            driver_version=driver_ver, supported_apis=apis
                        )
                        gpus_info_list.append(gpu_info)
                        gpu_idx_counter += 1
            else:
                logger.debug(f"WMIC for Intel failed. Stderr: {stderr.decode(errors='ignore')}")
        except FileNotFoundError: logger.info("WMIC not found for Intel detection.")
        except Exception as e: logger.error(f"❌ Intel WMIC detection failed: {e}")
        return gpus_info_list

    def _refine_intel_gpu_name(self, raw_name: str, device_id: Optional[str]) -> str:
        if device_id:
            dev_id_map = self.specs.get("name_refinement_device_id", {})
            if device_id in dev_id_map: return dev_id_map[device_id]

            dev_id_prefix_map = self.specs.get("name_refinement_device_id_prefix", {})
            for prefix, name_pattern in dev_id_prefix_map.items():
                if device_id.startswith(prefix.lower()): # ensure spec prefix matches case behavior
                    return f"{name_pattern} ({device_id})"

        raw_name_map = self.specs.get("name_refinement_raw_name_contains", {})
        for keyword, name_pattern in raw_name_map.items():
            if keyword in raw_name:
                 return name_pattern if name_pattern else raw_name # If pattern is null, use raw_name

        return raw_name if raw_name else f"Intel Graphics ({device_id or 'Unknown'})"

    async def _enrich_with_oneapi_tools(self, gpus: List[GPUInfo]):
        sycl_ls_path = "sycl-ls"
        try:
            proc = await asyncio.create_subprocess_exec(sycl_ls_path, "--verbose",
                                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                output = stdout.decode(errors='ignore')
                if "level_zero:gpu" in output.lower():
                    l0_driver_ver_match = re.search(r"Level-Zero driver\s+([\d\.]+)", output, re.IGNORECASE)
                    l0_ver_str = l0_driver_ver_match.group(1) if l0_driver_ver_match else ""
                    for gpu_info in gpus:
                        if gpu_info.vendor == VendorType.INTEL and gpu_info.name.split('(')[0].strip() in output:
                            if ComputeCapability.ONEAPI not in gpu_info.supported_apis:
                                gpu_info.supported_apis.append(ComputeCapability.ONEAPI)
                            if not gpu_info.oneapi_level_zero_version and l0_ver_str:
                                gpu_info.oneapi_level_zero_version = l0_ver_str
                            logger.info(f"Enriched Intel GPU {gpu_info.name} with ONEAPI info (L0 driver: {l0_ver_str or 'N/A'}).")
            else:
                logger.debug(f"{sycl_ls_path} failed. Stderr: {stderr.decode(errors='ignore').strip()}")
        except FileNotFoundError: logger.debug(f"{sycl_ls_path} not found.")
        except Exception as e: logger.warning(f"Error running {sycl_ls_path}: {e}")

    def _get_intel_memory_bandwidth(self, gpu_name: str, device_id: Optional[str] = None) -> float:
        name_lower = gpu_name.lower()
        bw_map = self.specs.get("memory_bandwidth_map", {})
        for model_key, bw in bw_map.items():
            if model_key in name_lower: return bw
        
        igpu_bw_estimates = self.specs.get("igpu_memory_bandwidth_estimates", {})
        if "iris xe" in name_lower or "uhd graphics" in name_lower:
            for keyword, bw in igpu_bw_estimates.items():
                if keyword in name_lower: return bw # e.g. "lpddr5" in "Iris Xe Graphics with LPDDR5"
            return igpu_bw_estimates.get("generic", 50.0) # Fallback for iGPU
        return 0.0

    def _get_intel_memory_estimate(self, gpu_name: str, device_id: Optional[str] = None) -> int:
        name_lower = gpu_name.lower().replace(' ', '').replace('graphics', '')
        mem_map = self.specs.get("memory_estimate_map", {})
        for model_key_norm, mem_mb in mem_map.items():
            if model_key_norm in name_lower:
                return mem_mb
        return 0 

    def _get_intel_gen_arch(self, gpu_name: str, device_id: Optional[str]) -> str:
        name_lower = gpu_name.lower()
        gen_arch_name_map = self.specs.get("gen_arch_name_map", {})
        for keyword, arch in gen_arch_name_map.items():
            if keyword in name_lower:
                return arch if arch else self._lookup_arch_by_device_id(device_id) # Fallback to device ID if name map leads to null

        return self._lookup_arch_by_device_id(device_id)

    def _lookup_arch_by_device_id(self, device_id: Optional[str]) -> str:
        if device_id:
            dev_id_map = self.specs.get("gen_arch_device_id_map", {})
            # Check for direct device_id match or prefix match
            cleaned_device_id = device_id.lower().replace("0x","")
            if cleaned_device_id in dev_id_map: return dev_id_map[cleaned_device_id]
            for prefix_len in sorted([len(k) for k in dev_id_map.keys()], reverse=True): # Prioritize longer prefix
                prefix = cleaned_device_id[:prefix_len]
                if prefix in dev_id_map: return dev_id_map[prefix]
        return "Unknown Gen"
    
    def _get_intel_eus(self, gpu_name: str, device_id: Optional[str]) -> Optional[int]:
        name_lower = gpu_name.lower().replace(' ', '').replace('graphics', '')
        
        eu_map_name = self.specs.get("eu_map_name", {})
        for model_key_norm, eus in eu_map_name.items():
            if model_key_norm in name_lower: return eus
        
        eu_match = re.search(r'(\d+)eu', name_lower)
        if eu_match: return int(eu_match.group(1))
        
        if device_id:
            eu_map_dev_id = self.specs.get("eu_map_device_id", {})
            cleaned_dev_id = device_id.lower().replace("0x","")
            if cleaned_dev_id in eu_map_dev_id: return eu_map_dev_id[cleaned_dev_id]
        
        # Specific fallbacks for common iGPU names if not in main map and no EU in name
        if "irisxe" in name_lower: return self.specs.get("eu_map_device_id", {}).get("default_irisxe_eu", 96) # fallback
        
        server_eu_map = self.specs.get("server_eu_map", {})
        for model_key_norm, eus in server_eu_map.items():
            if model_key_norm in name_lower: return eus
            
        return None

    def _determine_intel_performance_tier(self, gpu_name: str) -> PerformanceTier:
        name_lower = gpu_name.lower()
        if "data center gpu max" in name_lower or "ponte vecchio" in name_lower or "gaudi" in name_lower: return PerformanceTier.ENTERPRISE
        if "arc a770" in name_lower: return PerformanceTier.ENTHUSIAST
        if "arc a750" in name_lower or "arc a580" in name_lower: return PerformanceTier.MAINSTREAM
        if "arc a380" in name_lower or "arc a310" in name_lower or "iris xe max" in name_lower: return PerformanceTier.BUDGET
        if "iris xe graphics" in name_lower or "uhd graphics" in name_lower or "hd graphics" in name_lower: return PerformanceTier.INTEGRATED
        return PerformanceTier.UNKNOWN

    def get_telemetry(self, gpu: GPUInfo) -> TelemetryData:
        logger.debug(f"Intel telemetry for {gpu.name} via command line tools is complex and not fully implemented.")
        return TelemetryData(
            timestamp=datetime.now(), gpu_utilization=0.0, memory_utilization=0.0, 
            temperature=float(gpu.temperature or 0.0), power_draw=0.0 # Added or 0.0
        )
