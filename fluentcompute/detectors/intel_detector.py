import asyncio
import subprocess
import re
import platform
from typing import List, Optional
from datetime import datetime

from fluentcompute.models import GPUInfo, TelemetryData, VendorType, PerformanceTier, ComputeCapability
from .base import HardwareDetector
from fluentcompute.utils.logging_config import logger

class IntelDetector(HardwareDetector):
    async def detect_hardware(self) -> List[GPUInfo]:
        gpus = []
        if platform.system() == "Linux":
            gpus.extend(await self._detect_with_lspci_intel())
            # Enrichment steps like _enrich_with_clinfo or _enrich_with_sycl_ls
            # require those tools to be callable and output parsed.
            # For this pass, we'll keep the placeholders simple.
            if gpus:
                await self._enrich_with_oneapi_tools(gpus)

        elif platform.system() == "Windows":
            gpus.extend(await self._detect_with_wmic_windows())
        
        logger.info(f"Intel detector found {len(gpus)} potential devices.")
        return gpus

    async def _detect_with_lspci_intel(self) -> List[GPUInfo]:
        gpus_info_list = []
        try:
            # The grep pattern needs to be specific enough for Intel but not too broad.
            cmd = "lspci -nnk | grep -Ei 'VGA compatible controller.*Intel|Display controller.*Intel|3D controller.*Intel|Processing accelerators.*Intel'"
            process = await asyncio.create_subprocess_shell(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                executable='/bin/bash' # Good practice for shell features like pipes
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                output = stdout.decode(errors='ignore').strip()
                gpu_idx_counter = 0
                
                device_blocks = []
                current_block_lines = []
                # Split lspci output into blocks for each device
                # A new device usually starts a line with a PCI ID (e.g., 00:02.0)
                for line in output.splitlines():
                    if re.match(r"^\S+:\S+\.\S+", line) and current_block_lines:
                        device_blocks.append("\n".join(current_block_lines))
                        current_block_lines = [line]
                    else:
                        current_block_lines.append(line)
                if current_block_lines:
                    device_blocks.append("\n".join(current_block_lines))
                
                for block in device_blocks:
                    # Check for Intel Corporation and a device ID
                    intel_match = re.search(r"Intel Corporation", block)
                    pci_match = re.search(r"^(\S+)\s+(VGA compatible controller|3D controller|Display controller|Processing accelerators) \[.*?\]:\s*(?:Intel(?: Corporation)?)\s*(.*?)\s*\[(8086):(\S+)\]", block, re.MULTILINE | re.IGNORECASE)

                    if not intel_match or not pci_match : # Only process blocks confirmed to be Intel with device ID
                        continue
                        
                    pci_bus_id = pci_match.group(1)
                    raw_name_from_lspci = pci_match.group(3).strip()
                    device_id = pci_match.group(5).lower()
                    
                    name = self._refine_intel_gpu_name(raw_name_from_lspci, device_id)
                    
                    driver_match = re.search(r"Kernel driver in use:\s*(\S+)", block)
                    driver = driver_match.group(1) if driver_match else "unknown"
                    
                    # Assume OpenCL and Vulkan. ONEAPI added by enrichment if sycl-ls finds it.
                    apis = [ComputeCapability.OPENCL, ComputeCapability.VULKAN]

                    gpu_info = GPUInfo(
                        name=name, vendor=VendorType.INTEL, gpu_index=gpu_idx_counter,
                        uuid=f"intel-pci-{pci_bus_id}-{device_id}", pci_bus_id=pci_bus_id,
                        memory_total=self._get_intel_memory_estimate(name, device_id), memory_free=0, # lspci does not provide free
                        memory_bandwidth=self._get_intel_memory_bandwidth(name, device_id),
                        compute_capability=self._get_intel_gen_arch(name, device_id),
                        intel_eus=self._get_intel_eus(name, device_id),
                        performance_tier=self._determine_intel_performance_tier(name),
                        driver_version=driver, 
                        supported_apis=apis 
                    )
                    gpus_info_list.append(gpu_info)
                    gpu_idx_counter += 1
            else:
                 logger.debug(f"lspci command for Intel failed or no Intel GPUs found. stderr: {stderr.decode(errors='ignore').strip()}")
        except FileNotFoundError: 
            logger.info("lspci not found. Intel GPU detection via lspci not possible on this system.")
        except Exception as e: 
            logger.error(f"❌ Intel lspci detection failed: {e}")
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
                if len(lines) > 1: # Header + data
                    header = lines[0].lower().split(',')
                    try:
                        name_idx = header.index("name")
                        compat_idx = header.index("adaptercompatibility")
                        pnp_idx = header.index("pnpdeviceid")
                        driver_idx = header.index("driverversion")
                        ram_idx = header.index("adapterram")
                    except ValueError:
                        logger.error("WMIC output CSV header is missing expected columns.")
                        return []

                    gpu_idx_counter = 0
                    for line_content in lines[1:]: # Skip header
                        parts = line_content.split(',')
                        if len(parts) < max(name_idx, compat_idx, pnp_idx, driver_idx, ram_idx) + 1:
                            continue

                        adapter_compat = parts[compat_idx].strip()
                        raw_name_from_wmic = parts[name_idx].strip()

                        if "intel" not in adapter_compat.lower() and "intel" not in raw_name_from_wmic.lower():
                            continue 
                        
                        pnp_id = parts[pnp_idx].strip()
                        driver_ver = parts[driver_idx].strip()
                        adapter_ram_str = parts[ram_idx].strip()

                        dev_id_match = re.search(r"DEV_(\S{4})", pnp_id) # Extract 4-char device ID
                        device_id = dev_id_match.group(1).lower() if dev_id_match else None
                        name = self._refine_intel_gpu_name(raw_name_from_wmic, device_id)
                        
                        adapter_ram_mb = 0
                        if adapter_ram_str:
                            try: adapter_ram_mb = int(adapter_ram_str) // (1024 * 1024)
                            except ValueError: pass
                        
                        mem_total = self._get_intel_memory_estimate(name, device_id)
                        if mem_total == 0 and ("Arc" in name or "Iris Xe MAX" in name): # dGPU memory might be directly from AdapterRAM
                            mem_total = adapter_ram_mb
                        elif mem_total == 0 and "Graphics" in name: # iGPU
                            mem_total = adapter_ram_mb # WMIC might report available shared memory portion for iGPU

                        gpu_info = GPUInfo(
                            name=name, vendor=VendorType.INTEL, gpu_index=gpu_idx_counter,
                            uuid=f"intel-wmic-{pnp_id.replace('PCI\\','').replace('&','_')}", # Sanitize UUID a bit
                            pci_bus_id="unknown", # WMIC PNPDeviceID is not PCI Bus ID
                            memory_total=mem_total, memory_free=0, # WMIC does not provide free
                            memory_bandwidth=self._get_intel_memory_bandwidth(name, device_id),
                            compute_capability=self._get_intel_gen_arch(name, device_id),
                            intel_eus=self._get_intel_eus(name, device_id),
                            performance_tier=self._determine_intel_performance_tier(name),
                            driver_version=driver_ver,
                            supported_apis=[ComputeCapability.OPENCL, ComputeCapability.DIRECTML, ComputeCapability.VULKAN]
                        )
                        # Check for oneAPI drivers too.
                        if "oneAPI Level Zero" in driver_ver or "Intel Graphics Driver" in driver_ver : # Basic check
                            if ComputeCapability.ONEAPI not in gpu_info.supported_apis:
                                gpu_info.supported_apis.append(ComputeCapability.ONEAPI)
                        
                        gpus_info_list.append(gpu_info)
                        gpu_idx_counter += 1
            else:
                logger.debug(f"WMIC command for Intel failed. Stderr: {stderr.decode(errors='ignore')}")
        except FileNotFoundError: logger.info("WMIC not found. Intel GPU detection via WMIC not possible on Windows.")
        except Exception as e: logger.error(f"❌ Intel WMIC detection failed: {e}")
        return gpus_info_list

    def _refine_intel_gpu_name(self, raw_name: str, device_id: Optional[str]) -> str:
        # First, attempt to normalize based on known device ID prefixes or keywords.
        if device_id:
            # Alchemist (dGPUs, some server GPUs like Max series)
            if device_id.startswith("56") or device_id.startswith("0x56"): # Example: 56a0 (A770), 56b0 (A30M/A380 variant)
                if "A770" in raw_name or device_id in ["56a0"]: return "Intel Arc A770 Graphics"
                if "A750" in raw_name or device_id in ["56a1"]: return "Intel Arc A750 Graphics"
                if "A580" in raw_name or device_id in ["56a2"]: return "Intel Arc A580 Graphics"
                if "A380" in raw_name or device_id in ["56a5", "56a6", "56b0", "56b1"]: return "Intel Arc A380 Graphics" # A310/A350M might fall here
                if "Data Center GPU Max" in raw_name: return raw_name # Keep explicit names
                return f"Intel Arc Graphics ({device_id})" # Fallback for other 56xx
            # DG1 (Iris Xe MAX, Server SG1)
            if device_id in ["4905", "4907", "4c0c", "4c0d"]: return "Intel Iris Xe MAX Graphics" 
            # Tiger Lake, Alder Lake, Rocket Lake iGPUs (Xe-LP)
            if device_id.startswith("9a") or device_id.startswith("4c") or device_id.startswith("a7"): 
                 if "UHD Graphics" in raw_name and "Xe" not in raw_name: return f"Intel UHD Graphics ({device_id})" # e.g. some Alder Lake non-Xe UHD
                 return f"Intel Iris Xe Graphics ({device_id})"
            # Older gens like Skylake, Kaby Lake (Gen9)
            if device_id.startswith("19") or device_id.startswith("59") or device_id.startswith("3e"):
                 if "HD Graphics" in raw_name: return f"Intel HD Graphics ({device_id})"
                 if "Iris Plus Graphics" in raw_name: return f"Intel Iris Plus Graphics ({device_id})"
                 return f"Intel Graphics Gen9 ({device_id})"
        
        # Fallback to using keywords in the raw_name if device_id didn't give a clear match
        if "Arc" in raw_name: return raw_name 
        if "Iris Xe MAX" in raw_name: return raw_name
        if "Iris Xe" in raw_name: return raw_name
        if "UHD Graphics" in raw_name: return raw_name
        if "HD Graphics" in raw_name: return raw_name
        return raw_name if raw_name else f"Intel Graphics ({device_id or 'Unknown'})"

    async def _enrich_with_oneapi_tools(self, gpus: List[GPUInfo]):
        """Try to get oneAPI Level Zero version and confirm ONEAPI support using sycl-ls."""
        sycl_ls_path = "sycl-ls" # Assumed to be in PATH if oneAPI is installed
        try:
            proc = await asyncio.create_subprocess_exec(sycl_ls_path, "--verbose", # Verbose might give driver versions
                                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                output = stdout.decode(errors='ignore')
                # Example output for oneAPI Level Zero GPU:
                # [level_zero:gpu:0] Intel(R) Arc(TM) A770 Graphics, Intel(R) Level-Zero driver 1.3.26918, Runtimes: مستوى صفر /OpenCL 
                # Version might be on a line like: "Driver GFX Version: ..." or associated with Level-Zero driver.
                
                # Check for Level Zero devices
                if "level_zero:gpu" in output.lower():
                    # Attempt to extract a Level Zero driver version
                    l0_driver_ver_match = re.search(r"Level-Zero driver\s+([\d\.]+)", output, re.IGNORECASE)
                    l0_ver_str = l0_driver_ver_match.group(1) if l0_driver_ver_match else ""

                    for gpu_info in gpus:
                        if gpu_info.vendor == VendorType.INTEL:
                            # Heuristic: Match based on name. A more robust way would be mapping sycl-ls device to PCI ID if possible.
                            if gpu_info.name.split('(')[0].strip() in output: # "Intel Arc A770 Graphics" in "Intel(R) Arc(TM) A770 Graphics..."
                                if ComputeCapability.ONEAPI not in gpu_info.supported_apis:
                                    gpu_info.supported_apis.append(ComputeCapability.ONEAPI)
                                if not gpu_info.oneapi_level_zero_version and l0_ver_str:
                                    gpu_info.oneapi_level_zero_version = l0_ver_str
                                logger.info(f"Enriched Intel GPU {gpu_info.name} with ONEAPI info (L0 driver: {l0_ver_str or 'N/A'}).")
            else:
                logger.debug(f"{sycl_ls_path} failed or no oneAPI devices. Stderr: {stderr.decode(errors='ignore').strip()}")
        except FileNotFoundError:
            logger.debug(f"{sycl_ls_path} not found. Cannot enrich with oneAPI details.")
        except Exception as e:
            logger.warning(f"Error running or parsing {sycl_ls_path}: {e}")
        # Placeholder for clinfo (OpenCL) enrichment - more complex parsing needed
        pass

    def _get_intel_memory_bandwidth(self, gpu_name: str, device_id: Optional[str] = None) -> float: # GB/s
        name_lower = gpu_name.lower()
        # dGPUs
        if "arc a770" in name_lower: return 560.0 # 16 Gbps / 256-bit
        if "arc a750" in name_lower: return 512.0 # 16 Gbps / 256-bit
        if "arc a580" in name_lower: return 512.0 # 16 Gbps / 256-bit
        if "arc a380" in name_lower: return 186.0 # 15.5 Gbps / 96-bit
        if "iris xe max" in name_lower or "dg1" in name_lower : return 68.25 # LPDDR4X-4266 on 128-bit
        # Server GPUs
        if "data center gpu max 1550" in name_lower : return 4096.0 # HBM2e
        if "data center gpu max 1100" in name_lower : return 2048.0 # HBM2e

        # iGPUs depend on system RAM type and channels (DDR4, DDR5, LPDDR4x, LPDDR5)
        # Assuming dual channel for rough estimates.
        if "iris xe" in name_lower or "uhd graphics" in name_lower:
            if "lpddr5" in name_lower or "lp5" in name_lower: return 80.0 # ~LPDDR5-5200 dual channel
            if "lpddr4" in name_lower or "lp4" in name_lower: return 60.0 # ~LPDDR4x-4266 dual channel
            if "ddr5" in name_lower: return 70.0 # ~DDR5-4800 dual channel
            if "ddr4" in name_lower: return 45.0 # ~DDR4-3200 dual channel
            return 50.0 # Generic estimate
        return 0.0

    def _get_intel_memory_estimate(self, gpu_name: str, device_id: Optional[str] = None) -> int: # MB
        name_lower = gpu_name.lower().replace(' ', '') # remove spaces for easier matching
        # dGPUs
        if "arca77016gb" in name_lower: return 16 * 1024
        if "arca770" in name_lower or "arca770m" in name_lower: return 8 * 1024 # Common variant for A770/A770M
        if "arca750" in name_lower: return 8 * 1024
        if "arca580" in name_lower: return 8 * 1024
        if "arca380" in name_lower or "arca370m" in name_lower: return 6 * 1024
        if "arca350m" in name_lower or "arca310" in name_lower: return 4 * 1024
        if "irisxemax" in name_lower or "dg1" in name_lower : return 4 * 1024 
        # Server GPUs
        if "datacentergpumax1550" in name_lower: return 128 * 1024 # PVC
        if "datacentergpumax1100" in name_lower: return 48 * 1024 # PVC
        # iGPUs share system RAM, value returned is often max allocatable or a portion, not fixed VRAM.
        # This value should ideally come from an API if available. lspci/wmic provide some hints.
        return 0 

    def _get_intel_gen_arch(self, gpu_name: str, device_id: Optional[str]) -> str:
        name_lower = gpu_name.lower()
        if "data center gpu max" in name_lower or "ponte vecchio" in name_lower: return "Xe-HPC (Ponte Vecchio)"
        if "arc" in name_lower or (device_id and device_id.startswith("56")): return "Xe-HPG (Alchemist)"
        if "iris xe max" in name_lower or (device_id and device_id == "4905"): return "Xe-LP (DG1)" # dGPU variant of Xe-LP
        if "iris xe graphics" in name_lower or \
           (device_id and (device_id.startswith("9a") or device_id.startswith("4c8") or device_id.startswith("a7"))): 
             return "Xe-LP (iGPU)" # Covers TigerLake, RocketLake, AlderLake-U/P/H iGPUs
        if "uhd graphics" in name_lower: # Could be older or newer Xe-LP based
             if device_id and (device_id.startswith("4c8") or device_id.startswith("a7")): return "Xe-LP (iGPU)" # e.g. AlderLake UHD
             if device_id and device_id.startswith("3e9"): return "Gen9.5 (Kaby Lake/Coffee Lake)"
             if device_id and device_id.startswith("19"): return "Gen9 (Skylake)"
        if "hd graphics" in name_lower: # Typically older gens
             if device_id and device_id.startswith("59"): return "Gen9.5 (Kaby Lake/Coffee Lake)" # e.g. Kaby Lake HD 620
             if device_id and device_id.startswith("19"): return "Gen9 (Skylake)"
             # Add more Gen LUT here
        return "Unknown Gen"

    def _get_intel_eus(self, gpu_name: str, device_id: Optional[str]) -> Optional[int]:
        name_lower = gpu_name.lower().replace(' ', '').replace('graphics', '')
        # Arc GPUs usually described by Xe-cores (1 Xe-core = 16 EUs, but some Arc variants differ)
        # Vector Engines (renamed from EUs for some architectures like Alchemist)
        if "arca770" in name_lower: return 32 * 16 # 32 Xe-cores for A770 (512 Vector Engines)
        if "arca750" in name_lower: return 28 * 16 # 28 Xe-cores for A750 (448 VEs)
        if "arca580" in name_lower: return 24 * 16 # 24 Xe-cores for A580 (384 VEs)
        if "arca380" in name_lower: return 8 * 16  # 8 Xe-cores for A380 (128 VEs)
        if "arca370m" in name_lower: return 8 * 16 # A370M
        if "arca350m" in name_lower: return 6 * 16 # A350M
        if "arca310" in name_lower: return 6 * 16 # A310 desktop variant
        
        # Iris Xe MAX (DG1) has 96 EUs
        if "irisxemax" in name_lower or (device_id and device_id == "4905"): return 96
        
        # iGPUs often specify EU count in model names like "Iris Xe Graphics 96EU"
        eu_match = re.search(r'(\d+)eu', name_lower)
        if eu_match: return int(eu_match.group(1))
        
        # Fallback for common iGPU variants if not in name
        if "irisxe" in name_lower: # Generic Iris Xe (often 80 or 96 EUs for mobile)
            if device_id: # Tiger Lake examples
                if device_id.startswith("9a49"): return 96 # Common high-end TGL
                if device_id.startswith("9a7f"): return 48 # Lower-end TGL
            return 96 # Default guess for Iris Xe
        
        if "uhdgraphics" in name_lower:
            if device_id and device_id.startswith("4c8a"): return 32 # Alder Lake UHD 770
            if device_id and device_id.startswith("9bc5"): return 24 # Comet Lake UHD 630
            if device_id and device_id.startswith("3e92"): return 24 # Kaby Lake UHD 620
            
        # Server GPUs
        if "datacentergpumax1550" in name_lower : return 128 * 8 # 128 Xe-cores (PVC uses diff. core counts per slice) -> check Xe Vector Engines or Xe Matrix Extensions
        if "datacentergpumax1100" in name_lower : return 48 * 8  # Simplified
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
        # intel_gpu_top requires root or special permissions (CONFIG_DRM_I915_REQUEST_TIMEOUT in kernel).
        # And output is textual, not easily parsable JSON for all versions.
        # Fallback for now. A proper implementation would need to handle permissions and parse text.
        # Another option is oneAPI's Level Zero `zetool` or metrics APIs.
        logger.debug(f"Intel telemetry for {gpu.name} via command line tools is complex and not fully implemented.")
        
        # Placeholder based on initial detection or defaults
        return TelemetryData(
            timestamp=datetime.now(), 
            gpu_utilization=0.0, 
            memory_utilization=0.0, 
            temperature=float(gpu.temperature) if gpu.temperature else 0.0, # Use temp if detected at init
            power_draw=0.0
        )
