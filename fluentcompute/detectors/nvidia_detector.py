import asyncio
import subprocess
import re
from typing import List, Optional
from datetime import datetime

from fluentcompute.models import GPUInfo, TelemetryData, VendorType, PerformanceTier, ComputeCapability
from .base import HardwareDetector
from fluentcompute.utils.logging_config import logger
from fluentcompute.config.settings import PYNVML_AVAILABLE

if PYNVML_AVAILABLE:
    import pynvml

class NvidiaDetector(HardwareDetector):
    """Advanced NVIDIA GPU detection with NVML integration"""
    def __init__(self):
        self.nvml_initialized = False
        self._initialize_nvml()
    
    def _initialize_nvml(self):
        if PYNVML_AVAILABLE:
            try:
                if not self.nvml_initialized:
                    pynvml.nvmlInit()
                    self.nvml_initialized = True
                    logger.info("✅ NVML initialized successfully")
            except pynvml.NVMLError as e:
                logger.warning(f"⚠️  NVML initialization failed: {e}. Some NVIDIA features might be unavailable.")
                self.nvml_initialized = False
            except Exception as e:
                logger.error(f"❌ Unexpected error during NVML initialization: {e}")
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False


    async def detect_hardware(self) -> List[GPUInfo]:
        gpus = []
        if self.nvml_initialized:
            gpus.extend(await self._detect_with_nvml())
        else:
            logger.info("NVML not available or not initialized, falling back to nvidia-smi for NVIDIA GPU detection.")
            gpus.extend(await self._detect_with_nvidia_smi())
        
        if gpus and not any(gpu.cuda_version for gpu in gpus): # Check if any GPU already has CUDA version
            cuda_ver = self._get_cuda_version_from_smi()
            if cuda_ver:
                for gpu in gpus:
                    if not gpu.cuda_version: # Only set if not already set (e.g., by NVML)
                        gpu.cuda_version = cuda_ver
        return gpus

    def _get_cuda_version_from_smi(self) -> Optional[str]:
        try:
            # First, try to get CUDA version from nvidia-smi header
            smi_output = subprocess.check_output(['nvidia-smi'], text=True, timeout=5)
            match = re.search(r"CUDA Version:\s*([\d\.]+)", smi_output)
            if match:
                return match.group(1)
        except subprocess.CalledProcessError:
            logger.debug("nvidia-smi call failed for CUDA version detection from header.")
        except FileNotFoundError:
            logger.debug("nvidia-smi not found, cannot determine CUDA version from SMI.")
        except Exception as e:
            logger.debug(f"Could not determine CUDA version from nvidia-smi header: {e}")
        return None

    async def _detect_with_nvml(self) -> List[GPUInfo]:
        gpus = []
        if not self.nvml_initialized:
            return gpus
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            driver_ver_str = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            cuda_ver_str = ""
            try: 
                cuda_driver_ver_int = pynvml.nvmlSystemGetCudaDriverVersion() 
                cuda_ver_str = f"{cuda_driver_ver_int // 1000}.{(cuda_driver_ver_int % 1000) // 10}"
            except pynvml.NVMLError: 
                cuda_ver_str = self._get_cuda_version_from_smi() or ""


            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                uuid = pynvml.nvmlDeviceGetUUID(handle).decode('utf-8')
                pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
                pci_bus_id = pci_info.busId.decode('utf-8')
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                try: max_graphics_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                except pynvml.NVMLError: max_graphics_clock = 0
                try: max_mem_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except pynvml.NVMLError: max_mem_clock = 0
                
                power_limit_watts = 0
                try: power_limit_watts = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) // 1000 
                except pynvml.NVMLError:
                    try: power_limit_watts = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) // 1000
                    except pynvml.NVMLError: power_limit_watts = 0


                try: temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError: temperature = 0
                
                try: fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                except pynvml.NVMLError: fan_speed = 0 

                major, minor = 0,0
                try: major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                except pynvml.NVMLError: pass
                compute_cap_str = f"{major}.{minor}" if major > 0 else "unknown"
                
                nvlink_enabled = False 
                multi_gpu_capable = device_count > 1 
                
                vgpu_support = self._check_virtualization_support(name)
                try:
                    current_mig_mode, _ = pynvml.nvmlDeviceGetMigMode(handle) # Returns (current, pending)
                    if current_mig_mode == pynvml.NVML_DEVICE_MIG_ENABLE: 
                        vgpu_support = True 
                except pynvml.NVMLError: 
                    pass


                gpu = GPUInfo(
                    name=name, vendor=VendorType.NVIDIA, gpu_index=i, uuid=uuid, pci_bus_id=pci_bus_id,
                    memory_total=mem_info.total // (1024**2), memory_free=mem_info.free // (1024**2),
                    memory_bandwidth=self._calculate_memory_bandwidth(name, max_mem_clock),
                    compute_capability=compute_cap_str,
                    cuda_cores=self._get_cuda_cores(name), rt_cores=self._get_rt_cores(name), tensor_cores=self._get_tensor_cores(name),
                    boost_clock=max_graphics_clock, memory_clock=max_mem_clock,
                    performance_tier=self._determine_performance_tier(name),
                    power_limit=power_limit_watts, temperature=temperature, fan_speed=fan_speed,
                    driver_version=driver_ver_str, cuda_version=cuda_ver_str,
                    supported_apis=[ComputeCapability.CUDA, ComputeCapability.OPENCL, ComputeCapability.VULKAN],
                    nvlink_enabled=nvlink_enabled, multi_gpu_capable=multi_gpu_capable, virtualization_support=vgpu_support
                )
                gpus.append(gpu)
        except pynvml.NVMLError as e:
            logger.error(f"❌ NVML detection error: {e}. Ensure NVIDIA drivers are installed and NVML is accessible.")
        except Exception as e: 
            logger.error(f"❌ Unexpected error during NVML-based detection: {e}")
        return gpus

    async def _detect_with_nvidia_smi(self) -> List[GPUInfo]:
        gpus = []
        try:
            cmd = [
                'nvidia-smi', 
                '--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.free,compute_cap,driver_version,power.limit,temperature.gpu,fan.speed,clocks.max.graphics,clocks.max.mem',
                '--format=csv,noheader,nounits'
            ]
            result_proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await result_proc.communicate()

            if result_proc.returncode != 0:
                logger.warning(f"⚠️ nvidia-smi command failed. stderr: {stderr.decode(errors='ignore').strip()}")
                return []

            output = stdout.decode(errors='ignore').strip()
            for line_idx, line in enumerate(output.split('\n')):
                if not line.strip(): continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 13: 
                    logger.warning(f"nvidia-smi output line {line_idx+1} has too few parts ({len(parts)}/13): {line}")
                    continue

                def safe_int(val, default=0):
                    try: return int(float(val)) 
                    except (ValueError, TypeError): return default
                
                gpu_name = parts[1]
                max_mem_clk = safe_int(parts[12]) if parts[12] != '[N/A]' else 0
                
                index = safe_int(parts[0])
                uuid = parts[2] if parts[2] != '[N/A]' else f"nvidia-smi-{index}"
                pci_bus_id = parts[3] if parts[3] != '[N/A]' else "unknown"

                gpu = GPUInfo(
                    gpu_index=index, name=gpu_name, vendor=VendorType.NVIDIA, uuid=uuid, pci_bus_id=pci_bus_id,
                    memory_total=safe_int(parts[4]), memory_free=safe_int(parts[5]),
                    memory_bandwidth=self._calculate_memory_bandwidth(gpu_name, max_mem_clk),
                    compute_capability=parts[6] if parts[6] != '[N/A]' else "unknown",
                    cuda_cores=self._get_cuda_cores(gpu_name), rt_cores=self._get_rt_cores(gpu_name), tensor_cores=self._get_tensor_cores(gpu_name),
                    boost_clock=safe_int(parts[11]) if parts[11] != '[N/A]' else 0,
                    memory_clock=max_mem_clk,
                    performance_tier=self._determine_performance_tier(gpu_name),
                    power_limit=safe_int(parts[8].replace('W','').strip()) if parts[8] != '[N/A]' else 0, # smi outputs power with W suffix
                    temperature=safe_int(parts[9]) if parts[9] != '[N/A]' else 0, 
                    fan_speed=safe_int(parts[10].replace('%','').strip()) if parts[10] != '[N/A]' else 0,
                    driver_version=parts[7] if parts[7] != '[N/A]' else "",
                    supported_apis=[ComputeCapability.CUDA, ComputeCapability.OPENCL, ComputeCapability.VULKAN],
                    virtualization_support=self._check_virtualization_support(gpu_name)
                )
                gpus.append(gpu)
        except FileNotFoundError:
            logger.info("nvidia-smi not found. NVIDIA GPU detection via CLI is not possible.")
        except Exception as e:
            logger.error(f"❌ nvidia-smi detection failed: {e}")
        return gpus
    
    def get_telemetry(self, gpu: GPUInfo) -> TelemetryData:
        if not self.nvml_initialized:
            try:
                query_items = 'utilization.gpu,utilization.memory,temperature.gpu,power.draw,fan.speed,clocks.current.graphics,clocks.current.memory'
                id_selector = gpu.pci_bus_id if gpu.pci_bus_id and gpu.pci_bus_id.lower() != "unknown" and gpu.pci_bus_id.lower() != "n/a (apple silicon)" and gpu.pci_bus_id.lower() != "n/a (cloud)" else str(gpu.gpu_index)

                cmd = [
                    'nvidia-smi', f'--query-gpu={query_items}',
                    f'--id={id_selector}',
                    '--format=csv,noheader,nounits'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=2)
                parts = result.stdout.strip().split(',')
                
                def s_float(val_str, repl='', default=None):
                    try: return float(val_str.replace(repl, '').strip())
                    except: return default
                def s_int(val_str, repl='', default=None):
                    try: return int(val_str.replace(repl, '').strip())
                    except: return default

                if len(parts) >= 7:
                    return TelemetryData(
                        timestamp=datetime.now(),
                        gpu_utilization=s_float(parts[0], '%', 0.0),
                        memory_utilization=s_float(parts[1], '%', 0.0),
                        temperature=s_float(parts[2], default=gpu.temperature or 0.0),
                        power_draw=s_float(parts[3], 'W', 0.0),
                        fan_speed=s_float(parts[4], '%') if parts[4] != '[N/A]' else None,
                        clock_graphics=s_int(parts[5]) if parts[5] != '[N/A]' else None,
                        clock_memory=s_int(parts[6]) if parts[6] != '[N/A]' else None,
                    )
            except Exception as e:
                logger.debug(f"nvidia-smi telemetry fallback for {gpu.name} failed: {e}")
            return TelemetryData(datetime.now(), 0, 0, float(gpu.temperature), 0)

        try:
            handle = pynvml.nvmlDeviceGetHandleByPciBusId(gpu.pci_bus_id.encode('utf-8')) \
                if gpu.pci_bus_id and gpu.pci_bus_id.lower() != "unknown" and gpu.pci_bus_id.lower() != "n/a (apple silicon)" and gpu.pci_bus_id.lower() != "n/a (cloud)" \
                else pynvml.nvmlDeviceGetHandleByIndex(gpu.gpu_index)

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_util = ((mem_info.used) / mem_info.total) * 100 if mem_info.total > 0 else 0.0
            temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            
            fan_speed_val = None
            try: fan_speed_val = float(pynvml.nvmlDeviceGetFanSpeed(handle))
            except pynvml.NVMLError: pass 
            
            graphics_clock_val, memory_clock_val = None, None
            try: graphics_clock_val = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            except pynvml.NVMLError: pass
            try: memory_clock_val = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except pynvml.NVMLError: pass
            
            throttle_reasons_list = []
            try:
                reasons_val = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                if reasons_val & pynvml.nvmlClocksThrottleReasonGpuIdle: throttle_reasons_list.append("GpuIdle")
                if reasons_val & pynvml.nvmlClocksThrottleReasonApplicationsClocksSetting: throttle_reasons_list.append("AppClocks")
                if reasons_val & pynvml.nvmlClocksThrottleReasonSwPowerCap: throttle_reasons_list.append("SwPowerCap")
                if reasons_val & pynvml.nvmlClocksThrottleReasonHwSlowdown: throttle_reasons_list.append("HwSlowdown")
                if hasattr(pynvml, 'nvmlClocksThrottleReasonHwThermalSlowdown') and \
                   reasons_val & pynvml.nvmlClocksThrottleReasonHwThermalSlowdown: throttle_reasons_list.append("HwThermalSlowdown")
                if hasattr(pynvml, 'nvmlClocksThrottleReasonHwPowerBrakeSlowdown') and \
                   reasons_val & pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown: throttle_reasons_list.append("HwPowerBrake")
                if reasons_val & pynvml.nvmlClocksThrottleReasonSyncBoost: throttle_reasons_list.append("SyncBoost")
            except pynvml.NVMLError: pass 

            return TelemetryData(
                timestamp=datetime.now(), gpu_utilization=float(util.gpu), memory_utilization=mem_util,
                temperature=temp, power_draw=power, fan_speed=fan_speed_val,
                clock_graphics=graphics_clock_val, clock_memory=memory_clock_val, throttle_reasons=throttle_reasons_list
            )
        except pynvml.NVMLError as e:
            logger.error(f"❌ NVML telemetry failed for GPU {gpu.name} (idx {gpu.gpu_index}, pci {gpu.pci_bus_id}): {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected telemetry error for GPU {gpu.name}: {e}")
        return TelemetryData(datetime.now(), 0, 0, float(gpu.temperature), 0.0)

    def _calculate_memory_bandwidth(self, gpu_name: str, mem_clk_mhz: Optional[int] = None) -> float:
        bandwidth_map = { 
            'RTX 4090': 1008.0, 'RTX 4080': 716.8, 'RTX 4070 Ti SUPER': 672.3, 'RTX 4070 Ti': 504.2, 'RTX 4070 SUPER': 716.8, 'RTX 4070': 504.2,
            'RTX 3090 Ti': 1008.0, 'RTX 3090': 936.2, 'RTX 3080 Ti': 912.4, 'RTX 3080': 760.3, 'RTX 3070 Ti': 608.3, 'RTX 3070': 448.0,
            'A100': 1555.0, 'H100 SXM': 3350.0, 'H100 PCIe': 2000.0, 'V100': 900.0, 'T4': 320.0, 'A10G': 600.0, 'A10': 600.0, 'A40': 696.0, 'A6000': 768.0,
            'NVIDIA RTX 6000 Ada Generation': 960.0, 'NVIDIA RTX A6000': 768.0,
        }
        for model, bandwidth in bandwidth_map.items():
            if model in gpu_name: return bandwidth
        
        if mem_clk_mhz and mem_clk_mhz > 0:
            bus_width_bits = 384 
            if any(n in gpu_name for n in ['4070', '3070', 'Tesla T4', 'A10G']): bus_width_bits = 256
            elif any(n in gpu_name for n in ['4060', '3060']): bus_width_bits = 192
            elif any(n in gpu_name for n in ['Tesla V100']): bus_width_bits = 4096 # HBM2
            elif any(n in gpu_name for n in ['A100']): bus_width_bits = 5120 if "SXM" in gpu_name else 5120 # HBM2e, some sources use 6144 for 80GB
            
            # This calculation is simplified. Different memory types (GDDR vs HBM) have different multipliers.
            # Assuming mem_clk_mhz from NVML is effective data rate / 2 for GDDR. HBM is different.
            multiplier = 2 # For GDDR type cards
            if "HBM" in gpu_name or any(x in gpu_name for x in ["V100", "A100", "H100"]):
                 multiplier = 1 # HBM clocks are often reported differently (actual physical clock)
                 # Example: V100, 877 MHz memory clock (actual) * 4096-bit bus / 8 bits/byte = 448 GB/s (half of advertised 900)
                 # The 900 GB/s is (actual_clock * 2 (DDR)) * bus_width_bits / 8 / 1000
                 # For V100 NVML max mem clock seems to be data rate / 2
            
            # Check NVIDIA product pages for the clock value interpretation. Often, NVML GetMaxClockInfo is memory chip's clock.
            # E.g. RTX 3080 mem_clk (from NVML max) is around 9500 MHz (which is 19 Gbps / 2). Bandwidth = 9500 * 2 * 320 / 8 / 1000 = 760.
            # So, the (mem_clk_mhz * 2 * bus_width_bits) / 8000 formula generally holds for GDDR if mem_clk_mhz is half data rate.
            
            # If 'Tesla V100' or 'A100' etc, the map above is more reliable. This is a fallback.
            return (float(mem_clk_mhz) * multiplier * bus_width_bits) / 8.0 / 1000.0

        return 0.0

    def _get_cuda_cores(self, gpu_name: str) -> Optional[int]:
        cuda_cores_map = {
            'RTX 4090': 16384, 'RTX 4080 SUPER': 10240, 'RTX 4080': 9728, 'RTX 4070 Ti SUPER': 8448, 'RTX 4070 Ti': 7680, 'RTX 4070 SUPER': 7168, 'RTX 4070': 5888, 'RTX 4060 Ti': 4352, 'RTX 4060': 3072,
            'RTX 3090 Ti': 10752, 'RTX 3090': 10496, 'RTX 3080 Ti': 10240, 'RTX 3080': 8704,
            'RTX 3070 Ti': 6144, 'RTX 3070': 5888, 'RTX 3060 Ti': 4864, 'RTX 3060': 3584,
            'NVIDIA A100': 6912, 'NVIDIA H100': 14592, 'Tesla V100': 5120, 'Tesla T4': 2560, 'A10G': 9728, 'NVIDIA A10': 9216, 'NVIDIA A40': 10752, 'NVIDIA RTX A6000': 10752,
            'NVIDIA RTX 6000 Ada Generation': 18176, 'NVIDIA RTX 5000 Ada Generation': 12800, 'NVIDIA RTX 4000 Ada Generation': 6144
        }
        for model_key in cuda_cores_map: # Check for NVIDIA prefix as well
            if model_key in gpu_name or model_key.replace("NVIDIA ", "") in gpu_name:
                 return cuda_cores_map[model_key]
        return None

    def _get_rt_cores(self, gpu_name: str) -> Optional[int]:
        rt_cores_map = { 
            'RTX 4090': 128, 'RTX 4080': 76, 'RTX 4070 Ti': 60, 'RTX 4070': 46,
            'RTX 3090': 82, 'RTX 3080': 68, 'RTX 3070': 46,
            'NVIDIA A40': 84, 'NVIDIA RTX A6000': 84, 'NVIDIA RTX 6000 Ada Generation': 142
        } 
        for model_key in rt_cores_map:
            if model_key in gpu_name or model_key.replace("NVIDIA ", "") in gpu_name:
                 return rt_cores_map[model_key]
        if "Tesla" in gpu_name or "A100" in gpu_name or "H100" in gpu_name or "V100" in gpu_name: return None # No dedicated RT cores
        return None

    def _get_tensor_cores(self, gpu_name: str) -> Optional[int]:
        tensor_cores_map = {
            'RTX 4090': 512, 'RTX 4080': 304, 'RTX 4070 Ti': 240, 'RTX 4070': 184, 
            'RTX 3090': 328, 'RTX 3080': 272, 'RTX 3070': 184, 
            'NVIDIA A100': 432, 'NVIDIA H100': 456, 'Tesla V100': 640, 
            'Tesla T4': 320, 
            'NVIDIA A40': 336, 'NVIDIA RTX A6000': 336, 'NVIDIA RTX 6000 Ada Generation': 568
        }
        for model_key in tensor_cores_map:
            if model_key in gpu_name or model_key.replace("NVIDIA ", "") in gpu_name:
                 return tensor_cores_map[model_key]
        return None
    
    def _determine_performance_tier(self, gpu_name: str) -> PerformanceTier:
        name_upper = gpu_name.upper().replace("NVIDIA ", "")
        if any(card in name_upper for card in ['H100', 'A100', 'MI300', 'MI250', 'RTX 6000 ADA', 'A6000', 'RTX A6000', 'GV100', 'TESLA V100']): return PerformanceTier.ENTERPRISE
        if any(card in name_upper for card in ['RTX 4090', 'RTX 3090', 'RX 7900 XTX', 'RX 6950 XT', 'RX 6900 XT', 'RTX A5000', 'RTX 5000 ADA']): return PerformanceTier.PROFESSIONAL
        if any(card in name_upper for card in ['RTX 4080', 'RTX 3080 TI', 'RTX 3080', 'RX 7900 XT', 'RX 6800 XT', 'RTX 4070 TI', 'ARC A770']): return PerformanceTier.ENTHUSIAST
        if any(card in name_upper for card in ['RTX 4070', 'RTX 3070 TI', 'RTX 3070', 'RX 7800 XT', 'RX 6700 XT', 'RX 6750 XT', 'ARC A750', 'RTX 3060 TI']): return PerformanceTier.MAINSTREAM
        if any(card in name_upper for card in ['RTX 4060', 'RTX 3060', 'RTX 3050', 'RX 7600', 'RX 6600 XT', 'RX 6600', 'ARC A580', 'ARC A380', 'GTX 1660', 'GTX 1650', 'TESLA T4']): return PerformanceTier.BUDGET
        if "QUADRO" in name_upper or ("TESLA" in name_upper and "TESLA T4" not in name_upper and "TESLA V100" not in name_upper) : return PerformanceTier.ENTERPRISE
        return PerformanceTier.UNKNOWN

    def _check_virtualization_support(self, gpu_name: str) -> bool:
        # Normalize name for checks
        name_check = gpu_name.replace("NVIDIA ", "")
        vgpu_cards = ['A100', 'H100', 'A40', 'A30', 'A16', 'A10', 'A2', 
                      'RTX A6000', 'RTX A5000', 'RTX A4000', # Added A4000
                      'RTX 6000 Ada Generation', 'RTX 5000 Ada Generation', 'RTX 4000 Ada Generation', # Added Ada Prof line
                      'L40', 'L4']
        if any(card in name_check for card in vgpu_cards): return True
        if "TESLA M" in name_check or "TESLA P" in name_check or "TESLA V" in name_check : return True
        if "GRID" in name_check: return True
        return False

    def cleanup(self):
        if PYNVML_AVAILABLE and self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self.nvml_initialized = False
                logger.info("✅ NVML shutdown by NvidiaDetector.")
            except pynvml.NVMLError as e:
                logger.warning(f"⚠️ NVML shutdown error in NvidiaDetector: {e}")
            except Exception as e:
                logger.error(f"❌ Unexpected NVML shutdown error: {e}")
    
    def __del__(self):
        self.cleanup()
