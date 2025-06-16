import asyncio
import psutil
import socket
import platform
import sys
import threading
import time
import json
import logging # For log_level type hint
from typing import Dict, List, Optional, Tuple, Any

from fluentcompute.utils.logging_config import logger, get_logger
from fluentcompute.models import GPUInfo, TelemetryData, VendorType # Assuming models.__init__ exports these
from fluentcompute.detectors import (
    NvidiaDetector, AMDDetector, IntelDetector, AppleDetector, 
    CloudDetector, UnknownDetector, HardwareDetector
)
from fluentcompute.db.manager import DBManager
from fluentcompute.config.settings import DEFAULT_LOG_LEVEL

class HardwareManager:
    def __init__(self, enable_cloud_detection: bool = True, log_level: int = DEFAULT_LOG_LEVEL, db_base_path: Optional[str] = None):
        # Potentially reconfigure logger if a different level is passed
        global logger 
        if logger.level != log_level:
            logger = get_logger(level=log_level) 
        
        self.gpus: List[GPUInfo] = []
        self.detectors: List[HardwareDetector] = []
        self._register_detectors(enable_cloud_detection)
        
        self.telemetry_thread: Optional[threading.Thread] = None
        self.telemetry_stop_event = threading.Event()
        self.telemetry_interval_sec = 5
        self.telemetry_lock = threading.Lock() # Protects access to gpu.telemetry_history
        self.max_telemetry_history_per_gpu = 100

        self.os_type = platform.system()
        self.os_release = platform.release()
        self.os_version = platform.version()
        self.architecture = platform.machine()
        self.hostname = socket.gethostname()
        self.cpu_info_str = self._get_cpu_info_str()
        self.total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        
        db_path_obj = None
        if db_base_path:
            from pathlib import Path
            db_path_obj = Path(db_base_path)

        self.db_manager = DBManager(base_path=db_path_obj)


    def _get_cpu_info_str(self) -> str:
        if self.os_type == "Darwin":
            try: return subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
            except Exception: pass
        elif self.os_type == "Linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if "model name" in line: return line.split(':', 1)[1].strip()
            except Exception: pass
        
        # Fallback using psutil or platform for broader compatibility
        try:
            # platform.processor() can be generic; psutil often better if available.
            # On some systems psutil.cpu_freq() may not work or require permissions
            cpu_name = platform.processor()
            cores_logical = psutil.cpu_count(logical=True)
            cores_physical = psutil.cpu_count(logical=False)
            
            freq_str = ""
            if hasattr(psutil, 'cpu_freq'):
                try:
                    freq = psutil.cpu_freq()
                    if freq and freq.max > 0: # Check if freq.max is valid
                        freq_str = f", {freq.max:.0f} MHz" # Simplified: max freq
                except Exception: # psutil.cpu_freq() can fail (e.g. permissions, WSL)
                    pass
            
            return f"{cpu_name or 'Unknown CPU'} ({cores_physical or cores_logical}c/{cores_logical or '?'}t{freq_str})"

        except Exception as e_cpu: 
            logger.debug(f"Could not get detailed CPU info: {e_cpu}")
            return platform.processor() or "Unknown CPU"

    def _register_detectors(self, enable_cloud_detection: bool):
        self.detectors.append(NvidiaDetector())
        self.detectors.append(AMDDetector())
        self.detectors.append(IntelDetector())
        if platform.system() == "Darwin": 
            self.detectors.append(AppleDetector())
        if enable_cloud_detection: 
            self.detectors.append(CloudDetector())
        self.detectors.append(UnknownDetector())

    async def initialize_hardware(self):
        logger.info("🚀 Initializing hardware detection...")
        all_detected_gpus_by_detector: List[Tuple[str, List[GPUInfo]]] = []
        
        # Use a list to store coroutines for asyncio.gather
        detection_coroutines = [detector.detect_hardware() for detector in self.detectors]
        # Execute all detection coroutines concurrently
        results = await asyncio.gather(*detection_coroutines, return_exceptions=True)
        
        for i, result_item in enumerate(results):
            detector_name = self.detectors[i].__class__.__name__
            if isinstance(result_item, Exception):
                logger.error(f"❌ Error during hardware detection with {detector_name}: {result_item}", exc_info=True)
            elif isinstance(result_item, list):
                all_detected_gpus_by_detector.append((detector_name, result_item))
                if result_item: # Log only if GPUs were detected by this detector
                    for gpu_info in result_item:
                        logger.info(f"  🔎 Detected by {detector_name}: {gpu_info.name} (UUID: {gpu_info.uuid})")
                else:
                    logger.debug(f"  ⓘ {detector_name} found no devices.")
            else:
                logger.warning(f"⚠️ Unexpected result type from {detector_name}: {type(result_item)}")

        self.gpus = self._process_detected_gpus(all_detected_gpus_by_detector)
        
        if not self.gpus: 
            logger.warning("⚠️ No GPUs detected on this system.")
        else: 
            logger.info(f"✅ Hardware detection complete. Final unique GPU count: {len(self.gpus)}.")
        
        self.db_manager.log_hardware_snapshot(self.get_system_summary())


    def _process_detected_gpus(self, detected_gpus_by_detector: List[Tuple[str, List[GPUInfo]]]) -> List[GPUInfo]:
        final_gpus_map: Dict[str, GPUInfo] = {} 
        
        # Prioritize local detectors for detailed info
        local_detector_names = {NvidiaDetector.__name__, AMDDetector.__name__, IntelDetector.__name__, AppleDetector.__name__}

        # Pass 1: Add GPUs from local vendor detectors
        for detector_name, gpu_list in detected_gpus_by_detector:
            if detector_name not in local_detector_names:
                continue
            for gpu in gpu_list:
                # Keying by UUID primarily, fallback to PCI Bus ID if UUID is generic or missing
                key = gpu.uuid if gpu.uuid and not any(s in gpu.uuid for s in ["unknown", "pci-", "smi-", "rocm-", "wmic-"]) else gpu.pci_bus_id
                if key and key.lower() not in ["unknown", "n/a (apple silicon)", "n/a (cloud)"]: # Ensure key is somewhat unique
                    if key not in final_gpus_map:
                        final_gpus_map[key] = gpu
                    else:
                        # Merge logic: if an existing entry has less detail (e.g. from a fallback in the same detector),
                        # the new one might be better. For now, prefer first encountered.
                        logger.debug(f"Duplicate key {key} for GPU {gpu.name} from {detector_name}. Keeping first detailed entry.")
                else: # If key is not suitable, use a composite key (less ideal)
                    composite_key = f"{gpu.vendor.value}-{gpu.gpu_index}-{gpu.name[:10]}"
                    if composite_key not in final_gpus_map:
                        final_gpus_map[composite_key] = gpu


        # Pass 2: Process CloudDetector results and merge/enrich
        cloud_detector_name = CloudDetector.__name__
        for detector_name, gpu_list in detected_gpus_by_detector:
            if detector_name != cloud_detector_name:
                continue
            
            for cloud_gpu in gpu_list:
                enriched_existing = False
                # Try to match cloud GPU with an already detected local GPU
                # This logic is heuristic. A strong match requires more info from cloud metadata (like PCI ID).
                for local_key, local_gpu in final_gpus_map.items():
                    # Match if vendor matches, and (names are similar OR memory sizes are close)
                    # AND if local GPU doesn't already have cloud info (to avoid overwriting richer local cloud detection)
                    if local_gpu.vendor == cloud_gpu.vendor and not local_gpu.instance_type:
                        name_match = local_gpu.name in cloud_gpu.name or cloud_gpu.name in local_gpu.name
                        mem_match = abs(local_gpu.memory_total - cloud_gpu.memory_total) < (local_gpu.memory_total * 0.1) # within 10%
                        if name_match or mem_match:
                            local_gpu.instance_type = cloud_gpu.instance_type
                            local_gpu.cloud_region = cloud_gpu.cloud_region
                            local_gpu.spot_price = cloud_gpu.spot_price
                            # Potentially update name if cloud name is more specific
                            if "Cloud GPU" in local_gpu.name and "Cloud GPU" not in cloud_gpu.name:
                                local_gpu.name = cloud_gpu.name
                            logger.info(f"Enriched local GPU {local_gpu.name} (key: {local_key}) with cloud context from {cloud_gpu.name}.")
                            enriched_existing = True
                            break # Assume one cloud GPU entry maps to one local GPU
                
                if not enriched_existing: # If no match, add the cloud GPU as a new entry
                    key = cloud_gpu.uuid if cloud_gpu.uuid and "cloud-" in cloud_gpu.uuid else f"cloud-{cloud_gpu.instance_type}-{cloud_gpu.name.replace(' ','_')}-{cloud_gpu.gpu_index}"
                    if key not in final_gpus_map:
                        final_gpus_map[key] = cloud_gpu
                    else: # If key conflict, append index to make unique
                        final_gpus_map[f"{key}-{cloud_gpu.gpu_index}"] = cloud_gpu
        
        return sorted(list(final_gpus_map.values()), key=lambda gpu: (gpu.vendor.value, gpu.pci_bus_id if gpu.pci_bus_id else "", gpu.gpu_index))


    def get_all_gpus(self) -> List[GPUInfo]: return self.gpus
    def get_gpu_by_uuid(self, uuid: str) -> Optional[GPUInfo]:
        return next((gpu for gpu in self.gpus if gpu.uuid == uuid), None)

    def _telemetry_collector_task(self):
        logger.info("📊 Telemetry collector thread started.")
        # Create a map of GPU UUID to its responsible detector instance for telemetry
        gpu_detector_map: Dict[str, HardwareDetector] = {}
        
        for gpu in self.gpus:
            detector_class_name = ""
            if gpu.vendor == VendorType.NVIDIA: detector_class_name = NvidiaDetector.__name__
            elif gpu.vendor == VendorType.AMD: detector_class_name = AMDDetector.__name__
            elif gpu.vendor == VendorType.INTEL: detector_class_name = IntelDetector.__name__
            elif gpu.vendor == VendorType.APPLE: detector_class_name = AppleDetector.__name__
            elif gpu.vendor in [VendorType.CLOUD_AWS, VendorType.CLOUD_GCP, VendorType.CLOUD_AZURE]:
                 # If it's a cloud GPU that wasn't merged with a local one, use CloudDetector.
                 # If it *was* merged, its original vendor detector should handle it.
                 # This relies on the enrichment not changing gpu.vendor from, e.g. NVIDIA to CLOUD_AWS.
                 # The current _process_detected_gpus merges cloud info *into* local GPUs.
                 # So a NVIDIA GPU on AWS should still be handled by NvidiaDetector for telemetry.
                 # However, if it's purely a CloudDetector-found entry without local match, then use CloudDetector.
                 is_purely_cloud_detected = gpu.pci_bus_id == "N/A (Cloud)" and not any(
                     d_name in gpu.uuid for d_name in ["nvidia-", "amd-", "intel-", "apple-"] # Heuristic check if UUID implies local origin
                 )
                 if is_purely_cloud_detected:
                     detector_class_name = CloudDetector.__name__
                 else: # Fallback to trying to find a matching local detector if vendor implies it
                     if gpu.vendor == VendorType.NVIDIA: detector_class_name = NvidiaDetector.__name__ # ...etc. This could be more robust.


            if detector_class_name:
                detector_instance = next((det for det in self.detectors if det.__class__.__name__ == detector_class_name), None)
                if detector_instance:
                    gpu_detector_map[gpu.uuid] = detector_instance
                else:
                    logger.warning(f"Could not find detector instance for {detector_class_name} for GPU {gpu.uuid}")
                    gpu_detector_map[gpu.uuid] = UnknownDetector() # Fallback
            else:
                 gpu_detector_map[gpu.uuid] = UnknownDetector()

        last_db_log_time = time.monotonic()
        db_log_interval = 60 # Log to DB every 60 seconds (configurable)

        while not self.telemetry_stop_event.is_set():
            collected_telemetry_for_db: List[Tuple[str, TelemetryData]] = []
            if not self.gpus: # Handle case where GPUs might be removed dynamically (future)
                logger.info("No GPUs available for telemetry.")
                break

            for gpu in self.gpus:
                detector = gpu_detector_map.get(gpu.uuid)
                if not detector:
                    logger.warning(f"No telemetry detector found for GPU {gpu.uuid} ({gpu.name}). Skipping.")
                    continue
                try:
                    # Note: detector.get_telemetry() is currently synchronous.
                    # For a truly async telemetry system, this would need to be `await detector.get_telemetry_async(gpu)`
                    # and the collector task itself would be an asyncio task.
                    telemetry = detector.get_telemetry(gpu)
                    with self.telemetry_lock:
                        gpu.telemetry_history.append(telemetry)
                        if len(gpu.telemetry_history) > self.max_telemetry_history_per_gpu:
                            gpu.telemetry_history.pop(0)
                        gpu.temperature = int(telemetry.temperature) 
                        # Optionally update gpu.memory_free if telemetry provides current value.
                        # This would require get_telemetry to potentially return more, or a way to get memory info.
                        # For now, memory_free from initial detection remains.

                    collected_telemetry_for_db.append((gpu.uuid, telemetry))
                    logger.debug(f"Telemetry for {gpu.name} ({gpu.uuid}): Temp {telemetry.temperature}°C, Util {telemetry.gpu_utilization:.1f}%")
                except Exception as e: 
                    logger.error(f"Error getting telemetry for {gpu.name} ({gpu.uuid}): {e}", exc_info=False) # exc_info=False for less noise in telemetry loop
            
            current_time = time.monotonic()
            if collected_telemetry_for_db and (current_time - last_db_log_time >= db_log_interval):
                if self.db_manager.conn: # Ensure DB is available
                    self.db_manager.log_telemetry_batch(collected_telemetry_for_db)
                    last_db_log_time = current_time
                
            # Use telemetry_stop_event.wait for interruptible sleep
            # This makes shutdown quicker as it doesn't have to finish a full sleep interval.
            if self.telemetry_stop_event.wait(self.telemetry_interval_sec):
                break # Event was set, break the loop
        logger.info("📊 Telemetry collector thread stopped.")

    def start_telemetry_collection(self, interval_sec: int = 5, history_size: int = 100):
        if self.telemetry_thread and self.telemetry_thread.is_alive():
            logger.warning("Telemetry collection is already running.")
            return
        if not self.gpus:
            logger.info("No GPUs detected, telemetry collection will not start.")
            return
            
        self.telemetry_interval_sec = interval_sec
        self.max_telemetry_history_per_gpu = history_size
        self.telemetry_stop_event.clear() # Clear event before starting thread
        self.telemetry_thread = threading.Thread(target=self._telemetry_collector_task, daemon=True)
        self.telemetry_thread.start()

    def stop_telemetry_collection(self):
        if self.telemetry_thread and self.telemetry_thread.is_alive():
            logger.info("Stopping telemetry collection...")
            self.telemetry_stop_event.set()
            self.telemetry_thread.join(timeout=self.telemetry_interval_sec + 2) # Wait a bit longer than interval
            if self.telemetry_thread.is_alive(): 
                logger.warning("Telemetry thread did not stop in time.")
            self.telemetry_thread = None
        else: 
            logger.info("Telemetry collection is not running or already stopped.")
            
    def get_system_summary(self) -> Dict[str, Any]:
        return {
            "system": {
                "os": f"{self.os_type} {self.os_release} ({self.os_version})",
                "architecture": self.architecture, "hostname": self.hostname,
                "cpu": self.cpu_info_str, "total_ram_gb": self.total_ram_gb,
                "python_version": sys.version.split()[0],
                "fluentcompute_version": "0.1.0-MVP-Refactored" # Updated version
            },
            "gpus": [gpu.to_dict() for gpu in self.gpus] if self.gpus else []
        }

    def cleanup(self):
        logger.info("Cleaning up HardwareManager resources...")
        self.stop_telemetry_collection()
        if self.db_manager: 
            self.db_manager.close()
        
        for detector in self.detectors:
            try: 
                if hasattr(detector, 'cleanup') and callable(detector.cleanup):
                    detector.cleanup()
            except Exception as e: 
                logger.warning(f"Error cleaning up {detector.__class__.__name__}: {e}")
        
        # Shutdown CloudDetector's shared executor
        CloudDetector.shutdown_executor()
    
    # Implementing __del__ for automatic cleanup is often tricky with threads and external resources.
    # It's better to explicitly call cleanup(). However, if desired:
    # def __del__(self):
    #     self.cleanup()
```

---
**`fluentcompute_project/fluentcompute/__init__.py`**
---
```python
from .utils.logging_config import logger, get_logger, ColoredFormatter
from .core.hardware_manager import HardwareManager
from .models import (
    GPUInfo, TelemetryData, BenchmarkResult,
    VendorType, ComputeCapability, PerformanceTier, OptimizationStrategy
)
from .db.manager import DBManager

__version__ = "0.1.0-MVP-Refactored"

__all__ = [
    "logger", "get_logger", "ColoredFormatter",
    "HardwareManager",
    "GPUInfo", "TelemetryData", "BenchmarkResult",
    "VendorType", "ComputeCapability", "PerformanceTier", "OptimizationStrategy",
    "DBManager",
    "__version__"
]
