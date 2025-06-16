# fluentcompute/core/hardware_manager.py
import asyncio
import psutil
import socket
import platform
import sys
import subprocess # Keep for _get_cpu_info_str if needed
import threading
import time
import json
import logging # For log_level type hint
from typing import Dict, List, Optional, Tuple, Any

from fluentcompute.utils.logging_config import logger, get_logger
from fluentcompute.models import GPUInfo, TelemetryData, VendorType, PythonEnvironmentInfo # Added PythonEnvironmentInfo
from fluentcompute.detectors import (
    NvidiaDetector, AMDDetector, IntelDetector, AppleDetector,
    CloudDetector, UnknownDetector, HardwareDetector
)
from fluentcompute.environments.detector import EnvironmentDetector # New Import
from fluentcompute.db.manager import DBManager
from fluentcompute.config.settings import DEFAULT_LOG_LEVEL

class HardwareManager:
    def __init__(self, enable_cloud_detection: bool = True, log_level: int = DEFAULT_LOG_LEVEL, db_base_path: Optional[str] = None):
        global logger
        if logger.level != log_level:
            logger = get_logger(level=log_level)
        
        self.gpus: List[GPUInfo] = []
        self.detectors: List[HardwareDetector] = []
        self._register_detectors(enable_cloud_detection)
        
        # Initialize EnvironmentDetector
        self.env_detector = EnvironmentDetector()
        self.detected_environment: Optional[PythonEnvironmentInfo] = None # To store detected env info

        self.telemetry_thread: Optional[threading.Thread] = None
        self.telemetry_stop_event = threading.Event()
        self.telemetry_interval_sec = 5
        self.telemetry_lock = threading.Lock()
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
            try: return subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip() # Added text=True
            except Exception: pass
        elif self.os_type == "Linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if "model name" in line: return line.split(':', 1)[1].strip()
            except Exception: pass
        
        try:
            cpu_name = platform.processor()
            cores_logical = psutil.cpu_count(logical=True)
            cores_physical = psutil.cpu_count(logical=False)
            freq_str = ""
            if hasattr(psutil, 'cpu_freq'):
                try:
                    freq = psutil.cpu_freq()
                    if freq and freq.max > 0:
                        freq_str = f", {freq.max:.0f} MHz"
                except Exception: pass
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

    async def initialize_hardware_and_environment(self): # Renamed for clarity
        logger.info("🚀 Initializing hardware detection...")
        all_detected_gpus_by_detector: List[Tuple[str, List[GPUInfo]]] = []
        
        detection_coroutines = [detector.detect_hardware() for detector in self.detectors]
        results = await asyncio.gather(*detection_coroutines, return_exceptions=True)
        
        for i, result_item in enumerate(results):
            detector_name = self.detectors[i].__class__.__name__
            if isinstance(result_item, Exception):
                logger.error(f"❌ Error during hardware detection with {detector_name}: {result_item}", exc_info=True)
            elif isinstance(result_item, list):
                all_detected_gpus_by_detector.append((detector_name, result_item))
                if result_item:
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
        
        # Initialize environment after hardware
        logger.info("🚀 Initializing environment detection...")
        try:
            self.detected_environment = await self.env_detector.get_environment_info() # Now async
            if self.detected_environment:
                logger.info(f"✅ Environment detection complete. Type: {self.detected_environment.type}, Python: {self.detected_environment.python_version}")
                for fw in self.detected_environment.frameworks:
                    logger.info(f"  Framework: {fw.name} v{fw.version or 'N/A'}")
            else:
                logger.warning("⚠️ Environment detection did not return information.")
        except Exception as e_env:
            logger.error(f"❌ Error during environment detection: {e_env}", exc_info=True)

        self.db_manager.log_hardware_snapshot(await self.get_system_summary()) # Make get_system_summary async

    def _process_detected_gpus(self, detected_gpus_by_detector: List[Tuple[str, List[GPUInfo]]]) -> List[GPUInfo]:
        final_gpus_map: Dict[str, GPUInfo] = {} 
        local_detector_names = {NvidiaDetector.__name__, AMDDetector.__name__, IntelDetector.__name__, AppleDetector.__name__}

        for detector_name, gpu_list in detected_gpus_by_detector:
            if detector_name not in local_detector_names:
                continue
            for gpu in gpu_list:
                key = gpu.uuid if gpu.uuid and not any(s in gpu.uuid for s in ["unknown", "pci-", "smi-", "rocm-", "wmic-"]) else gpu.pci_bus_id
                if key and key.lower() not in ["unknown", "n/a (apple silicon)", "n/a (cloud)"]:
                    if key not in final_gpus_map:
                        final_gpus_map[key] = gpu
                    else:
                        logger.debug(f"Duplicate key {key} for GPU {gpu.name} from {detector_name}. Keeping first detailed entry.")
                else:
                    composite_key = f"{gpu.vendor.value}-{gpu.gpu_index}-{gpu.name[:10]}"
                    if composite_key not in final_gpus_map:
                        final_gpus_map[composite_key] = gpu

        cloud_detector_name = CloudDetector.__name__
        for detector_name, gpu_list in detected_gpus_by_detector:
            if detector_name != cloud_detector_name:
                continue
            
            for cloud_gpu in gpu_list:
                enriched_existing = False
                for local_key, local_gpu in final_gpus_map.items():
                    if local_gpu.vendor == cloud_gpu.vendor and not local_gpu.instance_type:
                        name_match = local_gpu.name in cloud_gpu.name or cloud_gpu.name in local_gpu.name
                        mem_match = abs(local_gpu.memory_total - cloud_gpu.memory_total) < (local_gpu.memory_total * 0.1)
                        if name_match or mem_match:
                            local_gpu.instance_type = cloud_gpu.instance_type
                            local_gpu.cloud_region = cloud_gpu.cloud_region
                            local_gpu.spot_price = cloud_gpu.spot_price
                            if "Cloud GPU" in local_gpu.name and "Cloud GPU" not in cloud_gpu.name:
                                local_gpu.name = cloud_gpu.name
                            logger.info(f"Enriched local GPU {local_gpu.name} (key: {local_key}) with cloud context from {cloud_gpu.name}.")
                            enriched_existing = True
                            break
                
                if not enriched_existing:
                    key = cloud_gpu.uuid if cloud_gpu.uuid and "cloud-" in cloud_gpu.uuid else f"cloud-{cloud_gpu.instance_type}-{cloud_gpu.name.replace(' ','_')}-{cloud_gpu.gpu_index}"
                    if key not in final_gpus_map:
                        final_gpus_map[key] = cloud_gpu
                    else:
                        final_gpus_map[f"{key}-{cloud_gpu.gpu_index}"] = cloud_gpu
        
        return sorted(list(final_gpus_map.values()), key=lambda gpu: (gpu.vendor.value, gpu.pci_bus_id if gpu.pci_bus_id else "", gpu.gpu_index))

    def get_all_gpus(self) -> List[GPUInfo]: return self.gpus
    def get_gpu_by_uuid(self, uuid: str) -> Optional[GPUInfo]:
        return next((gpu for gpu in self.gpus if gpu.uuid == uuid), None)

    async def _telemetry_collector_task_async(self): # Changed to async if get_telemetry becomes async
        logger.info("📊 Telemetry collector thread started.")
        gpu_detector_map: Dict[str, HardwareDetector] = {}
        
        for gpu in self.gpus:
            detector_class_name = ""
            if gpu.vendor == VendorType.NVIDIA: detector_class_name = NvidiaDetector.__name__
            elif gpu.vendor == VendorType.AMD: detector_class_name = AMDDetector.__name__
            elif gpu.vendor == VendorType.INTEL: detector_class_name = IntelDetector.__name__
            elif gpu.vendor == VendorType.APPLE: detector_class_name = AppleDetector.__name__
            elif gpu.vendor in [VendorType.CLOUD_AWS, VendorType.CLOUD_GCP, VendorType.CLOUD_AZURE]:
                 is_purely_cloud_detected = gpu.pci_bus_id == "N/A (Cloud)" and not any(
                     d_name in gpu.uuid for d_name in ["nvidia-", "amd-", "intel-", "apple-"]
                 )
                 if is_purely_cloud_detected: detector_class_name = CloudDetector.__name__
                 else: # Fallback based on vendor again (e.g. NVIDIA GPU on AWS)
                     if gpu.vendor == VendorType.NVIDIA: detector_class_name = NvidiaDetector.__name__
                     # Add similar for AMD/Intel on cloud if needed

            if detector_class_name:
                detector_instance = next((det for det in self.detectors if det.__class__.__name__ == detector_class_name), None)
                if detector_instance: gpu_detector_map[gpu.uuid] = detector_instance
                else:
                    logger.warning(f"Could not find detector for {detector_class_name} for GPU {gpu.uuid}")
                    gpu_detector_map[gpu.uuid] = UnknownDetector()
            else:
                 gpu_detector_map[gpu.uuid] = UnknownDetector()

        last_db_log_time = time.monotonic()
        db_log_interval = 60 

        while not self.telemetry_stop_event.is_set():
            collected_telemetry_for_db: List[Tuple[str, TelemetryData]] = []
            if not self.gpus:
                logger.info("No GPUs available for telemetry.")
                break

            telemetry_tasks = []
            for gpu in self.gpus:
                detector = gpu_detector_map.get(gpu.uuid)
                if detector:
                    # If detector.get_telemetry is async:
                    # telemetry_tasks.append(self._fetch_and_store_telemetry(gpu, detector))
                    # For now, assuming get_telemetry remains synchronous within the async collector task loop.
                    # This is not ideal for true async telemetry but matches current detector signatures.
                    try:
                        telemetry = detector.get_telemetry(gpu) # This is SYNC
                        with self.telemetry_lock:
                            gpu.telemetry_history.append(telemetry)
                            if len(gpu.telemetry_history) > self.max_telemetry_history_per_gpu:
                                gpu.telemetry_history.pop(0)
                            gpu.temperature = int(telemetry.temperature) # Type safety
                        collected_telemetry_for_db.append((gpu.uuid, telemetry))
                        logger.debug(f"Telemetry for {gpu.name} ({gpu.uuid}): Temp {telemetry.temperature}°C, Util {telemetry.gpu_utilization:.1f}%")
                    except Exception as e:
                        logger.error(f"Error getting sync telemetry for {gpu.name} ({gpu.uuid}): {e}", exc_info=False)
                else:
                     logger.warning(f"No telemetry detector found for GPU {gpu.uuid} ({gpu.name}). Skipping.")
            
            # await asyncio.gather(*telemetry_tasks) # If _fetch_and_store_telemetry was used

            current_time = time.monotonic()
            if collected_telemetry_for_db and (current_time - last_db_log_time >= db_log_interval):
                if self.db_manager.conn:
                    self.db_manager.log_telemetry_batch(collected_telemetry_for_db)
                    last_db_log_time = current_time
                
            # Asynchronous sleep, allows other asyncio tasks to run if any.
            # However, since _telemetry_collector_task_async is run in a thread, this interacts
            # with threading event.
            try:
                # To make this interruptible and work in a thread running an asyncio loop:
                # We need to run the telemetry loop itself as an asyncio task if it contains `await`.
                # For now, simplifying by keeping original threading model. The sleep here won't yield to other
                # *asyncio tasks within this thread* unless `asyncio.sleep` is used AND the surrounding
                # function is properly managed by an event loop (e.g., this whole telemetry task runs via asyncio.run_coroutine_threadsafe).
                
                # Using original threading.Event.wait for interruptible sleep.
                if self.telemetry_stop_event.wait(self.telemetry_interval_sec):
                    break # Event was set
            except RuntimeError as e_rt: # "Cannot run coroutine in a different loop" if mixing asyncio sleep in thread without setup
                 logger.debug(f"RuntimeError in telemetry sleep: {e_rt}. Falling back to time.sleep().")
                 time.sleep(self.telemetry_interval_sec) # Fallback if async sleep causes issues in current threading model
                 if self.telemetry_stop_event.is_set(): break
                 
        logger.info("📊 Telemetry collector thread stopped.")

    # Keeping telemetry collection in a separate thread as it was
    def start_telemetry_collection(self, interval_sec: int = 5, history_size: int = 100):
        if self.telemetry_thread and self.telemetry_thread.is_alive():
            logger.warning("Telemetry collection is already running.")
            return
        if not self.gpus:
            logger.info("No GPUs detected, telemetry collection will not start.")
            return
            
        self.telemetry_interval_sec = interval_sec
        self.max_telemetry_history_per_gpu = history_size
        self.telemetry_stop_event.clear()
        
        # The target function of the thread cannot directly be an async function (_telemetry_collector_task_async)
        # unless managed with asyncio.run_coroutine_threadsafe or by running an event loop inside the thread.
        # Sticking to the previous synchronous-looking _telemetry_collector_task target.
        # To make it truly async using `_telemetry_collector_task_async`:
        # self.telemetry_thread = threading.Thread(target=lambda: asyncio.run(self._telemetry_collector_task_async()), daemon=True)
        # This is a simplification; a proper async worker thread needs more robust loop management.
        # For now, let's keep the existing telemetry collection as a blocking task within the thread
        # and not make get_telemetry on detectors async yet.

        def _telemetry_task_wrapper(): # Wrapper for current synchronous telemetry logic
             loop = asyncio.new_event_loop()
             asyncio.set_event_loop(loop)
             try:
                 loop.run_until_complete(self._telemetry_collector_task_async())
             finally:
                 loop.close()
        # Still, direct asyncio calls within _telemetry_collector_task_async (like asyncio.sleep)
        # if run directly in a non-async aware thread are problematic.
        # For stability, reverting telemetry loop to be primarily synchronous and detectors.get_telemetry() also sync.
        # The telemetry collection is I/O bound (waiting for SMI/NVML), not CPU bound.
        # A simple thread is often sufficient if async complexity is to be avoided for now.
        # Renaming back _telemetry_collector_task_async to _telemetry_collector_task
        # and removing internal asyncio.sleep, using event.wait().

        self.telemetry_thread = threading.Thread(target=self._telemetry_collector_task, daemon=True)
        self.telemetry_thread.start()

    def _telemetry_collector_task(self): # Original synchronous version, best for threading for now
        logger.info("📊 Telemetry collector thread started.")
        # ... (Same content as original _telemetry_collector_task from existing codebase)
        # ... just ensure to use self.telemetry_stop_event.wait()
        # The GPU detector map logic here is fine.
        gpu_detector_map: Dict[str, HardwareDetector] = {} # Regenerate from scratch

        for gpu in self.gpus:
            detector_class_name = "" # Determine as before
            # ... (Existing logic for mapping gpu.vendor to detector class name)
            if gpu.vendor == VendorType.NVIDIA: detector_class_name = NvidiaDetector.__name__
            elif gpu.vendor == VendorType.AMD: detector_class_name = AMDDetector.__name__
            elif gpu.vendor == VendorType.INTEL: detector_class_name = IntelDetector.__name__
            elif gpu.vendor == VendorType.APPLE: detector_class_name = AppleDetector.__name__
            elif gpu.vendor in [VendorType.CLOUD_AWS, VendorType.CLOUD_GCP, VendorType.CLOUD_AZURE]:
                 is_purely_cloud_detected = gpu.pci_bus_id == "N/A (Cloud)" and not any(
                     d_name in gpu.uuid for d_name in ["nvidia-", "amd-", "intel-", "apple-"]
                 )
                 if is_purely_cloud_detected: detector_class_name = CloudDetector.__name__
                 else:
                     if gpu.vendor == VendorType.NVIDIA: detector_class_name = NvidiaDetector.__name__ # ...etc.

            if detector_class_name:
                detector_instance = next((det for det in self.detectors if det.__class__.__name__ == detector_class_name), None)
                if detector_instance:
                    gpu_detector_map[gpu.uuid] = detector_instance
                else:
                    logger.warning(f"Could not find detector instance for {detector_class_name} for GPU {gpu.uuid}")
                    gpu_detector_map[gpu.uuid] = UnknownDetector()
            else:
                 gpu_detector_map[gpu.uuid] = UnknownDetector()

        last_db_log_time = time.monotonic()
        db_log_interval = 60

        while not self.telemetry_stop_event.is_set():
            collected_telemetry_for_db: List[Tuple[str, TelemetryData]] = []
            if not self.gpus:
                logger.info("No GPUs available for telemetry.")
                break

            for gpu in self.gpus:
                detector = gpu_detector_map.get(gpu.uuid)
                if not detector:
                    logger.warning(f"No telemetry detector found for GPU {gpu.uuid} ({gpu.name}). Skipping.")
                    continue
                try:
                    telemetry = detector.get_telemetry(gpu) # SYNC CALL
                    with self.telemetry_lock:
                        gpu.telemetry_history.append(telemetry)
                        if len(gpu.telemetry_history) > self.max_telemetry_history_per_gpu:
                            gpu.telemetry_history.pop(0)
                        gpu.temperature = int(telemetry.temperature) # Type safety
                    collected_telemetry_for_db.append((gpu.uuid, telemetry))
                    logger.debug(f"Telemetry for {gpu.name} ({gpu.uuid}): Temp {telemetry.temperature}°C, Util {telemetry.gpu_utilization:.1f}%")
                except Exception as e:
                    logger.error(f"Error getting telemetry for {gpu.name} ({gpu.uuid}): {e}", exc_info=False)
            
            current_time = time.monotonic()
            if collected_telemetry_for_db and (current_time - last_db_log_time >= db_log_interval):
                if self.db_manager.conn:
                    self.db_manager.log_telemetry_batch(collected_telemetry_for_db)
                    last_db_log_time = current_time
            
            if self.telemetry_stop_event.wait(self.telemetry_interval_sec): # Uses threading.Event.wait
                break
        logger.info("📊 Telemetry collector thread stopped.")


    def stop_telemetry_collection(self):
        if self.telemetry_thread and self.telemetry_thread.is_alive():
            logger.info("Stopping telemetry collection...")
            self.telemetry_stop_event.set()
            self.telemetry_thread.join(timeout=self.telemetry_interval_sec + 2)
            if self.telemetry_thread.is_alive(): 
                logger.warning("Telemetry thread did not stop in time.")
            self.telemetry_thread = None
        else: 
            logger.info("Telemetry collection is not running or already stopped.")
            
    async def get_system_summary(self) -> Dict[str, Any]: # Changed to async
        # Ensure environment is detected if not already
        if not self.detected_environment and self.env_detector: # self.env_detector should exist
            try:
                self.detected_environment = await self.env_detector.get_environment_info()
            except Exception as e_sum_env:
                logger.error(f"Error getting environment info for summary: {e_sum_env}", exc_info=True)
                self.detected_environment = None # Explicitly set to None on error

        summary_data = {
            "system": {
                "os": f"{self.os_type} {self.os_release} ({self.os_version})",
                "architecture": self.architecture, "hostname": self.hostname,
                "cpu": self.cpu_info_str, "total_ram_gb": self.total_ram_gb,
                "python_version_fluentcompute": sys.version.split()[0], # Python running FluentCompute
                "fluentcompute_version": "0.1.0-Phase1-EnvDetect" # Updated version
            },
            "gpus": [gpu.to_dict() for gpu in self.gpus] if self.gpus else [],
            "environment": self.detected_environment.to_dict() if self.detected_environment else None
        }
        return summary_data

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
        
        CloudDetector.shutdown_executor()
