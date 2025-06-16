from abc import ABC, abstractmethod
from typing import List
from fluentcompute.models import GPUInfo, TelemetryData

class HardwareDetector(ABC):
    @abstractmethod
    async def detect_hardware(self) -> List[GPUInfo]:
        pass
    
    @abstractmethod
    def get_telemetry(self, gpu: GPUInfo) -> TelemetryData: # This should ideally be async too if it involves I/O
        pass

    def cleanup(self):
        """Optional cleanup method for detectors."""
        pass
