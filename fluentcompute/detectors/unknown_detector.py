from typing import List
from datetime import datetime

from fluentcompute.models import GPUInfo, TelemetryData
from .base import HardwareDetector
# from fluentcompute.utils.logging_config import logger # Not strictly needed if no logging

class UnknownDetector(HardwareDetector):
    async def detect_hardware(self) -> List[GPUInfo]:
        return [] # Does not detect anything new
    
    def get_telemetry(self, gpu: GPUInfo) -> TelemetryData:
        # Return minimal, neutral telemetry
        return TelemetryData(
            timestamp=datetime.now(),
            gpu_utilization=0.0,
            memory_utilization=0.0,
            temperature=float(gpu.temperature) if gpu.temperature else 0.0, # Use last known if any
            power_draw=0.0
        )

    # No cleanup needed
