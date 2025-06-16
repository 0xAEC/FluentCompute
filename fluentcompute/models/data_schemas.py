from dataclasses import dataclass, field, asdict as dataclass_asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from .enums import VendorType, ComputeCapability, PerformanceTier

@dataclass
class TelemetryData:
    """Real-time hardware telemetry"""
    timestamp: datetime
    gpu_utilization: float
    memory_utilization: float # Percentage
    temperature: float # Celsius
    power_draw: float # Watts
    fan_speed: Optional[float] = None # Percentage
    clock_graphics: Optional[int] = None # MHz
    clock_memory: Optional[int] = None # MHz
    throttle_reasons: List[str] = field(default_factory=list)

    def to_dict(self):
        data = dataclass_asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class BenchmarkResult:
    """Hardware benchmark results"""
    compute_score: float # e.g. FP32 TFLOPs
    memory_bandwidth: float # Measured GB/s
    mixed_precision_score: float # e.g. FP16/BF16 TFLOPs
    tensor_throughput: float # e.g. INT8 TOPs or Tensor Core TFLOPs
    energy_efficiency: float # Score / Watt
    benchmark_date: datetime

    def to_dict(self):
        data = dataclass_asdict(self)
        data['benchmark_date'] = self.benchmark_date.isoformat()
        return data

@dataclass
class GPUInfo:
    """Comprehensive GPU information"""
    name: str
    vendor: VendorType
    gpu_index: int # Internal index assigned by FluentCompute during detection for a given vendor
    uuid: str # Unique identifier (e.g., NVML UUID, generated for others)
    pci_bus_id: str
    
    memory_total: int  # MB
    memory_free: int   # MB (at detection time, or latest from telemetry)
    memory_bandwidth: float  # GB/s (Peak theoretical or from spec)
    
    compute_capability: str # e.g., "7.5" for CUDA, "GFX908" for ROCm, "Metal Shading Language vX.Y"
    cuda_cores: Optional[int] = None
    rocm_cus: Optional[int] = None # AMD Compute Units
    intel_eus: Optional[int] = None # Intel Execution Units
    apple_gpu_cores: Optional[int] = None # Apple M-series GPU cores
    rt_cores: Optional[int] = None
    tensor_cores: Optional[int] = None # Or equivalent like Intel XMX, Apple ANE (though ANE is separate)
    
    base_clock: int = 0  # MHz
    boost_clock: int = 0  # MHz
    memory_clock: int = 0  # MHz
    performance_tier: PerformanceTier = PerformanceTier.UNKNOWN
    
    power_limit: int = 0  # Watts (TDP or configurable limit)
    temperature: int = 0  # Celsius (at detection time, or latest from telemetry)
    fan_speed: int = 0   # % (at detection time, or latest from telemetry)
    
    driver_version: str = ""
    cuda_version: str = "" # If NVIDIA
    rocm_version: str = "" # If AMD
    metal_version: str = "" # If Apple
    oneapi_level_zero_version: str = "" # If Intel
    
    supported_apis: List[ComputeCapability] = field(default_factory=list)
    
    nvlink_enabled: bool = False
    multi_gpu_capable: bool = False # Generic flag, vendor-specific (SLI, CrossFire, NVLink, xGMI)
    virtualization_support: bool = False # e.g. SR-IOV, MIG, vGPU
    
    telemetry_history: List[TelemetryData] = field(default_factory=list)
    benchmark_results: Optional[BenchmarkResult] = None
    
    instance_type: Optional[str] = None # Cloud specific
    cloud_region: Optional[str] = None # Cloud specific
    spot_price: Optional[float] = None # Cloud specific (if fetchable)

    def to_dict(self) -> Dict[str, Any]:
        data = dataclass_asdict(self)
        data['vendor'] = self.vendor.value
        data['performance_tier'] = self.performance_tier.name
        data['supported_apis'] = [api.value for api in self.supported_apis]
        if self.benchmark_results:
            data['benchmark_results'] = self.benchmark_results.to_dict()
        
        data['telemetry_history'] = [th.to_dict() for th in self.telemetry_history]
        return data
