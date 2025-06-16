from enum import Enum, auto

class VendorType(Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    CLOUD_AWS = "aws"
    CLOUD_GCP = "gcp"
    CLOUD_AZURE = "azure"
    UNKNOWN = "unknown"

class ComputeCapability(Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    OPENCL = "opencl"
    METAL = "metal"
    ONEAPI = "oneapi" # For Intel Level Zero / SYCL
    DIRECTML = "directml" # For Windows ML
    VULKAN = "vulkan" # General purpose compute

class PerformanceTier(Enum):
    ENTERPRISE = auto()    # RTX 6000, A6000, H100, MI300X
    PROFESSIONAL = auto()  # RTX 4090, 4080, A4000, RX 7900XTX, M-series Ultra/Max
    ENTHUSIAST = auto()    # RTX 4070, 3080, RX 7800XT, Arc A770, M-series Pro
    MAINSTREAM = auto()    # RTX 4060, 3060, RX 6600, Arc A580, M-series base
    BUDGET = auto()        # GTX 1650, older cards, Arc A380
    INTEGRATED = auto()    # iGPUs (Intel Iris Xe, AMD Radeon Graphics)
    UNKNOWN = auto()

class OptimizationStrategy(Enum):
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    BALANCED = "balanced"
    POWER_SAVE = "power_save"
