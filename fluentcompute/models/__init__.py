# fluentcompute/models/__init__.py
from .enums import VendorType, ComputeCapability, PerformanceTier, OptimizationStrategy
from .data_schemas import (
    TelemetryData, 
    BenchmarkResult, 
    GPUInfo,
    FrameworkDependency, # New
    FrameworkInfo,       # New
    PythonEnvironmentInfo # New
)

__all__ = [
    "VendorType", "ComputeCapability", "PerformanceTier", "OptimizationStrategy",
    "TelemetryData", "BenchmarkResult", "GPUInfo",
    "FrameworkDependency", "FrameworkInfo", "PythonEnvironmentInfo" # Export new models
]
