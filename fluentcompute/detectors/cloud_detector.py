import asyncio
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import requests # External dependency

from fluentcompute.models import GPUInfo, TelemetryData, VendorType, PerformanceTier, ComputeCapability
from .base import HardwareDetector
from fluentcompute.utils.logging_config import logger


class CloudDetector(HardwareDetector):
    """Cloud instance detection (AWS, GCP, Azure)"""
    
    # Using a shared executor for all _async_http_get calls from this detector
    _executor = ThreadPoolExecutor(max_workers=3) # One for each cloud provider potential check

    async def detect_hardware(self) -> List[GPUInfo]:
        gpus = []
        cloud_provider = await self._detect_cloud_provider()
        
        if cloud_provider == VendorType.CLOUD_AWS:
            gpus.extend(await self._detect_aws_instances())
        elif cloud_provider == VendorType.CLOUD_GCP:
            gpus.extend(await self._detect_gcp_instances())
        elif cloud_provider == VendorType.CLOUD_AZURE:
            gpus.extend(await self._detect_azure_instances())
        
        if gpus: 
            logger.info(f"Detected running on {cloud_provider.value if cloud_provider else 'unknown cloud'}, found {len(gpus)} cloud GPU entries.")
        else:
            logger.debug("No specific cloud provider detected or no cloud GPU info found via metadata.")
        return gpus
    
    async def _detect_cloud_provider(self) -> Optional[VendorType]:
        # AWS: Check for instance identity document (most reliable) or specific metadata server files
        try:
            # Check for AWS Nitro-based instances (more modern metadata service)
            # The token is short-lived for security
            token_resp_text = await self._async_http_get('http://169.254.169.254/latest/api/token', 
                                                         headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'}, 
                                                         method='PUT', timeout=0.5)
            if token_resp_text:
                 # If token obtained, use it to fetch instance ID. If this succeeds, it's AWS.
                id_doc = await self._async_http_get('http://169.254.169.254/latest/dynamic/instance-identity/document', 
                                                    headers={'X-aws-ec2-metadata-token': token_resp_text}, timeout=0.5)
                if id_doc and "availabilityZone" in id_doc: return VendorType.CLOUD_AWS
        except Exception: pass # Could be older EC2 instance or not AWS

        try: # Fallback for older EC2 or if token method fails
            resp_aws = await self._async_http_get('http://169.254.169.254/latest/meta-data/instance-id', timeout=0.5)
            if resp_aws: return VendorType.CLOUD_AWS
        except Exception: pass
        
        # GCP: Check for metadata server with Google flavor
        try:
            resp_gcp = await self._async_http_get('http://metadata.google.internal/computeMetadata/v1/instance/id', 
                                               headers={'Metadata-Flavor': 'Google'}, timeout=0.5)
            if resp_gcp: return VendorType.CLOUD_GCP
        except Exception: pass

        # Azure: Check for metadata server with Azure-specific header/path
        try:
            resp_azure = await self._async_http_get('http://169.254.169.254/metadata/instance/compute/vmId?api-version=2021-02-01&format=text', 
                                                 headers={'Metadata': 'true'}, timeout=0.5)
            if resp_azure: return VendorType.CLOUD_AZURE
        except Exception: pass
        
        return None

    async def _async_http_get(self, url: str, headers: Optional[Dict] = None, 
                              timeout: float = 1.0, method: str = 'GET') -> Optional[str]:
        loop = asyncio.get_event_loop()
        try:
            if method.upper() == 'GET':
                future = loop.run_in_executor(self._executor, 
                                              lambda: requests.get(url, headers=headers, timeout=timeout))
            elif method.upper() == 'PUT': # For AWS token
                 future = loop.run_in_executor(self._executor, 
                                              lambda: requests.put(url, headers=headers, timeout=timeout))
            else:
                logger.warning(f"Unsupported HTTP method: {method} for {url}")
                return None
            
            response = await asyncio.wait_for(future, timeout=timeout + 0.2)
            response.raise_for_status() # Will raise HTTPError for bad responses (4xx or 5xx)
            return response.text
        except requests.exceptions.Timeout:
            logger.debug(f"Timeout ({timeout}s) fetching {url}")
        except requests.exceptions.RequestException as e:
            # More specific error for connection refused often means metadata server not there.
            if isinstance(e, requests.exceptions.ConnectionError):
                 logger.debug(f"Connection error for {url} (metadata endpoint likely not available).")
            else:
                 logger.debug(f"Request failed for {url}: {type(e).__name__} - {e}")
        except asyncio.TimeoutError: # This is the asyncio.wait_for timeout
            logger.debug(f"Async wrapper timeout for {url}")
        except Exception as e:
            logger.debug(f"Other error fetching {url}: {type(e).__name__} - {e}")
        return None

    async def _detect_aws_instances(self) -> List[GPUInfo]:
        gpus = []
        token = await self._async_http_get('http://169.254.169.254/latest/api/token', 
                                          headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'}, 
                                          method='PUT', timeout=0.5)
        aws_headers = {'X-aws-ec2-metadata-token': token} if token else {}

        try:
            instance_type = await self._async_http_get('http://169.254.169.254/latest/meta-data/instance-type', headers=aws_headers)
            region_az = await self._async_http_get('http://169.254.169.254/latest/meta-data/placement/availability-zone', headers=aws_headers)
            region = region_az[:-1] if region_az else None # e.g. us-east-1a -> us-east-1
            
            if not instance_type or not region: return []

            gpu_specs_list = self._get_aws_gpu_specs(instance_type)
            for i, spec in enumerate(gpu_specs_list):
                vendor_str = spec.get("vendor", "Unknown").lower()
                vendor_enum = VendorType.UNKNOWN
                if "nvidia" in vendor_str: vendor_enum = VendorType.NVIDIA
                elif "amd" in vendor_str: vendor_enum = VendorType.AMD
                elif "intel" in vendor_str or "habana" in vendor_str : vendor_enum = VendorType.INTEL


                gpu = GPUInfo(
                    name=spec.get("name", "AWS Cloud GPU"), vendor=vendor_enum, gpu_index=i,
                    uuid=f"aws-{instance_type}-{spec.get('name','GPU').replace(' ','_')}-{i}", 
                    pci_bus_id="N/A (Cloud)", # Could be found if local detectors also run
                    memory_total=spec.get("memory_mb", 0), memory_free=spec.get("memory_mb", 0),
                    memory_bandwidth=spec.get("memory_bandwidth_gbps", 0.0),
                    compute_capability=spec.get("compute_capability", "N/A"),
                    cuda_cores=spec.get("cuda_cores"), rocm_cus=spec.get("rocm_cus"), intel_eus=spec.get("intel_eus"),
                    performance_tier=self._determine_performance_tier_cloud(spec.get("name", "")),
                    driver_version="Cloud Provided", supported_apis=spec.get("supported_apis", []),
                    instance_type=instance_type, cloud_region=region,
                )
                gpus.append(gpu)
        except Exception as e: logger.error(f"❌ Error detecting AWS instances: {e}", exc_info=True)
        return gpus

    def _get_aws_gpu_specs(self, instance_type: str) -> List[Dict[str, Any]]:
        # Simplified map. This should ideally come from an external, updatable source or API.
        specs = { # Specs per *single* GPU unit within the instance type
            "p3.2xlarge": {"name": "NVIDIA Tesla V100", "memory_mb": 16384, "compute_capability": "7.0", "num_gpus": 1, "cuda_cores": 5120, "vendor": "NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            "p3.8xlarge": {"name": "NVIDIA Tesla V100", "memory_mb": 16384, "compute_capability": "7.0", "num_gpus": 4, "cuda_cores": 5120, "vendor": "NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            "p3.16xlarge": {"name": "NVIDIA Tesla V100", "memory_mb": 16384, "compute_capability": "7.0", "num_gpus": 8, "cuda_cores": 5120, "vendor": "NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            
            "g4dn.xlarge": {"name": "NVIDIA T4", "memory_mb": 16384, "compute_capability": "7.5", "num_gpus": 1, "cuda_cores": 2560, "vendor": "NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            # Add other g4dn sizes (e.g., g4dn.12xlarge has 4 T4s)

            "g5.xlarge": {"name": "NVIDIA A10G", "memory_mb": 24576, "compute_capability": "8.6", "num_gpus": 1, "cuda_cores": 9216, "vendor": "NVIDIA", "supported_apis": [ComputeCapability.CUDA]}, # A10G has 9216 for A10 not full GA102
            # Add other g5 sizes
            
            "p4d.24xlarge": {"name": "NVIDIA A100 40GB", "memory_mb": 40960, "compute_capability": "8.0", "num_gpus": 8, "cuda_cores": 6912, "vendor": "NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            "p4de.24xlarge": {"name": "NVIDIA A100 80GB", "memory_mb": 81920, "compute_capability": "8.0", "num_gpus": 8, "cuda_cores": 6912, "vendor": "NVIDIA", "supported_apis": [ComputeCapability.CUDA]},

            "p5.48xlarge": {"name": "NVIDIA H100", "memory_mb": 81920, "compute_capability": "9.0", "num_gpus": 8, "cuda_cores": 14592, "vendor": "NVIDIA", "supported_apis": [ComputeCapability.CUDA]}, # H100 SXM5 variant usually in P5

            "g4ad.xlarge": {"name": "AMD Radeon Pro V520", "memory_mb": 8192, "compute_capability": "gfx1011 (Navi 12)", "num_gpus": 1, "rocm_cus": 36, "vendor": "AMD", "supported_apis": [ComputeCapability.ROCM, ComputeCapability.OPENCL]},
            
            "dl1.24xlarge": {"name": "Habana Gaudi", "memory_mb": 32768, "compute_capability": "Intel Gaudi", "num_gpus": 8, "vendor":"INTEL", "supported_apis": []},
            "trn1.2xlarge": {"name": "AWS Trainium", "memory_mb": 32768, "compute_capability": "AWS Trainium", "num_gpus": 1, "vendor":"AWS", "supported_apis":[] }, # Placeholder, needs proper typing
        }
        
        instance_spec_data = specs.get(instance_type)
        if instance_spec_data:
            num_gpus = instance_spec_data.get("num_gpus", 1)
            single_gpu_spec_data = {k:v for k,v in instance_spec_data.items() if k != "num_gpus"}
            return [single_gpu_spec_data.copy() for _ in range(num_gpus)]
        
        # Heuristic for multi-GPU instances if only base is mapped (e.g. g5.xlarge -> g5.2xlarge = 2 GPUs)
        # This is very rough and prone to error, full map is better.
        base_type_match = re.match(r"([a-z0-9]+\.[a-z0-9]+)\.", instance_type) # e.g. p3 from p3.2xlarge
        if base_type_match:
             # Try a generic lookup using the first part, e.g., if "p3.2xlarge" exists, use its "name" for "p3.8xlarge"
            first_part_instance = base_type_match.group(1) + ".xlarge" # try common base size
            potential_base_spec = specs.get(first_part_instance)
            if potential_base_spec:
                 # Infer num_gpus from instance size string: 2xlarge = 1(base) * 2 / 2 = 1? No good general logic.
                 # Better to have full specs. For now, warning if not direct match.
                logger.warning(f"Using base spec for {first_part_instance} for {instance_type}, GPU count might be inaccurate.")
                # This doesn't properly handle num_gpus for derived types well without more logic.
                return [ {k:v for k,v in potential_base_spec.items() if k != "num_gpus"} ]


        logger.warning(f"No AWS GPU spec found for instance type: {instance_type}")
        return []

    async def _detect_gcp_instances(self) -> List[GPUInfo]:
        gpus_list = []
        headers_gcp = {'Metadata-Flavor': 'Google'}
        try:
            instance_name = await self._async_http_get('http://metadata.google.internal/computeMetadata/v1/instance/name', headers=headers_gcp)
            zone_full = await self._async_http_get('http://metadata.google.internal/computeMetadata/v1/instance/zone', headers=headers_gcp)
            machine_type_full = await self._async_http_get('http://metadata.google.internal/computeMetadata/v1/instance/machine-type', headers=headers_gcp)
            
            if not instance_name or not zone_full or not machine_type_full: return []
            zone = zone_full.split('/')[-1]
            machine_type_short = machine_type_full.split('/')[-1] # e.g. "a2-highgpu-1g"

            # GCP exposes attached GPUs under /instance/guest-attributes/accelerator/
            # Or directly under /instance/gpus/ for newer setups
            gpu_data_str = await self._async_http_get('http://metadata.google.internal/computeMetadata/v1/instance/gpus/?recursive=true', headers=headers_gcp)

            if gpu_data_str:
                try:
                    gpu_metadata_list = json.loads(gpu_data_str) # Expects a list of dicts for GPUs
                    if not isinstance(gpu_metadata_list, list): # some older GCP might return dict per GPU index
                        # if the output is like {"0": {"type": "nvidia-tesla-t4", ...}}
                        # then we need to handle it by making it a list. This seems unlikely for recursive=true.
                        logger.warning(f"Unexpected GCP GPU metadata format (expected list): {type(gpu_metadata_list)}")
                        # For simplicity, if not list, we assume it's like original iteration, where we got indices first
                        gpu_indices_resp = await self._async_http_get('http://metadata.google.internal/computeMetadata/v1/instance/gpus/', headers=headers_gcp)
                        if gpu_indices_resp:
                             gpu_indices = [idx.strip('/') for idx in gpu_indices_resp.strip().split('\n') if idx.strip().isdigit()]
                             gpu_metadata_list = []
                             for i_str in gpu_indices:
                                 gpu_type_resp = await self._async_http_get(f'http://metadata.google.internal/computeMetadata/v1/instance/gpus/{i_str}/type', headers=headers_gcp)
                                 if gpu_type_resp: gpu_metadata_list.append({"type": gpu_type_resp, "_fc_index": int(i_str)}) # add our own index
                    
                    for i, gpu_meta in enumerate(gpu_metadata_list):
                        gpu_type_name_raw = gpu_meta.get("type", gpu_meta.get("name")) # 'type' from /gpus, 'name' from guest-attributes?
                        if not gpu_type_name_raw: continue
                        
                        gpu_type_name = gpu_type_name_raw.split('/')[-1] # e.g. projects/../acceleratorTypes/nvidia-tesla-t4 -> nvidia-tesla-t4

                        spec = self._get_gcp_gpu_spec(gpu_type_name)
                        vendor_str = spec.get("vendor", "NVIDIA").lower() # Default to NVIDIA for GCP historic reasons
                        vendor_enum = VendorType.NVIDIA
                        if "amd" in vendor_str: vendor_enum = VendorType.AMD
                        elif "intel" in vendor_str: vendor_enum = VendorType.INTEL
                        
                        # Use index from metadata if available, else enumerate
                        gpu_idx = gpu_meta.get("_fc_index", gpu_meta.get("index", i)) 

                        gpu = GPUInfo(
                            name=spec.get("name", gpu_type_name), vendor=vendor_enum, gpu_index=gpu_idx,
                            uuid=f"gcp-{instance_name}-{spec.get('name','GPU').replace(' ','_')}-{gpu_idx}", 
                            pci_bus_id="N/A (Cloud)",
                            memory_total=spec.get("memory_mb", 0), memory_free=spec.get("memory_mb", 0),
                            memory_bandwidth=spec.get("memory_bandwidth_gbps", 0.0),
                            compute_capability=spec.get("compute_capability", "N/A"),
                            cuda_cores=spec.get("cuda_cores"),
                            performance_tier=self._determine_performance_tier_cloud(spec.get("name", "")),
                            driver_version=gpu_meta.get("driver_version", "Cloud Provided"), # Driver from metadata if present
                            supported_apis=spec.get("supported_apis", []),
                            instance_type=machine_type_short, cloud_region=zone,
                        )
                        gpus_list.append(gpu)
                except json.JSONDecodeError as e_json:
                     logger.warning(f"Failed to parse GCP GPU metadata JSON: {e_json}. Raw: {gpu_data_str[:200]}")
                except Exception as e_parse: # Catch other parsing errors
                     logger.error(f"Error parsing GCP GPU data: {e_parse}", exc_info=True)
            else: # Fallback to guest attributes for older systems (less likely but possible)
                logger.debug("GCP /instance/gpus/ empty or failed, guest attributes not yet implemented for GCP.")

        except Exception as e: logger.error(f"❌ Error detecting GCP instances: {e}", exc_info=True)
        return gpus_list

    def _get_gcp_gpu_spec(self, gpu_type_name_from_meta: str) -> Dict[str, Any]:
        # map key is the `type` field from GCP metadata e.g. "nvidia-tesla-t4"
        specs_map = {
            "nvidia-tesla-v100": {"name": "NVIDIA Tesla V100", "memory_mb": 16384, "compute_capability": "7.0", "cuda_cores": 5120, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            "nvidia-tesla-t4": {"name": "NVIDIA T4", "memory_mb": 16384, "compute_capability": "7.5", "cuda_cores": 2560, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            "nvidia-tesla-a100": {"name": "NVIDIA A100 40GB", "memory_mb": 40960, "compute_capability": "8.0", "cuda_cores": 6912, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            "nvidia-a100-80gb": {"name": "NVIDIA A100 80GB", "memory_mb": 81920, "compute_capability": "8.0", "cuda_cores": 6912, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            "nvidia-l4": {"name": "NVIDIA L4", "memory_mb": 24576, "compute_capability": "8.9", "cuda_cores": 7424, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            "nvidia-h100-80gb": {"name": "NVIDIA H100 80GB", "memory_mb": 81920, "compute_capability": "9.0", "cuda_cores": 14592, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]}, # Assuming PCIe variant for H100
        }
        
        # Normalize the name from metadata which might include project prefixes
        normalized_type = gpu_type_name_from_meta.split('/')[-1] 
        if normalized_type in specs_map: return specs_map[normalized_type].copy()
        
        logger.warning(f"No explicit GCP GPU spec found for type: {gpu_type_name_from_meta} (normalized: {normalized_type}). Inferring.")
        # Basic inference
        inferred_spec = {"name": normalized_type.replace('-', ' ').title(), "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]}
        if "amd" in normalized_type:
            inferred_spec["vendor"] = "AMD"
            inferred_spec["supported_apis"] = [ComputeCapability.ROCM, ComputeCapability.OPENCL]
        return inferred_spec

    async def _detect_azure_instances(self) -> List[GPUInfo]:
        gpus_list = []
        headers_azure = {'Metadata': 'true'}
        try:
            # Azure instance metadata is typically available without special tokens like AWS Nitro.
            vm_size = await self._async_http_get('http://169.254.169.254/metadata/instance/compute/vmSize?api-version=2021-02-01&format=text', headers=headers_azure)
            location = await self._async_http_get('http://169.254.169.254/metadata/instance/compute/location?api-version=2021-02-01&format=text', headers=headers_azure)
            
            # Additional SKU info for more precise GPU matching (especially for N-series v5)
            sku = await self._async_http_get('http://169.254.169.254/metadata/instance/compute/sku?api-version=2021-02-01&format=text', headers=headers_azure)

            if not vm_size or not location: return []

            # Azure sometimes puts GPU info in a more structured way for newer VMs,
            # but for now, rely on mapping vmSize to known GPU configurations.
            # The 'sku' can help differentiate variants, e.g. "Standard_ND96amsr_A100_v4"
            
            # Pass sku to a potentially more detailed getter
            gpu_specs_list = self._get_azure_gpu_specs(vm_size, sku) 
            for i, spec in enumerate(gpu_specs_list):
                vendor_str = spec.get("vendor", "Unknown").lower()
                vendor_enum = VendorType.NVIDIA # Azure primarily NVIDIA, some AMD MI series
                if "amd" in vendor_str: vendor_enum = VendorType.AMD
                elif "intel" in vendor_str : vendor_enum = VendorType.INTEL
                
                gpu = GPUInfo(
                    name=spec.get("name", "Azure Cloud GPU"), vendor=vendor_enum, gpu_index=i,
                    uuid=f"azure-{vm_size}-{spec.get('name','GPU').replace(' ','_')}-{i}", 
                    pci_bus_id="N/A (Cloud)",
                    memory_total=spec.get("memory_mb", 0), memory_free=spec.get("memory_mb", 0),
                    memory_bandwidth=spec.get("memory_bandwidth_gbps", 0.0),
                    compute_capability=spec.get("compute_capability", "N/A"),
                    cuda_cores=spec.get("cuda_cores"), rocm_cus=spec.get("rocm_cus"),
                    performance_tier=self._determine_performance_tier_cloud(spec.get("name", "")),
                    driver_version="Cloud Provided", supported_apis=spec.get("supported_apis", []),
                    instance_type=sku if sku else vm_size, # Prefer more specific SKU if available
                    cloud_region=location,
                )
                gpus_list.append(gpu)
        except Exception as e: logger.error(f"❌ Error detecting Azure instances: {e}", exc_info=True)
        return gpus_list

    def _get_azure_gpu_specs(self, vm_size: str, sku: Optional[str]=None) -> List[Dict[str, Any]]:
        # Azure instance SKUs (vmSize or the more detailed 'sku') map to GPU specs.
        # This map needs to be comprehensive. Using vm_size as primary key.
        # Format: { "vmSize_or_sku": { "name": ..., "num_gpus": X, ... } }
        # Source: Azure documentation on GPU-optimized virtual machine sizes.
        
        # Primary map using vmSize
        specs_map = {
            # V100
            "Standard_NC6s_v3": {"name": "NVIDIA Tesla V100", "memory_mb": 16384, "compute_capability": "7.0", "cuda_cores": 5120, "num_gpus": 1, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            "Standard_NC12s_v3": {"name": "NVIDIA Tesla V100", "memory_mb": 16384, "compute_capability": "7.0", "cuda_cores": 5120, "num_gpus": 2, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            "Standard_NC24s_v3": {"name": "NVIDIA Tesla V100", "memory_mb": 16384, "compute_capability": "7.0", "cuda_cores": 5120, "num_gpus": 4, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            "Standard_NC24rs_v3": {"name": "NVIDIA Tesla V100", "memory_mb": 16384, "compute_capability": "7.0", "cuda_cores": 5120, "num_gpus": 4, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]}, # RDMA

            # T4
            "Standard_NC4as_T4_v3": {"name": "NVIDIA T4", "memory_mb": 16384, "compute_capability": "7.5", "cuda_cores": 2560, "num_gpus": 1, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            # Add NC8as_T4_v3 (1), NC16as_T4_v3 (1), NC64as_T4_v3 (4) - note: NC T4s often have fractional GPUs in some SKUs. Here listed means full.
            
            # A100 (ND A100 v4 series)
            "Standard_ND96asr_v4": {"name": "NVIDIA A100 40GB", "memory_mb": 40960, "compute_capability": "8.0", "cuda_cores": 6912, "num_gpus": 8, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]}, # PCIe
            "Standard_ND96amsr_A100_v4": {"name": "NVIDIA A100 80GB", "memory_mb": 81920, "compute_capability": "8.0", "cuda_cores": 6912, "num_gpus": 8, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]}, # SXM

            # A10 (NVads A10 v5 series - these are often PARTITIONS of A10s, map gives full spec per reported GPU)
            # Example, NV6ads_A10_v5 is 1/6th of an A10. For simplicity, if FluentCompute treats these partitions as "a GPU",
            # the memory/cores would need to be divided. Current map gives full A10 spec. This is an area for refinement.
            "Standard_NV6ads_A10_v5": {"name": "NVIDIA A10 (Partition)", "memory_mb": 24576 // 6, "compute_capability": "8.6", "cuda_cores": 9216 // 6, "num_gpus": 1, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]},
            "Standard_NV12ads_A10_v5": {"name": "NVIDIA A10 (Partition)", "memory_mb": 24576 // 3, "compute_capability": "8.6", "cuda_cores": 9216 // 3, "num_gpus": 1, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]}, # effectively 1/3 of an A10 for this instance size
            # ... and so on for full A10 (NV72ads_A10_v5 implies 3 full A10s or 72/72=1 of a large type)

            # H100 (ND H100 v5 series)
             "Standard_ND96ix_H100_v5": {"name": "NVIDIA H100 80GB", "memory_mb": 81920, "compute_capability": "9.0", "cuda_cores": 14592, "num_gpus": 8, "vendor":"NVIDIA", "supported_apis": [ComputeCapability.CUDA]}, # SXM5 (Hopper)


            # AMD (ND MI200 v5 series)
            "Standard_ND96Mi_v5": {"name": "AMD Instinct MI200", "memory_mb": 65536, "compute_capability": "gfx90a", "rocm_cus": 104, "num_gpus": 1, "vendor": "AMD", "supported_apis": [ComputeCapability.ROCM, ComputeCapability.OPENCL]} # MI200 has 2 GCDs, here treating instance 'GPU' as one GCD

        }
        
        lookup_key = sku if sku and sku in specs_map else vm_size # Prefer SKU if it's in map
        base_spec_info = specs_map.get(lookup_key)

        if base_spec_info:
            num_gpus = base_spec_info.get("num_gpus", 1)
            single_gpu_spec_data = {k:v for k,v in base_spec_info.items() if k != "num_gpus"}
            return [single_gpu_spec_data.copy() for _ in range(num_gpus)]
        
        logger.warning(f"No Azure GPU spec found for VM Size: {vm_size} (SKU: {sku or 'N/A'}).")
        return []

    def _determine_performance_tier_cloud(self, gpu_name_str: str) -> PerformanceTier:
        name = gpu_name_str.upper()
        if any(x in name for x in ["A100", "H100", "V100", "GAUDI", "MI200", "TRAINIUM"]): return PerformanceTier.ENTERPRISE
        if any(x in name for x in ["A10G", "A10", "T4", "L4", "RADEON PRO V520"]): return PerformanceTier.PROFESSIONAL # T4 could also be enthusiast/mainstream depending on use
        return PerformanceTier.MAINSTREAM 

    def get_telemetry(self, gpu: GPUInfo) -> TelemetryData:
        # As with other cloud providers, true telemetry usually involves running vendor tools (nvidia-smi)
        # on the instance or using Azure Monitor metrics (which requires Azure SDK/API integration).
        # This is a placeholder. If local detectors run on the cloud VM, they'll provide better telemetry.
        logger.debug(f"Basic cloud telemetry for Azure GPU {gpu.name}. Detailed metrics would require Azure Monitor or on-VM tools.")
        return TelemetryData(
            timestamp=datetime.now(), 
            gpu_utilization=0.0, 
            memory_utilization=0.0, # Cloud instances report full memory at init; free is dynamic
            temperature=float(gpu.temperature) if gpu.temperature else 0.0, 
            power_draw=0.0
        )
    
    @classmethod
    def shutdown_executor(cls):
        """Should be called on application cleanup if CloudDetector was used."""
        logger.debug("Shutting down CloudDetector's ThreadPoolExecutor.")
        cls._executor.shutdown(wait=True)
