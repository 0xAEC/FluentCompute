# fluentcompute/detectors/cloud_detector.py
import asyncio
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path # For loading specs

import requests # External dependency

from fluentcompute.models import GPUInfo, TelemetryData, VendorType, PerformanceTier, ComputeCapability
from .base import HardwareDetector
from fluentcompute.utils.logging_config import logger


class CloudDetector(HardwareDetector):
    _SPEC_FILE_NAME = "cloud_specs.json"
    _executor = ThreadPoolExecutor(max_workers=3) 

    def __init__(self):
        self.specs: Dict[str, Any] = self._load_specs()

    def _load_specs(self) -> Dict[str, Any]:
        try:
            spec_path = Path(__file__).resolve().parent.parent / "data" / "specs" / self._SPEC_FILE_NAME
            if not spec_path.exists():
                logger.warning(f"Cloud spec file not found at {spec_path}. Using empty specs.")
                return {}
            with open(spec_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load Cloud specs from {self._SPEC_FILE_NAME}: {e}", exc_info=True)
            return {}
            
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
        try: # AWS Nitro
            token_resp_text = await self._async_http_get('http://169.254.169.254/latest/api/token', 
                                                         headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'}, 
                                                         method='PUT', timeout=0.5)
            if token_resp_text:
                id_doc = await self._async_http_get('http://169.254.169.254/latest/dynamic/instance-identity/document', 
                                                    headers={'X-aws-ec2-metadata-token': token_resp_text}, timeout=0.5)
                if id_doc and "availabilityZone" in id_doc: return VendorType.CLOUD_AWS
        except Exception: pass

        try: # AWS Legacy
            if await self._async_http_get('http://169.254.169.254/latest/meta-data/instance-id', timeout=0.5):
                return VendorType.CLOUD_AWS
        except Exception: pass
        
        try: # GCP
            if await self._async_http_get('http://metadata.google.internal/computeMetadata/v1/instance/id', 
                                               headers={'Metadata-Flavor': 'Google'}, timeout=0.5):
                return VendorType.CLOUD_GCP
        except Exception: pass

        try: # Azure
            if await self._async_http_get('http://169.254.169.254/metadata/instance/compute/vmId?api-version=2021-02-01&format=text', 
                                                 headers={'Metadata': 'true'}, timeout=0.5):
                return VendorType.CLOUD_AZURE
        except Exception: pass
        return None

    async def _async_http_get(self, url: str, headers: Optional[Dict] = None, 
                              timeout: float = 1.0, method: str = 'GET') -> Optional[str]:
        loop = asyncio.get_event_loop()
        try:
            request_func = requests.get if method.upper() == 'GET' else requests.put
            if method.upper() not in ['GET', 'PUT']:
                logger.warning(f"Unsupported HTTP method: {method} for {url}")
                return None
            
            future = loop.run_in_executor(self._executor, 
                                          lambda: request_func(url, headers=headers, timeout=timeout))
            response = await asyncio.wait_for(future, timeout=timeout + 0.2)
            response.raise_for_status()
            return response.text
        except requests.exceptions.Timeout: logger.debug(f"Timeout ({timeout}s) fetching {url}")
        except requests.exceptions.ConnectionError: logger.debug(f"Connection error for {url}.")
        except requests.exceptions.RequestException as e: logger.debug(f"Request failed for {url}: {e}")
        except asyncio.TimeoutError: logger.debug(f"Async wrapper timeout for {url}")
        except Exception as e: logger.debug(f"Other error fetching {url}: {e}")
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
            region = region_az[:-1] if region_az else None
            
            if not instance_type or not region: return []

            gpu_specs_list = self._get_aws_gpu_specs(instance_type)
            for i, spec in enumerate(gpu_specs_list):
                vendor_enum = VendorType[spec.get("vendor", "UNKNOWN").upper()] rescue VendorType.UNKNOWN
                
                # Convert string API names to Enum members
                api_enums = []
                for api_str in spec.get("supported_apis", []):
                    try: api_enums.append(ComputeCapability[api_str.upper()])
                    except KeyError: logger.warning(f"Unknown API string '{api_str}' in AWS spec for {spec.get('name')}")


                gpu = GPUInfo(
                    name=spec.get("name", "AWS Cloud GPU"), vendor=vendor_enum, gpu_index=i,
                    uuid=f"aws-{instance_type}-{spec.get('name','GPU').replace(' ','_')}-{i}", 
                    pci_bus_id="N/A (Cloud)",
                    memory_total=spec.get("memory_mb", 0), memory_free=spec.get("memory_mb", 0),
                    memory_bandwidth=spec.get("memory_bandwidth_gbps", 0.0),
                    compute_capability=spec.get("compute_capability", "N/A"),
                    cuda_cores=spec.get("cuda_cores"), rocm_cus=spec.get("rocm_cus"), intel_eus=spec.get("intel_eus"),
                    performance_tier=self._determine_performance_tier_cloud(spec.get("name", "")),
                    driver_version="Cloud Provided", supported_apis=api_enums,
                    instance_type=instance_type, cloud_region=region,
                )
                gpus.append(gpu)
        except Exception as e: logger.error(f"❌ Error detecting AWS instances: {e}", exc_info=True)
        return gpus

    def _get_aws_gpu_specs(self, instance_type: str) -> List[Dict[str, Any]]:
        aws_specs_data = self.specs.get("aws", {}).get("instance_specs", {})
        instance_spec_data = aws_specs_data.get(instance_type)
        if instance_spec_data:
            num_gpus = instance_spec_data.get("num_gpus", 1)
            single_gpu_spec_data = {k:v for k,v in instance_spec_data.items() if k != "num_gpus"}
            return [single_gpu_spec_data.copy() for _ in range(num_gpus)]
        
        # Heuristic fallback (simplified)
        base_type_match = re.match(r"([a-z0-9]+\.[a-z0-9]+)\.", instance_type)
        if base_type_match:
            first_part_instance = base_type_match.group(1) + ".xlarge"
            potential_base_spec = aws_specs_data.get(first_part_instance)
            if potential_base_spec:
                logger.warning(f"Using base spec for {first_part_instance} for {instance_type}, GPU count might be inaccurate.")
                return [ {k:v for k,v in potential_base_spec.items() if k != "num_gpus"} ]
        logger.warning(f"No AWS GPU spec found for instance type: {instance_type}")
        return []

    async def _detect_gcp_instances(self) -> List[GPUInfo]:
        gpus_list = []
        headers_gcp = {'Metadata-Flavor': 'Google'}
        try:
            instance_name_path = 'http://metadata.google.internal/computeMetadata/v1/instance/name'
            zone_path = 'http://metadata.google.internal/computeMetadata/v1/instance/zone'
            machine_type_path = 'http://metadata.google.internal/computeMetadata/v1/instance/machine-type'
            gpus_path = 'http://metadata.google.internal/computeMetadata/v1/instance/gpus/?recursive=true'

            instance_name, zone_full, machine_type_full, gpu_data_str = await asyncio.gather(
                self._async_http_get(instance_name_path, headers=headers_gcp),
                self._async_http_get(zone_path, headers=headers_gcp),
                self._async_http_get(machine_type_path, headers=headers_gcp),
                self._async_http_get(gpus_path, headers=headers_gcp)
            )
            
            if not all([instance_name, zone_full, machine_type_full]): return []
            zone = zone_full.split('/')[-1]
            machine_type_short = machine_type_full.split('/')[-1]

            if gpu_data_str:
                try:
                    gpu_metadata_list = json.loads(gpu_data_str)
                    if not isinstance(gpu_metadata_list, list): # Fallback parsing
                        gpu_indices_resp = await self._async_http_get('http://metadata.google.internal/computeMetadata/v1/instance/gpus/', headers=headers_gcp)
                        if gpu_indices_resp:
                             gpu_indices = [idx.strip('/') for idx in gpu_indices_resp.strip().split('\n') if idx.strip().isdigit()]
                             gpu_metadata_list = []
                             for i_str in gpu_indices:
                                 gpu_type_resp = await self._async_http_get(f'http://metadata.google.internal/computeMetadata/v1/instance/gpus/{i_str}/type', headers=headers_gcp)
                                 if gpu_type_resp: gpu_metadata_list.append({"type": gpu_type_resp, "_fc_index": int(i_str)})
                    
                    for i, gpu_meta in enumerate(gpu_metadata_list):
                        gpu_type_name_raw = gpu_meta.get("type", gpu_meta.get("name"))
                        if not gpu_type_name_raw: continue
                        gpu_type_name = gpu_type_name_raw.split('/')[-1]

                        spec = self._get_gcp_gpu_spec(gpu_type_name)
                        vendor_enum = VendorType[spec.get("vendor", "NVIDIA").upper()] rescue VendorType.UNKNOWN
                        gpu_idx = gpu_meta.get("_fc_index", gpu_meta.get("index", i)) 
                        
                        api_enums = []
                        for api_str in spec.get("supported_apis", []):
                            try: api_enums.append(ComputeCapability[api_str.upper()])
                            except KeyError: logger.warning(f"Unknown API string '{api_str}' in GCP spec for {spec.get('name')}")
                        
                        gpu = GPUInfo(
                            name=spec.get("name", gpu_type_name), vendor=vendor_enum, gpu_index=gpu_idx,
                            uuid=f"gcp-{instance_name}-{spec.get('name','GPU').replace(' ','_')}-{gpu_idx}", 
                            pci_bus_id="N/A (Cloud)",
                            memory_total=spec.get("memory_mb", 0), memory_free=spec.get("memory_mb", 0),
                            memory_bandwidth=spec.get("memory_bandwidth_gbps", 0.0),
                            compute_capability=spec.get("compute_capability", "N/A"),
                            cuda_cores=spec.get("cuda_cores"),
                            performance_tier=self._determine_performance_tier_cloud(spec.get("name", "")),
                            driver_version=gpu_meta.get("driver_version", "Cloud Provided"),
                            supported_apis=api_enums,
                            instance_type=machine_type_short, cloud_region=zone,
                        )
                        gpus_list.append(gpu)
                except json.JSONDecodeError as e: logger.warning(f"Failed to parse GCP GPU JSON: {e}.")
                except Exception as e_parse: logger.error(f"Error parsing GCP GPU data: {e_parse}", exc_info=True)
        except Exception as e: logger.error(f"❌ Error detecting GCP instances: {e}", exc_info=True)
        return gpus_list

    def _get_gcp_gpu_spec(self, gpu_type_name_from_meta: str) -> Dict[str, Any]:
        gcp_specs_data = self.specs.get("gcp", {}).get("gpu_type_specs", {})
        normalized_type = gpu_type_name_from_meta.split('/')[-1] 
        if normalized_type in gcp_specs_data: return gcp_specs_data[normalized_type].copy()
        
        logger.warning(f"No explicit GCP GPU spec for type: {normalized_type}. Inferring.")
        inferred_spec = {"name": normalized_type.replace('-', ' ').title(), "vendor":"NVIDIA", "supported_apis": ["cuda"]}
        if "amd" in normalized_type:
            inferred_spec["vendor"] = "AMD"; inferred_spec["supported_apis"] = ["rocm", "opencl"]
        return inferred_spec

    async def _detect_azure_instances(self) -> List[GPUInfo]:
        gpus_list = []
        headers_azure = {'Metadata': 'true'}
        base_url = 'http://169.254.169.254/metadata/instance/compute'
        api_ver_format = '?api-version=2021-02-01&format=text'
        try:
            vm_size_path = f"{base_url}/vmSize{api_ver_format}"
            location_path = f"{base_url}/location{api_ver_format}"
            sku_path = f"{base_url}/sku{api_ver_format}"

            vm_size, location, sku = await asyncio.gather(
                self._async_http_get(vm_size_path, headers=headers_azure),
                self._async_http_get(location_path, headers=headers_azure),
                self._async_http_get(sku_path, headers=headers_azure)
            )

            if not vm_size or not location: return []
            
            gpu_specs_list = self._get_azure_gpu_specs(vm_size, sku) 
            for i, spec in enumerate(gpu_specs_list):
                vendor_enum = VendorType[spec.get("vendor", "NVIDIA").upper()] rescue VendorType.UNKNOWN
                api_enums = []
                for api_str in spec.get("supported_apis", []):
                    try: api_enums.append(ComputeCapability[api_str.upper()])
                    except KeyError: logger.warning(f"Unknown API string '{api_str}' in Azure spec for {spec.get('name')}")

                gpu = GPUInfo(
                    name=spec.get("name", "Azure Cloud GPU"), vendor=vendor_enum, gpu_index=i,
                    uuid=f"azure-{vm_size}-{spec.get('name','GPU').replace(' ','_')}-{i}", 
                    pci_bus_id="N/A (Cloud)",
                    memory_total=spec.get("memory_mb", 0), memory_free=spec.get("memory_mb", 0),
                    memory_bandwidth=spec.get("memory_bandwidth_gbps", 0.0),
                    compute_capability=spec.get("compute_capability", "N/A"),
                    cuda_cores=spec.get("cuda_cores"), rocm_cus=spec.get("rocm_cus"),
                    performance_tier=self._determine_performance_tier_cloud(spec.get("name", "")),
                    driver_version="Cloud Provided", supported_apis=api_enums,
                    instance_type=sku if sku else vm_size, cloud_region=location,
                )
                gpus_list.append(gpu)
        except Exception as e: logger.error(f"❌ Error detecting Azure instances: {e}", exc_info=True)
        return gpus_list

    def _get_azure_gpu_specs(self, vm_size: str, sku: Optional[str]=None) -> List[Dict[str, Any]]:
        azure_specs_data = self.specs.get("azure", {}).get("vm_size_specs", {})
        lookup_key = sku if sku and sku in azure_specs_data else vm_size
        base_spec_info = azure_specs_data.get(lookup_key)

        if base_spec_info:
            num_gpus = base_spec_info.get("num_gpus", 1)
            single_gpu_spec_data = {k:v for k,v in base_spec_info.items() if k != "num_gpus"}
            return [single_gpu_spec_data.copy() for _ in range(num_gpus)]
        
        logger.warning(f"No Azure GPU spec found for VM Size: {vm_size} (SKU: {sku or 'N/A'}).")
        return []

    def _determine_performance_tier_cloud(self, gpu_name_str: str) -> PerformanceTier:
        name = gpu_name_str.upper()
        if any(x in name for x in ["A100", "H100", "V100", "GAUDI", "MI200", "TRAINIUM"]): return PerformanceTier.ENTERPRISE
        if any(x in name for x in ["A10G", "A10", "T4", "L4", "RADEON PRO V520"]): return PerformanceTier.PROFESSIONAL
        return PerformanceTier.MAINSTREAM 

    def get_telemetry(self, gpu: GPUInfo) -> TelemetryData:
        logger.debug(f"Basic cloud telemetry for {gpu.vendor.value} GPU {gpu.name}.")
        return TelemetryData(
            timestamp=datetime.now(), gpu_utilization=0.0, memory_utilization=0.0,
            temperature=float(gpu.temperature or 0.0), power_draw=0.0 # Added or 0.0
        )
    
    @classmethod
    def shutdown_executor(cls):
        logger.debug("Shutting down CloudDetector's ThreadPoolExecutor.")
        cls._executor.shutdown(wait=True)
