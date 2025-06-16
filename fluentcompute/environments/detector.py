# fluentcompute/environments/detector.py

import sys
import os
import importlib
import subprocess # Changed from asyncio.create_subprocess_exec for simplicity here; can be made async later if truly needed.
import json
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import tempfile # For temporary helper script

from fluentcompute.models import PythonEnvironmentInfo, FrameworkInfo, FrameworkDependency
from fluentcompute.utils.logging_config import logger

# Helper script content (this will be run by the target Python interpreter)
# Note the careful use of raw strings or escaped characters if this were a very complex script.
# For this complexity, a temporary file is cleaner than `python -c "long_script_string"`.
_FRAMEWORK_CHECKER_SCRIPT_CONTENT = """
import sys
import json
import importlib
import os
from pathlib import Path

def get_framework_details(framework_name):
    data = {
        "name": framework_name,
        "version": None,
        "path": None,
        "dependencies": [],
        "details": {}, # Changed from extra_details to match FrameworkInfo
        "error": None
    }
    try:
        module = importlib.import_module(framework_name)
        data["version"] = getattr(module, '__version__', 'N/A')
        # Some modules like jax might not have __file__ on the top-level import
        # if it's a namespace package. Check its components.
        if hasattr(module, '__file__') and module.__file__:
             data["path"] = str(Path(module.__file__).resolve().parent)
        elif hasattr(module, '__path__') and module.__path__: # For namespace packages
             data["path"] = str(Path(list(module.__path__)[0]).resolve())


        if framework_name == "tensorflow":
            import tensorflow as tf # Re-import for specific calls
            data["details"]["is_built_with_cuda"] = tf.test.is_built_with_cuda()
            try:
                data["details"]["is_gpu_available"] = bool(tf.config.list_physical_devices('GPU'))
            except Exception: # TF might raise if not properly set up, even for a query
                data["details"]["is_gpu_available"] = False

            try:
                build_info = tf.sysconfig.get_build_info()
                tf_cuda_version = build_info.get("cuda_version")
                tf_cudnn_version = build_info.get("cudnn_version")
                if tf_cuda_version and tf_cuda_version != "unknown": # TF sometimes reports "unknown"
                    data["dependencies"].append({"name": "CUDA (TensorFlow Build)", "version": str(tf_cuda_version)})
                if tf_cudnn_version and tf_cudnn_version != "unknown":
                    data["dependencies"].append({"name": "cuDNN (TensorFlow Build)", "version": str(tf_cudnn_version)})
            except Exception:
                 data["details"]["build_info_error"] = "Could not retrieve detailed TensorFlow build info."
                 # Fallback: check linked libraries if possible (platform dependent, complex)
                 # For now, rely on tf.test.is_built_with_cuda()

        elif framework_name == "torch":
            import torch # Re-import
            data["details"]["cuda_available_runtime"] = torch.cuda.is_available()
            if hasattr(torch.version, 'cuda') and torch.version.cuda: # Built with CUDA
                data["dependencies"].append({"name": "CUDA (PyTorch Build)", "version": torch.version.cuda})
            if hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available():
                cudnn_ver = torch.backends.cudnn.version()
                data["dependencies"].append({"name": "cuDNN (PyTorch Runtime)", "version": str(cudnn_ver)})
                data["details"]["cudnn_runtime_enabled"] = torch.backends.cudnn.enabled
            if hasattr(torch, '__config__') and torch.__config__.show:
                data["details"]["torch_build_summary"] = torch.__config__.show()


        elif framework_name == "jax":
            import jax # Re-import
            try:
                backend_platform = jax.lib.xla_bridge.get_backend().platform
                data["details"]["platform"] = backend_platform
                data["details"]["devices_detected_by_jax"] = str(jax.devices()) # list of devices JAX sees
                if backend_platform == 'gpu': # If JAX is using GPU, try to find jaxlib CUDA info
                     # jaxlib version is a key dependency indicator
                    try:
                        import jaxlib
                        data["dependencies"].append({"name": "jaxlib", "version": getattr(jaxlib, '__version__', 'N/A')})
                        # More direct CUDA version detection for JAX is complex as it might rely on env variables
                        # or system paths at jaxlib compile time.
                        # For now, noting GPU backend + jaxlib version is the primary info.
                    except ImportError:
                        data["details"]["jaxlib_import_error"] = "jaxlib could not be imported to check version."

            except Exception as e:
                data["details"]["jax_backend_info_error"] = str(e)


    except ImportError:
        data["error"] = f"'{framework_name}' not found by interpreter {sys.executable}"
    except Exception as e:
        data["error"] = str(e)
    
    # Python version that ran this script
    data["dependencies"].append({
        "name": "Python",
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "path": sys.executable
    })
    print(json.dumps(data))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        framework_to_check = sys.argv[1]
        get_framework_details(framework_to_check)
    else:
        # This error path should ideally not be hit if called correctly by EnvironmentDetector
        print(json.dumps({"error": "No framework name provided to checker script."}))
"""


class EnvironmentDetector:
    def __init__(self):
        self.detected_environment_info: Optional[PythonEnvironmentInfo] = None
        self._python_executable: str = sys.executable

    async def _run_command_json_output(self, cmd: List[str]) -> Optional[Any]:
        """Helper to run a command and parse its JSON output (sync subprocess)."""
        try:
            # Using synchronous subprocess for simplicity with temp files and immediate output.
            # If performance becomes an issue for many framework checks, could revert to async subprocess.
            process = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=15)
            if process.returncode == 0:
                try:
                    return json.loads(process.stdout)
                except json.JSONDecodeError:
                    logger.debug(f"Failed to parse JSON from command '{' '.join(cmd)}': {process.stdout[:500]}")
                    return None
            else:
                logger.debug(f"Command '{' '.join(cmd)}' failed. Return code: {process.returncode}. Error: {process.stderr[:500]}")
                return None
        except FileNotFoundError:
            logger.debug(f"Command '{cmd[0]}' not found for environment detection.")
            return None
        except subprocess.TimeoutExpired:
            logger.warning(f"Command '{' '.join(cmd)}' timed out.")
            return None
        except Exception as e:
            logger.warning(f"Error running command '{' '.join(cmd)}': {e}")
            return None

    async def _detect_conda_env(self) -> Tuple[Optional[str], Optional[str]]:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            conda_env_name = os.environ.get("CONDA_DEFAULT_ENV", Path(conda_prefix).name)
            if (Path(conda_prefix) / "conda-meta").exists():
                logger.debug(f"Detected Conda (env var): Name='{conda_env_name}', Prefix='{conda_prefix}'")
                return conda_env_name, conda_prefix
        
        conda_info = await self._run_command_json_output(["conda", "info", "--json"]) # _run_command... is now sync
        if conda_info and isinstance(conda_info, dict):
            active_prefix = conda_info.get("active_prefix")
            active_env_name = conda_info.get("active_prefix_name")
            if active_prefix and (Path(active_prefix) / "conda-meta").exists():
                 logger.debug(f"Detected Conda (conda info): Name='{active_env_name}', Prefix='{active_prefix}'")
                 return active_env_name, active_prefix
        return None, None

    async def _detect_venv(self) -> Optional[str]:
        virtual_env_path = os.environ.get("VIRTUAL_ENV")
        if virtual_env_path: # Check for pyvenv.cfg for venv or other markers for virtualenv
            # Standard venv marker
            if (Path(virtual_env_path) / "pyvenv.cfg").exists():
                 logger.debug(f"Detected venv: Path='{virtual_env_path}'")
                 return virtual_env_path
            # Legacy virtualenv might not have pyvenv.cfg but bin/activate exists
            elif (Path(virtual_env_path) / "bin" / "activate").exists(): # Linux/macOS
                 logger.debug(f"Detected virtualenv (legacy marker): Path='{virtual_env_path}'")
                 return virtual_env_path
            elif (Path(virtual_env_path) / "Scripts" / "activate.bat").exists(): # Windows
                 logger.debug(f"Detected virtualenv (legacy marker): Path='{virtual_env_path}'")
                 return virtual_env_path
        return None

    async def detect_python_environment(self) -> PythonEnvironmentInfo:
        logger.info("Detecting Python environment details...")
        current_process_python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        current_process_python_exe = sys.executable

        env_type = "system"
        env_name: Optional[str] = None
        # Start by assuming the current interpreter defines the environment
        env_path: Optional[str] = str(Path(current_process_python_exe).parent.parent) 
        self._python_executable = current_process_python_exe # This is the python we will use for framework checks.

        conda_name, conda_prefix = await self._detect_conda_env()
        if conda_prefix:
            env_type = "conda"
            env_name = conda_name
            env_path = conda_prefix
            # Determine the python executable within this conda environment
            conda_python_exe_candidates = [
                Path(conda_prefix) / "bin" / "python", 
                Path(conda_prefix) / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}",
                Path(conda_prefix) / "python.exe", # Windows
            ]
            found_conda_exe = None
            for cand_exe in conda_python_exe_candidates:
                if cand_exe.exists() and os.access(cand_exe, os.X_OK):
                    found_conda_exe = str(cand_exe)
                    break
            
            if found_conda_exe:
                self._python_executable = found_conda_exe
                logger.info(f"Using Conda Python for framework checks: {self._python_executable}")
            else:
                logger.warning(f"Conda env '{conda_name}' detected, but primary Python executable not found within. Using current process Python: {current_process_python_exe}")
                # Keep self._python_executable as current_process_python_exe if specific one not found

        else: # Not conda, check venv
            venv_path_prefix = await self._detect_venv()
            if venv_path_prefix:
                env_type = "venv"
                env_path = venv_path_prefix
                env_name = Path(venv_path_prefix).name
                venv_python_exe_candidates = [
                    Path(venv_path_prefix) / "bin" / "python",
                    Path(venv_path_prefix) / "Scripts" / "python.exe" # Windows
                ]
                found_venv_exe = None
                for cand_exe in venv_python_exe_candidates:
                    if cand_exe.exists() and os.access(cand_exe, os.X_OK):
                        found_venv_exe = str(cand_exe)
                        break

                if found_venv_exe:
                    self._python_executable = found_venv_exe
                    logger.info(f"Using venv Python for framework checks: {self._python_executable}")
                else:
                    logger.warning(f"Venv '{env_name}' detected, but primary Python executable not found within. Using current process Python: {current_process_python_exe}")
        
        # Get the Python version *of the target interpreter*
        target_python_version_info = await self._run_command_json_output(
            [self._python_executable, "-c", "import sys, json; print(json.dumps(list(sys.version_info)))"]
        )
        if target_python_version_info and len(target_python_version_info) >= 3:
            final_python_version = f"{target_python_version_info[0]}.{target_python_version_info[1]}.{target_python_version_info[2]}"
        else:
            logger.warning(f"Could not determine Python version for {self._python_executable}. Using current process version {current_process_python_version}.")
            final_python_version = current_process_python_version


        logger.info(f"Final environment for framework detection: Type='{env_type}', Name='{env_name or 'N/A'}', Path='{env_path or 'N/A'}', Python Exec='{self._python_executable}', Python Version='{final_python_version}'")

        self.detected_environment_info = PythonEnvironmentInfo(
            type=env_type, name=env_name, path=env_path,
            python_version=final_python_version, frameworks=[]
        )
        return self.detected_environment_info

    async def _execute_framework_check(self, framework_name: str) -> Optional[FrameworkInfo]:
        """Executes the helper script for a given framework."""
        if not self._python_executable:
            logger.error("Python executable for target environment not determined. Cannot check framework.")
            return None

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_script:
                tmp_script.write(_FRAMEWORK_CHECKER_SCRIPT_CONTENT)
                tmp_script_path = tmp_script.name
            
            cmd = [self._python_executable, tmp_script_path, framework_name]
            logger.debug(f"Executing framework check: {' '.join(cmd)}")
            
            # This is a blocking call, for true async, need asyncio.create_subprocess_exec
            # and to manage stdout/stderr asynchronously.
            raw_output = await self._run_command_json_output(cmd)

        finally:
            if 'tmp_script_path' in locals() and Path(tmp_script_path).exists():
                try:
                    os.remove(tmp_script_path)
                except Exception as e_rm:
                    logger.warning(f"Could not remove temporary script {tmp_script_path}: {e_rm}")

        if raw_output and isinstance(raw_output, dict):
            if raw_output.get("error"):
                logger.info(f"{framework_name} detection reported error: {raw_output['error']}")
                return None # Framework not found or error during its detection

            dependencies = [FrameworkDependency(**dep) for dep in raw_output.get("dependencies", [])]
            return FrameworkInfo(
                name=raw_output["name"],
                version=raw_output.get("version"),
                path=raw_output.get("path"),
                dependencies=dependencies,
                details=raw_output.get("details", {})
            )
        else:
            logger.warning(f"No valid output from {framework_name} checker script via {self._python_executable}.")
            return None

    async def detect_frameworks(self) -> List[FrameworkInfo]:
        if not self.detected_environment_info:
            await self.detect_python_environment()
        
        if not self.detected_environment_info:
            logger.error("Critical: Python environment not detected. Cannot proceed with framework detection.")
            return []
            
        logger.info(f"Detecting ML frameworks in Python env: {self.detected_environment_info.path or self._python_executable}")
        
        frameworks_to_check = ["tensorflow", "torch", "jax"]
        detected_frameworks: List[FrameworkInfo] = []

        for fw_name in frameworks_to_check:
            logger.info(f"--- Checking for {fw_name} ---")
            fw_info = await self._execute_framework_check(fw_name)
            if fw_info:
                detected_frameworks.append(fw_info)
                logger.info(f"Successfully detected {fw_name} v{fw_info.version or 'N/A'}")
            else:
                logger.info(f"{fw_name} not detected or error during check.")
        
        self.detected_environment_info.frameworks = detected_frameworks
        return detected_frameworks

    async def get_environment_info(self) -> Optional[PythonEnvironmentInfo]:
        if not self.detected_environment_info or not self.detected_environment_info.frameworks : #If python env not detected or frameworks not populated
             await self.detect_python_environment() # Detects python env, also sets self._python_executable
             if self.detected_environment_info: # Only proceed if python env was successfully set up
                await self.detect_frameworks()   # This will populate self.detected_environment_info.frameworks
        return self.detected_environment_info
