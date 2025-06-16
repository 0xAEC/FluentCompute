import sys
import os
import importlib
import subprocess
import json
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from fluentcompute.models import PythonEnvironmentInfo, FrameworkInfo, FrameworkDependency
from fluentcompute.utils.logging_config import logger

class EnvironmentDetector:
    def __init__(self):
        self.detected_environment_info: Optional[PythonEnvironmentInfo] = None
        self._python_executable: str = sys.executable # Store the executable used for detection

    async def _run_subprocess_json(self, cmd: List[str]) -> Optional[Any]:
        """Helper to run a subprocess and parse its JSON output."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                try:
                    return json.loads(stdout.decode())
                except json.JSONDecodeError:
                    logger.debug(f"Failed to parse JSON from command '{' '.join(cmd)}': {stdout.decode()}")
                    return None
            else:
                logger.debug(f"Command '{' '.join(cmd)}' failed with error: {stderr.decode()}")
                return None
        except FileNotFoundError:
            logger.debug(f"Command '{cmd[0]}' not found for environment detection.")
            return None
        except Exception as e:
            logger.warning(f"Error running command '{' '.join(cmd)}': {e}")
            return None

    async def _detect_conda_env(self) -> Tuple[Optional[str], Optional[str]]:
        """Detects active conda environment name and prefix."""
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            conda_env_name = os.environ.get("CONDA_DEFAULT_ENV", Path(conda_prefix).name)
            # Verify it's a conda env, e.g. by checking for conda-meta
            if (Path(conda_prefix) / "conda-meta").exists():
                logger.info(f"Detected Conda environment: Name='{conda_env_name}', Prefix='{conda_prefix}'")
                return conda_env_name, conda_prefix
        
        # Fallback if env vars not definitive, try 'conda info'
        conda_info = await self._run_subprocess_json(["conda", "info", "--json"])
        if conda_info and isinstance(conda_info, dict):
            active_prefix = conda_info.get("active_prefix")
            active_env_name = conda_info.get("active_prefix_name")
            if active_prefix and (Path(active_prefix) / "conda-meta").exists():
                 logger.info(f"Detected Conda environment (via conda info): Name='{active_env_name}', Prefix='{active_prefix}'")
                 return active_env_name, active_prefix
        return None, None

    async def _detect_venv(self) -> Optional[str]:
        """Detects active virtual environment (venv/virtualenv)."""
        virtual_env_path = os.environ.get("VIRTUAL_ENV")
        if virtual_env_path and (Path(virtual_env_path) / "pyvenv.cfg").exists():
            logger.info(f"Detected virtual environment: Path='{virtual_env_path}'")
            return virtual_env_path
        return None

    async def detect_python_environment(self) -> PythonEnvironmentInfo:
        """Detects the current Python environment (conda, venv, or system)."""
        logger.info("Detecting Python environment...")
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        python_exe_path = self._python_executable # sys.executable of the *current* process

        env_type = "system"
        env_name: Optional[str] = None
        env_path: Optional[str] = str(Path(python_exe_path).parent.parent) # Default to executable's root assumption

        conda_name, conda_prefix = await self._detect_conda_env()
        if conda_prefix:
            env_type = "conda"
            env_name = conda_name
            env_path = conda_prefix
             # Ensure python_exe_path is within this conda env if current process matches
            if not Path(python_exe_path).is_relative_to(Path(conda_prefix)): # python 3.9+ Path.is_relative_to
                 logger.warning(f"Active conda prefix '{conda_prefix}' detected, but current Python executable '{python_exe_path}' is outside.")
                 # Consider re-assigning python_exe_path if sure or using conda's reported python
                 conda_python_exe = Path(conda_prefix) / "bin" / "python" # or "python.exe" on Windows
                 if conda_python_exe.exists():
                     self._python_executable = str(conda_python_exe) # Update for framework detection
                     logger.info(f"Using Python executable from Conda env for framework checks: {self._python_executable}")


        else: # Not conda, check venv
            venv_path = await self._detect_venv()
            if venv_path:
                env_type = "venv"
                env_path = venv_path
                env_name = Path(venv_path).name
                 # Ensure python_exe_path is within this venv
                if not Path(python_exe_path).is_relative_to(Path(venv_path)):
                     logger.warning(f"Active venv '{venv_path}' detected, but current Python '{python_exe_path}' is outside.")
                     venv_python_exe = Path(venv_path) / "bin" / "python"
                     if venv_python_exe.exists():
                         self._python_executable = str(venv_python_exe)
                         logger.info(f"Using Python executable from venv for framework checks: {self._python_executable}")


        if env_type == "system":
            logger.info(f"Detected System Python: Path='{env_path}', Version='{python_version}'")
        
        self.detected_environment_info = PythonEnvironmentInfo(
            type=env_type,
            name=env_name,
            path=env_path,
            python_version=python_version,
            frameworks=[] # To be populated by detect_frameworks
        )
        return self.detected_environment_info

    async def detect_frameworks(self) -> List[FrameworkInfo]:
        """Detects installed ML frameworks."""
        if not self.detected_environment_info:
            await self.detect_python_environment()
        
        # This should always be true now unless detect_python_environment had a major issue
        if not self.detected_environment_info: 
            logger.error("Cannot detect frameworks without Python environment information.")
            return []

        logger.info(f"Detecting ML frameworks using Python: {self._python_executable}...")
        
        framework_infos: List[FrameworkInfo] = []
        
        # --- TensorFlow Detection (Example to be expanded) ---
        tf_info = await self._detect_tensorflow()
        if tf_info: framework_infos.append(tf_info)

        # --- PyTorch Detection (Example to be expanded) ---
        # torch_info = await self._detect_pytorch()
        # if torch_info: framework_infos.append(torch_info)

        # --- JAX Detection (Example to be expanded) ---
        # jax_info = await self._detect_jax()
        # if jax_info: framework_infos.append(jax_info)
        
        self.detected_environment_info.frameworks = framework_infos
        return framework_infos

    async def get_environment_info(self) -> Optional[PythonEnvironmentInfo]:
        """Gets all detected environment information."""
        if not self.detected_environment_info:
            await self.detect_python_environment() # Detects python env, initializes self.detected_environment_info
        
        # If frameworks haven't been populated yet for the detected env
        if self.detected_environment_info and not self.detected_environment_info.frameworks: 
            await self.detect_frameworks() # This will populate self.detected_environment_info.frameworks
        
        return self.detected_environment_info


    # --- Placeholder methods for individual framework detection ---
    async def _detect_tensorflow(self) -> Optional[FrameworkInfo]:
        logger.info("Attempting to detect TensorFlow...")
        # More detailed implementation will go here in the next step.
        # For now, just a basic import attempt:
        try:
            # We need to run the import in the context of the detected python_executable
            # This is tricky without creating a new process for each import.
            # A simple importlib.import_module("tensorflow") here would use the *current* interpreter.
            # For robust detection in isolated environments, one would run a small script via subprocess.
            
            # Simple import for now (may not reflect the target env if different from fluentcompute's env)
            # To be improved with subprocess or by ensuring fluentcompute runs *in* the target env
            import tensorflow as tf
            version = tf.__version__
            path = str(Path(tf.__file__).parent)
            details = {"is_built_with_cuda": tf.test.is_built_with_cuda()}
            # cuda_version = tf.version.GIT_VERSION # Often just "unknown" or "head" for prebuilts.
            # Better: Check tf.sysconfig for CUDA/cuDNN details
            # cuda_deps = [] # populate this
            logger.info(f"Detected TensorFlow: Version {version}, Path: {path}")
            return FrameworkInfo(name="TensorFlow", version=version, path=path, details=details, dependencies=[])
        except ImportError:
            logger.info("TensorFlow not found.")
            return None
        except Exception as e:
            logger.warning(f"Error during TensorFlow detection: {e}")
            return None

    # async def _detect_pytorch(self) -> Optional[FrameworkInfo]:
    #     logger.info("Attempting to detect PyTorch...")
    #     # Placeholder
    #     return None

    # async def _detect_jax(self) -> Optional[FrameworkInfo]:
    #     logger.info("Attempting to detect JAX...")
    #     # Placeholder
    #     return None
