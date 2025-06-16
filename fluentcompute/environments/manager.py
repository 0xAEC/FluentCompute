# fluentcompute/environments/manager.py
import subprocess
import json
from typing import List, Optional, Dict, Any, TypedDict

from fluentcompute.utils.logging_config import logger

# For stricter typing of Conda list output
class CondaEnvInfo(TypedDict):
    name: str
    prefix: str

class EnvironmentManager:
    """
    Manages listing and suggesting commands for Python environments (starting with Conda).
    """

    def __init__(self):
        # Could potentially take a config object for tool paths (e.g., conda_executable)
        pass

    async def _run_command_json_output(self, cmd: List[str]) -> Optional[Any]:
        """
        Helper to run a synchronous command (like conda) and parse its JSON output.
        Note: This is similar to the one in EnvironmentDetector, but kept separate for now.
        Could be refactored into a shared utility if more widely needed.
        Uses asyncio.create_subprocess_exec for non-blocking execution.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate() # await communication
            
            if process.returncode == 0:
                try:
                    return json.loads(stdout.decode())
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse JSON from command '{' '.join(cmd)}': {stdout.decode()[:500]}. Error: {e}")
                    return None
            else:
                stderr_output = stderr.decode()
                # Conda list environments might return 1 if no environments other than base, which is not an error for listing.
                # However, if it outputs valid JSON, we might still want to parse it.
                # Let's be a bit more lenient if we get JSON despite a non-zero exit code (for `conda env list` cases).
                if stdout:
                    try:
                        potential_json = json.loads(stdout.decode())
                        logger.debug(f"Command '{' '.join(cmd)}' had non-zero return code {process.returncode} but produced JSON. Stderr: {stderr_output[:500]}")
                        return potential_json # Return if parsable
                    except json.JSONDecodeError:
                        pass # Fall through to regular error logging
                
                logger.debug(f"Command '{' '.join(cmd)}' failed. Return code: {process.returncode}. Error: {stderr_output[:500]}")
                return None
        except FileNotFoundError:
            logger.error(f"Command '{cmd[0]}' not found. Ensure Conda (or the required tool) is installed and in PATH.")
            return None
        except asyncio.TimeoutError: # If communicate() times out (though not explicitly set here)
            logger.warning(f"Command '{' '.join(cmd)}' timed out during communication.")
            return None
        except Exception as e:
            logger.warning(f"Error running command '{' '.join(cmd)}': {e}", exc_info=True)
            return None

    async def list_conda_environments(self) -> List[Dict[str, str]]:
        """
        Lists available Conda environments.
        Returns a list of dictionaries, e.g., [{"name": "myenv", "path": "/path/to/myenv"}].
        """
        logger.info("Listing Conda environments...")
        conda_cmd = ["conda", "env", "list", "--json"]
        
        raw_output = await self._run_command_json_output(conda_cmd)
        
        environments: List[Dict[str, str]] = []
        if raw_output and "envs" in raw_output and isinstance(raw_output["envs"], list):
            for env_path in raw_output["envs"]:
                if not isinstance(env_path, str) or not env_path.strip():
                    continue

                path_obj = Path(env_path.strip())
                name = path_obj.name
                
                # Check for active environment marker from `conda info --json` if needed,
                # but `conda env list` provides paths directly.
                # Base environment often just shows the path to the main conda installation.
                # Conda info might be better for "active" but this is for "all known"
                
                # Heuristic: if path is the root of an anaconda/miniconda installation, it's often 'base'
                if (path_obj / "conda-meta" / "history").exists() and \
                   (path_obj.parent.name.lower() in ["anaconda3", "miniconda3", "miniconda", "anaconda"] or \
                    (path_obj / "pkgs").exists()): # Trying to identify base
                    if name == "envs": # e.g. /path/to/miniconda3/envs -> this is not an env, skip.
                        continue
                    # If the env_path itself looks like a base conda installation
                    # e.g. /opt/miniconda3 and there's no actual CONDA_DEFAULT_ENV
                    # then its name is likely 'base'.
                    # This gets tricky because `conda env list` just lists directories.
                    # We might need to cross-reference with `conda info --json`'s active_prefix_name if it matches path_obj.

                    # Simplification for now: Use folder name, unless it's a known root
                    # More reliable naming comes from comparing paths with `conda info --json` if `active_prefix` matches
                    # But for a simple list, folder names are generally used.

                environments.append({"name": name, "path": str(path_obj)})
            logger.info(f"Found {len(environments)} Conda environments.")
        elif raw_output:
            logger.warning(f"Unexpected output format from 'conda env list --json': {str(raw_output)[:200]}")
        else:
            logger.info("Could not list Conda environments (conda command failed or no environments found).")
            
        return environments

    def suggest_conda_activation_command(self, environment_name_or_path: str) -> str:
        """Suggests the command to activate a Conda environment."""
        # Conda activation can depend on the shell. `conda activate` is generally preferred.
        return f"conda activate {environment_name_or_path}"

    def suggest_conda_creation_command(
        self,
        environment_name: str,
        python_version: Optional[str] = None,
        packages: Optional[List[str]] = None,
        channel: Optional[str] = None
    ) -> str:
        """Suggests the command to create a new Conda environment."""
        if not environment_name:
            raise ValueError("Environment name must be provided for creation.")

        cmd_parts = ["conda", "create", "--name", environment_name]
        if python_version:
            cmd_parts.append(f"python={python_version}")
        if packages:
            cmd_parts.extend(packages)
        if channel:
            cmd_parts.extend(["-c", channel])
        cmd_parts.append("-y") # Assume yes to prompts for a suggested command
        
        return " ".join(cmd_parts)

    # --- venv related methods (MVP: placeholders or very basic) ---
    async def list_venv_environments(self) -> List[Dict[str, str]]:
        """
        Placeholder for listing venv environments.
        This is complex as venvs are not centrally registered.
        Might require user input for project directories.
        """
        logger.info("Listing venv environments is not fully supported in MVP. Focus is on Conda.")
        return []

    def suggest_venv_activation_command(self, venv_path: str) -> Dict[str, str]:
        """Suggests commands to activate a venv environment based on OS."""
        if not venv_path:
            raise ValueError("venv path must be provided.")
        
        path_obj = Path(venv_path)
        # Check for common activation script locations
        activate_scripts = {}
        
        # POSIX (Linux/macOS)
        posix_activate = path_obj / "bin" / "activate"
        if posix_activate.exists():
            activate_scripts["posix (bash/zsh/...)"] = f"source {posix_activate.resolve()}"
        
        # Windows
        win_activate_bat = path_obj / "Scripts" / "activate.bat"
        win_activate_ps1 = path_obj / "Scripts" / "Activate.ps1" # Note case for PowerShell

        if win_activate_bat.exists():
            activate_scripts["windows (cmd.exe)"] = str(win_activate_bat.resolve())
        if win_activate_ps1.exists():
            activate_scripts["windows (PowerShell)"] = str(win_activate_ps1.resolve()) # Needs `Set-ExecutionPolicy RemoteSigned -Scope Process` or similar sometimes.
            
        if not activate_scripts:
            logger.warning(f"Could not find standard activation scripts in venv path: {venv_path}")
            return {"error": "Activation scripts not found."}
            
        return activate_scripts

    def suggest_venv_creation_command(
        self,
        venv_path: str, # Usually a directory path like '.venv' or 'my_project_env'
        python_executable: Optional[str] = None # e.g., 'python3.9' or full path
    ) -> str:
        """Suggests the command to create a new venv environment."""
        if not venv_path:
            raise ValueError("venv path (directory for the new environment) must be provided.")
            
        python_cmd = python_executable or "python" # Defaults to 'python' in current PATH
        
        # Ensure venv_path is a directory name, not including 'python -m venv'
        # e.g., if user provides "./.venv", that's good.
        return f"{python_cmd} -m venv {venv_path}"
