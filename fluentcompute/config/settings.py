import os
import logging

# --- Global Settings ---
DB_FILE_NAME = "fluentcompute_data.db"
# Consider making this an absolute path or relative to a project root for robustness
# For simplicity, keeping it as a name to be joined with a base path later if needed.
# For example, in DBManager: self.db_file = Path(os.getcwd()) / DB_FILE_NAME or similar.
# A better approach for a real app would be to resolve this path from a known root.


# --- Advanced Feature Availability ---
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    import kubernetes
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False

# --- Logging Configuration ---
# Default log level, can be overridden by environment variable or HardwareManager init
DEFAULT_LOG_LEVEL = logging.INFO
LOG_LEVEL_ENV_VAR = "FC_LOG_LEVEL"
