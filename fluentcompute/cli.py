# fluentcompute/cli.py
import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from typing import Optional, List # Added List
from datetime import datetime # Added for monitor timestamp

import typer
# Typer version check for style consistency
try:
    from typer import colors as typer_colors
except ImportError: # Older Typer might have colors directly under typer.
    # In case typer_colors is not directly available, define a fallback or raise error
    class FallbackColors: # pragma: no cover
        RED = "red"
        GREEN = "green"
        YELLOW = "yellow"
        CYAN = "cyan"
        BRIGHT_GREEN = "bright_green" # typer might not have this, adjust if needed
    typer_colors = FallbackColors() # type: ignore


# Assuming this script is run from fluentcompute_project/fluentcompute/cli.py
# The project root for imports is fluentcompute_project/
project_root_for_imports = Path(__file__).resolve().parent.parent
if str(project_root_for_imports) not in sys.path:
    sys.path.insert(0, str(project_root_for_imports))

# The root directory of the project (fluentcompute_project/) for data storage etc.
PROJECT_DIRECTORY_ROOT = Path(__file__).resolve().parent.parent

try:
    from fluentcompute import HardwareManager, logger as fc_logger
    from fluentcompute.config.settings import DEFAULT_LOG_LEVEL, LOG_LEVEL_ENV_VAR
    from fluentcompute.models import PythonEnvironmentInfo
    from fluentcompute.environments import EnvironmentManager # New import
except ImportError as e:
    print(f"Error importing FluentCompute modules. Ensure fluentcompute_project is in PYTHONPATH. {e}")
    sys.exit(1)

app = typer.Typer(
    name="fluentcompute",
    help="FluentCompute CLI for hardware management and ML environment interaction.",
    rich_markup_mode="markdown" # Enable rich markup for help text
)

# Shared state object for context
class AppState:
    enable_cloud_detection: bool = True

@app.callback()
def main_callback(
    ctx: typer.Context,
    cloud_detect: bool = typer.Option(True, "--cloud/--no-cloud", help="Enable/disable cloud provider detection.")
):
    """
    FluentCompute CLI main entry point. Configure logging and global settings here.
    """
    ctx.obj = AppState()
    ctx.obj.enable_cloud_detection = cloud_detect

    log_level_str = os.environ.get(LOG_LEVEL_ENV_VAR, logging.getLevelName(DEFAULT_LOG_LEVEL)).upper()
    log_level_val = getattr(logging, log_level_str, DEFAULT_LOG_LEVEL)
    fc_logger.setLevel(log_level_val)

    if sys.platform == "win32" and sys.version_info >= (3, 8):
        try:
            current_policy = asyncio.get_event_loop_policy()
            if not isinstance(current_policy, asyncio.WindowsProactorEventLoopPolicy):
                # Use a unique attribute name to avoid clashes if other libraries also set this
                if not getattr(current_policy, '_fc_proactor_loop_policy_set', False):
                    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                    setattr(current_policy, '_fc_proactor_loop_policy_set', True)
        except Exception as e_policy:
            fc_logger.debug(f"Could not set WindowsProactorEventLoopPolicy: {e_policy}")

async def _get_manager_instance(ctx: typer.Context) -> HardwareManager:
    """Helper function to initialize and return a HardwareManager instance."""
    app_state: AppState = ctx.obj
    log_level_val = fc_logger.level

    # Store database in PROJECT_DIRECTORY_ROOT/data_cli/
    db_storage_path = PROJECT_DIRECTORY_ROOT / "data_cli"
    db_storage_path.mkdir(parents=True, exist_ok=True)

    manager = HardwareManager(
        enable_cloud_detection=app_state.enable_cloud_detection,
        log_level=log_level_val,
        db_base_path=str(db_storage_path)
    )
    await manager.initialize_hardware_and_environment()
    return manager

@app.command()
async def detect(ctx: typer.Context):
    """
    Detects system hardware, software environment, and prints a comprehensive summary.
    """
    fc_logger.info("Starting comprehensive system detection...")
    manager: Optional[HardwareManager] = None
    try:
        manager = await _get_manager_instance(ctx)
        system_summary = await manager.get_system_summary()

        typer.echo(typer.style("\n--- Full System, GPU, and Environment Summary ---", fg=typer_colors.CYAN, bold=True))
        try:
            # Using Typer's JSON rendering which handles Pydantic models better if we switch
            typer.echo(json.dumps(system_summary, indent=2))
        except TypeError as e_json:
            fc_logger.error(f"Could not serialize summary to JSON: {e_json}")
            typer.echo(str(system_summary)) # Fallback to raw string

        if not manager.get_all_gpus():
            typer.echo("No GPUs were detected on this system.")

    except Exception as e:
        fc_logger.critical(f"🚨 Error during 'detect' command: {e}", exc_info=True)
        raise typer.Exit(code=1)
    finally:
        if manager:
            manager.cleanup()

@app.command()
async def summary(ctx: typer.Context):
    """
    Provides a summary of system hardware, GPUs, and detected software environment.
    (Currently an alias for `detect` in terms of output content).
    """
    fc_logger.info("Fetching system summary...")
    # This command now has the same behavior as `detect`.
    # It's kept for potential future differentiation in output verbosity or content.
    await detect(ctx)


# --- Environment Subcommands ---
env_app = typer.Typer(
    name="env",
    help="Manage and inspect Python environments for ML frameworks. :wrench:",
    no_args_is_help=True # Show help if `fluentcompute env` is called without subcommand
)
app.add_typer(env_app, name="env")

@env_app.command("detect")
async def environment_detect_subcommand(ctx: typer.Context):
    """
    Detects and displays the current Python environment and ML framework information.
    """
    fc_logger.info("Starting focused environment detection...")
    manager: Optional[HardwareManager] = None
    try:
        manager = await _get_manager_instance(ctx)
        env_info: Optional[PythonEnvironmentInfo] = manager.detected_environment

        if env_info:
            typer.echo(typer.style("\n--- Detected Python Environment and Frameworks ---", fg=typer_colors.BRIGHT_GREEN, bold=True))
            try:
                typer.echo(json.dumps(env_info.to_dict(), indent=2))
            except TypeError as e_json:
                fc_logger.error(f"Could not serialize environment info to JSON: {e_json}")
                typer.echo(str(env_info))
        else:
            typer.echo(typer.style("Environment information could not be detected.", fg=typer_colors.YELLOW))

    except Exception as e:
        fc_logger.critical(f"🚨 Error during 'env detect' command: {e}", exc_info=True)
        raise typer.Exit(code=1)
    finally:
        if manager:
            manager.cleanup()

@env_app.command("list")
async def list_environments(
    ctx: typer.Context,
    conda: bool = typer.Option(True, help="List Conda environments. Enabled by default."),
    # venv: bool = typer.Option(False, help="List venv environments (future capability).")
):
    """Lists available Python environments (currently focused on Conda)."""
    env_manager = EnvironmentManager()
    output_displayed = False

    if conda:
        typer.echo(typer.style("\n--- Conda Environments ---", fg=typer_colors.CYAN, bold=True))
        try:
            conda_envs = await env_manager.list_conda_environments()
            if conda_envs:
                for env in conda_envs:
                    typer.echo(f"- Name: {env.get('name', 'N/A')}, Path: {env.get('path', 'N/A')}")
                output_displayed = True
            else:
                typer.echo("No Conda environments found or Conda is not available.")
        except Exception as e_conda_list: # Catch errors from conda command itself
            typer.echo(typer.style(f"Error listing Conda environments: {e_conda_list}", fg=typer_colors.RED))
            fc_logger.error(f"Failed to list conda environments: {e_conda_list}", exc_info=True)


    if not output_displayed:
        typer.echo("No environment types selected for listing or no environments found.")


@env_app.command("suggest-activate")
async def suggest_activation(
    ctx: typer.Context,
    name_or_path: str = typer.Option(
        ..., "--name", "-n", "--path", "-p", # merged options
        help="Name (e.g., 'myenv') or full path of the Conda environment to activate."
    ),
):
    """Suggests the Conda activation command for a given environment."""
    env_manager = EnvironmentManager()
    typer.echo(typer.style(f"\n--- Suggested Conda Activation for '{name_or_path}' ---", fg=typer_colors.CYAN, bold=True))
    command = env_manager.suggest_conda_activation_command(name_or_path)
    typer.echo("To activate (in a Conda-aware terminal):")
    typer.echo(typer.style(f"  {command}", fg=typer_colors.GREEN))
    typer.echo(typer.style("Note: Activation is shell-dependent. This is a common form.", dim=True))


@env_app.command("suggest-create")
async def suggest_creation(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", "-n", help="Name for the new Conda environment."),
    python_version: Optional[str] = typer.Option(None, "--python", "-pv", help="Python version (e.g., '3.9', '3.10.4')."),
    packages: Optional[List[str]] = typer.Option(
        None, "--package", "-pkg",
        help="Packages to install (e.g., 'tensorflow', 'pandas==1.5'). Can specify multiple times."
    ),
    channel: Optional[str] = typer.Option(None, "--channel", "-c", help="Conda channel to use (e.g., 'conda-forge')."),
):
    """Suggests the Conda command to create a new environment."""
    env_manager = EnvironmentManager()
    try:
        command = env_manager.suggest_conda_creation_command(
            environment_name=name,
            python_version=python_version,
            packages=packages,
            channel=channel
        )
        typer.echo(typer.style(f"\n--- Suggested Conda Creation Command for '{name}' ---", fg=typer_colors.CYAN, bold=True))
        typer.echo(typer.style(f"  {command}", fg=typer_colors.GREEN))
    except ValueError as e:
        typer.echo(typer.style(f"Error preparing creation command: {e}", fg=typer_colors.RED))
        raise typer.Exit(code=1)


# --- Monitor Command (remains top-level) ---
@app.command()
async def monitor(
    ctx: typer.Context,
    gpu_uuid: Optional[str] = typer.Argument(None, help="Specific GPU UUID to monitor. Monitors first GPU if not specified."),
    interval: int = typer.Option(3, "--interval", "-i", min=1, help="Telemetry polling interval in seconds."),
    duration: int = typer.Option(15, "--duration", "-d", min=0, help="Monitoring duration in seconds. Set to 0 for indefinite monitoring (Ctrl+C to stop).")
):
    """
    Monitors GPU telemetry for a specified duration or indefinitely. :chart_with_upwards_trend:
    """
    fc_logger.info("Starting GPU monitoring...")
    manager: Optional[HardwareManager] = None
    try:
        manager = await _get_manager_instance(ctx)
        gpus = manager.get_all_gpus()

        if not gpus:
            typer.echo(typer.style("No GPUs found to monitor.", fg=typer_colors.YELLOW))
            return

        target_gpu = None
        if gpu_uuid:
            target_gpu = manager.get_gpu_by_uuid(gpu_uuid)
            if not target_gpu:
                typer.echo(typer.style(f"Error: GPU with UUID '{gpu_uuid}' not found.", fg=typer_colors.RED))
                available_uuids = [gpu.uuid for gpu in gpus]
                if available_uuids:
                    typer.echo(f"Available GPU UUIDs: {', '.join(available_uuids)}")
                else:
                    typer.echo("No GPUs with identifiable UUIDs were found.")
                raise typer.Exit(code=1)
        else:
            target_gpu = gpus[0]
            typer.echo(f"No GPU UUID specified. Monitoring first detected GPU: {typer.style(target_gpu.name, bold=True)} (UUID: {target_gpu.uuid})")

        typer.echo(f"Monitoring GPU: {typer.style(target_gpu.name, bold=True)} (UUID: {target_gpu.uuid})")
        typer.echo(f"Telemetry DB: {manager.db_manager.db_file}")
        typer.echo(f"Polling interval: {interval}s. Duration: {duration if duration > 0 else 'Indefinite (Ctrl+C to stop)'}s.")
        
        manager.start_telemetry_collection(interval_sec=interval, history_size=20)
        
        # Determine loop iterations. Using float('inf') for current_iteration type consistency isn't ideal for linters.
        iterations_to_run = float('inf') if duration == 0 else (duration // interval)
        ran_iterations = 0

        while ran_iterations < iterations_to_run:
            await asyncio.sleep(interval)
            # Ensure target_gpu is still valid (it should be from above checks)
            if target_gpu and manager and manager.telemetry_lock: # Added manager check for telemetry_lock
                with manager.telemetry_lock:
                    if target_gpu.telemetry_history:
                        latest_tel = target_gpu.telemetry_history[-1]
                        ts = latest_tel.timestamp.strftime('%H:%M:%S')
                        util = f"{latest_tel.gpu_utilization:.1f}%".ljust(7)
                        mem_util = f"{latest_tel.memory_utilization:.1f}%".ljust(7)
                        temp = f"{latest_tel.temperature:.1f}°C".ljust(8)
                        pwr = f"{latest_tel.power_draw:.1f}W".ljust(7)
                        clk_g = f"{latest_tel.clock_graphics or 'N/A'}MHz".ljust(10)
                        clk_m = f"{latest_tel.clock_memory or 'N/A'}MHz".ljust(10)
                        fan = f"{latest_tel.fan_speed or 'N/A'}%".ljust(6)
                        thr = ','.join(latest_tel.throttle_reasons) if latest_tel.throttle_reasons else 'None'
                        
                        telemetry_line = (
                            f"[{ts}] Util: {util} MemUtil: {mem_util} Temp: {temp} Pwr: {pwr} "
                            f"GraphClk: {clk_g} MemClk: {clk_m} Fan: {fan} Throttle: {thr}"
                        )
                        typer.echo(telemetry_line)
                    else:
                        typer.echo(f"[{datetime.now().strftime('%H:%M:%S')}] No telemetry data yet for {target_gpu.name}...")
            if duration > 0:
                ran_iterations += 1

    except KeyboardInterrupt:
        typer.echo(typer.style("\nMonitoring interrupted by user.", fg=typer_colors.YELLOW))
    except Exception as e:
        fc_logger.critical(f"🚨 Error during 'monitor' command: {e}", exc_info=True)
        raise typer.Exit(code=1)
    finally:
        if manager:
            manager.cleanup()
        typer.echo("Monitoring stopped.")

if __name__ == "__main__":
    app()
