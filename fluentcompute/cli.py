# fluentcompute/cli.py
import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from typing import Optional

import typer
# Typer version check for style consistency
try:
    from typer import colors as typer_colors
except ImportError: # Older Typer might have colors directly under typer.
    typer_colors = typer 

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from fluentcompute import HardwareManager, logger as fc_logger
    from fluentcompute.config.settings import DEFAULT_LOG_LEVEL, LOG_LEVEL_ENV_VAR
    from fluentcompute.models import PythonEnvironmentInfo # For type hint if needed
except ImportError as e:
    print(f"Error importing FluentCompute modules. Ensure fluentcompute_project is in PYTHONPATH. {e}")
    sys.exit(1)

app = typer.Typer(name="fluentcompute", help="FluentCompute CLI for hardware management and ML environment interaction.")

# Shared state object for context
class AppState:
    enable_cloud_detection: bool = True
    # manager: Optional[HardwareManager] = None # Potentially share manager if commands need it serially without re-init

@app.callback()
def main_callback(
    ctx: typer.Context, 
    cloud_detect: bool = typer.Option(True, "--cloud/--no-cloud", help="Enable/disable cloud provider detection.")
):
    ctx.obj = AppState()
    ctx.obj.enable_cloud_detection = cloud_detect
    
    log_level_str = os.environ.get(LOG_LEVEL_ENV_VAR, logging.getLevelName(DEFAULT_LOG_LEVEL)).upper()
    log_level_val = getattr(logging, log_level_str, DEFAULT_LOG_LEVEL)
    fc_logger.setLevel(log_level_val)

    if sys.platform == "win32" and sys.version_info >= (3,8):
       try:
           current_policy = asyncio.get_event_loop_policy()
           if not isinstance(current_policy, asyncio.WindowsProactorEventLoopPolicy):
                if not hasattr(current_policy, '_proactor_loop_policy_set_globally_fc') or \
                   not current_policy._proactor_loop_policy_set_globally_fc:
                     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                     # type: ignore # Custom attribute
                     current_policy._proactor_loop_policy_set_globally_fc = True 
       except Exception as e_policy:
            fc_logger.debug(f"Could not set WindowsProactorEventLoopPolicy: {e_policy}")

async def _get_manager_instance(ctx: typer.Context) -> HardwareManager:
    app_state: AppState = ctx.obj
    
    # Check if a manager instance is already on app_state and if it's initialized appropriately.
    # For now, always create a new one to ensure commands are idempotent and pick up latest CLI flags.
    # if app_state.manager and app_state.manager.gpus: # or some other check for initialization
    #     return app_state.manager

    log_level_val = fc_logger.level # Use already configured logger level

    db_storage_path = project_root / "data_cli" # Keep CLI DB separate from demo script DB
    db_storage_path.mkdir(parents=True, exist_ok=True)
    
    manager = HardwareManager(
        enable_cloud_detection=app_state.enable_cloud_detection,
        log_level=log_level_val,
        db_base_path=str(db_storage_path)
    )
    # Changed method name
    await manager.initialize_hardware_and_environment() 
    # app_state.manager = manager # Store if you want to reuse across chained commands (advanced)
    return manager

@app.command()
async def detect(ctx: typer.Context):
    """
    Detects available hardware and prints detailed information for each GPU.
    Also includes basic environment detection in the full summary.
    """
    fc_logger.info("Starting combined hardware and environment detection for 'detect' command...")
    manager: Optional[HardwareManager] = None
    try:
        manager = await _get_manager_instance(ctx)
        gpus = manager.get_all_gpus()
        
        system_summary = await manager.get_system_summary() # Now contains environment
        typer.echo(typer.style("\n--- Full System, GPU, and Environment Summary ---", fg=typer_colors.CYAN, bold=True))
        try:
            typer.echo(json.dumps(system_summary, indent=2))
        except TypeError as e_json:
            typer.echo(f"Could not serialize summary to JSON: {e_json}")
            typer.echo(str(system_summary))
            
        if not gpus:
            typer.echo("No GPUs detected on this system (within the full summary).")

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
    (This command becomes very similar to `detect` now that `detect` also shows summary)
    Kept for consistency, could be an alias or offer a more concise output.
    """
    fc_logger.info("Fetching system summary...")
    manager: Optional[HardwareManager] = None
    try:
        manager = await _get_manager_instance(ctx)
        system_summary = await manager.get_system_summary()
        typer.echo(typer.style("\n--- System, GPU, and Environment Summary ---", fg=typer_colors.CYAN, bold=True))
        try:
            typer.echo(json.dumps(system_summary, indent=2))
        except TypeError as e_json:
            typer.echo(f"Could not serialize summary to JSON: {e_json}")
            typer.echo(str(system_summary))
    except Exception as e:
        fc_logger.critical(f"🚨 Error during 'summary' command: {e}", exc_info=True)
        raise typer.Exit(code=1)
    finally:
        if manager:
            manager.cleanup()

@app.command("env") # Group environment related commands, this is effectively 'env detect'
async def environment_detect(ctx: typer.Context):
    """
    Detects and displays Python environment and ML framework information.
    """
    fc_logger.info("Starting environment detection...")
    manager: Optional[HardwareManager] = None # Use manager to access its env_detector
    try:
        # We need HardwareManager to initialize EnvDetector, though this command
        # primarily focuses on the environment part.
        manager = await _get_manager_instance(ctx)
        
        # Access the already detected environment info from the manager
        env_info: Optional[PythonEnvironmentInfo] = manager.detected_environment

        if env_info:
            typer.echo(typer.style("\n--- Detected Python Environment and Frameworks ---", fg=typer_colors.BRIGHT_GREEN, bold=True))
            try:
                typer.echo(json.dumps(env_info.to_dict(), indent=2))
            except TypeError as e_json:
                typer.echo(f"Could not serialize environment info to JSON: {e_json}")
                typer.echo(str(env_info))
        else:
            typer.echo(typer.style("Environment information could not be detected.", fg=typer_colors.YELLOW))

    except Exception as e:
        fc_logger.critical(f"🚨 Error during 'env' command: {e}", exc_info=True)
        raise typer.Exit(code=1)
    finally:
        if manager:
            manager.cleanup()


@app.command()
async def monitor(
    ctx: typer.Context,
    gpu_uuid: Optional[str] = typer.Argument(None, help="Specific GPU UUID to monitor. Monitors first GPU if not specified."),
    interval: int = typer.Option(3, "--interval", "-i", help="Telemetry polling interval in seconds."),
    duration: int = typer.Option(15, "--duration", "-d", help="Monitoring duration in seconds. Set to 0 for indefinite monitoring (Ctrl+C to stop).")
):
    """
    Monitors GPU telemetry for a specified duration or indefinitely.
    """
    fc_logger.info("Starting GPU monitoring...")
    manager: Optional[HardwareManager] = None
    try:
        manager = await _get_manager_instance(ctx) # Initializes hardware & env
        gpus = manager.get_all_gpus()

        if not gpus:
            typer.echo("No GPUs found to monitor.")
            return

        target_gpu = None
        if gpu_uuid:
            target_gpu = manager.get_gpu_by_uuid(gpu_uuid)
            if not target_gpu:
                typer.echo(typer.style(f"Error: GPU with UUID '{gpu_uuid}' not found.", fg=typer_colors.RED))
                fc_logger.warning(f"Requested GPU UUID '{gpu_uuid}' not found. Available: {[g.uuid for g in gpus]}")
                return
        else:
            target_gpu = gpus[0]
            fc_logger.info(f"No GPU UUID specified, monitoring first detected GPU: {target_gpu.name} ({target_gpu.uuid})")

        typer.echo(f"Monitoring GPU: {target_gpu.name} (UUID: {target_gpu.uuid})")
        typer.echo(f"Telemetry DB: {manager.db_manager.db_file}")
        typer.echo(f"Polling interval: {interval}s. Monitoring duration: {duration if duration > 0 else 'Indefinite (Ctrl+C to stop)'}s.")
        
        manager.start_telemetry_collection(interval_sec=interval, history_size=20)
        
        num_iterations = duration // interval if duration > 0 else float('inf')
        current_iteration = 0
        
        while current_iteration < num_iterations: # type: ignore # float('inf') is fine with <
            # Using asyncio.sleep within Typer commands usually works well if the command itself is async.
            await asyncio.sleep(interval) 
            with manager.telemetry_lock:
                if target_gpu.telemetry_history: # target_gpu is guaranteed to be not None here
                    latest_tel = target_gpu.telemetry_history[-1]
                    telemetry_line = (
                        f"[{latest_tel.timestamp.strftime('%H:%M:%S')}] "
                        f"Util:{latest_tel.gpu_utilization:.1f}% MemUtil:{latest_tel.memory_utilization:.1f}% "
                        f"Temp:{latest_tel.temperature:.1f}°C Pwr:{latest_tel.power_draw:.1f}W "
                        f"ClkGraph:{latest_tel.clock_graphics or 'N/A'}MHz ClkMem:{latest_tel.clock_memory or 'N/A'}MHz "
                        f"Fan:{latest_tel.fan_speed or 'N/A'}% "
                        f"Throttle: {','.join(latest_tel.throttle_reasons) if latest_tel.throttle_reasons else 'None'}"
                    )
                    typer.echo(telemetry_line)
                else:
                    typer.echo(f"[{datetime.now().strftime('%H:%M:%S')}] No telemetry data yet for {target_gpu.name}...")
            if duration > 0:
                 current_iteration +=1

    except KeyboardInterrupt:
        typer.echo(typer.style("\nMonitoring interrupted by user.", fg=typer_colors.YELLOW))
    except Exception as e:
        fc_logger.critical(f"🚨 Error during 'monitor' command: {e}", exc_info=True)
        raise typer.Exit(code=1)
    finally:
        if manager: # Manager should always be defined unless _get_manager_instance failed severely
            manager.cleanup()
        typer.echo("Monitoring stopped.")


if __name__ == "__main__":
    app()
