import asyncio
import json
import os
import sys
import logging # For getattr
from pathlib import Path
from typing import Optional

import typer

# Assuming this script is run from fluentcompute_project/
# Add the project root to sys.path to allow imports from the fluentcompute package
project_root = Path(__file__).resolve().parent.parent # Adjust if cli.py is in fluentcompute/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from fluentcompute import HardwareManager, logger as fc_logger # Use the pre-configured logger
    from fluentcompute.config.settings import DEFAULT_LOG_LEVEL, LOG_LEVEL_ENV_VAR
except ImportError as e:
    print(f"Error importing FluentCompute modules. Ensure fluentcompute_project is in PYTHONPATH. {e}")
    sys.exit(1)

app = typer.Typer(name="fluentcompute", help="FluentCompute CLI for hardware management and ML environment interaction.")

# --- Helper for HardwareManager initialization and cleanup ---
async def _get_manager_instance(ctx: typer.Context) -> HardwareManager:
    log_level_str = os.environ.get(LOG_LEVEL_ENV_VAR, logging.getLevelName(DEFAULT_LOG_LEVEL)).upper()
    log_level_val = getattr(logging, log_level_str, DEFAULT_LOG_LEVEL)
    fc_logger.setLevel(log_level_val)

    # Use project_root which should be fluentcompute_project directory
    # Then data subdir for the database
    db_storage_path = project_root / "data" 
    db_storage_path.mkdir(parents=True, exist_ok=True)
    
    manager = HardwareManager(
        enable_cloud_detection=ctx.obj.get("enable_cloud_detection", True),
        log_level=log_level_val,
        db_base_path=str(db_storage_path)
    )
    await manager.initialize_hardware()
    return manager

@app.callback()
def main_callback(ctx: typer.Context, cloud_detect: bool = typer.Option(True, "--cloud/--no-cloud", help="Enable/disable cloud provider detection.")):
    """
    FluentCompute CLI main entry point.
    """
    ctx.obj = {"enable_cloud_detection": cloud_detect}
    
    # Ensure proper event loop policy for subprocesses on Windows
    if sys.platform == "win32" and sys.version_info >= (3,8):
       try:
           # Check if a ProactorEventLoop is already set or compatible
           current_policy = asyncio.get_event_loop_policy()
           if not isinstance(current_policy, asyncio.WindowsProactorEventLoopPolicy):
                # Only set if not already a proactor or if explicitly needed by library behavior
                # This check might be too simple, but helps avoid overriding if not needed.
                if not hasattr(current_policy, '_proactor_loop_policy_set_globally') or \
                   not current_policy._proactor_loop_policy_set_globally:
                     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                     current_policy._proactor_loop_policy_set_globally = True # Custom flag
       except Exception as e_policy:
            fc_logger.debug(f"Could not set WindowsProactorEventLoopPolicy: {e_policy} (might not be an issue)")


@app.command()
async def detect(ctx: typer.Context):
    """
    Detects available hardware and prints detailed information for each GPU.
    """
    fc_logger.info("Starting hardware detection...")
    manager: Optional[HardwareManager] = None
    try:
        manager = await _get_manager_instance(ctx)
        gpus = manager.get_all_gpus()
        if not gpus:
            typer.echo("No GPUs detected on this system.")
            return

        typer.echo(typer.style("\n--- Detected GPUs ---", fg=typer.colors.CYAN, bold=True))
        for gpu in gpus:
            typer.echo(f"\nGPU Index: {gpu.gpu_index}")
            try:
                # Pretty print dictionary representation
                typer.echo(json.dumps(gpu.to_dict(), indent=2))
            except TypeError as e_json:
                typer.echo(f"Could not serialize GPU info to JSON: {e_json}")
                typer.echo(str(gpu)) # Fallback to string representation
    except Exception as e:
        fc_logger.critical(f"🚨 Error during 'detect' command: {e}", exc_info=True)
        raise typer.Exit(code=1)
    finally:
        if manager:
            fc_logger.info("Cleaning up manager resources from 'detect' command.")
            manager.cleanup()

@app.command()
async def summary(ctx: typer.Context):
    """
    Provides a summary of the system hardware and detected GPUs.
    """
    fc_logger.info("Fetching system summary...")
    manager: Optional[HardwareManager] = None
    try:
        manager = await _get_manager_instance(ctx)
        system_summary = manager.get_system_summary()
        typer.echo(typer.style("\n--- System and GPU Summary ---", fg=typer.colors.CYAN, bold=True))
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
            fc_logger.info("Cleaning up manager resources from 'summary' command.")
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
        manager = await _get_manager_instance(ctx)
        gpus = manager.get_all_gpus()

        if not gpus:
            typer.echo("No GPUs found to monitor.")
            return

        target_gpu = None
        if gpu_uuid:
            target_gpu = manager.get_gpu_by_uuid(gpu_uuid)
            if not target_gpu:
                typer.echo(typer.style(f"Error: GPU with UUID '{gpu_uuid}' not found.", fg=typer.colors.RED))
                fc_logger.warning(f"Requested GPU UUID '{gpu_uuid}' for monitoring not found. Available UUIDs: {[g.uuid for g in gpus]}")
                return
        else:
            target_gpu = gpus[0]
            fc_logger.info(f"No GPU UUID specified, monitoring first detected GPU: {target_gpu.name} ({target_gpu.uuid})")

        typer.echo(f"Monitoring GPU: {target_gpu.name} (UUID: {target_gpu.uuid})")
        typer.echo(f"Telemetry DB: {manager.db_manager.db_file}")
        typer.echo(f"Polling interval: {interval}s. Monitoring duration: {duration if duration > 0 else 'Indefinite (Ctrl+C to stop)'}s.")
        
        manager.start_telemetry_collection(interval_sec=interval, history_size=20) # Keep history small for CLI display
        
        num_iterations = duration // interval if duration > 0 else float('inf')
        current_iteration = 0
        
        while current_iteration < num_iterations:
            await asyncio.sleep(interval)
            with manager.telemetry_lock: # Ensure thread-safe access to telemetry_history
                if target_gpu.telemetry_history:
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
        typer.echo(typer.style("\nMonitoring interrupted by user.", fg=typer.colors.YELLOW))
    except Exception as e:
        fc_logger.critical(f"🚨 Error during 'monitor' command: {e}", exc_info=True)
        raise typer.Exit(code=1)
    finally:
        if manager:
            fc_logger.info("Cleaning up manager resources from 'monitor' command.")
            manager.cleanup()
        typer.echo("Monitoring stopped.")


if __name__ == "__main__":
    app()
