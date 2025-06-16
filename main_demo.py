import asyncio
import json
import os
import sys
import logging # For getattr
from pathlib import Path

# Assuming this script is run from fluentcompute_project/
# Add the project root to sys.path to allow imports from the fluentcompute package
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from fluentcompute import HardwareManager, logger as fc_logger # Use the pre-configured logger
    from fluentcompute.config.settings import DEFAULT_LOG_LEVEL, LOG_LEVEL_ENV_VAR
except ImportError as e:
    print(f"Error importing FluentCompute modules. Make sure fluentcompute_project is in your PYTHONPATH or you are running from the project root. {e}")
    sys.exit(1)


async def main_fluent_compute_demo():
    """Main function to demonstrate FluentCompute capabilities."""
    print("Starting FluentCompute Hardware Integration Engine Demo (Refactored)")
    
    # Determine log level
    log_level_str = os.environ.get(LOG_LEVEL_ENV_VAR, logging.getLevelName(DEFAULT_LOG_LEVEL)).upper()
    log_level_val = getattr(logging, log_level_str, DEFAULT_LOG_LEVEL)
    
    # Reconfigure the logger if needed (it's already initialized by fluentcompute.utils.logging_config)
    fc_logger.setLevel(log_level_val)
    
    # Example: run the DB in a 'data' subdirectory of the project.
    db_storage_path = project_root / "data"
    db_storage_path.mkdir(parents=True, exist_ok=True)


    # Initialize HardwareManager
    # log_level is now handled by the fc_logger directly, HardwareManager can still take it if desired for its own logger instance though.
    manager = HardwareManager(enable_cloud_detection=True, log_level=log_level_val, db_base_path=str(db_storage_path)) 
    
    try:
        await manager.initialize_hardware()
        
        print("\n--- System and GPU Summary ---")
        summary = manager.get_system_summary()
        # Use try-except for JSON dumping in case of non-serializable data, though dataclasses should be fine.
        try:
            print(json.dumps(summary, indent=2))
        except TypeError as e_json:
            print(f"Could not serialize summary to JSON: {e_json}")
            print(summary) # Print raw dict as fallback

        if manager.get_all_gpus():
            manager.start_telemetry_collection(interval_sec=3, history_size=20) # Reduced history for demo brevity
            print(f"\nTelemetry collection started (DB at {manager.db_manager.db_file}). Monitoring for ~15 seconds. Press Ctrl+C to stop early.")
            
            for _ in range(5): # Show a few telemetry updates
                await asyncio.sleep(3)
                # Check telemetry history on the first available GPU for demo
                if manager.gpus and manager.gpus[0].telemetry_history:
                    latest_tel = manager.gpus[0].telemetry_history[-1]
                    print(f"  Live ({manager.gpus[0].name}): "
                          f"Util:{latest_tel.gpu_utilization:.1f}%, MemUtil:{latest_tel.memory_utilization:.1f}%, "
                          f"Temp:{latest_tel.temperature:.1f}°C, Pwr:{latest_tel.power_draw:.1f}W, "
                          f"Thr: {','.join(latest_tel.throttle_reasons) if latest_tel.throttle_reasons else 'None'}")
                elif manager.gpus:
                     print(f"  Live ({manager.gpus[0].name}): No telemetry data yet...")

        else:
            print("No GPUs found to monitor.")

    except KeyboardInterrupt:
        print("\nInterrupt received, shutting down...")
    except Exception as e:
        # Use fc_logger for consistency, it's already configured.
        fc_logger.critical(f"🚨 An unhandled critical error occurred in demo: {e}", exc_info=True)
    finally:
        print("Cleaning up and exiting demo.")
        if 'manager' in locals() and manager:
            manager.cleanup()

if __name__ == "__main__":
    # Ensure proper event loop policy for subprocesses on Windows if needed (especially for asyncio.create_subprocess_shell)
    if sys.platform == "win32" and sys.version_info >= (3,8):
       try:
           asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
       except Exception as e_policy:
            print(f"Could not set WindowsProactorEventLoopPolicy: {e_policy} (might not be an issue)")
    
    asyncio.run(main_fluent_compute_demo())
