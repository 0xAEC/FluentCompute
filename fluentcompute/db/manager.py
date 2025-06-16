import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from fluentcompute.utils.logging_config import logger
from fluentcompute.config.settings import DB_FILE_NAME
from fluentcompute.models import TelemetryData # For type hinting

class DBManager:
    def __init__(self, db_file_name: str = DB_FILE_NAME, base_path: Optional[Path] = None):
        if base_path is None:
            # Default to current working directory if no base_path is given
            # For a real application, this should be a well-defined application data directory
            base_path = Path.cwd() 
        self.db_file = base_path / db_file_name
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_db()

    def _initialize_db(self):
        try:
            # Ensure the directory for the database file exists
            self.db_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.conn = sqlite3.connect(self.db_file, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES, timeout=10)
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT, gpu_uuid TEXT NOT NULL, timestamp TIMESTAMP NOT NULL,
                gpu_utilization REAL, memory_utilization REAL, temperature REAL, power_draw REAL,
                fan_speed REAL, clock_graphics INTEGER, clock_memory INTEGER, throttle_reasons TEXT
            )""")
            # Index for faster querying by gpu_uuid and timestamp
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_telemetry_gpu_ts ON telemetry (gpu_uuid, timestamp DESC);
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS hardware_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT, snapshot_time TIMESTAMP NOT NULL, system_summary_json TEXT NOT NULL
            )""")
            self.conn.commit()
            logger.info(f"🗄️ Database initialized at {self.db_file}")
        except sqlite3.Error as e:
            logger.error(f"❌ Database initialization error: {e}")
            if self.conn: self.conn.close()
            self.conn = None
        except Exception as e: # Catch other potential errors like permission issues
            logger.error(f"❌ Failed to initialize database at {self.db_file}: {e}")
            if self.conn: self.conn.close()
            self.conn = None


    def log_telemetry_batch(self, telemetry_batch: List[Tuple[str, TelemetryData]]):
        if not self.conn or not telemetry_batch: return
        try:
            cursor = self.conn.cursor()
            records_to_insert = []
            for gpu_uuid, data in telemetry_batch:
                records_to_insert.append((
                    gpu_uuid, data.timestamp, data.gpu_utilization, data.memory_utilization, 
                    data.temperature, data.power_draw, data.fan_speed, 
                    data.clock_graphics, data.clock_memory, json.dumps(data.throttle_reasons or []) # Ensure throttle_reasons is list
                ))
            
            cursor.executemany("""
            INSERT INTO telemetry (gpu_uuid, timestamp, gpu_utilization, memory_utilization, temperature, power_draw, fan_speed, clock_graphics, clock_memory, throttle_reasons)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records_to_insert)
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"❌ Error batch logging telemetry to DB: {e}")
            # Consider rollback if transactions are used more extensively: if self.conn: self.conn.rollback()

    def log_hardware_snapshot(self, summary: Dict[str, Any]):
        if not self.conn: return
        try:
            cursor = self.conn.cursor()
            cursor.execute("INSERT INTO hardware_snapshots (snapshot_time, system_summary_json) VALUES (?, ?)",
                           (datetime.now(), json.dumps(summary, default=str)))
            self.conn.commit()
            logger.info("📝 Hardware snapshot logged to database.")
        except sqlite3.Error as e: 
            logger.error(f"❌ Error logging hardware snapshot to DB: {e}")
            # if self.conn: self.conn.rollback()
    
    def close(self):
        if self.conn:
            try:
                self.conn.close()
                logger.info("🗄️ Database connection closed.")
            except sqlite3.Error as e:
                logger.error(f"❌ Error closing database connection: {e}")
            self.conn = None
