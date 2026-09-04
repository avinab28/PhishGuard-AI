import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any
from backend.config import settings

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initializes the SQLite scan history table if not present."""
    settings.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_type TEXT NOT NULL,
                target TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                probability REAL NOT NULL,
                verdict TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

def record_scan(scan_type: str, target: str, risk_level: str, probability: float, verdict: str) -> int:
    """Records an audit scan entry into SQLite."""
    # Truncate stored target for privacy/storage if message is long
    saved_target = target[:200] + "..." if len(target) > 200 else target
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scan_history (scan_type, target, risk_level, probability, verdict, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (scan_type, saved_target, risk_level, round(probability, 4), verdict, now_iso))
        conn.commit()
        return cursor.lastrowid or 0

def get_recent_scans(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves the most recent scans."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, scan_type, target, risk_level, probability, verdict, created_at
            FROM scan_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def clear_scan_history() -> None:
    """Clears all historical scan entries."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM scan_history")
        conn.commit()
