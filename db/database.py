"""
db/database.py — SQLite schema and helper functions for CyberGuard AI.

Design decisions:
- SQLite chosen over PostgreSQL for zero-config hackathon setup.
- Single 'alerts' table stores all detections from all 3 modules.
- Thread-safe via check_same_thread=False for Gradio's threaded callbacks.
- DB file lives at project root as 'cyberguard.db'.
"""

import sys
import sqlite3
import os
from datetime import datetime

# Fix Windows console encoding for emoji/unicode characters
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cyberguard.db")


def get_connection():
    """Get a thread-safe SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the alerts table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source_module TEXT NOT NULL,
            severity TEXT NOT NULL,
            score REAL,
            details TEXT
        );
    """)
    conn.commit()
    conn.close()
    print("[DB] ✅ Database initialized at:", DB_PATH)


def insert_alert(source_module: str, severity: str, score: float, details: str):
    """Insert a new alert row with auto-generated timestamp."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO alerts (timestamp, source_module, severity, score, details) VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(sep=" ", timespec="seconds"), source_module, severity, score, details)
    )
    conn.commit()
    conn.close()


def get_alerts(limit: int = 50):
    """Fetch recent alerts sorted newest first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, timestamp, source_module, severity, score, details FROM alerts ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_alert_count():
    """Get total number of alerts in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM alerts")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def clear_alerts():
    """Clear all alerts — useful for demo resets."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()
