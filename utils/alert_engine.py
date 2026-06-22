"""
utils/alert_engine.py — Unified alert object/schema shared by all 3 detection modules.

All modules (deepfake, fraud, intrusion) call log_alert() to persist detections
into the shared SQLite alerts table, enabling the cross-module correlation that
is CyberGuard AI's key differentiator.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import insert_alert, get_alerts, get_alert_count, get_connection
from datetime import datetime, timedelta


def log_alert(source_module: str, severity: str, details: str, score: float):
    """
    Log a detection alert to the unified SQLite database.

    Args:
        source_module: One of "deepfake", "fraud", "intrusion"
        severity: One of "Low", "Medium", "High"
        details: Human-readable description of what was detected
        score: Numeric confidence/risk score (0.0 - 1.0 or 0-100 depending on module)
    """
    assert severity in ("Low", "Medium", "High"), f"Invalid severity: {severity}"
    assert source_module in ("deepfake", "fraud", "intrusion"), f"Invalid module: {source_module}"
    insert_alert(source_module, severity, score, details)
    print(f"[ALERT] {severity_emoji(severity)} [{source_module.upper()}] {details} (score={score})")


def severity_emoji(severity: str) -> str:
    """Return emoji prefix for severity level."""
    return {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(severity, "⚪")


def get_recent_alerts(limit: int = 50):
    """Fetch recent alerts with emoji-prefixed severity for display."""
    alerts = get_alerts(limit)
    for alert in alerts:
        alert["severity_display"] = f"{severity_emoji(alert['severity'])} {alert['severity']}"
    return alerts


def get_dashboard_stats():
    """
    Compute dashboard summary statistics.

    Returns:
        dict with keys: total_today, by_module (dict), highest_severity, total_all_time
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Total alerts all time
    cursor.execute("SELECT COUNT(*) FROM alerts")
    total_all_time = cursor.fetchone()[0]

    # Total alerts today (using ISO date prefix match)
    today_str = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) FROM alerts WHERE timestamp LIKE ?", (f"{today_str}%",))
    total_today = cursor.fetchone()[0]

    # Alerts by module
    by_module = {}
    for module in ("deepfake", "fraud", "intrusion"):
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE source_module = ?", (module,))
        by_module[module] = cursor.fetchone()[0]

    # Highest severity alert (priority: High > Medium > Low)
    highest_severity = "None"
    for sev in ("High", "Medium", "Low"):
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE severity = ?", (sev,))
        if cursor.fetchone()[0] > 0:
            highest_severity = sev
            break

    conn.close()
    return {
        "total_today": total_today,
        "total_all_time": total_all_time,
        "by_module": by_module,
        "highest_severity": highest_severity,
    }
