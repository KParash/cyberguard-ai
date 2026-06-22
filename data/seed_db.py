"""
data/seed_db.py — Populates cyberguard.db with sample alerts from the CSV datasets on first run.

This creates a realistic-looking alert history so the Unified Dashboard tab
has content immediately upon first launch, before any manual analysis is done.
Only seeds if the alerts table is empty (idempotent).
"""

import sys
import os
import random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import init_db, get_alert_count, insert_alert


def seed_db():
    """Populate the alerts table with realistic sample alerts if empty."""
    if get_alert_count() > 0:
        print("[SEED] ℹ️  Database already has alerts, skipping seed.")
        return

    print("[SEED] 🌱 Seeding database with sample alerts...")

    # Generate timestamps spread across the last 24 hours for realism
    now = datetime.now()
    def random_recent_timestamp():
        offset = timedelta(minutes=random.randint(5, 1440))
        return (now - offset).isoformat(sep=" ", timespec="seconds")

    # --- Sample Deepfake Alerts ---
    deepfake_alerts = [
        ("High", 0.92, "Deepfake detected in uploaded image — 92% fake probability. Face manipulation artifacts found."),
        ("High", 0.87, "Video frame analysis: 4/5 frames classified as deepfake (avg 87% fake probability)."),
        ("Medium", 0.65, "Uploaded image flagged as potentially manipulated — 65% fake probability. Manual review recommended."),
        ("Low", 0.35, "Image analysis complete — 35% fake probability. Classified as likely authentic."),
    ]

    # --- Sample Fraud Alerts ---
    fraud_alerts = [
        ("High", 95.0, "High-risk UPI transaction: ₹48,500 to new payee at 2:30 AM, location mismatch detected, velocity 12 txns/hr."),
        ("High", 88.0, "Suspicious transaction: ₹25,000 with device change + new payee + late-night timing."),
        ("Medium", 72.0, "Moderate-risk transaction: ₹15,000, new payee with account age < 30 days."),
        ("Medium", 68.0, "Flagged transaction: ₹8,200 with elevated velocity (8 txns/hr) and new payee."),
        ("Low", 30.0, "Low-risk transaction: ₹2,500 to known payee, normal patterns."),
    ]

    # --- Sample Intrusion Alerts ---
    intrusion_alerts = [
        ("High", -0.85, "Port scan detected: 45 unique ports contacted from 192.168.1.105 in 200ms burst."),
        ("High", -0.78, "Possible SYN flood: 1200 SYN packets with 0 RST responses from 10.0.0.55, 850 bytes."),
        ("Medium", -0.62, "Anomalous traffic pattern: 380 packets to port 445 (SMB) over 5 seconds from 172.16.0.22."),
        ("Low", -0.30, "Minor anomaly: Unusual protocol usage on port 8443, 45 packets, low threat score."),
    ]

    for severity, score, details in deepfake_alerts:
        insert_alert("deepfake", severity, score, details)
    for severity, score, details in fraud_alerts:
        insert_alert("fraud", severity, score, details)
    for severity, score, details in intrusion_alerts:
        insert_alert("intrusion", severity, score, details)

    # Update timestamps to be spread across recent hours for realism
    # (The insert_alert function uses current time, but for seeding we want variety)
    from db.database import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM alerts ORDER BY id")
    ids = [row["id"] for row in cursor.fetchall()]
    for alert_id in ids:
        cursor.execute(
            "UPDATE alerts SET timestamp = ? WHERE id = ?",
            (random_recent_timestamp(), alert_id)
        )
    conn.commit()
    conn.close()

    print(f"[SEED] ✅ Seeded {len(deepfake_alerts) + len(fraud_alerts) + len(intrusion_alerts)} sample alerts.")


if __name__ == "__main__":
    init_db()
    seed_db()
