"""
models/intrusion_detector.py — Network Intrusion Detection using IsolationForest.

DESIGN DECISIONS:
- Uses IsolationForest (unsupervised anomaly detection) — standard, well-justified
  choice for IDS and matches the pitch's "unsupervised anomaly detection for
  network intrusion" framing exactly.

- IsolationForest scores range from -1 (most anomalous) to +1 (most normal).
  We convert these to a 0-100 anomaly severity scale for user-friendly display.

- Trained on synthetic network flow data generated for this hackathon.
  Features mimic real NetFlow/IPFIX fields used in production IDS systems.

- contamination=0.15 matches the ~15% anomaly rate in the training data.

FEATURES USED:
- dst_port: Destination port number (0 for scans hitting many ports)
- packet_count: Number of packets in the flow
- byte_count: Total bytes transferred
- duration_ms: Flow duration in milliseconds
- flag_syn: SYN flag presence (TCP handshake)
- flag_rst: RST flag presence (connection reset)
- unique_ports_contacted: Number of unique destination ports from this source
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from utils.alert_engine import log_alert

# Feature columns used for training and prediction
FEATURE_COLS = [
    "dst_port", "packet_count", "byte_count", "duration_ms",
    "flag_syn", "flag_rst", "unique_ports_contacted"
]

# Global model and scaler
_model = None
_scaler = None


def load_model():
    """
    Train the IsolationForest on sample_network_logs.csv.
    Called once at app startup. Unsupervised — no labels used for training.
    """
    global _model, _scaler

    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "sample_network_logs.csv"
    )

    print("[IDS] 🔄 Training IsolationForest on synthetic network flow data...")

    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].values

    # Scale features
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)

    # Train IsolationForest
    # contamination=0.15 — expected proportion of anomalies in the data
    # random_state=42 for reproducibility
    _model = IsolationForest(
        n_estimators=100,
        contamination=0.15,
        random_state=42,
        max_samples="auto"
    )
    _model.fit(X_scaled)

    # Validation: check anomaly detection on training data
    predictions = _model.predict(X_scaled)
    anomaly_count = (predictions == -1).sum()
    labels = df["label"].values
    actual_anomalies = labels.sum()

    print(f"[IDS] ✅ Model trained. Samples: {len(X)}, "
          f"Detected anomalies: {anomaly_count}, "
          f"Actual anomalies in data: {actual_anomalies}")


def predict_traffic(
    dst_port: int,
    packet_count: int,
    byte_count: int,
    duration_ms: int,
    flag_syn: int,
    flag_rst: int,
    unique_ports_contacted: int
) -> dict:
    """
    Analyze a single network flow for intrusion/anomaly indicators.

    Returns:
        dict with keys: anomaly_score (0-100), verdict (str), reasoning (str),
                       raw_score (float), is_anomaly (bool)
    """
    if _model is None:
        raise RuntimeError("IDS model not loaded. Call load_model() first.")

    # Build feature vector
    features = np.array([[
        dst_port, packet_count, byte_count, duration_ms,
        flag_syn, flag_rst, unique_ports_contacted
    ]])

    # Scale features
    features_scaled = _scaler.transform(features)

    # Get anomaly score from IsolationForest
    # score_samples() returns values in roughly [-1, 0.5] range
    # More negative = more anomalous
    raw_score = _model.score_samples(features_scaled)[0]
    prediction = _model.predict(features_scaled)[0]  # -1 = anomaly, 1 = normal

    # Convert to 0-100 anomaly severity scale
    # raw_score typically ranges from about -0.8 (very anomalous) to 0.3 (very normal)
    # We map this to 0-100 where 100 = most anomalous
    anomaly_score = max(0, min(100, round((0.3 - raw_score) * 100 / 1.1, 1)))

    # Determine verdict
    if prediction == -1 and anomaly_score > 70:
        verdict = "🔴 MALICIOUS — High-confidence anomaly"
    elif prediction == -1:
        verdict = "🟡 SUSPICIOUS — Anomalous traffic pattern"
    else:
        verdict = "🟢 NORMAL — No anomaly detected"

    # Generate reasoning based on feature values
    reasoning = _generate_reasoning(
        dst_port, packet_count, byte_count, duration_ms,
        flag_syn, flag_rst, unique_ports_contacted, prediction
    )

    # Log alert if anomalous
    if prediction == -1:
        severity = "High" if anomaly_score > 70 else "Medium"
        log_alert(
            source_module="intrusion",
            severity=severity,
            details=reasoning,
            score=round(raw_score, 4)
        )

    return {
        "anomaly_score": anomaly_score,
        "verdict": verdict,
        "reasoning": reasoning,
        "raw_score": round(raw_score, 4),
        "is_anomaly": prediction == -1,
    }


def _generate_reasoning(
    dst_port, packet_count, byte_count, duration_ms,
    flag_syn, flag_rst, unique_ports_contacted, prediction
) -> str:
    """Generate human-readable reasoning for the IDS verdict."""
    flags = []

    # Port scan detection
    if unique_ports_contacted > 10:
        flags.append(f"Port scan detected: {unique_ports_contacted} unique ports contacted")

    # SYN flood indicators
    if packet_count > 500 and duration_ms < 500:
        flags.append(f"Possible SYN flood: {packet_count} packets in {duration_ms}ms")
    elif packet_count > 300:
        flags.append(f"High packet burst: {packet_count} packets")

    # Low byte-to-packet ratio (typical of scans/floods)
    if packet_count > 0 and byte_count / packet_count < 10:
        flags.append("Low bytes-per-packet ratio (scan/flood signature)")

    # Very short duration with high activity
    if duration_ms < 300 and packet_count > 100:
        flags.append(f"Burst traffic: {packet_count} packets in {duration_ms}ms")

    # Known dangerous ports
    if dst_port == 445:
        flags.append("Traffic to port 445 (SMB — common attack vector)")
    elif dst_port == 3389:
        flags.append("Traffic to port 3389 (RDP — potential brute force)")

    # SYN without RST (incomplete handshakes)
    if flag_syn and not flag_rst and packet_count > 200:
        flags.append("SYN flags without RST responses (incomplete handshakes)")

    if flags:
        return " | ".join(flags)
    elif prediction == -1:
        return "Statistical anomaly detected — traffic pattern deviates from learned baseline."
    else:
        return "Traffic pattern matches normal baseline. No anomalies detected."


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run intrusion detection on a batch of network flows from a CSV DataFrame.

    Expects DataFrame with columns matching FEATURE_COLS.
    Returns DataFrame with added anomaly_score, verdict, and reasoning columns,
    sorted by anomaly_score descending (most anomalous first).
    """
    if _model is None:
        raise RuntimeError("IDS model not loaded. Call load_model() first.")

    results = []
    for _, row in df.iterrows():
        try:
            result = predict_traffic(
                dst_port=int(row.get("dst_port", 80)),
                packet_count=int(row.get("packet_count", 50)),
                byte_count=int(row.get("byte_count", 30000)),
                duration_ms=int(row.get("duration_ms", 5000)),
                flag_syn=int(row.get("flag_syn", 1)),
                flag_rst=int(row.get("flag_rst", 0)),
                unique_ports_contacted=int(row.get("unique_ports_contacted", 1)),
            )
            results.append({
                "flow_id": row.get("flow_id", "N/A"),
                "src_ip": row.get("src_ip", "N/A"),
                "anomaly_score": result["anomaly_score"],
                "verdict": result["verdict"],
                "reasoning": result["reasoning"],
            })
        except Exception as e:
            results.append({
                "flow_id": row.get("flow_id", "N/A"),
                "src_ip": row.get("src_ip", "N/A"),
                "anomaly_score": -1,
                "verdict": "Error",
                "reasoning": str(e),
            })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    return result_df
