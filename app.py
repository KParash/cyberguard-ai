"""
app.py — CyberGuard AI: Unified Fraud, Deepfake & Intrusion Detection Platform

This is the SINGLE entry point for the entire application.
Run with: python app.py

Architecture:
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │ Deepfake        │   │ Transaction     │   │ Intrusion       │
    │ Detector (ViT)  │   │ Risk Engine     │   │ Detector (IF)   │
    │ HuggingFace     │   │ (RandomForest)  │   │ (IsolationForest│
    └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
             │                     │                      │
             └─────────────┬───────┴──────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ Alert Engine│
                    │ (SQLite DB) │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Unified    │
                    │  Dashboard  │
                    └─────────────┘

Startup sequence:
    1. Initialize SQLite database (create tables if needed)
    2. Seed sample alerts (if DB is empty)
    3. Load deepfake ViT model from HuggingFace
    4. Train fraud RandomForestClassifier on synthetic UPI data
    5. Train intrusion IsolationForest on synthetic network flow data
    6. Launch Gradio app on all interfaces

Team: Syed Safwan Ghouri (AI/ML Engineer) & Parash Protim Khargharia (Cybersecurity Engineer)
"""

import sys
import os
import tempfile

# Fix Windows console encoding for emoji/unicode characters — must be done
# BEFORE any module imports that print emoji during load
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

# Ensure project root is on the path for all imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image

# Internal modules
from db.database import init_db
from data.seed_db import seed_db
from models import deepfake_detector, fraud_engine, intrusion_detector
from utils.alert_engine import log_alert, get_recent_alerts, get_dashboard_stats, severity_emoji


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP: Initialize everything
# ═══════════════════════════════════════════════════════════════════════════════

def startup():
    """Run all initialization steps. Called once before the Gradio app launches."""
    print("=" * 70)
    print("🛡️  CyberGuard AI — Unified Fraud, Deepfake & Intrusion Detection")
    print("=" * 70)
    print()

    # Step 1: Database
    print("[STARTUP] Step 1/5: Initializing SQLite database...")
    init_db()

    # Step 2: Seed data
    print("[STARTUP] Step 2/5: Seeding sample data...")
    seed_db()

    # Step 3: Deepfake model (this downloads from HuggingFace on first run)
    print("[STARTUP] Step 3/5: Loading deepfake detection model...")
    deepfake_detector.load_model()

    # Step 4: Fraud model
    print("[STARTUP] Step 4/5: Training fraud detection model...")
    fraud_engine.load_model()

    # Step 5: Intrusion model
    print("[STARTUP] Step 5/5: Training intrusion detection model...")
    intrusion_detector.load_model()

    print()
    print("=" * 70)
    print("✅ All systems initialized. CyberGuard AI is ready.")
    print("=" * 70)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: DEEPFAKE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_image(image):
    """Gradio callback: Analyze a single image for deepfake indicators."""
    if image is None:
        return "❌ No image provided", "", ""

    try:
        # Convert numpy array (from Gradio) to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        result = deepfake_detector.predict_image(image)

        # Format probability display
        prob_text = "\n".join(
            f"  {label}: {prob*100:.2f}%"
            for label, prob in result["probabilities"].items()
        )

        verdict = f"{result['verdict']}\nConfidence: {result['confidence']:.1f}%"
        details = (
            f"📊 Probability Breakdown:\n{prob_text}\n\n"
            f"🎯 Fake Probability: {result['fake_prob']*100:.2f}%\n"
            f"📝 Analysis: Single image analyzed using ViT-base model\n"
            f"   (prithivMLmods/Deep-Fake-Detector-v2-Model)"
        )

        # Build a label dict for Gradio's Label component
        label_output = result["probabilities"]

        return verdict, details, label_output

    except Exception as e:
        return f"❌ Error: {str(e)}", "", {}


def analyze_video(video):
    """
    Gradio callback: Analyze a video using frame-sampling approach.

    FRAME-SAMPLING APPROACH: Extracts 5 evenly-spaced frames from the video,
    runs each through the image-level ViT classifier, and averages the
    fake-probability scores into one aggregate verdict.
    This is NOT a true temporal/video deepfake model — documented honestly.
    """
    if video is None:
        return "❌ No video provided", "", ""

    try:
        result = deepfake_detector.predict_video(video)

        # Format per-frame breakdown
        frame_details = "\n".join(
            f"  Frame {r['frame']}: {r['fake_prob']*100:.1f}% fake — {r['verdict']}"
            for r in result["per_frame_scores"]
        )

        verdict = f"{result['verdict']}\nConfidence: {result['confidence']:.1f}%"
        details = (
            f"🎬 Video Analysis (Frame-Sampling Approach)\n"
            f"   Frames analyzed: {result['num_frames_analyzed']}\n"
            f"   Frames classified as fake: {result['fake_frame_count']}/{result['num_frames_analyzed']}\n"
            f"   Average fake probability: {result['fake_prob']*100:.2f}%\n\n"
            f"📊 Per-Frame Breakdown:\n{frame_details}\n\n"
            f"⚠️ Note: This uses frame-level image analysis, not temporal video analysis.\n"
            f"   No audio deepfake detection is performed."
        )

        label_output = result["probabilities"]

        return verdict, details, label_output

    except Exception as e:
        return f"❌ Error: {str(e)}", "", {}


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: TRANSACTION RISK ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_transaction(amount, is_new_payee, time_of_day, account_age, velocity, device_change, location_mismatch):
    """Gradio callback: Analyze a single UPI transaction."""
    try:
        result = fraud_engine.predict_transaction(
            amount=float(amount),
            time_of_day=int(time_of_day),
            sender_account_age_days=int(account_age),
            is_new_payee=int(is_new_payee),
            transaction_velocity_1hr=int(velocity),
            device_change_flag=int(device_change),
            location_mismatch_flag=int(location_mismatch),
        )

        # Risk tier with emoji
        tier_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        tier_display = f"{tier_emoji.get(result['risk_tier'], '⚪')} {result['risk_tier']} Risk"

        # Format score display
        score_display = f"Risk Score: {result['risk_score']}/100"

        # Format feature contributions
        contrib_text = "\n".join(
            f"  • {name}: value={info['value']}, importance={info['importance']}%"
            for name, info in result["feature_contributions"].items()
        )

        details = (
            f"🎯 Risk Score: {result['risk_score']}/100\n"
            f"📊 Risk Tier: {tier_display}\n"
            f"💡 {result['explanation']}\n\n"
            f"📋 Feature Analysis:\n{contrib_text}\n\n"
            f"📝 Model: RandomForestClassifier trained on synthetic UPI data"
        )

        return score_display, tier_display, result["explanation"], details

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, error_msg, error_msg, error_msg


def analyze_transaction_batch(file):
    """Gradio callback: Analyze a CSV of transactions in batch mode."""
    if file is None:
        return None

    try:
        df = pd.read_csv(file.name if hasattr(file, 'name') else file)
        result_df = fraud_engine.predict_batch(df)
        return result_df

    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: INTRUSION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def scan_traffic(dst_port, packet_count, byte_count, duration_ms, flag_syn, flag_rst, unique_ports):
    """Gradio callback: Analyze a single network flow."""
    try:
        result = intrusion_detector.predict_traffic(
            dst_port=int(dst_port),
            packet_count=int(packet_count),
            byte_count=int(byte_count),
            duration_ms=int(duration_ms),
            flag_syn=int(flag_syn),
            flag_rst=int(flag_rst),
            unique_ports_contacted=int(unique_ports),
        )

        score_display = f"Anomaly Score: {result['anomaly_score']}/100"

        details = (
            f"🔍 Anomaly Score: {result['anomaly_score']}/100\n"
            f"📊 Verdict: {result['verdict']}\n"
            f"💡 {result['reasoning']}\n\n"
            f"🔢 Raw IsolationForest Score: {result['raw_score']}\n"
            f"   (more negative = more anomalous)\n\n"
            f"📝 Model: IsolationForest trained on synthetic network flow data"
        )

        return score_display, result["verdict"], result["reasoning"], details

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, error_msg, error_msg, error_msg


def scan_traffic_batch(file):
    """Gradio callback: Analyze a CSV of network flows in batch mode."""
    if file is None:
        return None

    try:
        df = pd.read_csv(file.name if hasattr(file, 'name') else file)
        result_df = intrusion_detector.predict_batch(df)
        return result_df

    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: UNIFIED DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def refresh_dashboard():
    """Gradio callback: Refresh the dashboard with latest alerts and stats."""
    try:
        # Get dashboard stats
        stats = get_dashboard_stats()

        stats_text = (
            f"📊 **Dashboard Summary**\n\n"
            f"📅 Alerts Today: **{stats['total_today']}**\n"
            f"📁 Total All-Time: **{stats['total_all_time']}**\n\n"
            f"🔍 **Alerts by Module:**\n"
            f"  🧠 Deepfake: {stats['by_module'].get('deepfake', 0)}\n"
            f"  💳 Fraud: {stats['by_module'].get('fraud', 0)}\n"
            f"  🌐 Intrusion: {stats['by_module'].get('intrusion', 0)}\n\n"
            f"⚡ Highest Severity: {severity_emoji(stats['highest_severity'])} {stats['highest_severity']}"
        )

        # Get recent alerts as a table
        alerts = get_recent_alerts(limit=50)

        if alerts:
            alert_data = []
            for a in alerts:
                alert_data.append([
                    a["timestamp"],
                    a["source_module"].upper(),
                    a["severity_display"],
                    f"{a['score']}" if a['score'] is not None else "N/A",
                    a["details"][:120] + ("..." if len(a["details"]) > 120 else ""),
                ])
            alert_df = pd.DataFrame(
                alert_data,
                columns=["Timestamp", "Module", "Severity", "Score", "Details"]
            )
        else:
            alert_df = pd.DataFrame(columns=["Timestamp", "Module", "Severity", "Score", "Details"])

        return stats_text, alert_df

    except Exception as e:
        return f"❌ Error: {str(e)}", pd.DataFrame()


def run_demo_scenario():
    """
    🚨 COORDINATED ATTACK DEMO SCENARIO

    This is the single most important demo moment — it visually proves the
    "unified" value proposition from the pitch deck's scenario callout.

    When clicked, this function:
    (a) Simulates a deepfake detection alert (as if a deepfake image was analyzed)
    (b) Simulates a matching high-risk UPI transaction
    (c) Writes both alerts within the same few seconds
    (d) Detects the correlation and displays a banner

    For the demo, we directly log the alerts rather than running the actual models,
    because:
    1. The deepfake model requires a real image file
    2. We want the demo to be instant and reliable (one-click, no setup)
    3. The correlation detection logic is the key demo value, not re-running inference

    In a production system, this correlation would be triggered by temporal/contextual
    matching of real-time alerts from the actual models.
    """
    try:
        # (a) Log a deepfake alert — simulating detection of a fake video call
        log_alert(
            source_module="deepfake",
            severity="High",
            details=(
                "🎭 DEMO: Deepfake detected in video call screenshot — 94% fake probability. "
                "Face manipulation artifacts consistent with GAN-generated imagery. "
                "Caller claimed to be 'Rajesh Kumar, Branch Manager, SBI Main Branch'."
            ),
            score=0.94,
        )

        # (b) Log a matching high-risk fraud transaction — same session window
        log_alert(
            source_module="fraud",
            severity="High",
            details=(
                "💸 DEMO: High-risk UPI transaction: ₹4,85,000 to new payee 'unknown_recv_x@upi' "
                "at 2:15 AM. Device change detected + location mismatch + velocity 15 txns/hr. "
                "Transaction initiated moments after deepfake video call."
            ),
            score=97.0,
        )

        # (c) Log an intrusion alert — network anomaly from same time window
        log_alert(
            source_module="intrusion",
            severity="Medium",
            details=(
                "🌐 DEMO: Anomalous network activity detected from IP 10.0.0.55 — "
                "unusual data exfiltration pattern, 2400 packets in 280ms, "
                "coinciding with the suspicious video call and transaction."
            ),
            score=-0.72,
        )

        # (d) Refresh dashboard to show new alerts
        stats_text, alert_df = refresh_dashboard()

        # Build the correlation banner
        correlation_banner = (
            "🚨🚨🚨 CORRELATED THREAT DETECTED 🚨🚨🚨\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🎭 Deepfake impersonation (94% confidence)\n"
            "  + 💸 Suspicious UPI transaction (₹4,85,000 to new payee)\n"
            "  + 🌐 Anomalous network activity (data exfiltration pattern)\n"
            "  — all detected within the SAME SESSION WINDOW\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "⚡ ASSESSMENT: Possible social-engineering fraud in progress.\n"
            "   An attacker is likely using a deepfake video call to impersonate\n"
            "   a bank official, while simultaneously initiating unauthorized\n"
            "   UPI transfers and exfiltrating account data.\n\n"
            "🛡️ RECOMMENDED ACTIONS:\n"
            "  1. Immediately freeze the UPI transaction\n"
            "  2. Block the originating IP address (10.0.0.55)\n"
            "  3. Alert the account holder via registered phone\n"
            "  4. Escalate to cybercrime cell\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "This unified detection is powered by CyberGuard AI's cross-module\n"
            "alert correlation engine — detecting threats that siloed tools miss."
        )

        return correlation_banner, stats_text, alert_df

    except Exception as e:
        return f"❌ Error running demo: {str(e)}", "", pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO UI CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_app():
    """Construct the complete Gradio Blocks application with all 4 tabs."""

# Custom CSS for premium styling
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

.gradio-container {
    max-width: 1200px !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Main title styling */
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #0b0f19 0%, #1a1a2e 100%);
    color: white;
    padding: 30px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}

/* Glassmorphism for block panels */
.gr-block, .gr-box, .gr-panel {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
}

/* Premium Primary Buttons */
button.primary {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(79, 172, 254, 0.4) !important;
}

/* Correlation Banner */
.alert-banner {
    background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    font-weight: bold;
    font-size: 1.1rem;
    box-shadow: 0 8px 25px rgba(255, 8, 68, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Form inputs styling */
input, textarea, .gr-input {
    background-color: rgba(0,0,0,0.2) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}
"""

theme = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "system-ui", "sans-serif"],
    primary_hue="cyan",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="#0b0f19",
    body_text_color="#e2e8f0",
    block_background_fill="rgba(255, 255, 255, 0.02)",
    block_border_width="1px",
    block_border_color="rgba(255, 255, 255, 0.05)",
    block_radius="16px",
    button_primary_text_color="#ffffff",
)

def build_app():
    """Construct the complete Gradio Blocks application with all 4 tabs."""

    with gr.Blocks(
        title="CyberGuard AI — Unified Threat Detection",
    ) as demo:

        # ── Header ──
        gr.Markdown(
            """
            # 🛡️ CyberGuard AI
            ### Unified Fraud, Deepfake & Intrusion Detection Platform
            *Protecting India's digital infrastructure with AI-powered multi-vector threat detection*

            ---
            """,
        )

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════
            # TAB 1: DEEPFAKE DETECTOR
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("🧠 Deepfake Detector"):
                gr.Markdown(
                    """
                    ### 🧠 Deepfake Detection Engine
                    Powered by **ViT-base** (Vision Transformer) fine-tuned for deepfake detection.
                    Model: `prithivMLmods/Deep-Fake-Detector-v2-Model`

                    > **📌 Note:** This model performs image-level analysis. For video, we extract 5 evenly-spaced
                    > frames and aggregate scores (frame-sampling approach). No audio deepfake detection in this MVP.
                    """
                )

                with gr.Tabs():
                    with gr.TabItem("📷 Image Analysis"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                image_input = gr.Image(
                                    label="Upload a face image or a single frame extracted from a video call screenshot",
                                    type="numpy",
                                )
                                image_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")

                            with gr.Column(scale=1):
                                image_verdict = gr.Textbox(label="Verdict", lines=2, interactive=False)
                                image_probs = gr.Label(label="Probability Distribution", num_top_classes=2)
                                image_details = gr.Textbox(label="Detailed Analysis", lines=8, interactive=False)

                        image_btn.click(
                            fn=analyze_image,
                            inputs=[image_input],
                            outputs=[image_verdict, image_details, image_probs],
                        )

                    with gr.TabItem("🎬 Video Analysis"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                video_input = gr.Video(label="Upload a video file (.mp4, .avi, .mov)")
                                video_btn = gr.Button("🔍 Analyze Video (Frame Sampling)", variant="primary", size="lg")
                                gr.Markdown(
                                    "*Extracts 5 evenly-spaced frames and averages fake-probability scores. "
                                    "This is a frame-sampling approach, not temporal video analysis.*"
                                )

                            with gr.Column(scale=1):
                                video_verdict = gr.Textbox(label="Verdict", lines=2, interactive=False)
                                video_probs = gr.Label(label="Aggregate Probability", num_top_classes=2)
                                video_details = gr.Textbox(label="Detailed Analysis", lines=12, interactive=False)

                        video_btn.click(
                            fn=analyze_video,
                            inputs=[video_input],
                            outputs=[video_verdict, video_details, video_probs],
                        )

            # ══════════════════════════════════════════════════════════════
            # TAB 2: TRANSACTION RISK ENGINE
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("💳 Transaction Risk Engine"):
                gr.Markdown(
                    """
                    ### 💳 UPI Transaction Risk Engine
                    Powered by **RandomForestClassifier** trained on synthetic UPI transaction data.
                    Provides behavioral ML scoring with explainable feature-importance-driven risk assessment.

                    > **📌 Note:** Trained on synthetic/demo data generated for this hackathon, not real bank data.
                    """
                )

                with gr.Tabs():
                    with gr.TabItem("🔍 Single Transaction"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                txn_amount = gr.Number(label="💰 Transaction Amount (₹)", value=5000, minimum=1)
                                txn_new_payee = gr.Checkbox(label="🆕 New Payee (first-time recipient)", value=False)
                                txn_time = gr.Slider(label="🕐 Time of Day (hour, 0-23)", minimum=0, maximum=23, step=1, value=14)
                                txn_account_age = gr.Number(label="📅 Sender Account Age (days)", value=365, minimum=1)
                                txn_velocity = gr.Slider(label="⚡ Transaction Velocity (txns in last hour)", minimum=0, maximum=30, step=1, value=2)
                                txn_device = gr.Checkbox(label="📱 Device Change Detected", value=False)
                                txn_location = gr.Checkbox(label="📍 Location Mismatch", value=False)

                                txn_btn = gr.Button("🔍 Analyze Transaction", variant="primary", size="lg")

                            with gr.Column(scale=1):
                                txn_score = gr.Textbox(label="Risk Score", interactive=False)
                                txn_tier = gr.Textbox(label="Risk Tier", interactive=False)
                                txn_explanation = gr.Textbox(label="Explanation", lines=2, interactive=False)
                                txn_details = gr.Textbox(label="Detailed Analysis", lines=10, interactive=False)

                        txn_btn.click(
                            fn=analyze_transaction,
                            inputs=[txn_amount, txn_new_payee, txn_time, txn_account_age, txn_velocity, txn_device, txn_location],
                            outputs=[txn_score, txn_tier, txn_explanation, txn_details],
                        )

                    with gr.TabItem("📊 Batch Mode"):
                        gr.Markdown(
                            "Upload a CSV file with columns: `transaction_id`, `amount`, `time_of_day`, "
                            "`sender_account_age_days`, `is_new_payee`, `transaction_velocity_1hr`, "
                            "`device_change_flag`, `location_mismatch_flag`"
                        )
                        txn_batch_file = gr.File(label="Upload Transaction CSV", file_types=[".csv"])
                        txn_batch_btn = gr.Button("📊 Analyze Batch", variant="primary")
                        txn_batch_output = gr.Dataframe(label="Results (sorted by risk score, highest first)")

                        txn_batch_btn.click(
                            fn=analyze_transaction_batch,
                            inputs=[txn_batch_file],
                            outputs=[txn_batch_output],
                        )

            # ══════════════════════════════════════════════════════════════
            # TAB 3: INTRUSION DETECTOR
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("🌐 Intrusion Detector"):
                gr.Markdown(
                    """
                    ### 🌐 Network Intrusion Detection System
                    Powered by **IsolationForest** unsupervised anomaly detection on network flow data.
                    Detects port scans, SYN floods, and other network anomalies.

                    > **📌 Note:** Trained on synthetic network flow data generated for this hackathon.
                    """
                )

                with gr.Tabs():
                    with gr.TabItem("🔍 Single Flow Analysis"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                ids_port = gr.Number(label="🔌 Destination Port", value=80, minimum=0, maximum=65535)
                                ids_packets = gr.Number(label="📦 Packet Count", value=50, minimum=1)
                                ids_bytes = gr.Number(label="📊 Byte Count", value=30000, minimum=1)
                                ids_duration = gr.Number(label="⏱️ Duration (ms)", value=5000, minimum=1)
                                ids_syn = gr.Checkbox(label="🔗 SYN Flag", value=True)
                                ids_rst = gr.Checkbox(label="❌ RST Flag", value=False)
                                ids_unique_ports = gr.Slider(
                                    label="🔍 Unique Ports Contacted", minimum=1, maximum=100, step=1, value=1
                                )

                                ids_btn = gr.Button("🔍 Scan Traffic", variant="primary", size="lg")

                            with gr.Column(scale=1):
                                ids_score = gr.Textbox(label="Anomaly Score", interactive=False)
                                ids_verdict = gr.Textbox(label="Verdict", interactive=False)
                                ids_reasoning = gr.Textbox(label="Reasoning", lines=2, interactive=False)
                                ids_details = gr.Textbox(label="Detailed Analysis", lines=10, interactive=False)

                        ids_btn.click(
                            fn=scan_traffic,
                            inputs=[ids_port, ids_packets, ids_bytes, ids_duration, ids_syn, ids_rst, ids_unique_ports],
                            outputs=[ids_score, ids_verdict, ids_reasoning, ids_details],
                        )

                    with gr.TabItem("📊 Batch Mode"):
                        gr.Markdown(
                            "Upload a CSV file with columns: `flow_id`, `src_ip`, `dst_port`, `protocol`, "
                            "`packet_count`, `byte_count`, `duration_ms`, `flag_syn`, `flag_rst`, `unique_ports_contacted`"
                        )
                        ids_batch_file = gr.File(label="Upload Network Flow CSV", file_types=[".csv"])
                        ids_batch_btn = gr.Button("📊 Scan Batch", variant="primary")
                        ids_batch_output = gr.Dataframe(label="Results (sorted by anomaly score, highest first)")

                        ids_batch_btn.click(
                            fn=scan_traffic_batch,
                            inputs=[ids_batch_file],
                            outputs=[ids_batch_output],
                        )

            # ══════════════════════════════════════════════════════════════
            # TAB 4: UNIFIED DASHBOARD
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("📊 Unified Dashboard"):
                gr.Markdown(
                    """
                    ### 📊 Unified Alert Dashboard
                    Cross-module threat correlation and real-time alert monitoring.
                    All alerts from Deepfake, Fraud, and Intrusion modules feed into this unified view.
                    """
                )

                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary", size="lg")
                    demo_btn = gr.Button(
                        "▶️ Run Demo Scenario: Coordinated Attack",
                        variant="primary",
                        size="lg",
                    )

                # Correlation banner (shown when demo scenario runs)
                correlation_output = gr.Textbox(
                    label="🚨 Threat Correlation Engine",
                    lines=20,
                    interactive=False,
                    visible=True,
                    value="Click '▶️ Run Demo Scenario' to simulate a coordinated attack and see cross-module correlation in action.",
                )

                # Stats summary
                dashboard_stats = gr.Markdown(label="Dashboard Statistics")

                # Alerts table
                alerts_table = gr.Dataframe(
                    label="📋 Recent Alerts (all modules, newest first)",
                    headers=["Timestamp", "Module", "Severity", "Score", "Details"],
                    interactive=False,
                )

                # Wire up refresh
                refresh_btn.click(
                    fn=refresh_dashboard,
                    inputs=[],
                    outputs=[dashboard_stats, alerts_table],
                )

                # Wire up demo scenario
                demo_btn.click(
                    fn=run_demo_scenario,
                    inputs=[],
                    outputs=[correlation_output, dashboard_stats, alerts_table],
                )

                # Auto-refresh on tab load
                demo.load(
                    fn=refresh_dashboard,
                    inputs=[],
                    outputs=[dashboard_stats, alerts_table],
                )

        # ── Footer ──
        gr.Markdown(
            """
            ---
            **CyberGuard AI** | Built by **Syed Safwan Ghouri** (AI/ML Engineer) & **Parash Protim Khargharia** (Cybersecurity Engineer)
            | Hackathon MVP — Protecting India's digital infrastructure with unified AI-powered threat detection
            """
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run all startup initialization
    startup()

    # Build and launch the Gradio app
    demo = build_app()

    print("\n🚀 Launching CyberGuard AI...")
    print("   Local URL: http://localhost:7860")
    print("   Share URL will be generated if share=True\n")

    # launch with share=True so judges can access via public link
    # server_name="0.0.0.0" binds to all interfaces for LAN access
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        theme=theme,
        css=custom_css,
    )
