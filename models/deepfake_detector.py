"""
models/deepfake_detector.py — Wraps the HuggingFace ViT deepfake detection model.

Model: prithivMLmods/Deep-Fake-Detector-v2-Model
Architecture: Vision Transformer (ViT-base), fine-tuned for binary classification (Real vs Fake)
Input: 224x224 RGB images

IMPORTANT NOTES:
- This model is IMAGE-LEVEL only. It does not analyze audio.
- For video input, we use a FRAME-SAMPLING APPROACH: extract 5 evenly-spaced frames
  from the video, run each through the image classifier, and aggregate scores.
- This gives a reasonable approximation for video deepfake detection but is NOT
  a true temporal/video deepfake model. Documented honestly as a hackathon MVP approach.
- No audio deepfake detection is wired in — listed as future work.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import cv2
import numpy as np
from utils.alert_engine import log_alert

# Global model/processor — loaded once at startup, cached in memory
_model = None
_processor = None


def load_model():
    """Load the ViT deepfake detection model from HuggingFace. Called once at startup."""
    global _model, _processor
    print("[DEEPFAKE] 🔄 Loading ViT deepfake detection model from HuggingFace...")
    _model = ViTForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
    _processor = ViTImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
    _model.eval()  # Set to evaluation mode — no training needed
    print("[DEEPFAKE] ✅ Model loaded successfully.")
    print(f"[DEEPFAKE]    Labels: {_model.config.id2label}")


def predict_image(image: Image.Image) -> dict:
    """
    Run deepfake detection on a single PIL Image.

    Returns:
        dict with keys: probabilities (dict), verdict (str), confidence (float), fake_prob (float)
    """
    if _model is None or _processor is None:
        raise RuntimeError("Deepfake model not loaded. Call load_model() first.")

    # Preprocess and run inference
    inputs = _processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
    labels = _model.config.id2label

    # Build probability dict: {label_name: probability}
    prob_dict = {labels[i]: round(probs[i], 4) for i in range(len(probs))}

    # Determine fake probability — handle various label naming conventions
    # The model may use "Fake"/"Real" or "fake"/"real" or other variations
    fake_prob = 0.0
    for label, prob in prob_dict.items():
        if "fake" in label.lower() or "deepfake" in label.lower():
            fake_prob = prob
            break

    # Generate verdict
    if fake_prob > 0.6:
        verdict = "⚠️ LIKELY DEEPFAKE"
        severity = "High" if fake_prob > 0.8 else "Medium"
    elif fake_prob > 0.4:
        verdict = "🟡 INCONCLUSIVE — Manual Review Recommended"
        severity = "Low"
    else:
        verdict = "✅ LIKELY AUTHENTIC"
        severity = None  # No alert for authentic images

    confidence = max(fake_prob, 1.0 - fake_prob) * 100

    # Log alert if deepfake detected
    if severity:
        log_alert(
            source_module="deepfake",
            severity=severity,
            details=f"Deepfake detected in uploaded image — {fake_prob*100:.1f}% fake probability.",
            score=round(fake_prob, 4)
        )

    return {
        "probabilities": prob_dict,
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "fake_prob": round(fake_prob, 4),
    }


def extract_frames_from_video(video_path: str, num_frames: int = 5) -> list:
    """
    Extract evenly-spaced frames from a video file using OpenCV.

    FRAME-SAMPLING APPROACH:
    This extracts `num_frames` frames evenly distributed across the video duration.
    Each frame is converted from BGR (OpenCV) to RGB (PIL) format.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract (default: 5)

    Returns:
        List of PIL Images
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError("Video has no frames")

    # Calculate evenly-spaced frame indices
    # E.g., for 100 frames and num_frames=5: [0, 25, 50, 75, 99]
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)

    cap.release()

    if len(frames) == 0:
        raise ValueError("Could not extract any frames from the video")

    return frames


def predict_video(video_path: str) -> dict:
    """
    Run deepfake detection on a video by extracting 5 frames and averaging scores.

    FRAME-SAMPLING APPROACH: This is NOT a true temporal video deepfake model.
    We extract 5 evenly-spaced frames, run each through the image-level ViT classifier,
    and aggregate (average) the fake-probability scores into one verdict.
    This provides a reasonable approximation for hackathon demo purposes.

    Args:
        video_path: Path to the video file (.mp4, .avi, .mov, etc.)

    Returns:
        dict with keys: probabilities (dict), verdict (str), confidence (float),
                       fake_prob (float), num_frames_analyzed (int), per_frame_scores (list)
    """
    frames = extract_frames_from_video(video_path, num_frames=5)

    # Run each frame through the model
    per_frame_results = []
    total_fake_prob = 0.0

    for i, frame in enumerate(frames):
        result = predict_image.__wrapped__(frame) if hasattr(predict_image, '__wrapped__') else _predict_image_no_alert(frame)
        per_frame_results.append({
            "frame": i + 1,
            "fake_prob": result["fake_prob"],
            "verdict": result["verdict"]
        })
        total_fake_prob += result["fake_prob"]

    # Average fake probability across all frames
    avg_fake_prob = total_fake_prob / len(frames)

    # Aggregate verdict based on average
    if avg_fake_prob > 0.6:
        verdict = "⚠️ LIKELY DEEPFAKE"
        severity = "High" if avg_fake_prob > 0.8 else "Medium"
    elif avg_fake_prob > 0.4:
        verdict = "🟡 INCONCLUSIVE — Manual Review Recommended"
        severity = "Low"
    else:
        verdict = "✅ LIKELY AUTHENTIC"
        severity = None

    confidence = max(avg_fake_prob, 1.0 - avg_fake_prob) * 100

    # Count how many frames were classified as fake
    fake_frame_count = sum(1 for r in per_frame_results if r["fake_prob"] > 0.5)

    # Log alert for video analysis if deepfake detected
    if severity:
        log_alert(
            source_module="deepfake",
            severity=severity,
            details=(
                f"Video frame analysis: {fake_frame_count}/{len(frames)} frames classified as deepfake "
                f"(avg {avg_fake_prob*100:.1f}% fake probability)."
            ),
            score=round(avg_fake_prob, 4)
        )

    return {
        "probabilities": {"Fake": round(avg_fake_prob, 4), "Real": round(1.0 - avg_fake_prob, 4)},
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "fake_prob": round(avg_fake_prob, 4),
        "num_frames_analyzed": len(frames),
        "per_frame_scores": per_frame_results,
        "fake_frame_count": fake_frame_count,
    }


def _predict_image_no_alert(image: Image.Image) -> dict:
    """
    Internal: Run prediction on a single image WITHOUT logging an alert.
    Used by predict_video to avoid logging individual frame alerts —
    only the aggregated video result gets logged.
    """
    if _model is None or _processor is None:
        raise RuntimeError("Deepfake model not loaded. Call load_model() first.")

    inputs = _processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
    labels = _model.config.id2label
    prob_dict = {labels[i]: round(probs[i], 4) for i in range(len(probs))}

    fake_prob = 0.0
    for label, prob in prob_dict.items():
        if "fake" in label.lower() or "deepfake" in label.lower():
            fake_prob = prob
            break

    if fake_prob > 0.6:
        verdict = "⚠️ LIKELY DEEPFAKE"
    elif fake_prob > 0.4:
        verdict = "🟡 INCONCLUSIVE"
    else:
        verdict = "✅ LIKELY AUTHENTIC"

    confidence = max(fake_prob, 1.0 - fake_prob) * 100

    return {
        "probabilities": prob_dict,
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "fake_prob": round(fake_prob, 4),
    }
