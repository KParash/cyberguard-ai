"""
models/fraud_engine.py — UPI Transaction Risk Engine using RandomForestClassifier.

DESIGN DECISIONS:
- Uses RandomForestClassifier (supervised) rather than IsolationForest (unsupervised)
  because we have labeled training data and RFC provides:
  1. More reliable demo results (deterministic, high accuracy on synthetic data)
  2. Feature importance scores for explainable risk explanations
  3. Probability outputs for granular risk scoring (0-100)

- Trained on synthetic UPI transaction data generated for this hackathon.
  This is NOT real bank data — clearly labeled as synthetic/demo data.

- Model is trained at app startup (~200 rows trains in <1 second)
  and cached in memory for the app's lifetime.

FEATURES USED (all numeric, no encoding needed):
- amount: Transaction amount in INR
- time_of_day: Hour of day (0-23)
- sender_account_age_days: How old the sender's account is
- is_new_payee: Binary — first-time recipient
- transaction_velocity_1hr: Number of transactions in the last hour
- device_change_flag: Binary — sender using a new/different device
- location_mismatch_flag: Binary — sender's location doesn't match usual pattern
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from utils.alert_engine import log_alert

# Feature columns used for training and prediction
FEATURE_COLS = [
    "amount", "time_of_day", "sender_account_age_days",
    "is_new_payee", "transaction_velocity_1hr",
    "device_change_flag", "location_mismatch_flag"
]

# Human-readable feature names for explanations
FEATURE_NAMES = {
    "amount": "High transaction amount",
    "time_of_day": "Unusual time of day (late night)",
    "sender_account_age_days": "New/young account",
    "is_new_payee": "New payee (first-time recipient)",
    "transaction_velocity_1hr": "High transaction velocity",
    "device_change_flag": "Device change detected",
    "location_mismatch_flag": "Location mismatch"
}

# Global model and scaler — trained once at startup
_model = None
_scaler = None
_feature_importances = None


def load_model():
    """
    Train the RandomForestClassifier on sample_transactions.csv.
    Called once at app startup. Trains in <1 second on ~200 rows.
    """
    global _model, _scaler, _feature_importances

    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "sample_transactions.csv"
    )

    print("[FRAUD] 🔄 Training RandomForestClassifier on synthetic UPI transaction data...")

    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].values
    y = df["label"].values

    # Scale features for consistent scoring
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)

    # Train RandomForestClassifier
    # n_estimators=100 is standard, random_state=42 for reproducibility
    _model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"  # Handle class imbalance (~15% fraud)
    )
    _model.fit(X_scaled, y)

    # Store feature importances for explanations
    _feature_importances = dict(zip(FEATURE_COLS, _model.feature_importances_))

    # Print training summary
    train_accuracy = _model.score(X_scaled, y)
    fraud_count = y.sum()
    print(f"[FRAUD] ✅ Model trained. Accuracy: {train_accuracy:.2%}, "
          f"Samples: {len(y)}, Fraud: {fraud_count} ({fraud_count/len(y):.1%})")
    print(f"[FRAUD]    Top features: {sorted(_feature_importances.items(), key=lambda x: -x[1])[:3]}")


def predict_transaction(
    amount: float,
    time_of_day: int,
    sender_account_age_days: int,
    is_new_payee: int,
    transaction_velocity_1hr: int,
    device_change_flag: int,
    location_mismatch_flag: int
) -> dict:
    """
    Analyze a single UPI transaction for fraud risk.

    Returns:
        dict with keys: risk_score (0-100), risk_tier (Low/Medium/High),
                       explanation (str), feature_contributions (dict)
    """
    if _model is None:
        raise RuntimeError("Fraud model not loaded. Call load_model() first.")

    # Build feature vector in the correct order
    features = np.array([[
        amount, time_of_day, sender_account_age_days,
        is_new_payee, transaction_velocity_1hr,
        device_change_flag, location_mismatch_flag
    ]])

    # Scale features using the fitted scaler
    features_scaled = _scaler.transform(features)

    # Get fraud probability (class 1 = fraud)
    fraud_prob = _model.predict_proba(features_scaled)[0][1]
    risk_score = round(fraud_prob * 100, 1)

    # Determine risk tier
    if risk_score > 70:
        risk_tier = "High"
    elif risk_score > 40:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    # Generate feature-importance-driven explanation
    explanation = _generate_explanation(features[0], risk_tier)

    # Build feature contribution details
    feature_contributions = {}
    for i, col in enumerate(FEATURE_COLS):
        importance = _feature_importances[col]
        value = features[0][i]
        feature_contributions[FEATURE_NAMES[col]] = {
            "value": value,
            "importance": round(importance * 100, 1)
        }

    # Log alert if high risk
    if risk_score > 70:
        log_alert(
            source_module="fraud",
            severity="High" if risk_score > 85 else "Medium",
            details=f"High-risk UPI transaction: ₹{amount:,.0f}, {explanation}",
            score=risk_score
        )

    return {
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "explanation": explanation,
        "feature_contributions": feature_contributions,
        "fraud_probability": round(fraud_prob, 4),
    }


def _generate_explanation(features: np.ndarray, risk_tier: str) -> str:
    """
    Generate a human-readable explanation based on which features are flagged.
    Uses actual feature values + importance to identify the key risk drivers.
    """
    flags = []

    amount, time_of_day, account_age, is_new, velocity, device_change, location_mismatch = features

    # Check each feature for suspicious values
    if is_new > 0.5:
        flags.append("new payee")
    if velocity > 5:
        flags.append(f"high transaction velocity ({int(velocity)} txns/hr)")
    if location_mismatch > 0.5:
        flags.append("location mismatch")
    if device_change > 0.5:
        flags.append("device change detected")
    if amount > 10000:
        flags.append(f"high amount (₹{amount:,.0f})")
    if time_of_day >= 23 or time_of_day <= 4:
        flags.append(f"late-night timing ({int(time_of_day)}:00)")
    if account_age < 30:
        flags.append(f"new account ({int(account_age)} days old)")

    if flags:
        return f"Flagged due to: {' + '.join(flags)}"
    elif risk_tier == "Low":
        return "No significant risk factors detected. Transaction appears normal."
    else:
        return "Moderate risk based on combined feature analysis."


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run fraud risk scoring on a batch of transactions from a CSV DataFrame.

    Expects DataFrame with columns matching FEATURE_COLS.
    Returns the DataFrame with added risk_score, risk_tier, and explanation columns,
    sorted by risk_score descending (highest risk first).
    """
    if _model is None:
        raise RuntimeError("Fraud model not loaded. Call load_model() first.")

    results = []
    for _, row in df.iterrows():
        try:
            result = predict_transaction(
                amount=float(row.get("amount", 0)),
                time_of_day=int(row.get("time_of_day", 12)),
                sender_account_age_days=int(row.get("sender_account_age_days", 365)),
                is_new_payee=int(row.get("is_new_payee", 0)),
                transaction_velocity_1hr=int(row.get("transaction_velocity_1hr", 1)),
                device_change_flag=int(row.get("device_change_flag", 0)),
                location_mismatch_flag=int(row.get("location_mismatch_flag", 0)),
            )
            results.append({
                "transaction_id": row.get("transaction_id", "N/A"),
                "amount": row.get("amount", 0),
                "risk_score": result["risk_score"],
                "risk_tier": result["risk_tier"],
                "explanation": result["explanation"],
            })
        except Exception as e:
            results.append({
                "transaction_id": row.get("transaction_id", "N/A"),
                "amount": row.get("amount", 0),
                "risk_score": -1,
                "risk_tier": "Error",
                "explanation": str(e),
            })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return result_df
