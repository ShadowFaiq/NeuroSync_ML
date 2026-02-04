"""
Flask backend for Make.com + Firestore ingestion.
- Existing: webhook blueprint.
- Added: /api/ingest-health-log endpoint storing raw and derived docs.
Run locally: python make_backend.py  (set FIREBASE_CREDENTIALS if not firebase_credentials.json)
"""
import os
from datetime import datetime
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore

from integration.make_webhook_handler import webhook_bp


def init_firebase():
    """Initialize Firebase app once using local JSON or FIREBASE_CREDENTIALS env."""
    if firebase_admin._apps:
        return firestore.client()

    cred_path = os.getenv("FIREBASE_CREDENTIALS", "firebase_credentials.json")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    return firestore.client()


app = Flask(__name__)
app.register_blueprint(webhook_bp)
db = init_firebase()


@app.route("/api/ingest-health-log", methods=["POST"])
def ingest_health_log():
    data = request.json or {}
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    now = datetime.utcnow()

    # 1) Raw submission
    db.collection("health_logs").document(f"{user_id}_{now.strftime('%Y%m%d%H%M%S')}").set({
        **data,
        "timestamp": now,
    })

    # 2) Burnout assessment (simple example)
    stress = data.get("stress_level", 0)
    workload = data.get("workload_rating", 0)
    burnout_score = ((stress + workload) / 2) * 10
    db.collection("burnout_assessment").document(user_id).set(
        {
            "burnout_score": burnout_score,
            "last_updated": now,
        }
    )

    # 3) Sleep-productivity correlation (guard against div by zero)
    sleep_hours = data.get("sleep_hours_per_night")
    productivity = data.get("productivity_score")
    if sleep_hours is not None and productivity:
        correlation = sleep_hours / max(productivity, 1)
    else:
        correlation = 0
    db.collection("sleep_productivity_correlation").document(user_id).set(
        {
            "correlation": correlation,
            "last_updated": now,
        }
    )

    # 4) Fitness progress (simple example)
    db.collection("fitness_progress").document(user_id).set(
        {
            "exercise_days_per_week": data.get("exercise_days_per_week", 0),
            "last_updated": now,
        }
    )

    return jsonify({"message": "Data stored successfully", "status": "ok", "user_id": user_id})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
