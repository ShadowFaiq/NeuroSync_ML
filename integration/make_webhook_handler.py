"""
Make.com Webhook Handler
Processes health data from Google Forms via Make.com
Calculates composite burnout risk by synthesizing outputs from ALL ML models:
- Sentiment Journaling (VADER sentiment scores)
- Nutrition Tracker (food health labels & scores)
- Sleep-Productivity Correlator (sleep/productivity relationship)
- Fitness Trends (exercise frequency)
Also exposes an ingest endpoint for Make.com to write health logs into Firestore
using a service account (backend-first, no Make Firestore module).
"""

import logging
import os
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from composite_burnout_assessment import CompositeBurnoutAssessment, calculate_burnout_risk
import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)
webhook_bp = Blueprint('webhooks', __name__, url_prefix='/api')


_db = None


def get_db():
    """Initialize and cache Firestore client using service account credentials."""
    global _db
    if _db is not None:
        return _db

    cred_path = os.getenv("FIREBASE_CREDENTIALS", "firebase_credentials.json")
    try:
        firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase app initialized from %s", cred_path)
    _db = firestore.client()
    return _db


@webhook_bp.route('/health', methods=['GET'])
def health_check():
    """Lightweight readiness probe for Make/Render."""
    try:
        get_db()
        return jsonify({'status': 'ok'}), 200
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return jsonify({'status': 'error', 'message': str(exc)}), 500


def _coerce_timestamp(ts_value):
    """Return ISO timestamp string; defaults to now when parsing fails."""
    if ts_value:
        try:
            # Accept ISO strings or epoch seconds
            if isinstance(ts_value, (int, float)):
                return datetime.fromtimestamp(ts_value).isoformat()
            return datetime.fromisoformat(str(ts_value)).isoformat()
        except Exception:
            logger.warning("Could not parse timestamp %s; using now()", ts_value)
    return datetime.now().isoformat()


@webhook_bp.route('/process-health-data', methods=['POST'])
def process_health_data():
    """
    Process health data already stored in Firestore for a user.

    Expected payload from Make.com:
    {
        "user_id": "user_123"
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        user_id = data.get('user_id')

        if not user_id:
            logger.warning("No user_id provided in webhook request")
            return jsonify({'error': 'user_id required'}), 400

        response, status = run_health_analysis(user_id)
        return jsonify(response), status

    except Exception as e:
        logger.error(f"❌ Unhandled error in process_health_data: {str(e)}", exc_info=True)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@webhook_bp.route('/ingest-health-log', methods=['POST'])
def ingest_health_log():
    """
    Ingest health metrics from Make.com, persist to Firestore via service account,
    then return analysis results.

    Expected payload (numbers as strings are accepted):
    {
        "user_id": "user_123",
        "timestamp": "2026-01-19T20:00:00Z",
        "sleep_hours_per_night": 7.2,
        "sleep_quality_rating": 4,
        "productivity_score": 6,
        "stress_level": 5,
        "workload_rating": 6,
        "exercise_days_per_week": 3,
        "social_isolation": 2,
        "work_mental_health_interference": "Sometimes",
        "work_hours_per_week": 42
    }
    """
    payload = request.get_json(silent=True) or {}
    user_id = payload.get('user_id')

    if not user_id:
        return jsonify({'error': 'user_id required'}), 400

    timestamp = _coerce_timestamp(payload.get('timestamp'))
    db = get_db()

    try:
        save_health_log(user_id, payload, timestamp)
    except Exception as e:
        logger.error(f"Error saving health log: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to save health log'}), 500

    # Write to remaining 3 collections (burnout, sleep-productivity, fitness)
    # NOTE: Burnout is now a COMPOSITE assessment synthesizing ALL model outputs
    try:
        # 1) Composite Burnout Assessment (synthesizes sentiment, nutrition, sleep, fitness)
        burnout_result = calculate_composite_burnout_for_user(user_id, db)
        
        db.collection("burnout_assessment").document(user_id).set({
            "burnout_risk_score": burnout_result['burnout_risk_score'],
            "risk_level": burnout_result['risk_level'],
            "risk_interpretation": burnout_result['risk_interpretation'],
            "confidence_pct": burnout_result['confidence_pct'],
            "components": burnout_result['components'],
            "recommendations": burnout_result['recommendations'],
            "last_updated": timestamp,
            "user_id": user_id
        })
        logger.info(f"Composite burnout assessment saved for {user_id}: "
                   f"score={burnout_result['burnout_risk_score']}, "
                   f"level={burnout_result['risk_level']}")

        # 2) Sleep-productivity correlation
        sleep_hours = payload.get("sleep_hours_per_night")
        productivity = payload.get("productivity_score")
        if sleep_hours is not None and productivity:
            try:
                sleep_h = float(sleep_hours)
                prod_s = float(productivity)
                correlation = sleep_h / max(prod_s, 1)
            except (ValueError, TypeError):
                correlation = 0
        else:
            correlation = 0
        db.collection("sleep_productivity_correlation").document(user_id).set({
            "sleep_hours_per_night": sleep_hours,
            "productivity_score": productivity,
            "correlation": correlation,
            "last_updated": timestamp,
            "user_id": user_id
        })
        logger.info(f"Sleep-productivity correlation saved for {user_id}: correlation={correlation}")

        # 3) Fitness progress
        exercise_days = payload.get("exercise_days_per_week", 0)
        db.collection("fitness_progress").document(user_id).set({
            "exercise_days_per_week": exercise_days,
            "last_updated": timestamp,
            "user_id": user_id
        })
        logger.info(f"Fitness progress saved for {user_id}: exercise_days={exercise_days}")

    except Exception as e:
        logger.error(f"Error writing to derived collections: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to write derived data'}), 500

    logger.info("Health log ingested for user %s", user_id)
    response, status = run_health_analysis(user_id)
    return jsonify(response), status


def calculate_sleep_productivity_correlation(health_data):
    """
    Calculate correlation between sleep duration and productivity score
    
    Args:
        health_data: DataFrame with columns [sleep_hours_per_night, productivity_score, ...]
    
    Returns:
        dict: {correlation, avg_sleep, avg_productivity, trend, last_updated}
    """
    try:
        # Ensure numeric data
        sleep_data = pd.to_numeric(health_data['sleep_hours_per_night'], errors='coerce')
        productivity_data = pd.to_numeric(health_data['productivity_score'], errors='coerce')
        
        # Remove NaN values
        valid_mask = ~(sleep_data.isna() | productivity_data.isna())
        sleep_clean = sleep_data[valid_mask]
        productivity_clean = productivity_data[valid_mask]
        
        if len(sleep_clean) < 2:
            logger.warning("Insufficient data for correlation calculation")
            return {
                'correlation': 0.0,
                'avg_sleep': float(sleep_clean.mean()) if len(sleep_clean) > 0 else 0,
                'avg_productivity': float(productivity_clean.mean()) if len(productivity_clean) > 0 else 0,
                'trend': 'neutral',
                'last_updated': datetime.now().isoformat()
            }
        
        correlation = float(sleep_clean.corr(productivity_clean))
        
        # Determine trend
        if correlation > 0.3:
            trend = 'positive'
        elif correlation < -0.3:
            trend = 'negative'
        else:
            trend = 'neutral'
        
        return {
            'correlation': correlation,
            'avg_sleep': float(sleep_clean.mean()),
            'avg_productivity': float(productivity_clean.mean()),
            'trend': trend,
            'last_updated': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error calculating sleep-productivity correlation: {str(e)}")
        return {
            'correlation': 0.0,
            'avg_sleep': 0,
            'avg_productivity': 0,
            'trend': 'error',
            'last_updated': datetime.now().isoformat()
        }


def run_health_analysis(user_id, days=30):
    """Shared analysis pipeline: fetch data, compute correlations, burnout, and persist."""
    try:
        health_data = get_health_data(user_id, days=days)
    except Exception as e:
        logger.error(f"Failed to fetch health data: {str(e)}")
        return {
            'status': 'error',
            'message': 'Failed to fetch health data'
        }, 500

    if health_data is None or health_data.empty:
        logger.warning(f"No health data found for user {user_id}")
        return {
            'status': 'warning',
            'message': 'No health data available for analysis',
            'user_id': user_id
        }, 202

    try:
        sleep_productivity_corr = calculate_sleep_productivity_correlation(health_data)
        fitness_progress = calculate_fitness_progress(health_data)
    except Exception as e:
        logger.error(f"Error calculating correlations: {str(e)}")
        return {'error': f'Correlation calculation failed: {str(e)}'}, 500

    try:
        update_correlation_data(user_id, sleep_productivity_corr)
        update_fitness_progress(user_id, fitness_progress)
    except Exception as e:
        logger.error(f"Error updating Firebase correlation data: {str(e)}")

    try:
        burnout_predictor = BurnoutPredictor()
        features = aggregate_health_features(health_data)
        burnout_result = burnout_predictor.predict_burnout(features)
    except Exception as e:
        logger.error(f"Error predicting burnout: {str(e)}")
        return {'error': f'Burnout prediction failed: {str(e)}'}, 500

    try:
        save_burnout_results(user_id, burnout_result)
    except Exception as e:
        logger.error(f"Error saving burnout results to Firebase: {str(e)}")

    response = {
        'status': 'success',
        'user_id': user_id,
        'timestamp': datetime.now().isoformat(),
        'results': {
            'burnout_risk': burnout_result['burnout_risk'],
            'burnout_score': burnout_result['burnout_score'],
            'confidence': float(burnout_result['confidence']),
            'interpretation': burnout_result['interpretation'],
            'sleep_productivity_correlation': float(sleep_productivity_corr['correlation']),
            'fitness_trend': fitness_progress['trend'],
            'avg_sleep': float(sleep_productivity_corr['avg_sleep']),
            'avg_productivity': float(sleep_productivity_corr['avg_productivity']),
            'avg_exercise_days': float(fitness_progress['avg_exercise_days'])
        }
    }

    logger.info("✅ Successfully processed health data for user %s", user_id)
    return response, 200


def calculate_fitness_progress(health_data):
    """
    Calculate fitness progress and trends
    
    Args:
        health_data: DataFrame with column [exercise_days_per_week, ...]
    
    Returns:
        dict: {avg_exercise_days, weekly_average, monthly_average, trend, last_updated}
    """
    try:
        exercise_data = pd.to_numeric(health_data['exercise_days_per_week'], errors='coerce')
        exercise_data = exercise_data.dropna()
        
        if len(exercise_data) == 0:
            return {
                'avg_exercise_days': 0,
                'weekly_average': 0,
                'monthly_average': 0,
                'trend': 'no_data',
                'last_updated': datetime.now().isoformat()
            }
        
        avg_exercise = exercise_data.mean()
        
        # Determine trend (compare last week to overall average)
        if len(exercise_data) >= 7:
            last_week = exercise_data.iloc[-7:].mean()
            if last_week > avg_exercise * 1.1:
                trend = 'improving'
            elif last_week < avg_exercise * 0.9:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'avg_exercise_days': float(avg_exercise),
            'weekly_average': float(exercise_data.iloc[-7:].mean()) if len(exercise_data) >= 7 else float(avg_exercise),
            'monthly_average': float(avg_exercise),
            'trend': trend,
            'last_updated': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error calculating fitness progress: {str(e)}")
        return {
            'avg_exercise_days': 0,
            'weekly_average': 0,
            'monthly_average': 0,
            'trend': 'error',
            'last_updated': datetime.now().isoformat()
        }


def aggregate_health_features(health_data):
    """
    Aggregate health data into features for burnout prediction
    
    Args:
        health_data: DataFrame with all health metrics
    
    Returns:
        dict: Features for burnout predictor
    """
    try:
        features = {
            'sleep_duration': float(pd.to_numeric(health_data['sleep_hours_per_night'], errors='coerce').mean()),
            'sleep_consistency': calculate_sleep_consistency(health_data),
            'sleep_quality': float(pd.to_numeric(health_data['sleep_quality_rating'], errors='coerce').mean()),
            'stress_level': float(pd.to_numeric(health_data['stress_level'], errors='coerce').mean()),
            'workload_rating': float(pd.to_numeric(health_data['workload_rating'], errors='coerce').mean()),
            'productivity_score': float(pd.to_numeric(health_data['productivity_score'], errors='coerce').mean()),
            'fitness_activity_level': float(pd.to_numeric(health_data['exercise_days_per_week'], errors='coerce').mean() / 7),
            'workout_frequency': float(pd.to_numeric(health_data['exercise_days_per_week'], errors='coerce').mean()),
            'social_isolation': float(pd.to_numeric(health_data['social_isolation'], errors='coerce').mean()),
            'work_interference': calculate_work_interference_score(health_data),
            'work_hours': float(pd.to_numeric(health_data['work_hours_per_week'], errors='coerce').iloc[0]) if len(health_data) > 0 else 40.0
        }
        
        logger.info(f"Aggregated features: {features}")
        return features
    
    except Exception as e:
        logger.error(f"Error aggregating health features: {str(e)}")
        # Return default features
        return {
            'sleep_duration': 7.0,
            'sleep_consistency': 0.7,
            'sleep_quality': 3,
            'stress_level': 5,
            'workload_rating': 5,
            'productivity_score': 3,
            'fitness_activity_level': 0.4,
            'workout_frequency': 2.5,
            'social_isolation': 2.5,
            'work_interference': 2.5,
            'work_hours': 40
        }


def calculate_sleep_consistency(health_data):
    """
    Calculate sleep consistency (lower variation = higher consistency)
    Returns 0-1 scale where 1 = perfectly consistent
    
    Args:
        health_data: DataFrame with sleep_hours_per_night column
    
    Returns:
        float: Consistency score 0-1
    """
    try:
        sleep_data = pd.to_numeric(health_data['sleep_hours_per_night'], errors='coerce').dropna()
        
        if len(sleep_data) < 2:
            return 0.5  # Neutral
        
        sleep_std = sleep_data.std()
        # Convert to 0-1 scale (assuming 3 hours is max acceptable variation)
        consistency = max(0, 1 - (sleep_std / 3))
        
        return float(consistency)
    
    except Exception as e:
        logger.error(f"Error calculating sleep consistency: {str(e)}")
        return 0.5


def calculate_work_interference_score(health_data):
    """
    Convert work-mental health interference to numeric score (0-5)
    
    Args:
        health_data: DataFrame with work_mental_health_interference column
    
    Returns:
        float: Average interference score
    """
    try:
        interference_map = {
            'Never': 1,
            'Rarely': 2,
            'Sometimes': 3,
            'Often': 4,
            'Always': 5
        }
        
        interference_col = health_data['work_mental_health_interference']
        interference_scores = [
            interference_map.get(str(val).strip(), 3)  # Default to "Sometimes" if unclear
            for val in interference_col
        ]
        
        if not interference_scores:
            return 3.0  # Default to neutral
        
        avg_score = float(sum(interference_scores) / len(interference_scores))
        return avg_score
    
    except Exception as e:
        logger.error(f"Error calculating work interference score: {str(e)}")
        return 3.0


def save_health_log(user_id, payload, timestamp):
    """Persist the raw Make.com payload into Firestore health_logs."""
    db = get_db()
    numeric_keys = [
        'sleep_hours_per_night', 'sleep_quality_rating', 'productivity_score',
        'stress_level', 'workload_rating', 'exercise_days_per_week',
        'social_isolation', 'work_hours_per_week'
    ]

    normalized = {}
    for key, value in payload.items():
        if key in numeric_keys:
            try:
                normalized[key] = float(value)
            except Exception:
                normalized[key] = value
        else:
            normalized[key] = value

    normalized['user_id'] = user_id
    normalized['timestamp'] = timestamp

    # Ensure work_mental_health_interference stored as string for mapping
    if 'work_mental_health_interference' in payload:
        normalized['work_mental_health_interference'] = str(payload['work_mental_health_interference'])

    db.collection('health_logs').add(normalized)


def save_burnout_results(user_id, burnout_result):
    """Save burnout assessment results to Firestore."""
    db = get_db()
    doc_ref = db.collection('burnout_assessment').document(user_id)
    doc_ref.set({
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'burnout_risk': burnout_result['burnout_risk'],
        'burnout_score': burnout_result['burnout_score'],
        'confidence': float(burnout_result['confidence']),
        'interpretation': burnout_result['interpretation'],
        'recommendations': burnout_result['recommendations'],
        'factors_analysis': burnout_result.get('factors_analysis', {})
    })
    logger.info(f"✅ Saved burnout results for user {user_id}")


def update_correlation_data(user_id, correlation_data):
    """Update sleep-productivity correlation in Firestore."""
    db = get_db()
    doc_ref = db.collection('sleep_productivity_correlation').document(user_id)
    doc_ref.set(correlation_data, merge=True)
    logger.info(f"✅ Updated correlation data for user {user_id}")


def update_fitness_progress(user_id, fitness_data):
    """Update fitness progress in Firestore."""
    db = get_db()
    doc_ref = db.collection('fitness_progress').document(user_id)
    doc_ref.set(fitness_data, merge=True)
    logger.info(f"✅ Updated fitness progress for user {user_id}")


# Health data fetching function (add to firebase_manager.py)
def get_health_data(user_id, days=7):
    """
    Fetch health data for user from last N days
    
    Args:
        user_id: User identifier
        days: Number of days to retrieve
    
    Returns:
        pd.DataFrame: Health records or None if no data
    """
    try:
        db = get_db()
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = db.collection('health_logs')\
            .where('user_id', '==', user_id)\
            .where('timestamp', '>=', cutoff_date)
        
        docs = query.stream()
        
        data = []
        for doc in docs:
            doc_dict = doc.to_dict()
            data.append(doc_dict)
        
        if not data:
            logger.warning(f"No health data found for user {user_id}")
            return None
        
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"Retrieved {len(df)} health records for user {user_id}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching health data: {str(e)}")
        return None



def calculate_composite_burnout_for_user(user_id: str, db) -> dict:
    """
    Calculate COMPOSITE burnout assessment by synthesizing outputs from ALL ML models:
    
    1. Sentiment Journaling → sentiment_label, sentiment_compound
    2. Nutrition Tracker → health_label, health_score
    3. Sleep-Productivity Correlator → sleep_hours, productivity_score, correlation
    4. Fitness Trends → exercise_days_per_week
    5. Form Data → stress_level, workload_rating, social_isolation
    
    Args:
        user_id: User identifier
        db: Firestore client
    
    Returns:
        dict: Composite burnout result with score, level, and recommendations
    """
    
    try:
        logger.info(f"Calculating composite burnout for {user_id}")
        
        # 1) Fetch sentiment logs (last 7 days)
        sentiment_logs = []
        try:
            sentiment_docs = db.collection('sentiment_journal')\
                .where('user_id', '==', user_id)\
                .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                .limit(30).stream()
            
            for doc in sentiment_docs:
                sentiment_logs.append(doc.to_dict())
            logger.debug(f"Retrieved {len(sentiment_logs)} sentiment logs for {user_id}")
        except Exception as e:
            logger.debug(f"Could not fetch sentiment logs: {e}")
        
        # 2) Fetch nutrition logs (last 7 days)
        nutrition_logs = []
        try:
            nutrition_docs = db.collection('nutrition_logs')\
                .where('user_id', '==', user_id)\
                .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                .limit(50).stream()
            
            for doc in nutrition_docs:
                nutrition_logs.append(doc.to_dict())
            logger.debug(f"Retrieved {len(nutrition_logs)} nutrition logs for {user_id}")
        except Exception as e:
            logger.debug(f"Could not fetch nutrition logs: {e}")
        
        # 3) Fetch latest health log for sleep/productivity/stress/exercise
        sleep_data = {}
        stress_level = None
        workload_rating = None
        social_isolation = None
        exercise_days = None
        
        try:
            health_docs = db.collection('health_logs')\
                .where('user_id', '==', user_id)\
                .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                .limit(1).stream()
            
            for doc in health_docs:
                latest_health = doc.to_dict()
                sleep_data['sleep_hours_per_night'] = latest_health.get('sleep_hours_per_night')
                sleep_data['productivity_score'] = latest_health.get('productivity_score')
                stress_level = latest_health.get('stress_level')
                workload_rating = latest_health.get('workload_rating')
                social_isolation = latest_health.get('social_isolation')
                exercise_days = latest_health.get('exercise_days_per_week')
                logger.debug(f"Retrieved latest health log for {user_id}")
        except Exception as e:
            logger.debug(f"Could not fetch health logs: {e}")
        
        # 4) Calculate composite burnout using all model outputs
        assessor = CompositeBurnoutAssessment()
        result = assessor.calculate_from_user_data(
            user_id=user_id,
            sentiment_logs=sentiment_logs if sentiment_logs else None,
            nutrition_logs=nutrition_logs if nutrition_logs else None,
            sleep_data=sleep_data if sleep_data.get('sleep_hours_per_night') else None,
            exercise_days_per_week=exercise_days,
            stress_level=stress_level,
            workload_rating=workload_rating,
            social_isolation=social_isolation,
            days_lookback=7
        )
        
        logger.info(f"Composite burnout calculated for {user_id}: "
                   f"score={result['burnout_risk_score']}, level={result['risk_level']}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error calculating composite burnout: {e}", exc_info=True)
        
        # Return safe default if calculation fails
        return {
            'user_id': user_id,
            'burnout_risk_score': 50.0,
            'risk_level': 'MODERATE',
            'risk_interpretation': 'Unable to calculate (insufficient data)',
            'confidence_pct': 0,
            'components': {
                'sentiment_health': 0,
                'nutrition_health': 0,
                'sleep_fitness_balance': 0,
                'stress_workload': 0,
                'social_connection': 0,
            },
            'timestamp': datetime.now().isoformat(),
            'recommendations': ['Please submit more data for accurate assessment.']
        }
