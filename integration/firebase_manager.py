"""
Firebase Firestore Manager for Health Data Integration
Handles fetching and caching health logs from Firestore database.
"""

import streamlit as st
import pandas as pd
import logging
from typing import Optional

# Lazy import Firebase (saves ~3-5 seconds on startup)
firebase_admin = None
credentials = None
firestore = None

logger = logging.getLogger(__name__)


@st.cache_resource
def initialize_firebase():
    """
    Initialize Firebase Admin SDK.
    Requires credentials JSON file in project root or environment variable.
    """
    global firebase_admin, credentials, firestore
    
    # Lazy import Firebase only when needed
    if firebase_admin is None:
        logger.info("First-time Firebase import (this takes ~3-5 seconds)...")
        import firebase_admin as fa
        from firebase_admin import credentials as creds, firestore as fs
        firebase_admin = fa
        credentials = creds
        firestore = fs
        logger.info("Firebase loaded successfully")
    
    try:
        # Check if Firebase app is already initialized
        firebase_admin.get_app()
    except ValueError:
        # Firebase app not initialized, attempt to initialize
        try:
            # Try to get credentials from environment or file
            cred = credentials.Certificate("firebase_credentials.json")
            firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized successfully")
        except FileNotFoundError:
            logger.warning("firebase_credentials.json not found. Firebase features will be unavailable.")
            return None
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            return None
    
    return firestore.client()


def get_health_data() -> Optional[pd.DataFrame]:
    """
    Fetch health data from Firestore 'health_logs' collection.
    
    Returns:
        DataFrame with standardized columns: timestamp, sleep_hours, productivity_score, 
        stress_level, exercise_days_per_week, etc.
        Returns None if Firebase is not initialized or data is unavailable.
    """
    db = initialize_firebase()
    
    if db is None:
        logger.warning("Firebase not initialized. Using placeholder data.")
        return None
    
    try:
        from firebase_admin import firestore as fs
        
        logs_ref = db.collection("health_logs")
        docs = logs_ref.order_by("timestamp", direction=fs.Query.DESCENDING).limit(30).stream()
        
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data["id"] = doc.id
            data.append(doc_data)
        
        if not data:
            logger.warning("No documents found in health_logs collection")
            return None
        
        df = pd.DataFrame(data)
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Map Firebase field names to standardized names for backward compatibility
        field_mappings = {
            'sleep_hours_per_night': 'sleep_hours',
            'exercise_days_per_week': 'exercise_frequency',
            'sleep_quality_rating': 'sleep_quality'
        }
        df = df.rename(columns=field_mappings)

        # Coerce numeric fields to numbers for charts
        for col in ["sleep_hours", "productivity_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Log the columns to help with debugging
        logger.info(f"Successfully fetched {len(df)} health log entries from Firestore")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        logger.info(f"First row data: {df.iloc[0].to_dict() if len(df) > 0 else 'No data'}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching health data from Firestore: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def save_health_log(log_data: dict) -> bool:
    """
    Save a single health log entry to Firestore.
    
    Args:
        log_data: Dictionary containing health metrics
        
    Returns:
        True if successful, False otherwise
    """
    db = initialize_firebase()
    
    if db is None:
        logger.error("Firebase not initialized. Cannot save log.")
        return False
    
    try:
        db.collection("daily_logs").add(log_data)
        logger.info("Health log saved to Firestore")
        return True
    except Exception as e:
        logger.error(f"Error saving health log to Firestore: {e}")
        return False


def save_daily_checkin(checkin_data: dict, user_id: str = "default_user") -> bool:
    """
    Save daily check-in data to Firestore.
    
    Args:
        checkin_data: Dictionary with keys like sleep_hours, productivity_score, etc.
        user_id: User identifier (default: "default_user")
        
    Returns:
        True if successful, False otherwise
    """
    db = initialize_firebase()
    
    if db is None:
        logger.error("Firebase not initialized. Cannot save check-in.")
        return False
    
    try:
        from datetime import datetime
        
        # Add metadata
        checkin_data['user_id'] = user_id
        checkin_data['timestamp'] = datetime.now()
        
        # Save to Firestore
        db.collection("daily_logs").add(checkin_data)
        logger.info("Daily check-in saved to Firestore")
        return True
    except Exception as e:
        logger.error(f"Error saving daily check-in: {e}")
        return False


def save_sentiment_entry(sentiment_data: dict, user_id: str = "default_user") -> bool:
    """
    Save sentiment journal entry to Firestore.
    
    Args:
        sentiment_data: Dictionary with keys: entry_text, sentiment_label, 
                       sentiment_compound, sentiment_scores, entry_type
        user_id: User identifier
        
    Returns:
        True if successful, False otherwise
    """
    db = initialize_firebase()
    
    if db is None:
        logger.error("Firebase not initialized. Cannot save sentiment entry.")
        return False
    
    try:
        from datetime import datetime
        
        # Validate required fields
        required_fields = ['entry_text', 'sentiment_label', 'sentiment_compound', 'sentiment_scores']
        missing = [f for f in required_fields if f not in sentiment_data]
        if missing:
            logger.error(f"Missing required fields: {missing}")
            return False
        
        # Add metadata
        sentiment_data['user_id'] = user_id
        sentiment_data['timestamp'] = datetime.now()
        
        logger.info(f"Saving sentiment entry for user={user_id}, text_length={len(sentiment_data.get('entry_text', ''))}")
        
        # Save to Firestore
        doc_ref = db.collection("sentiment_journal").add(sentiment_data)
        logger.info(f"Sentiment entry saved to Firestore: {doc_ref}")
        return True
    except Exception as e:
        logger.error(f"Error saving sentiment entry: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def save_nutrition_entry(
    user_id: str,
    food_name: str,
    confidence: float,
    nutritional_info: dict,
    health_score: float,
    health_label: str
) -> bool:
    """
    Save nutrition entry to Firestore with detailed nutritional information.
    
    Args:
        user_id: User identifier (str)
        food_name: Predicted food class name (str)
        confidence: Model's confidence score (float, 0-1)
        nutritional_info: Dict with keys 'calories', 'protein', 'carbs', 'fats'
        health_score: Calculated health score (float, 0-10)
        health_label: Health classification (str) - "Healthy", "Moderate", or "Junk"
        
    Returns:
        bool: True if successful, False if failed
    """
    db = initialize_firebase()
    
    if db is None:
        error_msg = "❌ Firebase not initialized. Check firebase_credentials.json"
        logger.error(error_msg)
        try:
            st.error(error_msg)
        except:
            pass
        return False
    
    try:
        # Get server timestamp
        global firestore
        if firestore is None:
            from firebase_admin import firestore as fs
            firestore = fs
        
        # Create document with all required fields
        entry_data = {
            'user_id': user_id,
            'food_name': food_name,
            'confidence': float(confidence),
            'health_score': float(health_score),
            'health_label': health_label,
            'nutritional_info': {
                'calories': nutritional_info.get('calories', 0),
                'protein': nutritional_info.get('protein', 0),
                'carbs': nutritional_info.get('carbs', 0),
                'fats': nutritional_info.get('fats', 0)
            },
            'timestamp': firestore.SERVER_TIMESTAMP  # Server-side timestamp
        }
        
        logger.info(f"Attempting to save nutrition entry: {food_name} for user {user_id}")
        logger.info(f"Entry data: {entry_data}")
        
        # Save to Firestore 'nutrition_logs' collection
        try:
            doc_ref, doc_id = db.collection("nutrition_logs").add(entry_data)
            logger.info(f"✅ Nutrition entry saved successfully to Firestore with doc_id: {doc_id}")
            logger.info(f"✅ Nutrition entry saved successfully to Firestore: {food_name} (score: {health_score}/10)")
            
            # Show success toast in Streamlit
            try:
                st.success(f"✅ Logged {food_name} to nutrition database!")
                st.info(f"📊 Check 'Nutrition Log' tab to view your entry")
            except:
                pass
            
            return True
        except Exception as save_error:
            logger.error(f"❌ Failed during Firestore write: {str(save_error)}")
            raise save_error
        
    except Exception as e:
        error_msg = f"❌ Error saving nutrition entry: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Show detailed error in Streamlit
        try:
            st.error(error_msg)
            st.error(f"Details: {str(e)}")
        except:
            pass
        
        return False


def save_nutrition_log(nutrition_data: dict, user_id: str = "default_user") -> bool:
    """
    Save nutrition/food log to Firestore (legacy function).
    
    Args:
        nutrition_data: Dictionary with keys: food_class, confidence, 
                       health_score, macro_type, is_healthy
        user_id: User identifier
        
    Returns:
        True if successful, False otherwise
    """
    db = initialize_firebase()
    
    if db is None:
        logger.error("Firebase not initialized. Cannot save nutrition log.")
        return False
    
    try:
        from datetime import datetime
        
        # Add metadata
        nutrition_data['user_id'] = user_id
        nutrition_data['timestamp'] = datetime.now()
        
        # Save to Firestore
        db.collection("nutrition_logs").add(nutrition_data)
        logger.info("Nutrition log saved to Firestore")
        return True
    except Exception as e:
        logger.error(f"Error saving nutrition log: {e}")
        return False


def save_burnout_assessment(burnout_data: dict, user_id: str = "default_user") -> bool:
    """
    Save burnout risk assessment to Firestore.
    
    Args:
        burnout_data: Dictionary with keys: risk_level, risk_score, 
                     features, recommendations
        user_id: User identifier
        
    Returns:
        True if successful, False otherwise
    """
    db = initialize_firebase()
    
    if db is None:
        logger.error("Firebase not initialized. Cannot save burnout assessment.")
        return False
    
    try:
        from datetime import datetime
        
        # Add metadata
        burnout_data['user_id'] = user_id
        burnout_data['timestamp'] = datetime.now()
        
        # Save to Firestore
        db.collection("burnout_assessments").add(burnout_data)
        logger.info("Burnout assessment saved to Firestore")
        return True
    except Exception as e:
        logger.error(f"Error saving burnout assessment: {e}")
        return False


def get_sleep_productivity_correlation(user_id: Optional[str] = None, days: int = 90) -> Optional[pd.DataFrame]:
    """Fetch sleep vs productivity pairs/correlation logs from Firestore.

    Args:
        user_id: optional user filter
        days: lookback window for timestamp filtering

    Returns:
        DataFrame with at least sleep_hours, productivity_score, timestamp, user_id
    """
    db = initialize_firebase()
    if db is None:
        logger.error("Firebase not initialized. Cannot fetch sleep_productivity_correlation.")
        return None

    from firebase_admin import firestore as fs
    from datetime import datetime, timedelta

    cutoff = datetime.now() - timedelta(days=days)

    try:
        query = db.collection("sleep_productivity_correlation").where("timestamp", ">=", cutoff)
        if user_id:
            query = query.where("user_id", "==", user_id)
        docs = query.order_by("timestamp", direction=fs.Query.DESCENDING).limit(200).stream()

        rows = []
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data["id"] = doc.id
            rows.append(doc_data)

        if not rows:
            logger.warning("No documents found in sleep_productivity_correlation collection")
            return None

        df = pd.DataFrame(rows)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Normalize field names if present
        field_mappings = {
            "sleep_hours_per_night": "sleep_hours",
        }
        df = df.rename(columns=field_mappings)

        # Coerce key numeric fields
        for col in ["sleep_hours", "productivity_score", "correlation"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"Fetched {len(df)} sleep_productivity_correlation entries (days={days}, user={user_id or 'any'})")
        return df

    except Exception as e:
        logger.error(f"Error fetching sleep_productivity_correlation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


@st.cache_data(ttl=300)
def get_sentiment_history(user_id: Optional[str] = None, days: int = 7) -> Optional[pd.DataFrame]:
    """
    Fetch sentiment journal entries for a user. Defaults to last 7 days.
    If user_id is None, returns all sentiment entries.
    Tries a date-filtered query first; if Firestore index/rule issues arise,
    falls back to a simpler query so the UI can still render history.
    """
    db = initialize_firebase()
    if db is None:
        logger.error("Firebase not initialized. Cannot fetch sentiment history.")
        return None

    # Ensure firestore reference exists (defensive: initialize_firebase may have returned client while global 'firestore' stayed None)
    global firestore
    if firestore is None:
        try:
            from firebase_admin import firestore as fs
            firestore = fs
        except Exception as e:
            logger.error(f"Firestore module unavailable: {e}")
            return None

    from datetime import datetime, timedelta

    def _to_df(docs):
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data["id"] = doc.id
            data.append(doc_data)
        if not data:
            return None
        df_local = pd.DataFrame(data)
        if "timestamp" in df_local.columns:
            df_local["timestamp"] = pd.to_datetime(df_local["timestamp"])
        return df_local

    cutoff = datetime.now() - timedelta(days=days)

    # Strategy 1: timestamp-only range, then filter user_id client-side (avoids composite index requirement)
    try:
        docs = (
            db.collection("sentiment_journal")
            .where("timestamp", ">=", cutoff)
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(500)
            .stream()
        )
        df = _to_df(docs)
        if df is not None:
            if user_id is not None:
                df = df[df.get("user_id") == user_id]
            if not df.empty:
                df = df.sort_values(by="timestamp", ascending=False)
                logger.info(f"Fetched {len(df)} sentiment entries (timestamp filter, {days}d, user={user_id})")
                return df
    except Exception as e:
        logger.warning(f"Timestamp-filtered sentiment query failed, trying next fallback: {e}")

    # Strategy 2: user filter without order_by (no composite index), then sort client-side
    if user_id is not None:
        try:
            docs = db.collection("sentiment_journal").where("user_id", "==", user_id).stream()
            df = _to_df(docs)
            if df is not None:
                df = df.sort_values(by="timestamp", ascending=False)
                logger.info(f"Fetched {len(df)} sentiment entries (user filter, client-sorted, user={user_id})")
                return df
        except Exception as e:
            logger.error(f"Fallback sentiment query failed: {e}")

    logger.warning(f"No sentiment entries found for user={user_id}")
    return None


@st.cache_data(ttl=300)
def get_nutrition_history(user_id: str = "default_user", days: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch nutrition logs for a user.
    
    Args:
        user_id: User identifier
        days: Number of days of history to fetch
        
    Returns:
        DataFrame with nutrition entries or None
    """
    db = initialize_firebase()
    
    if db is None:
        return None
    
    try:
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        
        docs = (db.collection("nutrition_logs")
                .where("user_id", "==", user_id)
                .where("timestamp", ">=", cutoff)
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .stream())
        
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data["id"] = doc.id
            data.append(doc_data)
        
        if not data:
            logger.warning("No nutrition logs found")
            return None
        
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        logger.info(f"Fetched {len(df)} nutrition logs")
        return df
    except Exception as e:
        logger.error(f"Error fetching nutrition history: {e}")
        return None


@st.cache_data(ttl=300)
def get_burnout_history(user_id: str = "default_user", limit: int = 10) -> Optional[pd.DataFrame]:
    """
    Fetch burnout assessments for a user.
    
    Args:
        user_id: User identifier
        limit: Maximum number of assessments to fetch
        
    Returns:
        DataFrame with burnout assessments or None
    """
    db = initialize_firebase()
    
    if db is None:
        return None
    
    try:
        docs = (db.collection("burnout_assessments")
                .where("user_id", "==", user_id)
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(limit)
                .stream())
        
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data["id"] = doc.id
            data.append(doc_data)
        
        if not data:
            logger.warning("No burnout assessments found")
            return None
        
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        logger.info(f"Fetched {len(df)} burnout assessments")
        return df
    except Exception as e:
        logger.error(f"Error fetching burnout history: {e}")
        return None


@st.cache_data(ttl=60)
def get_nutrition_logs(user_id: Optional[str] = "default_user") -> Optional[pd.DataFrame]:
    """
    Fetch nutrition logs for a user from Firestore.
    If user_id is None, return all nutrition logs (unfiltered).
    
    Returns:
        DataFrame with columns: food_name, health_label, health_score, confidence, timestamp, user_id
    """
    db = initialize_firebase()
    if db is None:
        logger.error("Firebase not initialized. Cannot fetch nutrition logs.")
        return None
    
    try:
        global firestore
        if firestore is None:
            from firebase_admin import firestore as fs
            firestore = fs
        
        docs = (
            db.collection("nutrition_logs")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(500)
            .stream()
        )
        
        data = []
        for doc in docs:
            doc_data = doc.to_dict()
            if user_id is None or doc_data.get("user_id") == user_id:
                doc_data["id"] = doc.id
                data.append(doc_data)
        
        if not data:
            logger.warning(f"No nutrition logs found for user {user_id}")
            return None
        
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=False)
        
        if "health_score" in df.columns:
            df["health_score"] = pd.to_numeric(df["health_score"], errors="coerce")
        
        logger.info(f"Fetched {len(df)} nutrition logs for user {user_id}")
        return df
    except Exception as e:
        logger.error(f"Error fetching nutrition logs: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
