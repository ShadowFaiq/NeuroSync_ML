"""
Firebase Manager - Additional Functions for Make.com Integration
Add these functions to your existing firebase_manager.py
"""

import logging
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


def get_health_data(user_id, days=7):
    """
    Fetch health data for user from last N days from Firestore
    
    Args:
        user_id: User identifier (string)
        days: Number of days to retrieve (default 7)
    
    Returns:
        pd.DataFrame: Health records with all metrics
        None: If no data found or error occurs
    
    Example:
        >>> df = get_health_data('user_123', days=30)
        >>> print(df.columns)
        ['timestamp', 'user_id', 'sleep_hours_per_night', 'stress_level', ...]
    """
    try:
        from integration.firebase_manager import db
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # ⚠️ CRITICAL: Firestore must store timestamps as ISO strings (not Timestamp objects)
        # If Make.com stores Firestore Timestamps, this query will return empty results
        # Ensure Make.com passes timestamp as string: "{{now | formatDate('YYYY-MM-DDTHH:mm:ss')}}"
        cutoff_timestamp = cutoff_date.isoformat()
        
        logger.info(f"Fetching health data for user {user_id} from last {days} days")
        
        # Query Firestore
        query = db.collection('health_logs')\
            .where('user_id', '==', user_id)\
            .where('timestamp', '>=', cutoff_timestamp)\
            .order_by('timestamp', direction='DESCENDING')
        
        docs = query.stream()
        
        # Convert to list
        data_list = []
        for doc in docs:
            doc_dict = doc.to_dict()
            data_list.append(doc_dict)
        
        if not data_list:
            logger.warning(f"No health data found for user {user_id}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data_list)
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp ascending for analysis
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Retrieved {len(df)} health records for user {user_id}")
        logger.debug(f"Columns in dataframe: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching health data for user {user_id}: {str(e)}", exc_info=True)
        return None


def save_burnout_results(user_id, burnout_result):
    """
    Save burnout assessment results to Firestore
    Stores results in burnout_assessment collection
    
    Args:
        user_id: User identifier (string)
        burnout_result: dict with keys:
            - burnout_risk: "Low" / "Medium" / "High"
            - burnout_score: 0-100
            - confidence: 0-1
            - interpretation: str
            - recommendations: list of str
            - factors_analysis: dict (optional)
    
    Example:
        >>> result = {
        ...     'burnout_risk': 'Medium',
        ...     'burnout_score': 62,
        ...     'confidence': 0.85,
        ...     'interpretation': 'Moderate burnout risk...',
        ...     'recommendations': ['Get more sleep', 'Exercise regularly']
        ... }
        >>> save_burnout_results('user_123', result)
    """
    try:
        from integration.firebase_manager import db
        
        logger.info(f"Saving burnout results for user {user_id}")
        
        doc_ref = db.collection('burnout_assessment').document(user_id)
        
        # Prepare data for storage
        data_to_save = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'burnout_risk': burnout_result.get('burnout_risk', 'Unknown'),
            'burnout_score': int(burnout_result.get('burnout_score', 0)),
            'confidence': float(burnout_result.get('confidence', 0)),
            'interpretation': burnout_result.get('interpretation', ''),
            'recommendations': burnout_result.get('recommendations', []),
            'factors_analysis': burnout_result.get('factors_analysis', {})
        }
        
        # Set document (overwrites if exists)
        doc_ref.set(data_to_save)
        
        logger.info(f"✅ Saved burnout results for user {user_id}: {burnout_result['burnout_risk']}")
    
    except Exception as e:
        logger.error(f"Error saving burnout results: {str(e)}", exc_info=True)
        raise


def update_correlation_data(user_id, correlation_data):
    """
    Update sleep-productivity correlation in Firestore
    Stores results in sleep_productivity_correlation collection
    
    Args:
        user_id: User identifier (string)
        correlation_data: dict with keys:
            - correlation: float (-1 to 1)
            - avg_sleep: float (hours)
            - avg_productivity: float (1-5)
            - trend: "positive" / "negative" / "neutral"
            - last_updated: ISO timestamp (auto-set if not provided)
    
    Example:
        >>> corr = {
        ...     'correlation': 0.72,
        ...     'avg_sleep': 7.3,
        ...     'avg_productivity': 3.8,
        ...     'trend': 'positive'
        ... }
        >>> update_correlation_data('user_123', corr)
    """
    try:
        from integration.firebase_manager import db
        
        logger.info(f"Updating sleep-productivity correlation for user {user_id}")
        
        doc_ref = db.collection('sleep_productivity_correlation').document(user_id)
        
        # Add/update timestamp
        data_to_save = correlation_data.copy()
        if 'last_updated' not in data_to_save:
            data_to_save['last_updated'] = datetime.now().isoformat()
        
        # Merge with existing data (don't overwrite)
        doc_ref.set(data_to_save, merge=True)
        
        logger.info(f"✅ Updated correlation data for user {user_id}: correlation={correlation_data.get('correlation', 0):.2f}")
    
    except Exception as e:
        logger.error(f"Error updating correlation data: {str(e)}", exc_info=True)
        raise


def update_fitness_progress(user_id, fitness_data):
    """
    Update fitness progress tracking in Firestore
    Stores results in fitness_progress collection
    
    Args:
        user_id: User identifier (string)
        fitness_data: dict with keys:
            - avg_exercise_days: float (0-7)
            - weekly_average: float
            - monthly_average: float
            - trend: "improving" / "declining" / "stable"
            - last_updated: ISO timestamp (auto-set if not provided)
    
    Example:
        >>> fitness = {
        ...     'avg_exercise_days': 3.2,
        ...     'weekly_average': 3.5,
        ...     'monthly_average': 3.1,
        ...     'trend': 'stable'
        ... }
        >>> update_fitness_progress('user_123', fitness)
    """
    try:
        from integration.firebase_manager import db
        
        logger.info(f"Updating fitness progress for user {user_id}")
        
        doc_ref = db.collection('fitness_progress').document(user_id)
        
        # Add/update timestamp
        data_to_save = fitness_data.copy()
        if 'last_updated' not in data_to_save:
            data_to_save['last_updated'] = datetime.now().isoformat()
        
        # Merge with existing data
        doc_ref.set(data_to_save, merge=True)
        
        logger.info(f"✅ Updated fitness progress for user {user_id}: trend={fitness_data.get('trend', 'unknown')}")
    
    except Exception as e:
        logger.error(f"Error updating fitness progress: {str(e)}", exc_info=True)
        raise


def get_correlation_data(user_id):
    """
    Fetch sleep-productivity correlation for a user
    
    Args:
        user_id: User identifier
    
    Returns:
        dict: Correlation data or None if not found
    """
    try:
        from integration.firebase_manager import db
        
        doc = db.collection('sleep_productivity_correlation').document(user_id).get()
        
        if doc.exists:
            logger.info(f"Retrieved correlation data for user {user_id}")
            return doc.to_dict()
        else:
            logger.warning(f"No correlation data found for user {user_id}")
            return None
    
    except Exception as e:
        logger.error(f"Error fetching correlation data: {str(e)}")
        return None


def get_fitness_progress(user_id):
    """
    Fetch fitness progress for a user
    
    Args:
        user_id: User identifier
    
    Returns:
        dict: Fitness data or None if not found
    """
    try:
        from integration.firebase_manager import db
        
        doc = db.collection('fitness_progress').document(user_id).get()
        
        if doc.exists:
            logger.info(f"Retrieved fitness progress for user {user_id}")
            return doc.to_dict()
        else:
            logger.warning(f"No fitness progress found for user {user_id}")
            return None
    
    except Exception as e:
        logger.error(f"Error fetching fitness progress: {str(e)}")
        return None


def get_burnout_assessment(user_id):
    """
    Fetch latest burnout assessment for a user
    
    Args:
        user_id: User identifier
    
    Returns:
        dict: Burnout assessment or None if not found
    """
    try:
        from integration.firebase_manager import db
        
        doc = db.collection('burnout_assessment').document(user_id).get()
        
        if doc.exists:
            logger.info(f"Retrieved burnout assessment for user {user_id}")
            return doc.to_dict()
        else:
            logger.warning(f"No burnout assessment found for user {user_id}")
            return None
    
    except Exception as e:
        logger.error(f"Error fetching burnout assessment: {str(e)}")
        return None


def get_burnout_history(user_id, days=30):
    """
    Fetch burnout assessment history for a user
    
    Args:
        user_id: User identifier
        days: Number of days to retrieve
    
    Returns:
        list: Burnout assessment history (chronological order)
    """
    try:
        from integration.firebase_manager import db
        
        # This assumes you store historical data in a subcollection
        # Or you can modify to store multiple assessments with timestamps
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = db.collection('burnout_assessment_history')\
            .where('user_id', '==', user_id)\
            .where('timestamp', '>=', cutoff_date)\
            .order_by('timestamp')
        
        docs = query.stream()
        
        history = [doc.to_dict() for doc in docs]
        
        logger.info(f"Retrieved {len(history)} burnout assessments for user {user_id}")
        return history
    
    except Exception as e:
        logger.error(f"Error fetching burnout history: {str(e)}")
        return []


def create_health_log_entry(user_id, health_data):
    """
    Create a new health log entry in Firestore
    Called after Google Form submission via Make.com
    
    Args:
        user_id: User identifier
        health_data: dict with all health metrics from form
    
    Returns:
        str: Document ID if successful, None if failed
    """
    try:
        from integration.firebase_manager import db
        
        logger.info(f"Creating health log entry for user {user_id}")
        
        # Create document with auto-generated ID
        doc_ref = db.collection('health_logs').document()
        
        # Add timestamp if not present
        if 'timestamp' not in health_data:
            health_data['timestamp'] = datetime.now().isoformat()
        
        # Ensure user_id is included
        health_data['user_id'] = user_id
        
        # Set document
        doc_ref.set(health_data)
        
        logger.info(f"✅ Created health log entry: {doc_ref.id}")
        return doc_ref.id
    
    except Exception as e:
        logger.error(f"Error creating health log entry: {str(e)}", exc_info=True)
        return None


# Firestore Security Rules (copy to Firebase Console)
FIRESTORE_SECURITY_RULES = """
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    
    // Health logs - user can read/write their own
    match /health_logs/{document=**} {
      allow read: if request.auth.uid == resource.data.user_id;
      allow create: if request.auth.uid == request.resource.data.user_id;
      allow update, delete: if false; // Logs are immutable
    }
    
    // Sleep-productivity correlation - user can read their own
    match /sleep_productivity_correlation/{userId} {
      allow read: if request.auth.uid == userId;
      allow write: if false; // Only backend/Make.com can write
    }
    
    // Fitness progress - user can read their own
    match /fitness_progress/{userId} {
      allow read: if request.auth.uid == userId;
      allow write: if false; // Only backend/Make.com can write
    }
    
    // Burnout assessment - user can read their own
    match /burnout_assessment/{userId} {
      allow read: if request.auth.uid == userId;
      allow write: if false; // Only backend/Make.com can write
    }
    
    // Burnout assessment history - user can read their own
    match /burnout_assessment_history/{document=**} {
      allow read: if request.auth.uid == resource.data.user_id;
      allow write: if false; // Only backend/Make.com can write
    }
    
    // Deny all other access
    match /{document=**} {
      allow read, write: if false;
    }
  }
}
"""
