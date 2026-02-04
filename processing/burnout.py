"""
Burnout Risk Assessment Module
Combines features from nutrition, sentiment, sleep, and fitness data.
Uses Random Forest Classifier to predict burnout risk.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Lazy import sklearn and joblib (saves ~5-8 seconds on startup)
joblib = None
RandomForestClassifier = None
StandardScaler = None

logger = logging.getLogger(__name__)


class BurnoutPredictor:
    """Predicts burnout risk using Random Forest Classifier."""
    
    # Feature names expected by the model
    FEATURE_NAMES = [
        'healthy_food_ratio',
        'junk_food_ratio',
        'average_sentiment',
        'sentiment_variability',
        'sleep_consistency',
        'sleep_duration',
        'productivity_score',
        'fitness_activity_level',
        'workout_frequency',
        'stress_level',
        'workload_rating',
        'work_hours'
    ]
    
    def __init__(self, model_path='models/burnout_rf.pkl'):
        """
        Initialize Burnout Predictor with trained model.
        
        Args:
            model_path (str): Path to trained Random Forest model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        # Don't load model immediately - wait until first use (saves ~5-8 seconds)
        logger.info("BurnoutPredictor initialized (model will load on first use)")
    
    def load_model(self):
        """Load the pre-trained Random Forest model."""
        global joblib, RandomForestClassifier, StandardScaler
        
        # Lazy import sklearn and joblib only when model is loaded
        if joblib is None:
            logger.info("First-time sklearn/joblib import (this takes ~5-8 seconds)...")
            import joblib as joblib_module
            from sklearn.ensemble import RandomForestClassifier as RFC
            from sklearn.preprocessing import StandardScaler as SS
            joblib = joblib_module
            RandomForestClassifier = RFC
            StandardScaler = SS
            logger.info("sklearn/joblib loaded successfully")
        
        # Initialize scaler
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        try:
            if Path(self.model_path).exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}. Creating placeholder model.")
                self.model = self._create_placeholder_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = self._create_placeholder_model()
    
    def _create_placeholder_model(self):
        """Create a placeholder Random Forest model for testing."""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Create dummy training data
        X_dummy = np.random.randn(100, len(self.FEATURE_NAMES))
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        logger.info("Placeholder Random Forest model created for testing")
        return model
    
    def aggregate_features(self, df):
        """
        Aggregate features from Firebase DataFrame into a single feature vector for prediction.
        
        This is the "Feature Bridge" that maps Firestore field names to model features.
        
        Args:
            df (pd.DataFrame): Firebase DataFrame from get_health_data()
        
        Returns:
            dict: Aggregated features mapped to model requirements, or None if df is invalid
        
        Field Mappings:
            Firebase Field → Model Feature (Reason)
            healthy_ratio → healthy_food_ratio (nutrition resilience)
            junk_ratio → junk_food_ratio (nutrition resilience)
            sentiment_avg → average_sentiment (emotional exhaustion indicator)
            sentiment_variability → sentiment_variability (mood stability)
            sleep_hours → sleep_duration (strong burnout predictor)
            sleep_consistency → sleep_consistency (sleep pattern stability)
            activity_level → fitness_activity_level (stress protective factor)
            workouts_per_week → workout_frequency (exercise frequency)
            stress_score/stress_level → stress_level (direct stress calibration)
            productivity_score → productivity_score (output capacity)
            workload_rating → workload_rating (work pressure)
            work_hours → work_hours (time demand)
        """
        try:
            if df is None or df.empty:
                logger.warning("Cannot aggregate features from empty DataFrame")
                return None
            
            # Get the most recent record
            latest_record = df.iloc[-1]
            
            features = {}
            
            # Nutrition features
            features['healthy_food_ratio'] = float(latest_record.get('healthy_ratio', 0.5))
            features['junk_food_ratio'] = float(latest_record.get('junk_ratio', 0.2))
            
            # Sentiment features
            features['average_sentiment'] = float(latest_record.get('sentiment_avg', latest_record.get('average_sentiment', 0)))
            features['sentiment_variability'] = float(latest_record.get('sentiment_variability', 0.5))
            
            # Sleep features
            features['sleep_consistency'] = float(latest_record.get('sleep_consistency', 0.5))
            features['sleep_duration'] = float(latest_record.get('sleep_hours', latest_record.get('sleep_duration', 7)))
            
            # Fitness features
            features['fitness_activity_level'] = float(latest_record.get('activity_level', latest_record.get('fitness_activity_level', 0.5)))
            features['workout_frequency'] = float(latest_record.get('workouts_per_week', latest_record.get('workout_frequency', 2)))
            
            # Survey/Daily Check-in features
            features['productivity_score'] = float(latest_record.get('productivity_score', 5))
            features['stress_level'] = float(latest_record.get('stress_score', latest_record.get('stress_level', 5)))
            features['workload_rating'] = float(latest_record.get('workload_rating', 5))
            features['work_hours'] = float(latest_record.get('work_hours', 8))
            
            logger.info("Features aggregated successfully from Firebase DataFrame")
            return features
        except Exception as e:
            logger.error(f"Error aggregating features: {e}")
            return None
    
    def predict_burnout(self, features_dict):
        """
        Predict burnout risk based on aggregated features.
        
        Args:
            features_dict (dict or pd.Series): Dictionary/Series of aggregated features
                Can be a manual dict or a Pandas Series from Firebase DataFrame
        
        Returns:
            dict: Burnout prediction with risk score
        """
        try:
            # Lazy load model on first prediction
            if self.model is None:
                logger.info("Loading model for first prediction...")
                self.load_model()
            
            # Convert features dict/Series to ordered array
            feature_array = np.array([
                features_dict.get(name, 0) if isinstance(features_dict, dict) else features_dict.get(name, 0) 
                for name in self.FEATURE_NAMES
            ]).reshape(1, -1)
            
            # Normalize features
            feature_array_scaled = self.scaler.fit_transform(feature_array)
            
            # Get prediction and probability
            prediction = self.model.predict(feature_array_scaled)[0]
            probabilities = self.model.predict_proba(feature_array_scaled)[0]
            
            # Calculate burnout score (0-100)
            burnout_score = int(probabilities[1] * 100)
            
            result = {
                'burnout_risk': 'High' if burnout_score >= 70 else 'Medium' if burnout_score >= 40 else 'Low',
                'burnout_score': burnout_score,
                'confidence': float(max(probabilities)),
                'interpretation': self._interpret_risk(burnout_score),
                'recommendations': self._generate_recommendations(burnout_score, features_dict)
            }
            
            logger.info(f"Burnout prediction: {result['burnout_risk']} ({burnout_score}%)")
            return result
        except Exception as e:
            logger.error(f"Error predicting burnout: {e}")
            return None
    
    def predict_burnout_from_dataframe(self, df, use_latest=True):
        """
        Predict burnout risk from a Firestore DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame from Firebase with health metrics
            use_latest (bool): If True, use the latest row; if False, return predictions for all rows
            
        Returns:
            dict or list: Burnout prediction(s)
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided for burnout prediction")
            return None
        
        try:
            # Firestore field name mappings
            field_mappings = {
                'healthy_food_ratio': 'healthy_food_ratio',
                'junk_food_ratio': 'junk_food_ratio',
                'average_sentiment': 'average_sentiment',
                'sentiment_variability': 'sentiment_variability',
                'sleep_consistency': 'sleep_consistency',
                'sleep_duration': 'sleep_hours',  # Map Firestore 'sleep_hours' to model feature 'sleep_duration'
                'productivity_score': 'productivity_score',
                'fitness_activity_level': 'fitness_activity_level',
                'workout_frequency': 'workouts_per_week',  # Map Firestore field
                'stress_level': 'stress_level',
                'workload_rating': 'workload_rating',
                'work_hours': 'work_hours'
            }
            
            if use_latest:
                # Get the most recent row
                latest_row = df.iloc[-1]
                
                # Map Firestore fields to model features
                features = {}
                for model_feature, firestore_field in field_mappings.items():
                    if firestore_field in latest_row.index:
                        features[model_feature] = float(latest_row[firestore_field])
                    else:
                        features[model_feature] = 0  # Default if field doesn't exist
                
                return self.predict_burnout(features)
            else:
                # Return predictions for all rows
                results = []
                for idx, row in df.iterrows():
                    features = {}
                    for model_feature, firestore_field in field_mappings.items():
                        if firestore_field in row.index:
                            features[model_feature] = float(row[firestore_field])
                        else:
                            features[model_feature] = 0
                    
                    prediction = self.predict_burnout(features)
                    if prediction:
                        prediction['date'] = row.get('timestamp', idx)
                        results.append(prediction)
                
                return results
        except Exception as e:
            logger.error(f"Error predicting burnout from DataFrame: {e}")
            return None
    
    def _interpret_risk(self, score):
        """Interpret burnout risk score."""
        if score >= 80:
            return "Critical burnout risk detected. Immediate intervention recommended."
        elif score >= 60:
            return "High burnout risk. Consider reducing workload or seeking support."
        elif score >= 40:
            return "Moderate burnout risk. Monitor closely and maintain self-care."
        else:
            return "Low burnout risk. Keep maintaining healthy habits."
    
    def _generate_recommendations(self, score, features):
        """Generate personalized recommendations based on burnout risk."""
        recommendations = []
        
        # Sentiment-based recommendations
        if features.get('average_sentiment', 0) < -0.2:
            recommendations.append("Consider mental health support or counseling.")
        
        # Sleep-based recommendations
        if features.get('sleep_duration', 7) < 6:
            recommendations.append("Improve sleep hygiene. Aim for 7-9 hours of sleep.")
        
        # Fitness-based recommendations
        if features.get('fitness_activity_level', 0.5) < 0.3:
            recommendations.append("Increase physical activity. Start with 30 minutes of exercise daily.")
        
        # Nutrition-based recommendations
        if features.get('junk_food_ratio', 0.2) > 0.4:
            recommendations.append("Improve diet quality. Reduce processed foods and increase vegetables.")
        
        # Work-based recommendations
        if features.get('work_hours', 8) > 10:
            recommendations.append("Discuss work-life balance with your manager. Consider reducing hours.")
        
        if features.get('stress_level', 5) > 7:
            recommendations.append("Practice stress management techniques: meditation, yoga, or breathing exercises.")
        
        if not recommendations:
            recommendations.append("Continue your current healthy lifestyle practices.")
        
        return recommendations
    
    def get_feature_importance(self):
        """Get feature importance from the Random Forest model."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = dict(zip(self.FEATURE_NAMES, self.model.feature_importances_))
                sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                return dict(sorted_features)
            return None
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None


# Example usage function
def example_burnout_usage():
    """Example of how to use BurnoutPredictor."""
    predictor = BurnoutPredictor()
    
    # Example features
    sample_features = {
        'healthy_food_ratio': 0.6,
        'junk_food_ratio': 0.15,
        'average_sentiment': -0.1,
        'sentiment_variability': 0.3,
        'sleep_consistency': 0.7,
        'sleep_duration': 6.5,
        'productivity_score': 4,
        'fitness_activity_level': 0.4,
        'workout_frequency': 2,
        'stress_level': 7,
        'workload_rating': 8,
        'work_hours': 10
    }
    
    result = predictor.predict_burnout(sample_features)
    print(f"Burnout Prediction: {result}")


if __name__ == '__main__':
    example_burnout_usage()
