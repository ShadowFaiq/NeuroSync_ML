"""
Burnout Risk Score Predictor
Uses trained model to predict burnout risk from user survey responses
"""

import joblib
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BurnoutPredictor:
    """Loads trained model and predicts burnout risk from survey responses"""
    
    def __init__(self):
        """Load pre-trained model and scaler"""
        try:
            self.model = joblib.load('models/burnout_predictor.pkl')
            self.scaler = joblib.load('models/burnout_scaler.pkl')
            self.feature_names = joblib.load('models/burnout_features.pkl')
            logger.info("✅ Model loaded successfully")
        except FileNotFoundError:
            logger.error("❌ Model files not found! Run burnout_training_pipeline.py first")
            self.model = None
            self.scaler = None
            self.feature_names = None
    
    def predict_from_survey(self, survey_responses):
        """
        Predict burnout risk from user survey responses
        
        Args:
            survey_responses (dict): Survey answers mapped to features
            Example:
            {
                'age': 28,
                'work_hours': 45,
                'exercise_days': 3,
                'sleep_hours': 6.5,
                'sleep_quality': 7,
                'productivity': 6,
                'mental_clarity': 5,
                'social_isolation': 2,
                'support_system': 8,
                'emotional_exhaustion': 6
            }
        
        Returns:
            dict: Prediction result with burnout risk score and interpretation
        """
        
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded. Run training pipeline first.'
            }
        
        try:
            # Create feature vector matching trained features
            features = np.zeros((1, len(self.feature_names)))
            
            for i, feature_name in enumerate(self.feature_names):
                if feature_name in survey_responses:
                    features[0, i] = survey_responses[feature_name]
                else:
                    # Use default middle value for missing features
                    features[0, i] = 0
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Interpret result
            risk_level = self._interpret_risk(prediction, probabilities)
            
            return {
                'success': True,
                'burnout_risk_class': int(prediction),
                'burnout_probabilities': {
                    'low': float(probabilities[0]) if len(probabilities) > 0 else 0,
                    'medium': float(probabilities[1]) if len(probabilities) > 1 else 0,
                    'high': float(probabilities[2]) if len(probabilities) > 2 else 0,
                },
                'risk_level': risk_level['level'],
                'risk_score': risk_level['score'],
                'recommendations': risk_level['recommendations']
            }
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _interpret_risk(self, prediction, probabilities):
        """Interpret risk level and provide recommendations"""
        
        # Map prediction to risk level
        if prediction == 0:
            level = "LOW BURNOUT RISK"
            score = "✅ Good"
            color = "green"
        elif prediction == 1:
            level = "MODERATE BURNOUT RISK"
            score = "⚠️ Monitor"
            color = "yellow"
        elif prediction == 2:
            level = "HIGH BURNOUT RISK"
            score = "🔴 Concerning"
            color = "orange"
        else:  # prediction == 3
            level = "SEVERE BURNOUT RISK"
            score = "🚨 Critical"
            color = "red"
        
        # Generate recommendations based on prediction
        recommendations = self._get_recommendations(prediction)
        
        return {
            'level': level,
            'score': score,
            'color': color,
            'numeric': int(prediction),
            'recommendations': recommendations
        }
    
    def _get_recommendations(self, risk_level):
        """Generate health recommendations based on risk level"""
        
        recommendations = {
            0: [
                "✅ Maintain current healthy habits",
                "Continue regular exercise routine",
                "Keep up with sleep schedule",
                "Maintain strong social connections"
            ],
            1: [
                "⚠️ Consider increasing exercise to 4+ days/week",
                "Aim for 7-8 hours of sleep consistently",
                "Schedule regular breaks during work",
                "Strengthen your support network",
                "Practice stress management (meditation, deep breathing)"
            ],
            2: [
                "🔴 Increase physical activity to 5+ days/week",
                "Prioritize 8+ hours of sleep per night",
                "Take regular breaks and consider mental health support",
                "Evaluate work-life balance",
                "Consider speaking with a counselor or therapist"
            ],
            3: [
                "🚨 URGENT: Consult healthcare professional",
                "Consider reducing work hours if possible",
                "Seek professional mental health support immediately",
                "Prioritize sleep (aim for 9+ hours)",
                "Engage in daily physical activity",
                "Build strong support system",
                "Consider medical evaluation for stress-related issues"
            ]
        }
        
        return recommendations.get(risk_level, [])
    
    def predict_batch(self, survey_responses_list):
        """
        Predict for multiple users
        
        Args:
            survey_responses_list (list): List of survey response dicts
        
        Returns:
            list: List of prediction results
        """
        return [self.predict_from_survey(responses) for responses in survey_responses_list]


def example_usage():
    """Example: How to use the predictor"""
    
    # Initialize predictor
    predictor = BurnoutPredictor()
    
    # Example survey response (from Google Form)
    user_survey = {
        'age': 28,
        'work_hours': 50,
        'exercise_days': 2,
        'sleep_hours': 5.5,
        'sleep_quality': 4,
        'productivity': 4,
        'mental_clarity': 3,
        'social_isolation': 4,
        'support_system': 5,
        'emotional_exhaustion': 7
    }
    
    # Get prediction
    result = predictor.predict_from_survey(user_survey)
    
    # Display results
    if result['success']:
        print("\n" + "="*70)
        print("BURNOUT RISK ASSESSMENT")
        print("="*70)
        print(f"\n🎯 Risk Level: {result['risk_level']}")
        print(f"📊 Score: {result['risk_score']}")
        print(f"\nProbabilities:")
        for risk_type, prob in result['burnout_probabilities'].items():
            print(f"  {risk_type.capitalize()}: {prob*100:.1f}%")
        print(f"\n💡 Recommendations:")
        for rec in result['recommendations']:
            print(f"  • {rec}")
        print("\n" + "="*70)
    else:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    example_usage()
