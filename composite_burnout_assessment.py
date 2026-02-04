"""
Composite Burnout Assessment
Synthesizes outputs from ALL integrated ML models to calculate burnout risk:
- Sentiment Journaling (VADER sentiment scores)
- Nutrition Tracker (food health labels & scores)
- Sleep-Productivity Correlator (sleep/productivity relationship)
- Fitness Trends (exercise frequency)

This is a META-ANALYSIS approach: burnout is derived from OTHER model outputs,
not trained on external datasets. This creates a real-time, holistic burnout score.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CompositeBurnoutAssessment:
    """
    Calculates burnout risk by analyzing outputs from all integrated ML models.
    
    Burnout is a composite of:
    1. Sentiment Health (positive/negative trend)
    2. Nutrition Health (food quality choices)
    3. Sleep-Productivity Balance (sleep adequacy vs output demands)
    4. Fitness Consistency (exercise frequency)
    5. Form-Based Metrics (stress, workload, social isolation)
    
    Output: Single burnout_risk_score (0-100)
    """
    
    # Weight each dimension (must sum to 1.0)
    WEIGHTS = {
        'sentiment': 0.25,      # Mental health trend (25%)
        'nutrition': 0.15,      # Physical health via food (15%)
        'sleep_fitness': 0.25,  # Recovery & activity balance (25%)
        'stress_form': 0.20,    # Direct stress indicators (20%)
        'social': 0.15,         # Social connection (15%)
    }
    
    def __init__(self):
        """Initialize assessment calculator."""
        if sum(self.WEIGHTS.values()) != 1.0:
            raise ValueError("WEIGHTS must sum to 1.0")
        logger.info("✅ CompositeBurnoutAssessment initialized")
    
    def calculate_from_user_data(
        self,
        user_id: str,
        sentiment_logs: list = None,
        nutrition_logs: list = None,
        sleep_data: Dict = None,
        exercise_days_per_week: float = None,
        stress_level: int = None,
        workload_rating: int = None,
        social_isolation: int = None,
        days_lookback: int = 7
    ) -> Dict:
        """
        Calculate composite burnout risk score from all available model outputs.
        
        Args:
            user_id: User identifier
            sentiment_logs: List of dicts with 'sentiment_compound' and 'timestamp'
            nutrition_logs: List of dicts with 'health_score' and 'health_label'
            sleep_data: Dict with 'sleep_hours_per_night' and 'productivity_score'
            exercise_days_per_week: Numeric (0-7)
            stress_level: Integer (1-10)
            workload_rating: Integer (1-10)
            social_isolation: Integer (1-10) where 1=isolated, 10=connected
            days_lookback: How many days to analyze for trends
        
        Returns:
            Dict with burnout risk score, component breakdown, and recommendations
        """
        
        logger.info(f"Calculating composite burnout for {user_id}")
        
        # Calculate each component (0-100 scale)
        sentiment_score = self._calculate_sentiment_component(sentiment_logs, days_lookback)
        nutrition_score = self._calculate_nutrition_component(nutrition_logs, days_lookback)
        sleep_fitness_score = self._calculate_sleep_fitness_component(
            sleep_data, exercise_days_per_week
        )
        stress_form_score = self._calculate_stress_form_component(
            stress_level, workload_rating
        )
        social_score = self._calculate_social_component(social_isolation)
        
        # Weighted composite (inverted so higher = more at-risk)
        composite_burnout = (
            self.WEIGHTS['sentiment'] * sentiment_score +
            self.WEIGHTS['nutrition'] * nutrition_score +
            self.WEIGHTS['sleep_fitness'] * sleep_fitness_score +
            self.WEIGHTS['stress_form'] * stress_form_score +
            self.WEIGHTS['social'] * social_score
        )
        
        # ENHANCEMENT: Apply interaction effects (factors that amplify each other)
        # Low sleep + High stress = severe risk (multiplicative effect)
        if sleep_fitness_score < 40 and stress_form_score < 40:
            composite_burnout *= 1.2  # 20% increase for compounding risk
            logger.debug("Applied interaction penalty: low sleep + high stress")
        
        # Poor mental health + Poor nutrition = severe neglect indicator
        if sentiment_score < 35 and nutrition_score < 35:
            composite_burnout *= 1.15  # 15% increase
            logger.debug("Applied interaction penalty: poor sentiment + poor nutrition")
        
        # Low social support + High stress = critical risk
        if social_score < 30 and stress_form_score < 30:
            composite_burnout *= 1.15  # 15% increase
            logger.debug("Applied interaction penalty: isolated + high stress")
        
        # Recovery capacity: if sleep is good but everything else is bad, still concerning
        recovery_capacity = (sleep_fitness_score + social_score) / 2
        if recovery_capacity < 35 and composite_burnout > 60:
            composite_burnout *= 1.1  # 10% increase for poor recovery potential
            logger.debug("Applied recovery capacity penalty: low recovery with high burnout")
        
        # Normalize to 0-100 range
        composite_burnout = min(100, composite_burnout)
        composite_burnout = max(0, composite_burnout)
        
        # Classify risk level
        risk_level, risk_interpretation = self._classify_risk(composite_burnout)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            sentiment_score,
            nutrition_score,
            sleep_fitness_score,
            stress_form_score,
            social_score,
            composite_burnout
        )
        
        return {
            'user_id': user_id,
            'burnout_risk_score': round(composite_burnout, 1),  # 0-100
            'risk_level': risk_level,  # LOW, MODERATE, HIGH, SEVERE
            'risk_interpretation': risk_interpretation,
            'confidence_pct': self._calculate_confidence(sentiment_logs, nutrition_logs),
            'components': {
                'sentiment_health': round(sentiment_score, 1),      # 0=depressed, 100=thriving
                'nutrition_health': round(nutrition_score, 1),      # 0=all junk, 100=all healthy
                'sleep_fitness_balance': round(sleep_fitness_score, 1),  # 0=exhausted, 100=optimal
                'stress_workload': round(stress_form_score, 1),     # 0=calm, 100=overwhelmed
                'social_connection': round(social_score, 1),        # 0=isolated, 100=connected
            },
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'recommendations': recommendations
        }
    
    def _calculate_sentiment_component(
        self, 
        sentiment_logs: list = None, 
        days_lookback: int = 7
    ) -> float:
        """
        Calculate sentiment health (0-100, lower = more at-risk).
        
        VADER sentiment_compound: -1.0 (very negative) to +1.0 (very positive)
        Burnout risk: High negative sentiment or high volatility indicates burnout.
        
        Returns:
            0 = consistently negative (high risk)
            50 = neutral/mixed
            100 = consistently positive (low risk)
        """
        
        if not sentiment_logs:
            logger.debug("No sentiment logs provided; assuming neutral")
            return 50.0
        
        df = pd.DataFrame(sentiment_logs)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_lookback)
            df = df[df['timestamp'] >= cutoff]
        
        if df.empty:
            logger.debug("No recent sentiment logs; assuming neutral")
            return 50.0
        
        compounds = pd.to_numeric(df['sentiment_compound'], errors='coerce').dropna()
        
        if compounds.empty:
            return 50.0
        
        avg_sentiment = compounds.mean()  # -1 to +1
        sentiment_std = compounds.std()
        
        # Convert to 0-100 scale (higher = less burnout)
        # Positive sentiment reduces burnout, negative increases it
        sentiment_base = ((avg_sentiment + 1) / 2) * 100  # -1→0, 0→50, +1→100
        
        # ENHANCED: High volatility is a strong burnout indicator (emotional dysregulation)
        # Standard deviation penalty - scales with severity
        volatility_penalty = min(sentiment_std * 30, 40)  # Up to 40 point penalty (was 20)
        
        # ENHANCED: Trending analysis - check if sentiment is declining
        if len(compounds) >= 3:
            recent_avg = compounds.iloc[-3:].mean()
            older_avg = compounds.iloc[:-3].mean() if len(compounds) > 3 else avg_sentiment
            
            # If sentiment is declining, add burnout penalty
            if recent_avg < older_avg - 0.2:  # Significant decline
                volatility_penalty += 15  # Additional 15 point penalty
                logger.debug(f"Sentiment declining trend detected: {older_avg:.2f} → {recent_avg:.2f}")
        
        sentiment_score = max(0, sentiment_base - volatility_penalty)
        
        logger.debug(
            f"Sentiment: avg={avg_sentiment:.2f}, std={sentiment_std:.2f}, "
            f"volatility_penalty={volatility_penalty:.1f}, score={sentiment_score:.1f}"
        )
        
        return sentiment_score
    
    def _calculate_nutrition_component(
        self, 
        nutrition_logs: list = None, 
        days_lookback: int = 7
    ) -> float:
        """
        Calculate nutrition health (0-100, lower = more at-risk).
        
        Food health labels: healthy (9), moderate (5), junk (2)
        Poor nutrition correlates with burnout (self-care neglect).
        
        Returns:
            0 = all junk food
            50 = balanced mixed diet
            100 = all healthy food
        """
        
        if not nutrition_logs:
            logger.debug("No nutrition logs; assuming balanced diet")
            return 50.0
        
        df = pd.DataFrame(nutrition_logs)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_lookback)
            df = df[df['timestamp'] >= cutoff]
        
        if df.empty:
            logger.debug("No recent nutrition logs; assuming balanced")
            return 50.0
        
        # Use health_score if available (0-10), otherwise use health_label
        if 'health_score' in df.columns:
            scores = pd.to_numeric(df['health_score'], errors='coerce').dropna()
            nutrition_score = (scores.mean() / 10) * 100 if not scores.empty else 50.0
        elif 'health_label' in df.columns:
            labels = df['health_label'].str.lower()
            healthy_count = (labels == 'healthy').sum()
            moderate_count = (labels == 'moderate').sum()
            junk_count = (labels == 'junk').sum()
            total = len(labels)
            
            if total > 0:
                nutrition_score = (
                    (healthy_count * 100 + moderate_count * 50 + junk_count * 0) / total
                )
            else:
                nutrition_score = 50.0
        else:
            nutrition_score = 50.0
        
        logger.debug(f"Nutrition: score={nutrition_score:.1f}")
        return nutrition_score
    
    def _calculate_sleep_fitness_component(
        self,
        sleep_data: Dict = None,
        exercise_days_per_week: float = None
    ) -> float:
        """
        Calculate sleep-fitness recovery balance (0-100, lower = more at-risk).
        
        Sleep needs differ by activity level:
        - High activity → need 8-9 hours sleep
        - Low activity → need 7-8 hours sleep
        - Insufficient sleep + high workload = burnout
        
        Returns:
            0 = sleep deprived + sedentary (high risk)
            50 = adequate sleep or moderate activity
            100 = excellent sleep + active fitness
        """
        
        # Extract sleep hours and productivity
        sleep_hours = None
        productivity_score = None
        
        if sleep_data:
            sleep_hours = sleep_data.get('sleep_hours_per_night')
            productivity_score = sleep_data.get('productivity_score')
        
        if sleep_hours is None:
            sleep_hours = 7.0  # Default assumption
        
        if productivity_score is None:
            productivity_score = 5.0  # Default assumption
        
        if exercise_days_per_week is None:
            exercise_days_per_week = 2.0  # Default: sedentary
        
        sleep_hours = float(sleep_hours)
        productivity_score = float(productivity_score)
        exercise_days_per_week = float(exercise_days_per_week)
        
        # Optimal sleep is 7-9 hours; penalize deviations
        sleep_quality = 0.0
        if 7 <= sleep_hours <= 9:
            sleep_quality = 100.0
        elif 6 <= sleep_hours < 7 or 9 < sleep_hours <= 10:
            sleep_quality = 80.0  # Slightly insufficient
        elif 5 <= sleep_hours < 6 or 10 < sleep_hours <= 11:
            sleep_quality = 50.0  # Concerning
        else:
            sleep_quality = 20.0  # Severe sleep deprivation
        
        # Exercise consistency: optimal is 4-5 days/week
        exercise_quality = 0.0
        if 4 <= exercise_days_per_week <= 5:
            exercise_quality = 100.0
        elif 3 <= exercise_days_per_week < 4 or 5 < exercise_days_per_week <= 6:
            exercise_quality = 80.0
        elif 2 <= exercise_days_per_week < 3 or 6 < exercise_days_per_week <= 7:
            exercise_quality = 60.0
        elif 1 <= exercise_days_per_week < 2:
            exercise_quality = 40.0
        else:
            exercise_quality = 20.0  # No exercise
        
        # Combine: both matter equally
        sleep_fitness_score = (sleep_quality + exercise_quality) / 2
        
        # Adjustment based on productivity: high productivity with low sleep = unsustainable
        if productivity_score >= 7 and sleep_hours < 7:
            sleep_fitness_score *= 0.8  # 20% penalty for overwork
        
        logger.debug(
            f"Sleep-Fitness: sleep={sleep_hours}h, exercise={exercise_days_per_week}d/w, "
            f"productivity={productivity_score}, score={sleep_fitness_score:.1f}"
        )
        
        return sleep_fitness_score
    
    def _calculate_stress_form_component(
        self,
        stress_level: int = None,
        workload_rating: int = None
    ) -> float:
        """
        Calculate stress & workload burden (0-100, lower = more at-risk).
        
        Direct indicators from Google Form.
        ENHANCED: Non-linear scoring emphasizes critical stress levels.
        
        Scale: 1=calm, 10=overwhelmed
        Returns: 0=very overwhelmed (100%), 100=very calm (0%)
        """
        
        if stress_level is None:
            stress_level = 5  # Default neutral
        
        if workload_rating is None:
            workload_rating = 5
        
        stress_level = int(stress_level)
        workload_rating = int(workload_rating)
        
        # ENHANCED: Non-linear conversion - high stress has exponential impact
        # Level 8-10 are critical and heavily penalized
        if stress_level >= 8:
            stress_score = (10 - stress_level) ** 2  # Squared penalty for critical levels
        else:
            stress_score = ((10 - stress_level) / 9) * 100
        
        if workload_rating >= 8:
            workload_score = (10 - workload_rating) ** 2
        else:
            workload_score = ((10 - workload_rating) / 9) * 100
        
        # Combined - they reinforce each other
        # When both are high, the penalty is amplified
        if stress_level >= 7 and workload_rating >= 7:
            stress_form_score = min((stress_score + workload_score) / 2.5, 0)  # Harsher penalty
        else:
            stress_form_score = (stress_score + workload_score) / 2
        
        stress_form_score = min(100, max(0, stress_form_score))
        
        logger.debug(
            f"Stress-Form: stress={stress_level}, workload={workload_rating}, "
            f"score={stress_form_score:.1f}"
        )
        
        return stress_form_score
    
    def _calculate_social_component(
        self,
        social_isolation: int = None
    ) -> float:
        """
        Calculate social connection health (0-100, lower = more at-risk).
        
        Social isolation is a major burnout indicator.
        
        Input scale: 1=very isolated, 10=very connected
        Returns: 0=isolated (high risk), 100=connected (low risk)
        """
        
        if social_isolation is None:
            social_isolation = 5  # Default neutral
        
        social_isolation = int(social_isolation)
        
        # Convert 1-10 scale to 0-100 where 10=connected=100
        social_score = ((social_isolation - 1) / 9) * 100
        
        logger.debug(f"Social: isolation={social_isolation}, score={social_score:.1f}")
        
        return social_score
    
    def _classify_risk(self, burnout_score: float) -> Tuple[str, str]:
        """
        Classify burnout risk level based on composite score.
        
        Args:
            burnout_score: 0-100 where 0=no risk, 100=severe risk
        
        Returns:
            Tuple of (risk_level, interpretation)
        """
        
        if burnout_score < 25:
            return "LOW", "✅ You're managing well. Keep up your healthy habits!"
        elif burnout_score < 50:
            return "MODERATE", "⚠️ Some burnout signs. Consider increasing self-care."
        elif burnout_score < 75:
            return "HIGH", "🔴 Significant burnout risk. Action needed soon."
        else:
            return "SEVERE", "🚨 Critical burnout risk. Please seek support immediately."
    
    def _calculate_confidence(
        self,
        sentiment_logs: list = None,
        nutrition_logs: list = None
    ) -> float:
        """
        Calculate confidence in assessment based on data availability.
        
        More logs = higher confidence.
        Minimum 3 days of data for moderate confidence.
        """
        
        confidence = 50.0  # Base confidence
        
        # Sentiment data
        if sentiment_logs and len(sentiment_logs) >= 3:
            confidence += 20
        elif sentiment_logs and len(sentiment_logs) >= 1:
            confidence += 10
        
        # Nutrition data
        if nutrition_logs and len(nutrition_logs) >= 5:
            confidence += 20
        elif nutrition_logs and len(nutrition_logs) >= 2:
            confidence += 10
        
        return min(confidence, 100.0)
    
    def _generate_recommendations(
        self,
        sentiment_score: float,
        nutrition_score: float,
        sleep_fitness_score: float,
        stress_form_score: float,
        social_score: float,
        composite_burnout: float
    ) -> list:
        """
        Generate personalized recommendations based on weakest components.
        
        Returns:
            List of actionable recommendations, prioritized by impact
        """
        
        recommendations = []
        
        # Identify lowest-scoring dimensions
        components = [
            ('Sentiment & Mental Health', sentiment_score),
            ('Nutrition & Self-Care', nutrition_score),
            ('Sleep & Fitness Balance', sleep_fitness_score),
            ('Stress & Workload', stress_form_score),
            ('Social Connection', social_score),
        ]
        
        # Sort by score (lowest first = highest priority)
        components.sort(key=lambda x: x[1])
        
        # Generate specific recommendations
        if components[0][1] < 40:
            if components[0][0] == 'Sentiment & Mental Health':
                recommendations.append(
                    "🧠 Consider talking to a counselor or therapist. "
                    "Your mood trend suggests emotional strain."
                )
            elif components[0][0] == 'Nutrition & Self-Care':
                recommendations.append(
                    "🥗 Prioritize nutrition. Eating healthier foods will boost "
                    "your energy and mood."
                )
            elif components[0][0] == 'Sleep & Fitness Balance':
                recommendations.append(
                    "😴 Prioritize sleep (7-9 hours) and exercise (4-5 days/week). "
                    "Your recovery is critical."
                )
            elif components[0][0] == 'Stress & Workload':
                recommendations.append(
                    "📋 Review your workload. Consider setting boundaries, "
                    "delegating, or asking for support."
                )
            elif components[0][0] == 'Social Connection':
                recommendations.append(
                    "👥 Increase social connections. Reach out to friends, "
                    "family, or community."
                )
        
        # General recommendations based on overall risk
        if composite_burnout >= 75:
            recommendations.append(
                "🚨 Your burnout risk is critical. Please speak with HR, "
                "a manager, or mental health professional."
            )
        elif composite_burnout >= 50:
            recommendations.append(
                "⚠️ You're showing signs of moderate burnout. "
                "Consider a brief break or vacation."
            )
        
        # Positive reinforcement
        if composite_burnout < 30:
            recommendations.append(
                "🎉 You're managing stress well! Keep maintaining "
                "your current healthy habits."
            )
        
        return recommendations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Return weights for each component."""
        return self.WEIGHTS.copy()


# Standalone utility function for quick calculation
def calculate_burnout_risk(
    user_id: str,
    sentiment_logs: list = None,
    nutrition_logs: list = None,
    sleep_hours: float = None,
    productivity_score: float = None,
    exercise_days: float = None,
    stress_level: int = None,
    workload_rating: int = None,
    social_isolation: int = None
) -> Dict:
    """
    Quick wrapper to calculate composite burnout score.
    
    Example:
        result = calculate_burnout_risk(
            user_id="user_123",
            sentiment_logs=[...],
            nutrition_logs=[...],
            sleep_hours=7.5,
            exercise_days=3,
            stress_level=6
        )
    """
    
    assessor = CompositeBurnoutAssessment()
    
    sleep_data = None
    if sleep_hours is not None or productivity_score is not None:
        sleep_data = {
            'sleep_hours_per_night': sleep_hours,
            'productivity_score': productivity_score
        }
    
    return assessor.calculate_from_user_data(
        user_id=user_id,
        sentiment_logs=sentiment_logs,
        nutrition_logs=nutrition_logs,
        sleep_data=sleep_data,
        exercise_days_per_week=exercise_days,
        stress_level=stress_level,
        workload_rating=workload_rating,
        social_isolation=social_isolation
    )
