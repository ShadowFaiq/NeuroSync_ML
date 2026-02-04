"""
Nutrition Tracker & Image Classification Module
Handles food image classification using transfer learning and rule-based expert system.
Preprocesses images, loads trained model, and flags dietary concerns.
"""

import numpy as np
from PIL import Image
import logging
from pathlib import Path
import os
import sys

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


class FoodImageAnalyzer:
    """Food image analyzer using transfer learning classifier."""
    
    def __init__(self):
        """Initialize the Food Image Analyzer with transfer learning model."""
        self.classifier = None
        self._initialized = False
        self.last_error = None
        logger.info("FoodImageAnalyzer initialized")
    
    def _load_classifier(self):
        """Lazy load the food classifier on first use."""
        if self._initialized:
            return True
        
        try:
            from models.food_classifier import get_food_classifier
            self.classifier = get_food_classifier()
            self._initialized = True
            logger.info("Food classifier loaded successfully")
            return True
        except Exception as e:
            logger.exception(f"Error loading food classifier: {str(e)}")
            import traceback
            traceback.print_exc()
            self.last_error = str(e)
            return False
    
    def analyze_food_image(self, image_input):
        """
        Analyze food image and return health classification.
        
        Args:
            image_input: Either file path (str) or PIL Image object
        
        Returns:
            dict: Analysis results with health classification and nutrition score
        """
        if not self._load_classifier():
            error_msg = getattr(self, 'last_error', 'Unknown error')
            return {
                'success': False,
                'error': f'Food classifier not available: {error_msg}',
                'health_score': 0,
                'classification': 'unknown'
            }
        
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                if not os.path.exists(image_input):
                    return {
                        'success': False,
                        'error': f'Image file not found: {image_input}'
                    }
                result = self.classifier.predict_image(image_input)
            elif isinstance(image_input, Image.Image):
                # PIL Image - convert to array
                image_array = np.array(image_input)
                result = self.classifier.predict_from_array(image_array)
            else:
                # Assume numpy array
                result = self.classifier.predict_from_array(image_input)
            
            # Get health score
            health_info = self.classifier.get_health_score(result['classification'])
            
            return {
                'success': True,
                'primary_food': result['primary_food'],
                'classification': result['classification'],
                'confidence': result['confidence'],
                'health_score': health_info['score'],
                'health_description': health_info['description'],
                'health_emoji': health_info['emoji'],
                'all_predictions': result['predictions'],
                'nutritional_benefits': self._get_nutritional_info(result['primary_food']),
                'recommendations': self._get_recommendations(result['classification'])
            }
        
        except Exception as e:
            logger.error(f"Error analyzing food image: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'health_score': 0
            }
    
    def _get_nutritional_info(self, food_name):
        """Get nutritional information for identified food."""
        food_nutrition = {
            'sushi': {
                'nutrients': ['Protein', 'Omega-3 Fatty Acids', 'Vitamins B12, D'],
                'benefits': 'High in protein and healthy fats',
                'calories_per_serving': '200-300'
            },
            'salad': {
                'nutrients': ['Fiber', 'Vitamins A, C', 'Minerals'],
                'benefits': 'Low calorie, rich in nutrients',
                'calories_per_serving': '100-200'
            },
            'edamame': {
                'nutrients': ['Protein', 'Fiber', 'Iron'],
                'benefits': 'Plant-based protein, low calorie',
                'calories_per_serving': '95'
            },
            'apple_pie': {
                'nutrients': ['Carbohydrates', 'Vitamin C'],
                'concerns': 'High sugar and fat content',
                'calories_per_serving': '300-400'
            },
            'ice_cream': {
                'nutrients': ['Calcium', 'Vitamin D'],
                'concerns': 'High sugar and saturated fat',
                'calories_per_serving': '150-300'
            },
            'french_toast': {
                'nutrients': ['Carbohydrates', 'Protein'],
                'concerns': 'High calorie and fat content',
                'calories_per_serving': '400-600'
            },
            'cannoli': {
                'nutrients': ['Carbohydrates'],
                'concerns': 'Very high sugar and fat',
                'calories_per_serving': '250-350'
            },
            'falafel': {
                'nutrients': ['Protein', 'Fiber', 'Iron'],
                'benefits': 'Good plant protein source',
                'concerns': 'Often deep-fried, high in calories',
                'calories_per_serving': '170-200'
            },
            'ramen': {
                'nutrients': ['Carbohydrates', 'Protein'],
                'concerns': 'High sodium content',
                'calories_per_serving': '400-500'
            },
            'bibimbap': {
                'nutrients': ['Protein', 'Vegetables', 'Carbohydrates'],
                'benefits': 'Balanced meal with vegetables',
                'calories_per_serving': '450-550'
            }
        }
        
        return food_nutrition.get(food_name.lower(), {
            'nutrients': ['Unknown'],
            'benefits': 'Unable to determine nutritional info'
        })
    
    def _get_recommendations(self, classification):
        """Get dietary recommendations based on food classification."""
        recommendations = {
            'healthy': [
                '✅ Great choice for daily consumption',
                '✅ Rich in essential nutrients',
                '✅ Supports overall wellness',
                '💡 Pair with whole grains for complete nutrition'
            ],
            'moderate': [
                '⚠️ Good in moderation',
                '⚠️ Balance with lighter meals',
                '💡 Pair with vegetables or fruits',
                '💡 Watch portion sizes'
            ],
            'junk': [
                '⚠️ High in calories and sugar',
                '⚠️ Enjoy occasionally, not daily',
                '⚠️ Balance with healthy meals',
                '💡 Stay hydrated and exercise regularly'
            ],
            'unknown': [
                '❓ Unable to classify food',
                '💡 Check nutritional label for details',
                '💡 Consult dietary guidelines'
            ]
        }
        
        return recommendations.get(classification, recommendations['unknown'])


# Legacy NutritionCNN class for backward compatibility

        
        # Lazy import TensorFlow only when model is actually loaded
        if tf is None:
            logger.info("First-time TensorFlow import (this takes ~10-15 seconds)...")
            import tensorflow as tf_module
            from tensorflow.keras.models import load_model as load_model_func
            tf = tf_module
            load_model = load_model_func
            logger.info("TensorFlow loaded successfully")
        
        try:
            if Path(self.model_path).exists():
                self.model = load_model(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}. Using placeholder model.")
                self.model = self._create_placeholder_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = self._create_placeholder_model()
    
    def _create_placeholder_model(self):
        """Create a placeholder model for testing."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128, 128, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.class_labels), activation='softmax')
        ])
        logger.info("Placeholder model created for testing")
        return model
    
    def preprocess_image(self, image_input):
        """
        Preprocess image for model input.
        
        Args:
            image_input: PIL Image or numpy array
        
        Returns:
            numpy array: Preprocessed image
        """
        try:
            # Convert to PIL Image if numpy array
            if isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input.astype('uint8'))
            else:
                image = image_input
            
            # Resize to model input size
            image = image.resize(self.input_size)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array and normalize
            image_array = np.array(image) / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            logger.info("Image preprocessed successfully")
            return image_array
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_input):
        """
        Predict food category from image.
        
        Args:
            image_input: PIL Image or numpy array
        
        Returns:
            dict: Prediction results with class and confidence
        """
        try:
            # Lazy load model on first prediction
            if self.model is None:
                logger.info("Loading model for first prediction...")
                self.load_model()
            
            preprocessed_image = self.preprocess_image(image_input)
            if preprocessed_image is None:
                return None
            
            predictions = self.model.predict(preprocessed_image)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            result = {
                'class': self.class_labels[predicted_class_idx],
                'confidence': confidence,
                'all_predictions': {
                    label: float(prob) for label, prob in zip(self.class_labels, predictions[0])
                }
            }
            
            logger.info(f"Prediction: {result['class']} ({result['confidence']:.2%})")
            return result
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None


class NutritionExpertSystem:
    """Rule-based expert system for dietary analysis."""
    
    # Define nutrition categories
    HEALTHY_CATEGORIES = {'Leafy Greens', 'Vegetables', 'Fruits', 'Proteins', 'Grains', 'Dairy'}
    UNHEALTHY_CATEGORIES = {'Processed Sugar', 'Fried Foods', 'Fast Food', 'Beverages'}
    
    def __init__(self):
        """Initialize the expert system with rules."""
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize dietary rules and recommendations."""
        rules = {
            'Leafy Greens': {'health_score': 10, 'macro': 'Fiber', 'deficiency_risk': []},
            'Vegetables': {'health_score': 9, 'macro': 'Vitamins', 'deficiency_risk': ['Vitamin C']},
            'Fruits': {'health_score': 8, 'macro': 'Fiber', 'deficiency_risk': ['Fiber']},
            'Proteins': {'health_score': 8, 'macro': 'Protein', 'deficiency_risk': ['Iron']},
            'Grains': {'health_score': 7, 'macro': 'Carbs', 'deficiency_risk': ['Fiber']},
            'Dairy': {'health_score': 7, 'macro': 'Calcium', 'deficiency_risk': ['Calcium']},
            'Processed Sugar': {'health_score': 2, 'macro': 'Sugar', 'deficiency_risk': ['Weight Gain']},
            'Fried Foods': {'health_score': 3, 'macro': 'Fat', 'deficiency_risk': ['Heart Health']},
            'Fast Food': {'health_score': 2, 'macro': 'Fat & Sodium', 'deficiency_risk': ['Heart Health']},
            'Beverages': {'health_score': 5, 'macro': 'Varies', 'deficiency_risk': ['Sugar Intake']}
        }
        return rules
    
    def analyze(self, cnn_prediction):
        """
        Analyze food prediction and generate recommendations.
        
        Args:
            cnn_prediction (dict): Prediction from NutritionCNN
        
        Returns:
            dict: Analysis results with recommendations
        """
        try:
            food_class = cnn_prediction['class']
            confidence = cnn_prediction['confidence']
            
            if confidence < 0.6:
                logger.warning(f"Low confidence prediction: {confidence:.2%}")
            
            rule = self.rules.get(food_class, {})
            
            analysis = {
                'food_class': food_class,
                'confidence': confidence,
                'health_score': rule.get('health_score', 0),
                'macro_type': rule.get('macro', 'Unknown'),
                'is_healthy': food_class in self.HEALTHY_CATEGORIES,
                'deficiency_risk': rule.get('deficiency_risk', []),
                'recommendations': self._generate_recommendations(food_class)
            }
            
            logger.info(f"Analysis complete: {food_class}")
            return analysis
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return None
    
    def _generate_recommendations(self, food_class):
        """Generate dietary recommendations based on food class."""
        recommendations = {
            'Leafy Greens': "Excellent choice! Continue incorporating leafy greens into your diet.",
            'Vegetables': "Great source of vitamins and minerals. Keep it up!",
            'Fruits': "Good choice! Ensure a variety of fruits for different nutrients.",
            'Proteins': "Essential for muscle building and recovery.",
            'Grains': "Good source of carbohydrates. Choose whole grains when possible.",
            'Dairy': "Good for calcium intake. Consider alternatives if needed.",
            'Processed Sugar': "⚠️ High sugar content. Limit intake for better health.",
            'Fried Foods': "⚠️ High in unhealthy fats. Consider healthier cooking methods.",
            'Fast Food': "⚠️ Often high in sodium and unhealthy fats. Reduce frequency.",
            'Beverages': "Check sugar content. Prefer water or unsweetened beverages."
        }
        return recommendations.get(food_class, "No specific recommendation available.")


class NutritionTracker:
    """Main Nutrition Tracker combining food image analysis and Expert System."""
    
    def __init__(self, model_path='models/food_classifier_model.h5'):
        """
        Initialize the Nutrition Tracker.
        
        Args:
            model_path (str): Path to the trained food classifier model
        """
        self.analyzer = FoodImageAnalyzer()
        self.expert_system = NutritionExpertSystem()
    
    def analyze_meal(self, image_input):
        """
        Analyze a meal image end-to-end.
        
        Args:
            image_input: PIL Image or numpy array or file path
        
        Returns:
            dict: Complete analysis with prediction and recommendations
        """
        try:
            # Get food image analysis
            result = self.analyzer.analyze_food_image(image_input)
            if result is None:
                return None
            
            # Get expert system recommendations
            analysis = self.expert_system.analyze(result)
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing meal: {e}")
            return None


class NutritionAnalyzer:
    """Lightweight nutrition analyzer for tests and E2E pipeline."""

    HEALTHY_FOODS = {
        'apple', 'banana', 'orange', 'strawberry', 'blueberry', 'broccoli',
        'carrot', 'spinach', 'kale', 'lettuce', 'tomato', 'cucumber',
        'chicken', 'salmon', 'fish', 'turkey', 'beef', 'egg', 'eggs',
        'rice', 'oats', 'quinoa', 'wheat', 'bread', 'pasta',
        'milk', 'yogurt', 'cheese', 'butter', 'nuts', 'almonds',
        'olive', 'oil', 'beans', 'lentils', 'peas', 'chickpea'
    }

    UNHEALTHY_FOODS = {
        'burger', 'fries', 'chips', 'candy', 'soda', 'coke',
        'donut', 'cake', 'cookie', 'ice cream', 'pizza', 'hotdog',
        'fried', 'fries', 'chicken nuggets', 'fast food', 'processed',
        'sugar', 'salt', 'fat', 'saturated', 'trans fat'
    }

    def calculate_nutrition_score(self, foods: list) -> float:
        """Calculate nutrition score (0-1) based on food list."""
        if not foods:
            return 0.5
        
        healthy_count = sum(1 for f in foods if f.lower() in self.HEALTHY_FOODS)
        unhealthy_count = sum(1 for f in foods if f.lower() in self.UNHEALTHY_FOODS)
        
        # Score: 1.0 = all healthy, 0.0 = all unhealthy
        if len(foods) == 0:
            return 0.5
        
        score = (healthy_count - unhealthy_count) / len(foods)
        return max(0.0, min(1.0, (score + 1) / 2))  # Normalize to 0-1

    def classify_meal(self, foods: list) -> str:
        """Classify meal as healthy, moderate, or unhealthy."""
        score = self.calculate_nutrition_score(foods)
        if score >= 0.7:
            return 'Healthy'
        elif score >= 0.4:
            return 'Moderate'
        else:
            return 'Unhealthy'


# Example usage function
def example_nutrition_usage():
    """Example of how to use NutritionTracker."""
    tracker = NutritionTracker()
    
    # Example with a sample image
    # image = Image.open('sample_meal.jpg')
    # result = tracker.analyze_meal(image)
    # print(result)
    
    logger.info("Nutrition tracker initialized and ready to use")


if __name__ == '__main__':
    example_nutrition_usage()
