"""
Utility functions for Neuro Sync project.
Provides helper functions for data processing, file handling, and common operations.
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path='.env'):
    """Load environment configuration from .env file."""
    load_dotenv(config_path)
    config = {
        'firebase_credentials': os.getenv('FIREBASE_CREDENTIALS_PATH'),
        'google_sheets_credentials': os.getenv('GOOGLE_SHEETS_CREDENTIALS_PATH'),
        'model_path': os.getenv('MODEL_PATH', 'models/'),
        'data_path': os.getenv('DATA_PATH', 'data/'),
        'debug_mode': os.getenv('DEBUG_MODE', 'False') == 'True',
    }
    return config


def ensure_directory(path):
    """Ensure a directory exists; create it if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory ensured: {path}")


def load_json(file_path):
    """Load JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        return None


def save_json(data, file_path):
    """Save data as JSON file."""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")


def normalize_image(image_array, target_size=(128, 128)):
    """Normalize and resize image array for CNN input."""
    from PIL import Image
    import numpy as np
    
    # Convert to PIL Image if numpy array
    if isinstance(image_array, np.ndarray):
        image = Image.fromarray(image_array.astype('uint8'))
    else:
        image = image_array
    
    # Resize to target size
    image = image.resize(target_size)
    
    # Normalize pixel values to [0, 1]
    image_array = np.array(image) / 255.0
    
    return image_array


def get_sentiment_label(sentiment_score):
    """Convert sentiment score to human-readable label."""
    if sentiment_score > 0.1:
        return "Positive"
    elif sentiment_score < -0.1:
        return "Negative"
    else:
        return "Neutral"


def calculate_bmi(weight_kg, height_m):
    """Calculate Body Mass Index."""
    if height_m <= 0:
        return None
    return weight_kg / (height_m ** 2)


def log_user_action(action, details=None):
    """Log user action for debugging and analytics."""
    log_data = {
        'action': action,
        'details': details,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    logger.info(f"User Action: {json.dumps(log_data)}")


# Import pandas for timestamp
import pandas as pd
