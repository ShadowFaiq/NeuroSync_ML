"""
Burnout Risk Prediction Pipeline
Trains ML model on existing datasets matching the 12-feature survey structure
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BurnoutDataPipeline:
    """Pipeline to prepare data and train burnout predictor"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.model = None
        self.feature_names = [
            'age', 'work_hours', 'exercise_days', 'sleep_hours', 'sleep_quality',
            'productivity', 'mental_clarity', 'social_isolation', 'support_system',
            'emotional_exhaustion'
        ]
    
    def load_mental_health_dataset(self):
        """Load and prepare Mental Health in Tech survey"""
        logger.info("\n[1/4] Loading Mental Health in Tech survey...")
        
        try:
            df = pd.read_csv('Mental health in tech survey.csv')
            logger.info(f"✅ Loaded {len(df)} responses")
            
            # Create features from available columns
            processed = pd.DataFrame()
            
            # Demographics
            if 'Age' in df.columns:
                processed['age'] = df['Age']
            
            # Work
            if 'work_interfere' in df.columns:
                # Map work interference to numeric (will be target variable)
                work_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
                processed['work_interfere'] = df['work_interfere'].map(work_map)
            
            # Mental health indicators (can infer some features)
            if 'mental_health_consequence' in df.columns:
                consequence_map = {'No': 0, 'Maybe': 2, 'Yes': 4}
                processed['emotional_exhaustion'] = df['mental_health_consequence'].map(consequence_map)
            
            # Remove nulls
            processed = processed.dropna(subset=['work_interfere'])
            
            logger.info(f"✅ Prepared {len(processed)} valid records")
            return processed
        
        except Exception as e:
            logger.error(f"❌ Error loading mental health data: {e}")
            return None
    
    def load_sleep_lifestyle_dataset(self):
        """Load and prepare Sleep Health and Lifestyle dataset"""
        logger.info("\n[2/4] Loading Sleep Health and Lifestyle dataset...")
        
        try:
            df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
            logger.info(f"✅ Loaded {len(df)} records")
            
            processed = pd.DataFrame()
            
            # Map available columns
            if 'Age' in df.columns:
                processed['age'] = df['Age']
            
            if 'Sleep Duration' in df.columns:
                processed['sleep_hours'] = df['Sleep Duration']
            
            if 'Quality of Sleep' in df.columns:
                processed['sleep_quality'] = df['Quality of Sleep']
            
            if 'Physical Activity Level' in df.columns:
                processed['exercise_days'] = df['Physical Activity Level']
            
            if 'Stress Level' in df.columns:
                processed['stress_level'] = df['Stress Level']
            
            # Create target: high stress = high burnout risk
            if 'Stress Level' in df.columns:
                processed['work_interfere'] = (df['Stress Level'] >= 7).astype(int) * 3
            
            processed = processed.dropna(subset=['work_interfere'])
            
            logger.info(f"✅ Prepared {len(processed)} valid records")
            return processed
        
        except Exception as e:
            logger.error(f"❌ Error loading sleep data: {e}")
            return None
    
    def load_nova_pulse_dataset(self):
        """Load and prepare NOVA PULSE Integrated Dataset"""
        logger.info("\n[3/4] Loading NOVA PULSE dataset...")
        
        try:
            df = pd.read_csv('NOVA_PULSE_Integrated_Dataset.csv')
            logger.info(f"✅ Loaded {len(df)} records")
            
            processed = pd.DataFrame()
            
            # Map available columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            logger.info(f"Available numeric columns: {list(numeric_cols)[:10]}")
            
            # Try to map columns based on names
            for col in numeric_cols:
                col_lower = col.lower()
                if 'age' in col_lower:
                    processed['age'] = df[col]
                elif 'sleep' in col_lower and 'hour' in col_lower:
                    processed['sleep_hours'] = df[col]
                elif 'stress' in col_lower:
                    processed['stress_level'] = df[col]
                elif 'product' in col_lower or 'perform' in col_lower:
                    processed['productivity'] = df[col]
            
            # Create target if stress available
            if 'stress_level' in processed.columns:
                processed['work_interfere'] = (processed['stress_level'] >= 6).astype(int) * 3
            
            processed = processed.dropna(subset=['work_interfere'])
            
            if len(processed) > 0:
                logger.info(f"✅ Prepared {len(processed)} valid records")
                return processed
            else:
                logger.warning("⚠️ No valid records after processing")
                return None
        
        except Exception as e:
            logger.error(f"❌ Error loading NOVA PULSE data: {e}")
            return None
    
    def combine_datasets(self):
        """Load and combine all datasets"""
        logger.info("\n" + "="*70)
        logger.info("BURNOUT PREDICTION - DATA PREPARATION PIPELINE")
        logger.info("="*70)
        
        datasets = []
        
        # Load each dataset
        mh_data = self.load_mental_health_dataset()
        if mh_data is not None and len(mh_data) > 0:
            datasets.append(mh_data)
        
        sleep_data = self.load_sleep_lifestyle_dataset()
        if sleep_data is not None and len(sleep_data) > 0:
            datasets.append(sleep_data)
        
        nova_data = self.load_nova_pulse_dataset()
        if nova_data is not None and len(nova_data) > 0:
            datasets.append(nova_data)
        
        if not datasets:
            logger.error("❌ No datasets could be loaded!")
            return None
        
        # Combine datasets
        logger.info(f"\n✅ Combining {len(datasets)} datasets...")
        combined = pd.concat(datasets, ignore_index=True, sort=False)
        combined = combined.dropna(subset=['work_interfere'])
        
        logger.info(f"✅ Combined dataset: {len(combined)} records, {len(combined.columns)} columns")
        logger.info(f"Feature columns: {list(combined.columns)}")
        
        return combined
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        logger.info("\n[4/4] Preparing features for training...")
        
        # Fill missing values
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        # Separate features and target
        target_col = 'work_interfere'
        if target_col not in df.columns:
            logger.error(f"❌ Target column '{target_col}' not found!")
            return None, None, None
        
        y = df[target_col].astype(int)
        
        # Select only numeric features for X
        feature_cols = [col for col in df.columns 
                       if col != target_col and df[col].dtype in [np.float64, np.int64]]
        
        if len(feature_cols) == 0:
            logger.error("❌ No numeric features found!")
            return None, None, None
        
        X = df[feature_cols].copy()
        
        logger.info(f"✅ Features selected: {len(feature_cols)}")
        logger.info(f"Features: {feature_cols}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_cols
    
    def train_models(self, X, y, feature_cols):
        """Train Random Forest and Gradient Boosting models"""
        logger.info("\n" + "="*70)
        logger.info("MODEL TRAINING")
        logger.info("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
        
        # Train Random Forest
        logger.info("\n[1/2] Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_score = rf_model.score(X_test, y_test)
        logger.info(f"✅ Random Forest Accuracy: {rf_score:.3f}")
        
        # Train Gradient Boosting
        logger.info("\n[2/2] Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        gb_score = gb_model.score(X_test, y_test)
        logger.info(f"✅ Gradient Boosting Accuracy: {gb_score:.3f}")
        
        # Use best model
        if gb_score >= rf_score:
            logger.info("\n🏆 Gradient Boosting selected (better accuracy)")
            self.model = gb_model
            best_score = gb_score
        else:
            logger.info("\n🏆 Random Forest selected (better accuracy)")
            self.model = rf_model
            best_score = rf_score
        
        # Feature importance
        logger.info("\n📊 Feature Importance:")
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for idx, row in importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Detailed evaluation
        logger.info("\n" + "="*70)
        logger.info("MODEL EVALUATION")
        logger.info("="*70)
        
        y_pred = self.model.predict(X_test)
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        # Save model
        model_path = 'models/burnout_predictor.pkl'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, 'models/burnout_scaler.pkl')
        joblib.dump(feature_cols, 'models/burnout_features.pkl')
        
        logger.info(f"\n✅ Model saved to {model_path}")
        
        return best_score
    
    def run(self):
        """Execute full pipeline"""
        # Combine data
        data = self.combine_datasets()
        if data is None or len(data) == 0:
            logger.error("❌ Failed to prepare data!")
            return False
        
        # Prepare features
        X, y, feature_cols = self.prepare_features(data)
        if X is None:
            logger.error("❌ Failed to prepare features!")
            return False
        
        # Train models
        score = self.train_models(X, y, feature_cols)
        
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nModel saved and ready for production!")
        logger.info(f"Final Accuracy: {score:.3f}")
        logger.info("\nNext: Users fill survey → Model predicts burnout risk!")
        
        return True


if __name__ == "__main__":
    pipeline = BurnoutDataPipeline()
    pipeline.run()
