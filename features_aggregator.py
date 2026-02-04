"""
Feature Aggregator for Burnout Model
Builds feature vectors aligned with processing/burnout.py FEATURE_NAMES
from available datasets (sleep health, mental health, NOVA PULSE).
"""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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

DEFAULTS = {
    'healthy_food_ratio': 0.5,
    'junk_food_ratio': 0.2,
    'average_sentiment': 0.0,
    'sentiment_variability': 0.5,
    'sleep_consistency': 0.5,
    'sleep_duration': 7.0,
    'productivity_score': 5.0,
    'fitness_activity_level': 0.5,
    'workout_frequency': 2.0,
    'stress_level': 5.0,
    'workload_rating': 5.0,
    'work_hours': 8.0,
}


def _fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    for name in FEATURE_NAMES:
        if name not in df.columns:
            df[name] = DEFAULTS[name]
        else:
            df[name] = df[name].fillna(DEFAULTS[name])
    return df[FEATURE_NAMES]


def aggregate_from_sleep_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Aggregate features and target from Sleep Health and Lifestyle dataset.
    
    Expected columns (best-effort mapping):
    - 'Sleep Duration' → sleep_duration (hours)
    - 'Quality of Sleep' → sleep_consistency (scaled 0-1)
    - 'Physical Activity Level' → fitness_activity_level (scaled 0-1)
    - 'Stress Level' → stress_level (1-10)
    
    Target mapping (ordinal 0..3 from Stress Level):
    - 1-3 → 0 (Low)
    - 4-5 → 1 (Moderate)
    - 6-7 → 2 (High)
    - 8-10 → 3 (Severe)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int)

    features = pd.DataFrame()

    # Sleep
    if 'Sleep Duration' in df.columns:
        features['sleep_duration'] = pd.to_numeric(df['Sleep Duration'], errors='coerce')
    
    if 'Quality of Sleep' in df.columns:
        q = pd.to_numeric(df['Quality of Sleep'], errors='coerce')
        # Scale quality (1-10) to consistency (0-1)
        features['sleep_consistency'] = (q.clip(lower=0, upper=10)) / 10.0
        # Use quality also as a proxy for productivity if missing
        features['productivity_score'] = q

    # Fitness
    if 'Physical Activity Level' in df.columns:
        pal = pd.to_numeric(df['Physical Activity Level'], errors='coerce')
        # Normalize PAL to 0..1 by dividing by max (avoid divide-by-zero)
        max_pal = pal.max() if pal.max() and pal.max() > 0 else 1.0
        features['fitness_activity_level'] = (pal / max_pal).clip(lower=0, upper=1)
        # Derive workout_frequency (approximate 0..7 scale)
        features['workout_frequency'] = (features['fitness_activity_level'] * 7.0).clip(lower=0, upper=7)

    # Stress
    if 'Stress Level' in df.columns:
        sl = pd.to_numeric(df['Stress Level'], errors='coerce')
        features['stress_level'] = sl
        # Workload rating proxy if not present
        features['workload_rating'] = sl

    # Work hours not available → default
    # Nutrition & sentiment not available → defaults
    features = _fill_defaults(features)

    # Target from Stress Level → ordinal 0..3
    if 'Stress Level' in df.columns:
        bins = [0, 3, 5, 7, 10]
        labels = [0, 1, 2, 3]
        y = pd.cut(pd.to_numeric(df['Stress Level'], errors='coerce'), bins=bins, labels=labels, include_lowest=True)
        y = y.astype('Int64')
    else:
        y = pd.Series(dtype='Int64')

    # Drop rows with NaN target
    valid_mask = ~y.isna()
    return features[valid_mask], y[valid_mask].astype(int)


def aggregate_from_mental_health_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Aggregate features and target from Mental Health in Tech survey.
    
    Target mapping:
    - 'work_interfere' → {Never:0, Rarely:1, Sometimes:2, Often:3}
    
    Features: use defaults, fill what we can if present.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int)

    features = pd.DataFrame()

    # Work hours: if any approximate field exists
    for col in df.columns:
        if 'work' in col.lower() and 'hour' in col.lower():
            features['work_hours'] = pd.to_numeric(df[col], errors='coerce')
            break
    
    # Stress-like proxies: avoid leaking target, use defaults otherwise
    # Productivity proxy: none → default

    features = _fill_defaults(features)

    # Target
    y = pd.Series(dtype='Int64')
    if 'work_interfere' in df.columns:
        map_interfere = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
        y = df['work_interfere'].map(map_interfere)
        y = y.astype('Int64')

    valid_mask = ~y.isna()
    return features[valid_mask], y[valid_mask].astype(int)


def aggregate_from_nova_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Aggregate features from NOVA PULSE dataset if available.
    Attempts heuristic mapping based on column names.
    Target: derive from stress if available (ordinal 0..3).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int)

    features = pd.DataFrame()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Heuristic mappings
    for col in numeric_cols:
        lc = col.lower()
        series = pd.to_numeric(df[col], errors='coerce')
        if 'sleep' in lc and ('hour' in lc or 'duration' in lc):
            features['sleep_duration'] = series
        elif 'sleep' in lc and ('quality' in lc or 'consistency' in lc):
            features['sleep_consistency'] = (series.clip(lower=0, upper=10)) / 10.0
        elif 'stress' in lc:
            features['stress_level'] = series
            features['workload_rating'] = series
        elif 'product' in lc or 'perform' in lc:
            features['productivity_score'] = series
        elif 'fitness' in lc or 'activity' in lc or 'exercise' in lc:
            features['fitness_activity_level'] = series
            max_val = series.max() if series.max() and series.max() > 0 else 1.0
            features['fitness_activity_level'] = (series / max_val).clip(lower=0, upper=1)
            features['workout_frequency'] = (features['fitness_activity_level'] * 7.0).clip(lower=0, upper=7)
        elif 'work' in lc and 'hour' in lc:
            features['work_hours'] = series

    features = _fill_defaults(features)

    # Target from stress if available
    y = pd.Series(dtype='Int64')
    if 'stress_level' in features.columns and 'stress_level' in df.columns:
        bins = [0, 3, 5, 7, 10]
        labels = [0, 1, 2, 3]
        y = pd.cut(pd.to_numeric(df[df.columns[[c.lower()=="stress_level" for c in df.columns]][0]], errors='coerce'), bins=bins, labels=labels, include_lowest=True)
        y = y.astype('Int64')

    valid_mask = ~y.isna()
    return features[valid_mask], y[valid_mask].astype(int)


def combine_aggregations(*datasets: Tuple[pd.DataFrame, pd.Series]) -> Tuple[pd.DataFrame, pd.Series]:
    """Combine multiple feature/target datasets into one."""
    Xs = []
    ys = []
    for X, y in datasets:
        if X is not None and y is not None and len(X) > 0 and len(y) > 0:
            # Ensure correct column order
            Xs.append(X[FEATURE_NAMES])
            ys.append(y)
    if not Xs:
        return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int)
    X = pd.concat(Xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    return X, y
