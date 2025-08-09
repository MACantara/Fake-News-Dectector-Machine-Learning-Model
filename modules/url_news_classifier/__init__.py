"""
URL News Classifier Module
Modular implementation for URL-based news article classification using reinforcement learning
"""

from .classifier import URLNewsClassifier
from .utils import (
    extract_url_features,
    create_feature_vector,
    heuristic_prediction,
    update_feature_weights,
    load_feedback_data,
    save_feedback_data
)

__all__ = [
    'URLNewsClassifier',
    'extract_url_features',
    'create_feature_vector', 
    'heuristic_prediction',
    'update_feature_weights',
    'load_feedback_data',
    'save_feedback_data'
]
