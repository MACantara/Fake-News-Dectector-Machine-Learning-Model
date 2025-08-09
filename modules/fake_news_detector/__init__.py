"""
Fake News Detector Module
Provides fake news detection capabilities using machine learning
"""

from .detector import FakeNewsDetector
from .utils import (
    preprocess_text,
    verify_lfs_file,
    check_lfs_files,
    get_feedback_stats,
    load_feedback_data,
    save_feedback_data,
    load_pattern_cache,
    save_pattern_cache
)

__all__ = [
    'FakeNewsDetector',
    'preprocess_text',
    'verify_lfs_file', 
    'check_lfs_files',
    'get_feedback_stats',
    'load_feedback_data',
    'save_feedback_data',
    'load_pattern_cache',
    'save_pattern_cache'
]
