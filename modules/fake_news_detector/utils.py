"""
Utility functions for fake news detection
"""

import os
import json
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime


def preprocess_text(text, stemmer=None, stop_words=None):
    """Clean and preprocess text data"""
    if pd.isna(text) or text is None:
        return ""
    
    # Initialize defaults if not provided
    if stemmer is None:
        stemmer = PorterStemmer()
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenize and remove stopwords
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)


def verify_lfs_file(filepath):
    """Verify if a file is properly downloaded from Git LFS"""
    try:
        if not os.path.exists(filepath):
            return False, f"File {filepath} does not exist"
        
        # Check file size - LFS pointer files are very small (~100 bytes)
        file_size = os.path.getsize(filepath)
        if file_size < 1000:  # Less than 1KB might be an LFS pointer
            print(f"Warning: {filepath} is very small ({file_size} bytes), might be an LFS pointer file")
            return False, f"File appears to be an LFS pointer (size: {file_size} bytes)"
        
        # Try to read the beginning of the file to check for LFS pointer content
        with open(filepath, 'rb') as f:
            first_bytes = f.read(100)
            if b'version https://git-lfs.github.com/spec/' in first_bytes:
                return False, "File is a Git LFS pointer, not the actual file"
        
        print(f"✓ {filepath} verified as proper file (size: {file_size} bytes)")
        return True, "File verified successfully"
        
    except Exception as e:
        return False, f"Error verifying file: {str(e)}"


def check_lfs_files():
    """Check if Git LFS files are properly available"""
    print("=== Checking Git LFS Files ===")
    
    lfs_files = [
        'models/fake_news_model.pkl',
        'models/political_news_classifier.pkl', 
        'datasets/WELFake_Dataset.csv',
        'datasets/News_Category_Dataset_v3.json'
    ]
    
    available_files = []
    missing_files = []
    
    for file_path in lfs_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > 1000:  # Larger than typical LFS pointer
                print(f"✓ {file_path} - Available ({file_size:,} bytes)")
                available_files.append(file_path)
            else:
                print(f"⚠ {file_path} - Possibly LFS pointer ({file_size} bytes)")
                missing_files.append(file_path)
        else:
            print(f"✗ {file_path} - Not found")
            missing_files.append(file_path)
    
    print(f"\nSummary: {len(available_files)} available, {len(missing_files)} missing/incomplete")
    
    if missing_files:
        print("\nMissing or incomplete files:", missing_files)
        print("\nThis might indicate:")
        print("1. Git LFS is not properly configured")
        print("2. Files were not pulled during deployment")
        print("3. Repository access issues")
        print("\nThe application will attempt to train models from available data.")
    
    return available_files, missing_files


def load_feedback_data(feedback_file='user_feedback.json'):
    """Load existing feedback data"""
    try:
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            print(f"Loaded {len(feedback_data)} feedback entries")
            return feedback_data
    except Exception as e:
        print(f"Error loading feedback data: {str(e)}")
    return []


def save_feedback_data(feedback_data, feedback_file='user_feedback.json'):
    """Save feedback data to file"""
    try:
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving feedback data: {str(e)}")


def load_pattern_cache(pattern_cache_file='datasets/news_pattern_cache.json'):
    """Load pattern cache for news articles"""
    try:
        if os.path.exists(pattern_cache_file):
            with open(pattern_cache_file, 'r', encoding='utf-8') as f:
                pattern_cache = json.load(f)
            print(f"Loaded {len(pattern_cache)} cached news patterns")
            return pattern_cache
    except Exception as e:
        print(f"Error loading pattern cache: {str(e)}")
    return []


def save_pattern_cache(pattern_cache, pattern_cache_file='datasets/news_pattern_cache.json'):
    """Save pattern cache to file"""
    try:
        with open(pattern_cache_file, 'w', encoding='utf-8') as f:
            json.dump(pattern_cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving pattern cache: {str(e)}")


def get_feedback_stats(feedback_data, retrain_threshold=10):
    """Get statistics about user feedback"""
    if not feedback_data:
        return {
            'total_feedback': 0,
            'used_for_training': 0,
            'pending_training': 0,
            'accuracy_improvement': None
        }
    
    used_count = len([f for f in feedback_data if f.get('used_for_training', False)])
    pending_count = len(feedback_data) - used_count
    
    return {
        'total_feedback': len(feedback_data),
        'used_for_training': used_count,
        'pending_training': pending_count,
        'retrain_threshold': retrain_threshold,
        'needs_retraining': pending_count >= retrain_threshold
    }


def add_pattern_to_cache(pattern_cache, text, label, metadata=None, stemmer=None, stop_words=None):
    """Add a new pattern to the cache"""
    pattern_entry = {
        'timestamp': datetime.now().isoformat(),
        'text': text,
        'label': label,
        'processed_text': preprocess_text(text, stemmer, stop_words),
        'metadata': metadata or {}
    }
    
    pattern_cache.append(pattern_entry)
    
    print(f"Pattern added to cache. Total cached patterns: {len(pattern_cache)}")
    
    return pattern_entry


def get_pattern_cache_stats(pattern_cache, pattern_cache_threshold=5):
    """Get statistics about the pattern cache"""
    if not pattern_cache:
        return {
            'total_patterns': 0,
            'cache_threshold': pattern_cache_threshold,
            'needs_retraining': False
        }
    
    return {
        'total_patterns': len(pattern_cache),
        'cache_threshold': pattern_cache_threshold,
        'needs_retraining': len(pattern_cache) >= pattern_cache_threshold,
        'oldest_pattern': pattern_cache[0]['timestamp'] if pattern_cache else None,
        'newest_pattern': pattern_cache[-1]['timestamp'] if pattern_cache else None
    }


def clear_used_patterns(pattern_cache):
    """Clear patterns that have been used for training"""
    original_count = len(pattern_cache)
    remaining_patterns = [p for p in pattern_cache if not p.get('used_for_training', False)]
    cleared_count = original_count - len(remaining_patterns)
    
    if cleared_count > 0:
        print(f"🧹 Cleared {cleared_count} used patterns from cache")
    
    return remaining_patterns, cleared_count
