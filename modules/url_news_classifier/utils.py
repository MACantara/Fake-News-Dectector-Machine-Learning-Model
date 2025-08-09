"""
URL News Classifier Utility Functions
Contains utility functions for URL feature extraction, feedback management, and helper operations
"""

import numpy as np
import json
import re
import os
from urllib.parse import urlparse, parse_qs
from datetime import datetime

def extract_url_features(url):
    """Extract comprehensive features from URL for classification (optimized, no content fetching)"""
    features = {}
    parsed_url = urlparse(url.lower())
    
    # Define patterns
    news_url_patterns = [
        r'/news/', r'/article/', r'/story/', r'/post/', r'/blog/',
        r'/breaking/', r'/latest/', r'/update/', r'/report/', r'/press/',
        r'\d{4}/\d{2}/\d{2}/', r'\d{4}-\d{2}-\d{2}', r'\d{4}/\d{1,2}/',
        r'/\d+/', r'article-\d+', r'story-\d+', r'news-\d+',
        r'/politics/', r'/sports/', r'/business/', r'/tech/', r'/health/',
        r'/entertainment/', r'/opinion/', r'/world/', r'/local/'
    ]
    
    news_domains = {
        'cnn.com', 'bbc.com', 'reuters.com', 'ap.org', 'npr.org',
        'nytimes.com', 'washingtonpost.com', 'theguardian.com',
        'abs-cbn.com', 'gma.tv', 'inquirer.net', 'rappler.com',
        'philstar.com', 'manilabulletin.ph', 'cnnphilippines.com',
        'news.com', 'newsweek.com', 'time.com', 'bloomberg.com',
        'forbes.com', 'wsj.com', 'usatoday.com', 'nbcnews.com'
    }
    
    non_news_patterns = [
        r'/about/', r'/contact/', r'/privacy/', r'/terms/', r'/help/',
        r'/login/', r'/register/', r'/profile/', r'/settings/', r'/account/',
        r'/shop/', r'/buy/', r'/cart/', r'/checkout/', r'/product/',
        r'/search/', r'/category/', r'/tag/', r'/archive/', r'/sitemap/',
        r'/api/', r'/admin/', r'/dashboard/', r'/upload/', r'/download/'
    ]
    
    # Basic URL structure features
    features['domain'] = parsed_url.netloc
    features['path'] = parsed_url.path
    features['path_length'] = len(parsed_url.path)
    features['path_segments'] = len([seg for seg in parsed_url.path.split('/') if seg])
    features['has_query'] = bool(parsed_url.query)
    features['has_fragment'] = bool(parsed_url.fragment)
    features['url_length'] = len(url)
    
    # Domain analysis
    domain_parts = features['domain'].split('.')
    features['domain_length'] = len(features['domain'])
    features['has_subdomain'] = len(domain_parts) > 2
    features['is_known_news_domain'] = any(domain in features['domain'] for domain in news_domains)
    features['domain_has_news_keyword'] = any(keyword in features['domain'] for keyword in ['news', 'press', 'media', 'journal', 'times', 'post', 'herald'])
    
    # Path analysis
    path_lower = features['path'].lower()
    features['path_depth'] = path_lower.count('/')
    features['has_file_extension'] = bool(re.search(r'\.[a-zA-Z0-9]{2,5}$', path_lower))
    features['extension_is_html'] = path_lower.endswith(('.html', '.htm', '.php', '.asp', '.aspx', '.jsp'))
    
    # News pattern matching
    features['has_news_pattern'] = any(re.search(pattern, url.lower()) for pattern in news_url_patterns)
    features['has_non_news_pattern'] = any(re.search(pattern, url.lower()) for pattern in non_news_patterns)
    features['news_pattern_count'] = sum(1 for pattern in news_url_patterns if re.search(pattern, url.lower()))
    
    # Date and time patterns
    features['has_date_pattern'] = bool(re.search(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', url))
    features['has_year'] = bool(re.search(r'/20\d{2}/', url))
    features['has_timestamp'] = bool(re.search(r'\d{10,13}', url))  # Unix timestamp
    
    # Article identifier patterns
    features['has_article_id'] = bool(re.search(r'(article|story|post|news)[-_]?\d+', url.lower()))
    features['has_numeric_id'] = bool(re.search(r'/\d{4,}/', url))
    features['has_slug'] = bool(re.search(r'/[a-z0-9]+-[a-z0-9-]+', url.lower()))
    
    # Query parameter analysis
    if parsed_url.query:
        query_params = parse_qs(parsed_url.query)
        features['query_param_count'] = len(query_params)
        features['has_id_param'] = any(key in query_params for key in ['id', 'article_id', 'post_id', 'story_id'])
        features['has_tracking_params'] = any(key in query_params for key in ['utm_source', 'utm_campaign', 'ref', 'source'])
    else:
        features['query_param_count'] = 0
        features['has_id_param'] = False
        features['has_tracking_params'] = False
    
    # URL pattern scoring
    features['news_score'] = 0
    if features['is_known_news_domain']:
        features['news_score'] += 3
    if features['domain_has_news_keyword']:
        features['news_score'] += 2
    if features['has_news_pattern']:
        features['news_score'] += 2
    if features['has_date_pattern']:
        features['news_score'] += 2
    if features['has_article_id']:
        features['news_score'] += 1
    if features['has_slug']:
        features['news_score'] += 1
    if features['has_non_news_pattern']:
        features['news_score'] -= 2
    
    return features

def create_feature_vector(url):
    """Create a comprehensive feature vector for the URL (optimized, URL-only)"""
    url_features = extract_url_features(url)
    
    # Convert to numerical vector with consistent ordering
    feature_vector = []
    feature_names = []
    
    # Define feature order for consistency
    feature_order = [
        'path_length', 'path_segments', 'has_query', 'has_fragment', 'url_length',
        'domain_length', 'has_subdomain', 'is_known_news_domain', 'domain_has_news_keyword',
        'path_depth', 'has_file_extension', 'extension_is_html',
        'has_news_pattern', 'has_non_news_pattern', 'news_pattern_count',
        'has_date_pattern', 'has_year', 'has_timestamp',
        'has_article_id', 'has_numeric_id', 'has_slug',
        'query_param_count', 'has_id_param', 'has_tracking_params',
        'news_score'
    ]
    
    for key in feature_order:
        value = url_features.get(key, 0)
        if isinstance(value, bool):
            feature_vector.append(1 if value else 0)
        elif isinstance(value, (int, float)):
            feature_vector.append(value)
        else:
            # For string features, use length
            feature_vector.append(len(str(value)))
        feature_names.append(key)
    
    return np.array(feature_vector), feature_names

def heuristic_prediction(url):
    """Fast heuristic prediction when model is not trained (optimized)"""
    url_features = extract_url_features(url)
    
    # Use the pre-calculated news_score from features
    score = url_features.get('news_score', 0)
    
    # Additional scoring refinements
    if url_features.get('has_slug'):
        score += 0.5
    if url_features.get('has_tracking_params'):
        score += 0.5  # News articles often have tracking
    if url_features.get('path_depth', 0) > 5:
        score -= 1  # Very deep paths are less likely to be news
    if url_features.get('url_length', 0) > 200:
        score -= 0.5  # Very long URLs are less likely to be news
    
    # Normalize score to probability (0-10 scale)
    probability = min(max(score / 8, 0), 1)
    prediction = probability > 0.5
    
    # Calculate confidence consistently with trained model
    confidence = max(probability, 1 - probability)
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'is_news_article': prediction,
        'probability_news': probability,
        'probability_not_news': 1 - probability,
        'feature_vector': [],
        'heuristic_based': True,
        'news_score': score
    }

def update_feature_weights(feature_weights, feedback_entry, learning_rate=0.1):
    """Update feature weights based on feedback (RL component)"""
    if not feedback_entry['was_correct']:
        # If prediction was wrong, adjust weights
        reward = -1
    else:
        # If prediction was correct, reinforce weights
        reward = 1
    
    # Apply learning rate and user confidence
    learning_signal = learning_rate * reward * feedback_entry['user_confidence']
    
    # Update weights (simplified approach)
    for feature_name in feature_weights:
        feature_weights[feature_name] += learning_signal * 0.1
        # Keep weights positive
        feature_weights[feature_name] = max(0.1, feature_weights[feature_name])
    
    return feature_weights

def load_feedback_data(feedback_path='url_classifier_feedback.json'):
    """Load existing feedback data"""
    try:
        if os.path.exists(feedback_path):
            with open(feedback_path, 'r') as f:
                return json.load(f)
        return []
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error loading feedback data: {str(e)}")
        return []

def save_feedback_data(feedback_data, feedback_path='url_classifier_feedback.json'):
    """Save feedback data to file"""
    try:
        with open(feedback_path, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving feedback data: {str(e)}")
        return False

def calculate_accuracy_from_feedback(feedback_data):
    """Calculate accuracy from feedback data"""
    if not feedback_data:
        return 0.0
    
    correct_predictions = sum(1 for entry in feedback_data if entry.get('was_correct', False))
    return correct_predictions / len(feedback_data)

def get_recent_feedback(feedback_data, limit=10):
    """Get recent feedback samples"""
    return feedback_data[-limit:] if feedback_data else []

def validate_url(url):
    """Validate if URL is properly formatted"""
    try:
        parsed = urlparse(url)
        return bool(parsed.netloc and parsed.scheme in ['http', 'https'])
    except:
        return False
