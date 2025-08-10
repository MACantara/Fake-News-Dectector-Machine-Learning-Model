"""
URL News Classifier Utility Functions
Contains utility functions for URL feature extraction, feedback management, and helper operations
"""

import numpy as np
import json
import re
import os
import sqlite3
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

def initialize_feedback_db(db_path='datasets/url_classifier_feedback.db'):
    """Initialize SQLite database for feedback storage"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create feedback table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                url TEXT NOT NULL,
                predicted_label BOOLEAN NOT NULL,
                actual_label BOOLEAN NOT NULL,
                user_confidence REAL NOT NULL,
                was_correct BOOLEAN NOT NULL,
                prediction_confidence REAL,
                feature_vector TEXT
            )
        ''')
        
        # Create index on timestamp for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)
        ''')
        
        # Create index on url for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_url ON feedback(url)
        ''')
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error initializing feedback database: {str(e)}")
        return False

def save_feedback_to_db(feedback_data, db_path='datasets/url_classifier_feedback.db'):
    """Save feedback data to SQLite database"""
    try:
        # Initialize database if it doesn't exist
        initialize_feedback_db(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Insert new feedback entries
        for entry in feedback_data:
            # Convert feature vector to JSON string
            feature_vector_str = json.dumps(entry.get('feature_vector', []))
            
            cursor.execute('''
                INSERT OR REPLACE INTO feedback 
                (timestamp, url, predicted_label, actual_label, user_confidence, 
                 was_correct, prediction_confidence, feature_vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.get('timestamp'),
                entry.get('url'),
                entry.get('predicted_label'),
                entry.get('actual_label'),
                entry.get('user_confidence'),
                entry.get('was_correct'),
                entry.get('prediction_confidence'),
                feature_vector_str
            ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving feedback to database: {str(e)}")
        return False

def load_feedback_from_db(db_path='datasets/url_classifier_feedback.db'):
    """Load feedback data from SQLite database"""
    try:
        if not os.path.exists(db_path):
            # If database doesn't exist, initialize it and return empty list
            initialize_feedback_db(db_path)
            return []
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, url, predicted_label, actual_label, user_confidence,
                   was_correct, prediction_confidence, feature_vector
            FROM feedback
            ORDER BY timestamp
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        feedback_data = []
        for row in rows:
            # Parse feature vector from JSON string
            try:
                feature_vector = json.loads(row[7]) if row[7] else []
            except (json.JSONDecodeError, TypeError):
                feature_vector = []
            
            feedback_entry = {
                'timestamp': row[0],
                'url': row[1],
                'predicted_label': bool(row[2]),
                'actual_label': bool(row[3]),
                'user_confidence': float(row[4]),
                'was_correct': bool(row[5]),
                'prediction_confidence': float(row[6]) if row[6] is not None else 0.5,
                'feature_vector': feature_vector
            }
            feedback_data.append(feedback_entry)
        
        return feedback_data
    except Exception as e:
        print(f"Error loading feedback from database: {str(e)}")
        return []

def add_single_feedback_to_db(feedback_entry, db_path='datasets/url_classifier_feedback.db'):
    """Add a single feedback entry to the database"""
    try:
        # Initialize database if it doesn't exist
        initialize_feedback_db(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Convert feature vector to JSON string
        feature_vector_str = json.dumps(feedback_entry.get('feature_vector', []))
        
        cursor.execute('''
            INSERT INTO feedback 
            (timestamp, url, predicted_label, actual_label, user_confidence, 
             was_correct, prediction_confidence, feature_vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_entry.get('timestamp'),
            feedback_entry.get('url'),
            feedback_entry.get('predicted_label'),
            feedback_entry.get('actual_label'),
            feedback_entry.get('user_confidence'),
            feedback_entry.get('was_correct'),
            feedback_entry.get('prediction_confidence'),
            feature_vector_str
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error adding feedback to database: {str(e)}")
        return False

def get_recent_feedback_from_db(limit=10, db_path='datasets/url_classifier_feedback.db'):
    """Get recent feedback samples from database"""
    try:
        if not os.path.exists(db_path):
            return []
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, url, predicted_label, actual_label, user_confidence,
                   was_correct, prediction_confidence, feature_vector
            FROM feedback
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        feedback_data = []
        for row in rows:
            # Parse feature vector from JSON string
            try:
                feature_vector = json.loads(row[7]) if row[7] else []
            except (json.JSONDecodeError, TypeError):
                feature_vector = []
            
            feedback_entry = {
                'timestamp': row[0],
                'url': row[1],
                'predicted_label': bool(row[2]),
                'actual_label': bool(row[3]),
                'user_confidence': float(row[4]),
                'was_correct': bool(row[5]),
                'prediction_confidence': float(row[6]) if row[6] is not None else 0.5,
                'feature_vector': feature_vector
            }
            feedback_data.append(feedback_entry)
        
        return feedback_data
    except Exception as e:
        print(f"Error getting recent feedback from database: {str(e)}")
        return []

def get_feedback_count_from_db(db_path='datasets/url_classifier_feedback.db'):
    """Get total count of feedback entries in database"""
    try:
        if not os.path.exists(db_path):
            return 0
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM feedback')
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    except Exception as e:
        print(f"Error getting feedback count from database: {str(e)}")
        return 0

def migrate_json_to_db(json_path='datasets/url_classifier_feedback.json', db_path='datasets/url_classifier_feedback.db'):
    """Migrate existing JSON feedback data to SQLite database"""
    try:
        # Load existing JSON data
        json_data = load_feedback_data(json_path)
        
        if not json_data:
            print("No JSON data to migrate")
            return True
        
        # Initialize database
        initialize_feedback_db(db_path)
        
        # Save to database
        result = save_feedback_to_db(json_data, db_path)
        
        if result:
            print(f"Successfully migrated {len(json_data)} feedback entries from JSON to SQLite")
            # Optionally backup and remove JSON file
            backup_path = json_path + '.backup'
            if os.path.exists(json_path):
                os.rename(json_path, backup_path)
                print(f"JSON file backed up to {backup_path}")
        
        return result
    except Exception as e:
        print(f"Error migrating JSON to database: {str(e)}")
        return False

def calculate_accuracy_from_feedback(feedback_data):
    """Calculate accuracy from feedback data"""
    if not feedback_data:
        return 0.0
    
    correct_predictions = sum(1 for entry in feedback_data if entry.get('was_correct', False))
    return correct_predictions / len(feedback_data)

def get_recent_feedback(feedback_data, limit=10):
    """Get recent feedback samples (legacy function - use get_recent_feedback_from_db for database)"""
    return feedback_data[-limit:] if feedback_data else []

def load_feedback_data(feedback_path='datasets/url_classifier_feedback.json'):
    """Load existing feedback data from JSON (legacy function)"""
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

def save_feedback_data(feedback_data, feedback_path='datasets/url_classifier_feedback.json'):
    """Save feedback data to JSON file (legacy function)"""
    try:
        with open(feedback_path, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving feedback data: {str(e)}")
        return False

def validate_url(url):
    """Validate if URL is properly formatted"""
    try:
        parsed = urlparse(url)
        return bool(parsed.netloc and parsed.scheme in ['http', 'https'])
    except:
        return False
