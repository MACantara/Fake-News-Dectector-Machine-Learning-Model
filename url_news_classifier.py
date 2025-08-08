"""
URL News Article Classifier using Reinforcement Learning
This module implements a reinforcement learning approach to classify whether a URL is a news article or not.
"""

import numpy as np
import pandas as pd
import json
import pickle
import re
import requests
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from datetime import datetime
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class URLNewsClassifier:
    """
    Reinforcement Learning-based URL News Article Classifier
    Uses user feedback to continuously improve classification accuracy
    """
    
    def __init__(self, model_path='url_news_classifier.pkl', feedback_path='url_classifier_feedback.json'):
        self.model_path = model_path
        self.feedback_path = feedback_path
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.is_trained = False
        
        # Reinforcement Learning Parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.1  # Epsilon for epsilon-greedy exploration
        self.confidence_threshold = 0.7
        
        # Feature weights (updated through RL)
        self.feature_weights = {
            'url_structure': 1.0,
            'domain_credibility': 1.0,
            'content_indicators': 1.0,
            'meta_tags': 1.0,
            'text_features': 1.0
        }
        
        # Feedback storage
        self.feedback_data = []
        self.training_history = []
        
        # Load existing model and feedback
        self.load_feedback()
        self.load_model()
        
        # News-related patterns
        self.news_url_patterns = [
            r'/news/', r'/article/', r'/story/', r'/post/', r'/blog/',
            r'/breaking/', r'/latest/', r'/update/', r'/report/',
            r'\d{4}/\d{2}/\d{2}/', r'\d{4}-\d{2}-\d{2}',
            r'/\d+/', r'article-\d+', r'story-\d+'
        ]
        
        self.news_domains = {
            'cnn.com', 'bbc.com', 'reuters.com', 'ap.org', 'npr.org',
            'nytimes.com', 'washingtonpost.com', 'theguardian.com',
            'abs-cbn.com', 'gma.tv', 'inquirer.net', 'rappler.com',
            'philstar.com', 'manilabulletin.ph', 'cnnphilippines.com'
        }
        
        self.non_news_patterns = [
            r'/about/', r'/contact/', r'/privacy/', r'/terms/',
            r'/login/', r'/register/', r'/profile/', r'/settings/',
            r'/shop/', r'/buy/', r'/cart/', r'/checkout/',
            r'/search/', r'/category/', r'/tag/', r'/archive/'
        ]
    
    def extract_url_features(self, url):
        """Extract features from URL for classification"""
        features = {}
        parsed_url = urlparse(url)
        
        # URL structure features
        features['domain'] = parsed_url.netloc.lower()
        features['path'] = parsed_url.path.lower()
        features['path_length'] = len(parsed_url.path)
        features['path_segments'] = len([seg for seg in parsed_url.path.split('/') if seg])
        features['has_query'] = bool(parsed_url.query)
        features['has_fragment'] = bool(parsed_url.fragment)
        
        # Domain credibility
        features['is_known_news_domain'] = any(domain in features['domain'] for domain in self.news_domains)
        features['has_subdomain'] = len(features['domain'].split('.')) > 2
        features['domain_length'] = len(features['domain'])
        
        # URL pattern matching
        features['has_news_pattern'] = any(re.search(pattern, url) for pattern in self.news_url_patterns)
        features['has_non_news_pattern'] = any(re.search(pattern, url) for pattern in self.non_news_patterns)
        
        # Date patterns
        features['has_date_pattern'] = bool(re.search(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', url))
        features['has_year'] = bool(re.search(r'/20\d{2}/', url))
        
        # Article ID patterns
        features['has_article_id'] = bool(re.search(r'(article|story|post)[-_]?\d+', url))
        features['has_numeric_id'] = bool(re.search(r'/\d{4,}/', url))
        
        return features
    
    def extract_content_features(self, url):
        """Extract content-based features from the webpage"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            features = {}
            
            # Meta tag analysis
            title = soup.find('title')
            features['has_title'] = bool(title)
            features['title_length'] = len(title.get_text()) if title else 0
            
            # Article-specific meta tags
            features['has_article_tag'] = bool(soup.find('article'))
            features['has_time_tag'] = bool(soup.find('time'))
            features['has_author_meta'] = bool(soup.find('meta', attrs={'name': 'author'}))
            features['has_article_meta'] = bool(soup.find('meta', property='article:published_time'))
            
            # Content structure
            paragraphs = soup.find_all('p')
            features['paragraph_count'] = len(paragraphs)
            features['avg_paragraph_length'] = np.mean([len(p.get_text()) for p in paragraphs]) if paragraphs else 0
            
            # News-specific elements
            features['has_byline'] = bool(soup.find(class_=re.compile(r'byline|author')))
            features['has_dateline'] = bool(soup.find(class_=re.compile(r'date|time|published')))
            features['has_headline'] = bool(soup.find(['h1', 'h2'], class_=re.compile(r'headline|title')))
            
            return features
            
        except Exception as e:
            # Return default features if content extraction fails
            return {
                'has_title': False, 'title_length': 0, 'has_article_tag': False,
                'has_time_tag': False, 'has_author_meta': False, 'has_article_meta': False,
                'paragraph_count': 0, 'avg_paragraph_length': 0, 'has_byline': False,
                'has_dateline': False, 'has_headline': False
            }
    
    def create_feature_vector(self, url):
        """Create a comprehensive feature vector for the URL"""
        url_features = self.extract_url_features(url)
        content_features = self.extract_content_features(url)
        
        # Combine all features
        all_features = {**url_features, **content_features}
        
        # Convert to numerical vector
        feature_vector = []
        feature_names = []
        
        for key, value in all_features.items():
            if isinstance(value, bool):
                feature_vector.append(1 if value else 0)
            elif isinstance(value, (int, float)):
                feature_vector.append(value)
            else:
                # For string features, use length
                feature_vector.append(len(str(value)))
            feature_names.append(key)
        
        return np.array(feature_vector), feature_names
    
    def predict_with_confidence(self, url):
        """Predict if URL is a news article with confidence score"""
        if not self.is_trained:
            # Use heuristic-based prediction for untrained model
            return self.heuristic_prediction(url)
        
        feature_vector, _ = self.create_feature_vector(url)
        feature_vector = feature_vector.reshape(1, -1)
        
        # Get prediction and probability
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        confidence = max(probabilities)
        
        return {
            'prediction': bool(prediction),
            'confidence': float(confidence),
            'is_news_article': bool(prediction),
            'probability_news': float(probabilities[1]) if len(probabilities) > 1 else 0.5,
            'probability_not_news': float(probabilities[0]) if len(probabilities) > 1 else 0.5,
            'feature_vector': feature_vector.flatten().tolist()
        }
    
    def heuristic_prediction(self, url):
        """Fallback heuristic prediction when model is not trained"""
        url_features = self.extract_url_features(url)
        
        # Simple scoring based on heuristics
        score = 0
        
        if url_features['is_known_news_domain']:
            score += 3
        if url_features['has_news_pattern']:
            score += 2
        if url_features['has_date_pattern']:
            score += 2
        if url_features['has_article_id']:
            score += 1
        if url_features['has_non_news_pattern']:
            score -= 2
        
        # Normalize score to probability
        probability = min(max(score / 6, 0), 1)
        prediction = probability > 0.5
        
        return {
            'prediction': prediction,
            'confidence': abs(probability - 0.5) * 2,  # Distance from uncertainty
            'is_news_article': prediction,
            'probability_news': probability,
            'probability_not_news': 1 - probability,
            'feature_vector': [],
            'heuristic_based': True
        }
    
    def add_feedback(self, url, predicted_label, actual_label, user_confidence=1.0):
        """Add user feedback for reinforcement learning"""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'url': url,
            'predicted_label': bool(predicted_label),
            'actual_label': bool(actual_label),
            'user_confidence': float(user_confidence),
            'was_correct': bool(predicted_label) == bool(actual_label)
        }
        
        # Get current prediction details
        prediction_result = self.predict_with_confidence(url)
        feedback_entry.update({
            'prediction_confidence': prediction_result['confidence'],
            'feature_vector': prediction_result.get('feature_vector', [])
        })
        
        self.feedback_data.append(feedback_entry)
        self.save_feedback()
        
        # Update feature weights based on feedback
        self.update_feature_weights(feedback_entry)
        
        # Trigger retraining if enough feedback collected
        if len(self.feedback_data) % 5 == 0 and len(self.feedback_data) >= 3:  # Retrain every 5 feedback samples
            self.retrain_model()
        
        return feedback_entry
    
    def update_feature_weights(self, feedback_entry):
        """Update feature weights based on feedback (RL component)"""
        if not feedback_entry['was_correct']:
            # If prediction was wrong, adjust weights
            reward = -1
        else:
            # If prediction was correct, reinforce weights
            reward = 1
        
        # Apply learning rate and user confidence
        learning_signal = self.learning_rate * reward * feedback_entry['user_confidence']
        
        # Update weights (simplified approach)
        for feature_name in self.feature_weights:
            self.feature_weights[feature_name] += learning_signal * 0.1
            # Keep weights positive
            self.feature_weights[feature_name] = max(0.1, self.feature_weights[feature_name])
    
    def retrain_model(self):
        """Retrain the model with accumulated feedback"""
        if len(self.feedback_data) < 3:
            return  # Need minimum samples
        
        print(f"Retraining model with {len(self.feedback_data)} feedback samples...")
        
        # Prepare training data
        X = []
        y = []
        
        for feedback in self.feedback_data:
            if feedback.get('feature_vector'):
                X.append(feedback['feature_vector'])
                y.append(1 if feedback['actual_label'] else 0)
        
        if len(X) < 3:
            return  # Not enough feature vectors
        
        X = np.array(X)
        y = np.array(y)
        
        # Train new model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Evaluate on training data (for logging)
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Log training history
        training_entry = {
            'timestamp': datetime.now().isoformat(),
            'samples_count': len(X),
            'accuracy': float(accuracy),
            'feature_weights': self.feature_weights.copy()
        }
        self.training_history.append(training_entry)
        
        # Save updated model
        self.save_model()
        
        print(f"Model retrained. Accuracy: {accuracy:.3f}")
        return accuracy
    
    def save_model(self):
        """Save the trained model"""
        if self.model:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'feature_weights': self.feature_weights,
                'training_history': self.training_history,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, self.model_path)
    
    def load_model(self):
        """Load existing model if available"""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.vectorizer = model_data.get('vectorizer', self.vectorizer)
            self.feature_weights = model_data.get('feature_weights', self.feature_weights)
            self.training_history = model_data.get('training_history', [])
            self.is_trained = model_data.get('is_trained', False)
            print(f"Loaded existing model with {len(self.training_history)} training iterations")
        except FileNotFoundError:
            print("No existing model found. Starting fresh.")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def save_feedback(self):
        """Save feedback data to file"""
        try:
            with open(self.feedback_path, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
        except Exception as e:
            print(f"Error saving feedback: {e}")
    
    def load_feedback(self):
        """Load existing feedback data"""
        try:
            with open(self.feedback_path, 'r') as f:
                self.feedback_data = json.load(f)
            print(f"Loaded {len(self.feedback_data)} feedback samples")
        except FileNotFoundError:
            self.feedback_data = []
        except Exception as e:
            print(f"Error loading feedback: {e}")
            self.feedback_data = []
    
    def get_model_stats(self):
        """Get model statistics and performance metrics"""
        stats = {
            'is_trained': self.is_trained,
            'feedback_count': len(self.feedback_data),
            'training_iterations': len(self.training_history),
            'feature_weights': self.feature_weights.copy(),
            'last_accuracy': None,
            'correct_predictions': 0,
            'total_predictions': 0
        }
        
        if self.training_history:
            stats['last_accuracy'] = self.training_history[-1]['accuracy']
        
        # Calculate accuracy from feedback
        if self.feedback_data:
            correct = sum(1 for f in self.feedback_data if f['was_correct'])
            total = len(self.feedback_data)
            stats['correct_predictions'] = correct
            stats['total_predictions'] = total
            stats['feedback_accuracy'] = correct / total if total > 0 else 0
        
        return stats
    
    def get_recent_feedback(self, limit=10):
        """Get recent feedback samples"""
        return self.feedback_data[-limit:] if self.feedback_data else []
