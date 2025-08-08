"""
URL News Article Classifier using Reinforcement Learning
This module implements a reinforcement learning approach to classify whether a URL is a news article or not.
"""

import numpy as np
import pandas as pd
import json
import re
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
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
        
        # Parallel processing parameters
        self.max_workers = 4  # Number of parallel workers
        self.batch_size = 50  # Default batch size for processing
        
        # Feature weights (updated through RL) - URL-only features
        self.feature_weights = {
            'url_structure': 1.0,
            'domain_credibility': 1.0,
            'path_patterns': 1.0,
            'url_length': 1.0,
            'special_indicators': 1.0
        }
        
        # Feedback storage
        self.feedback_data = []
        self.training_history = []
        
        # Load existing model and feedback
        self.load_feedback()
        self.load_model()
        
        # News-related patterns (optimized for URL-only analysis)
        self.news_url_patterns = [
            r'/news/', r'/article/', r'/story/', r'/post/', r'/blog/',
            r'/breaking/', r'/latest/', r'/update/', r'/report/', r'/press/',
            r'\d{4}/\d{2}/\d{2}/', r'\d{4}-\d{2}-\d{2}', r'\d{4}/\d{1,2}/',
            r'/\d+/', r'article-\d+', r'story-\d+', r'news-\d+',
            r'/politics/', r'/sports/', r'/business/', r'/tech/', r'/health/',
            r'/entertainment/', r'/opinion/', r'/world/', r'/local/'
        ]
        
        self.news_domains = {
            'cnn.com', 'bbc.com', 'reuters.com', 'ap.org', 'npr.org',
            'nytimes.com', 'washingtonpost.com', 'theguardian.com',
            'abs-cbn.com', 'gma.tv', 'inquirer.net', 'rappler.com',
            'philstar.com', 'manilabulletin.ph', 'cnnphilippines.com',
            'news.com', 'newsweek.com', 'time.com', 'bloomberg.com',
            'forbes.com', 'wsj.com', 'usatoday.com', 'nbcnews.com'
        }
        
        self.non_news_patterns = [
            r'/about/', r'/contact/', r'/privacy/', r'/terms/', r'/help/',
            r'/login/', r'/register/', r'/profile/', r'/settings/', r'/account/',
            r'/shop/', r'/buy/', r'/cart/', r'/checkout/', r'/product/',
            r'/search/', r'/category/', r'/tag/', r'/archive/', r'/sitemap/',
            r'/api/', r'/admin/', r'/dashboard/', r'/upload/', r'/download/'
        ]
    
    def extract_url_features(self, url):
        """Extract comprehensive features from URL for classification (optimized, no content fetching)"""
        features = {}
        parsed_url = urlparse(url.lower())
        
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
        features['is_known_news_domain'] = any(domain in features['domain'] for domain in self.news_domains)
        features['domain_has_news_keyword'] = any(keyword in features['domain'] for keyword in ['news', 'press', 'media', 'journal', 'times', 'post', 'herald'])
        
        # Path analysis
        path_lower = features['path'].lower()
        features['path_depth'] = path_lower.count('/')
        features['has_file_extension'] = bool(re.search(r'\.[a-zA-Z0-9]{2,5}$', path_lower))
        features['extension_is_html'] = path_lower.endswith(('.html', '.htm', '.php', '.asp', '.aspx', '.jsp'))
        
        # News pattern matching
        features['has_news_pattern'] = any(re.search(pattern, url.lower()) for pattern in self.news_url_patterns)
        features['has_non_news_pattern'] = any(re.search(pattern, url.lower()) for pattern in self.non_news_patterns)
        features['news_pattern_count'] = sum(1 for pattern in self.news_url_patterns if re.search(pattern, url.lower()))
        
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
    
    def create_feature_vector(self, url):
        """Create a comprehensive feature vector for the URL (optimized, URL-only)"""
        url_features = self.extract_url_features(url)
        
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
        
        # Calculate confidence as the maximum probability (consistent with heuristic)
        confidence = max(probabilities)
        
        return {
            'prediction': bool(prediction),
            'confidence': float(confidence),
            'is_news_article': bool(prediction),
            'probability_news': float(probabilities[1]) if len(probabilities) > 1 else 0.5,
            'probability_not_news': float(probabilities[0]) if len(probabilities) > 1 else 0.5,
            'feature_vector': feature_vector.flatten().tolist()
        }
    
    def predict_batch(self, urls, use_parallel=True):
        """Predict multiple URLs efficiently with batch processing"""
        if not urls:
            return []
        
        if use_parallel and len(urls) > 10:
            return self._predict_batch_parallel(urls)
        else:
            return self._predict_batch_sequential(urls)
    
    def _predict_batch_sequential(self, urls):
        """Sequential batch prediction"""
        results = []
        for url in urls:
            try:
                result = self.predict_with_confidence(url)
                result['url'] = url
                results.append(result)
            except Exception as e:
                results.append({
                    'url': url,
                    'error': str(e),
                    'prediction': False,
                    'confidence': 0.0,
                    'is_news_article': False
                })
        return results
    
    def _predict_batch_parallel(self, urls):
        """Parallel batch prediction using ThreadPoolExecutor"""
        def predict_single(url):
            try:
                result = self.predict_with_confidence(url)
                result['url'] = url
                return result
            except Exception as e:
                return {
                    'url': url,
                    'error': str(e),
                    'prediction': False,
                    'confidence': 0.0,
                    'is_news_article': False
                }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(predict_single, urls))
        
        return results
    
    def heuristic_prediction(self, url):
        """Fast heuristic prediction when model is not trained (optimized)"""
        url_features = self.extract_url_features(url)
        
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
    
    def add_feedback(self, url, predicted_label, actual_label, user_confidence=1.0):
        """Add user feedback for reinforcement learning (optimized)"""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'url': url,
            'predicted_label': bool(predicted_label),
            'actual_label': bool(actual_label),
            'user_confidence': float(user_confidence),
            'was_correct': bool(predicted_label) == bool(actual_label)
        }
        
        # Get current prediction details (fast, no content fetching)
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
        if len(self.feedback_data) % 10 == 0 and len(self.feedback_data) >= 10:  # Retrain every 10 feedback samples
            self.retrain_model()
        
        return feedback_entry
    
    def add_batch_feedback(self, feedback_list):
        """Add multiple feedback entries efficiently"""
        for feedback in feedback_list:
            url = feedback['url']
            predicted_label = feedback['predicted_label']
            actual_label = feedback['actual_label']
            user_confidence = feedback.get('user_confidence', 1.0)
            
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'url': url,
                'predicted_label': bool(predicted_label),
                'actual_label': bool(actual_label),
                'user_confidence': float(user_confidence),
                'was_correct': bool(predicted_label) == bool(actual_label)
            }
            
            # For batch processing, generate feature vector only if needed
            if not self.is_trained:
                feedback_entry['feature_vector'] = []
                feedback_entry['prediction_confidence'] = 0.5
            else:
                prediction_result = self.predict_with_confidence(url)
                feedback_entry.update({
                    'prediction_confidence': prediction_result['confidence'],
                    'feature_vector': prediction_result.get('feature_vector', [])
                })
            
            self.feedback_data.append(feedback_entry)
            self.update_feature_weights(feedback_entry)
        
        self.save_feedback()
        
        # Check if retraining is needed
        if len(self.feedback_data) >= 10 and len(self.feedback_data) % 10 == 0:
            self.retrain_model()
        
        return len(feedback_list)
    
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
        """Retrain the model with accumulated feedback (optimized with parallel processing)"""
        if len(self.feedback_data) < 10:
            return  # Need minimum samples
        
        print(f"Retraining model with {len(self.feedback_data)} feedback samples...")
        
        # Prepare training data with parallel feature generation
        feedback_to_process = []
        X = []
        y = []
        
        for feedback in self.feedback_data:
            feature_vector = feedback.get('feature_vector')
            
            # If feature_vector is empty or missing, collect URLs for batch processing
            if not feature_vector or len(feature_vector) == 0:
                feedback_to_process.append(feedback)
            else:
                X.append(feature_vector)
                y.append(1 if feedback['actual_label'] else 0)
        
        # Process missing feature vectors in parallel
        if feedback_to_process:
            urls_to_process = [f['url'] for f in feedback_to_process]
            
            def generate_feature_vector(url):
                try:
                    feature_vector, _ = self.create_feature_vector(url)
                    return feature_vector.tolist()
                except Exception as e:
                    print(f"Error creating feature vector for {url}: {e}")
                    return None
            
            if len(urls_to_process) > 10:
                # Use parallel processing for large batches
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    feature_vectors = list(executor.map(generate_feature_vector, urls_to_process))
            else:
                # Sequential processing for small batches
                feature_vectors = [generate_feature_vector(url) for url in urls_to_process]
            
            # Add successfully generated feature vectors
            for i, feature_vector in enumerate(feature_vectors):
                if feature_vector is not None:
                    X.append(feature_vector)
                    y.append(1 if feedback_to_process[i]['actual_label'] else 0)
        
        if len(X) < 10:
            print(f"Not enough valid feature vectors ({len(X)}) to train model")
            return  # Not enough feature vectors
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Training with {len(X)} feature vectors...")
        
        # Train optimized model
        self.model = RandomForestClassifier(
            n_estimators=50,  # Reduced for faster training
            max_depth=10,     # Limit depth to prevent overfitting
            random_state=42,
            class_weight='balanced',
            n_jobs=-1         # Use all available cores
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
            'feature_weights': self.feature_weights.copy(),
            'model_params': {
                'n_estimators': 50,
                'max_depth': 10,
                'feature_count': X.shape[1] if X.ndim > 1 else len(X)
            }
        }
        self.training_history.append(training_entry)
        
        # Save updated model
        self.save_model()
        
        print(f"Model retrained successfully! Accuracy: {accuracy:.3f}")
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
    
    def analyze_urls_batch(self, urls, confidence_threshold=0.7):
        """Analyze a batch of URLs and return categorized results"""
        results = self.predict_batch(urls)
        
        analysis = {
            'total_urls': len(urls),
            'news_articles': [],
            'non_news': [],
            'high_confidence': [],
            'low_confidence': [],
            'errors': [],
            'summary': {
                'total_urls': len(urls),
                'news_count': 0,
                'non_news_count': 0,
                'high_confidence_count': 0,
                'low_confidence_count': 0,
                'error_count': 0,
                'avg_confidence': 0.0
            }
        }
        
        confidences = []
        
        for result in results:
            if 'error' in result:
                analysis['errors'].append(result)
                analysis['summary']['error_count'] += 1
                continue
            
            confidence = result.get('confidence', 0.0)
            confidences.append(confidence)
            
            if result.get('is_news_article', False):
                analysis['news_articles'].append(result)
                analysis['summary']['news_count'] += 1
            else:
                analysis['non_news'].append(result)
                analysis['summary']['non_news_count'] += 1
            
            if confidence >= confidence_threshold:
                analysis['high_confidence'].append(result)
                analysis['summary']['high_confidence_count'] += 1
            else:
                analysis['low_confidence'].append(result)
                analysis['summary']['low_confidence_count'] += 1
        
        if confidences:
            analysis['summary']['avg_confidence'] = np.mean(confidences)
        
        return analysis
    
    def get_performance_stats(self):
        """Get detailed performance statistics"""
        stats = self.get_model_stats()
        
        # Add performance metrics
        if self.feedback_data:
            recent_feedback = self.feedback_data[-50:]  # Last 50 feedback entries
            recent_accuracy = sum(1 for f in recent_feedback if f['was_correct']) / len(recent_feedback)
            stats['recent_accuracy'] = recent_accuracy
            
            # Confidence vs accuracy correlation
            high_conf_feedback = [f for f in recent_feedback if f.get('prediction_confidence', 0) > 0.7]
            if high_conf_feedback:
                high_conf_accuracy = sum(1 for f in high_conf_feedback if f['was_correct']) / len(high_conf_feedback)
                stats['high_confidence_accuracy'] = high_conf_accuracy
        
        # Model training stats
        if self.training_history:
            stats['training_progression'] = [t['accuracy'] for t in self.training_history]
            stats['model_improvement'] = (
                self.training_history[-1]['accuracy'] - self.training_history[0]['accuracy']
                if len(self.training_history) > 1 else 0
            )
        
        return stats
