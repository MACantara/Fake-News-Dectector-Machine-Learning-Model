"""
URL News Classifier Main Class
Reinforcement Learning-based URL News Article Classifier
"""

import numpy as np
import pandas as pd
import json
import os
import joblib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from concurrent.futures import ThreadPoolExecutor
import warnings

from .utils import (
    extract_url_features,
    create_feature_vector,
    heuristic_prediction,
    update_feature_weights,
    initialize_feedback_db,
    load_feedback_from_db,
    save_feedback_to_db,
    add_single_feedback_to_db,
    get_recent_feedback_from_db,
    get_feedback_count_from_db,
    migrate_json_to_db,
    calculate_accuracy_from_feedback,
    get_recent_feedback,
    validate_url
)

warnings.filterwarnings('ignore')

class URLNewsClassifier:
    """
    Reinforcement Learning-based URL News Article Classifier
    Uses user feedback to continuously improve classification accuracy
    """
    
    def __init__(self, model_path='models/url_news_classifier.pkl', feedback_db_path='datasets/url_classifier_feedback.db'):
        self.model_path = model_path
        self.feedback_db_path = feedback_db_path
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
        
        # Feedback storage - now using database
        self.feedback_data = []
        self.training_history = []
        
        # Initialize database and migrate from JSON if needed
        self._initialize_feedback_system()
        
        # Load existing model and feedback
        self.load_feedback()
        self.load_model()
    
    def _initialize_feedback_system(self):
        """Initialize feedback database and migrate from JSON if needed"""
        # Initialize SQLite database
        initialize_feedback_db(self.feedback_db_path)
        
        # Check if we need to migrate from JSON
        json_path = 'datasets/url_classifier_feedback.json'
        if os.path.exists(json_path):
            print("Found existing JSON feedback file. Migrating to SQLite database...")
            migrate_json_to_db(json_path, self.feedback_db_path)
    
    def predict_with_confidence(self, url):
        """Predict if URL is a news article with confidence score"""
        if not validate_url(url):
            return {
                'prediction': False,
                'confidence': 0.0,
                'is_news_article': False,
                'probability_news': 0.0,
                'probability_not_news': 1.0,
                'error': 'Invalid URL format'
            }
        
        if not self.is_trained:
            # Use heuristic-based prediction for untrained model
            return heuristic_prediction(url)
        
        feature_vector, _ = create_feature_vector(url)
        feature_vector = feature_vector.reshape(1, -1)
        
        # Get prediction and probability
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        # Calculate confidence as the maximum probability
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
    
    def add_feedback(self, url, predicted_label, actual_label, user_confidence=1.0):
        """Add user feedback for reinforcement learning"""
        if not validate_url(url):
            raise ValueError("Invalid URL format")
        
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
        
        # Save individual feedback entry to database
        add_single_feedback_to_db(feedback_entry, self.feedback_db_path)
        
        # Update feature weights based on feedback
        self.feature_weights = update_feature_weights(
            self.feature_weights, 
            feedback_entry, 
            self.learning_rate
        )
        
        # Trigger retraining if enough feedback collected
        feedback_count = get_feedback_count_from_db(self.feedback_db_path)
        if feedback_count % 100 == 0 and feedback_count >= 100:
            self.retrain_model()
        
        return feedback_entry
    
    def add_batch_feedback(self, feedback_list):
        """Add multiple feedback entries efficiently"""
        processed_count = 0
        errors = []
        
        for feedback in feedback_list:
            try:
                url = feedback['url']
                predicted_label = feedback['predicted_label']
                actual_label = feedback['actual_label']
                user_confidence = feedback.get('user_confidence', 1.0)
                
                if not validate_url(url):
                    errors.append(f"Invalid URL: {url}")
                    continue
                
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
                self.feature_weights = update_feature_weights(
                    self.feature_weights, 
                    feedback_entry, 
                    self.learning_rate
                )
                processed_count += 1
                
            except Exception as e:
                errors.append(f"Error processing feedback for {feedback.get('url', 'unknown')}: {str(e)}")
        
        # Save all feedback entries to database in batch
        if processed_count > 0:
            # Get the new entries we just added to feedback_data
            new_entries = self.feedback_data[-processed_count:]
            save_feedback_to_db(new_entries, self.feedback_db_path)
        
        # Check if retraining is needed
        feedback_count = get_feedback_count_from_db(self.feedback_db_path)
        if feedback_count >= 100 and feedback_count % 100 == 0:
            self.retrain_model()
        
        return {
            'processed_count': processed_count,
            'errors': errors,
            'total_feedback': feedback_count
        }
    
    def retrain_model(self):
        """Retrain the model with accumulated feedback"""
        # Load all feedback from database
        all_feedback = load_feedback_from_db(self.feedback_db_path)
        
        if len(all_feedback) < 10:
            print(f"Need at least 10 feedback samples, have {len(all_feedback)}")
            return None
        
        print(f"Retraining model with {len(all_feedback)} feedback samples...")
        
        # Prepare training data
        X = []
        y = []
        
        for feedback in all_feedback:
            feature_vector = feedback.get('feature_vector')
            
            # Generate feature vector if missing
            if not feature_vector or len(feature_vector) == 0:
                try:
                    feature_vector, _ = create_feature_vector(feedback['url'])
                    feedback['feature_vector'] = feature_vector.tolist()
                except Exception as e:
                    print(f"Error generating features for {feedback['url']}: {str(e)}")
                    continue
            
            X.append(feature_vector)
            y.append(1 if feedback['actual_label'] else 0)
        
        if len(X) < 10:
            print(f"Not enough valid feature vectors: {len(X)}")
            return None
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Training with {len(X)} feature vectors...")
        
        # Train optimized model
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Evaluate on training data
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
            try:
                model_data = {
                    'model': self.model,
                    'is_trained': self.is_trained,
                    'feature_weights': self.feature_weights,
                    'training_history': self.training_history,
                    'timestamp': datetime.now().isoformat()
                }
                joblib.dump(model_data, self.model_path)
                print(f"Model saved to {self.model_path}")
                return True
            except Exception as e:
                print(f"Error saving model: {str(e)}")
                return False
        return False
    
    def load_model(self):
        """Load existing model if available"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Model file {self.model_path} not found")
                return False
            
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.is_trained = model_data.get('is_trained', False)
            self.feature_weights = model_data.get('feature_weights', self.feature_weights)
            self.training_history = model_data.get('training_history', [])
            
            print(f"Model loaded from {self.model_path}")
            return True
        except FileNotFoundError:
            print(f"Model file {self.model_path} not found")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def save_feedback(self):
        """Save feedback data to database (legacy method maintained for compatibility)"""
        if self.feedback_data:
            return save_feedback_to_db(self.feedback_data, self.feedback_db_path)
        return True
    
    def load_feedback(self):
        """Load existing feedback data from database"""
        self.feedback_data = load_feedback_from_db(self.feedback_db_path)
        print(f"Loaded {len(self.feedback_data)} feedback entries from database")
    
    def get_model_stats(self):
        """Get model statistics and performance metrics"""
        # Get feedback count from database
        feedback_count = get_feedback_count_from_db(self.feedback_db_path)
        
        stats = {
            'is_trained': self.is_trained,
            'feedback_count': feedback_count,
            'training_iterations': len(self.training_history),
            'feature_weights': self.feature_weights.copy(),
            'last_accuracy': None,
            'correct_predictions': 0,
            'total_predictions': 0,
            'accuracy': 0.0
        }
        
        if self.training_history:
            stats['last_accuracy'] = self.training_history[-1]['accuracy']
        
        # Calculate accuracy from feedback
        if feedback_count > 0:
            # Load all feedback for accuracy calculation
            all_feedback = load_feedback_from_db(self.feedback_db_path)
            stats['accuracy'] = calculate_accuracy_from_feedback(all_feedback)
            stats['correct_predictions'] = sum(1 for entry in all_feedback if entry.get('was_correct', False))
            stats['total_predictions'] = len(all_feedback)
        
        return stats
    
    def get_recent_feedback(self, limit=10):
        """Get recent feedback samples from database"""
        return get_recent_feedback_from_db(limit, self.feedback_db_path)
    
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
            analysis['summary']['avg_confidence'] = sum(confidences) / len(confidences)
        
        return analysis
    
    def retrain_with_feedback(self, urls, labels):
        """Retrain URL classifier with feedback data"""
        if len(urls) != len(labels):
            raise ValueError("URLs and labels must have the same length")
        
        if len(urls) < 5:
            raise ValueError(f"Need at least 5 feedback entries, have {len(urls)}")
        
        # Add feedback entries
        feedback_list = []
        for url, label in zip(urls, labels):
            # Get current prediction
            prediction_result = self.predict_with_confidence(url)
            predicted_label = prediction_result['is_news_article']
            
            feedback_list.append({
                'url': url,
                'predicted_label': predicted_label,
                'actual_label': bool(label),
                'user_confidence': 1.0
            })
        
        # Add batch feedback
        result = self.add_batch_feedback(feedback_list)
        
        # Force retrain
        accuracy = self.retrain_model()
        
        return {
            'processed_feedback': result['processed_count'],
            'errors': result['errors'],
            'final_accuracy': accuracy
        }
    
    def get_performance_stats(self):
        """Get detailed performance statistics"""
        stats = self.get_model_stats()
        
        # Add performance metrics
        feedback_count = get_feedback_count_from_db(self.feedback_db_path)
        if feedback_count > 0:
            recent_feedback = get_recent_feedback_from_db(20, self.feedback_db_path)
            if recent_feedback:
                recent_accuracy = calculate_accuracy_from_feedback(recent_feedback)
                stats['recent_accuracy'] = recent_accuracy
        
        # Model training stats
        if self.training_history:
            stats['training_history'] = self.training_history[-5:]  # Last 5 training sessions
            stats['model_improvement'] = []
            
            if len(self.training_history) > 1:
                for i in range(1, len(self.training_history)):
                    prev_acc = self.training_history[i-1]['accuracy']
                    curr_acc = self.training_history[i]['accuracy']
                    improvement = curr_acc - prev_acc
                    stats['model_improvement'].append(improvement)
        
        return stats
    
    def migrate_from_json(self, json_path='datasets/url_classifier_feedback.json'):
        """Migrate feedback data from JSON file to SQLite database"""
        return migrate_json_to_db(json_path, self.feedback_db_path)
    
    def get_database_info(self):
        """Get information about the feedback database"""
        return {
            'database_path': self.feedback_db_path,
            'database_exists': os.path.exists(self.feedback_db_path),
            'feedback_count': get_feedback_count_from_db(self.feedback_db_path),
            'database_size_bytes': os.path.getsize(self.feedback_db_path) if os.path.exists(self.feedback_db_path) else 0
        }
