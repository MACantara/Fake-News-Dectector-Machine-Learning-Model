"""
Fake News Detector class for detecting fake news using machine learning
"""

import os
import json
import threading
import pandas as pd
import joblib
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from .utils import (
    preprocess_text,
    verify_lfs_file,
    load_feedback_data,
    save_feedback_data,
    get_feedback_stats
)


class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False
        self.accuracy = None
        self.feedback_file = 'user_feedback.json'
        self.feedback_data = []
        self.retrain_threshold = 100  # Retrain after collecting 100 feedback samples
        
        # Reference to URL classifier for retraining
        self.url_classifier = None
        
        self.load_feedback_data()
    
    def set_url_classifier(self, url_classifier):
        """Set the URL classifier reference for retraining functionality"""
        self.url_classifier = url_classifier
        print("✅ URL classifier connected to fake news detector")
    
    def load_feedback_data(self):
        """Load existing feedback data"""
        self.feedback_data = load_feedback_data(self.feedback_file)
    
    def save_feedback_data(self):
        """Save feedback data to file"""
        save_feedback_data(self.feedback_data, self.feedback_file)
    
    def retrain_url_classifier_with_patterns(self):
        """Retrain URL classifier using URL classifier feedback data"""
        try:
            print("🚀 Starting URL classifier retraining with feedback data...")
            
            # Check if URL classifier is available
            if not self.url_classifier:
                print("❌ URL classifier not available - cannot perform retraining")
                print("💡 The URL classifier needs to be initialized first. Please check the app initialization.")
                return False
            
            # Check if URL classifier has the required methods
            if not hasattr(self.url_classifier, 'add_feedback') or not hasattr(self.url_classifier, 'retrain_model'):
                print("❌ URL classifier doesn't have required methods for retraining")
                return False
            
            # Load URL classifier feedback data
            feedback_urls = []
            feedback_labels = []
            
            try:
                if hasattr(self.url_classifier, 'feedback_data') and self.url_classifier.feedback_data:
                    print(f"📊 Processing {len(self.url_classifier.feedback_data)} URL classifier feedback entries...")
                    for feedback in self.url_classifier.feedback_data:
                        url = feedback.get('url', '')
                        actual_label = feedback.get('actual_label', None)
                        if url and actual_label is not None:
                            feedback_urls.append(url)
                            feedback_labels.append(bool(actual_label))
                    
                    print(f"✅ Extracted {len(feedback_urls)} URLs from feedback data")
                else:
                    print("ℹ️ No URL classifier feedback data available")
            except Exception as e:
                print(f"⚠️ Could not load URL classifier feedback: {e}")
            
            if len(feedback_urls) < 5:
                print(f"❌ Insufficient training data for URL classifier: {len(feedback_urls)} URLs (minimum 5 required)")
                return False
            
            print(f"🔄 Training URL classifier with {len(feedback_urls)} URLs...")
            print(f"   📰 News URLs: {sum(feedback_labels)} | 🚫 Non-news URLs: {len(feedback_labels) - sum(feedback_labels)}")
            
            # Add some negative examples (non-news URLs) if we have mostly positive examples
            positive_ratio = sum(feedback_labels) / len(feedback_labels) if feedback_labels else 0
            if positive_ratio > 0.8:  # If more than 80% are news URLs
                print("⚖️ Adding negative examples to balance the dataset...")
                # Add some common non-news URL patterns
                non_news_urls = [
                    'https://example.com/about-us',
                    'https://example.com/contact',
                    'https://example.com/privacy-policy',
                    'https://example.com/shop/products',
                    'https://example.com/login',
                    'https://example.com/register',
                    'https://example.com/search?q=test'
                ]
                feedback_urls.extend(non_news_urls)
                feedback_labels.extend([False] * len(non_news_urls))
                print(f"✅ Added {len(non_news_urls)} negative examples")
            
            # Create training data for URL classifier
            training_data = []
            for url, label in zip(feedback_urls, feedback_labels):
                training_data.append({
                    'url': url,
                    'actual_label': label,
                    'predicted_label': label,  # Initial assumption
                    'user_confidence': 1.0,
                    'source': 'feedback_retraining'
                })
            
            # Store current model stats for comparison
            old_stats = self.url_classifier.get_model_stats() if hasattr(self.url_classifier, 'get_model_stats') else {}
            
            # Batch add feedback to URL classifier (this will trigger retraining)
            print("🔄 Adding batch feedback to URL classifier...")
            added_count = self.url_classifier.add_batch_feedback(training_data)
            
            print(f"✅ Added {added_count} feedback entries to URL classifier")
            
            # Force retraining if it hasn't been triggered automatically
            if hasattr(self.url_classifier, 'retrain_model'):
                print("🔄 Triggering URL classifier model retraining...")
                self.url_classifier.retrain_model()
            
            # Get new model stats
            new_stats = self.url_classifier.get_model_stats() if hasattr(self.url_classifier, 'get_model_stats') else {}
            
            print("✅ URL classifier retraining completed!")
            if old_stats and new_stats:
                print(f"📊 Model improvement stats:")
                print(f"   Feedback entries: {old_stats.get('total_feedback', 0)} → {new_stats.get('total_feedback', 0)}")
                if 'accuracy' in new_stats:
                    print(f"   Model accuracy: {new_stats.get('accuracy', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during URL classifier retraining: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def add_feedback(self, text, predicted_label, actual_label, confidence, user_comment=None):
        """Add user feedback for model improvement"""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'predicted_label': predicted_label,
            'actual_label': actual_label,
            'confidence': confidence,
            'user_comment': user_comment,
            'processed_text': self.preprocess_text(text)
        }
        
        self.feedback_data.append(feedback_entry)
        self.save_feedback_data()
        
        print(f"Feedback added. Total feedback entries: {len(self.feedback_data)}")
        
        # Check if we should retrain the model
        unprocessed_feedback = [f for f in self.feedback_data if not f.get('used_for_training', False)]
        if len(unprocessed_feedback) >= self.retrain_threshold:
            print(f"Threshold reached ({self.retrain_threshold} new feedback entries). Scheduling model retraining...")
            # Schedule retraining in a separate thread to avoid blocking
            threading.Thread(target=self.retrain_with_feedback, daemon=True).start()
    
    def retrain_with_feedback(self):
        """Retrain the model incorporating user feedback"""
        try:
            print("Starting model retraining with user feedback...")
            
            # Load original dataset
            df = self.load_and_prepare_data('datasets/WELFake_Dataset.csv')
            
            # Add feedback data to training set
            unprocessed_feedback = [f for f in self.feedback_data if not f.get('used_for_training', False)]
            
            if unprocessed_feedback:
                feedback_df = pd.DataFrame([
                    {
                        'processed_text': f['processed_text'],
                        'label': 1 if f['actual_label'].lower() == 'real' else 0
                    }
                    for f in unprocessed_feedback
                    if f['processed_text'].strip()  # Only include non-empty processed text
                ])
                
                if not feedback_df.empty:
                    # Combine original data with feedback data
                    combined_df = pd.concat([
                        df[['processed_text', 'label']],
                        feedback_df
                    ], ignore_index=True)
                    
                    print(f"Training with {len(df)} original samples + {len(feedback_df)} feedback samples")
                    
                    # Retrain the model
                    old_accuracy = self.accuracy
                    new_accuracy = self.train_best_model(combined_df)
                    
                    # Save the updated model
                    model_data = {
                        'model': self.model,
                        'accuracy': new_accuracy,
                        'stemmer': self.stemmer,
                        'stop_words': self.stop_words,
                        'training_samples': len(combined_df),
                        'feedback_samples': len(feedback_df),
                        'last_retrain': datetime.now().isoformat()
                    }
                    joblib.dump(model_data, 'models/fake_news_model.pkl')
                    
                    # Mark feedback as used for training
                    for feedback in unprocessed_feedback:
                        feedback['used_for_training'] = True
                        feedback['training_date'] = datetime.now().isoformat()
                    
                    self.save_feedback_data()
                    
                    print(f"Model retrained successfully!")
                    print(f"Previous accuracy: {old_accuracy:.4f}")
                    print(f"New accuracy: {new_accuracy:.4f}")
                    print(f"Improvement: {(new_accuracy - old_accuracy):.4f}")
                    
        except Exception as e:
            print(f"Error during retraining: {str(e)}")
    
    def retrain_from_new_pattern(self, training_data):
        """Add new patterns to cache for future model improvements"""
        try:
            if not training_data:
                print("No training data provided for pattern learning")
                return
            
            # Ensure training_data is a list
            if not isinstance(training_data, list):
                print("Converting training_data to list format...")
                training_data = [training_data] if training_data else []
            
            if len(training_data) == 0:
                print("No valid training data provided for pattern learning")
                return
            
            print(f"Processing {len(training_data)} new patterns for caching...")
            
            # Process each training sample and add to pattern cache
            patterns_added = 0
            
            for item in training_data:
                if not isinstance(item, dict):
                    print(f"Warning: Skipping invalid training item: {type(item)}")
                    continue
                
                # Extract text content
                text_content = item.get('text', '') or item.get('content', '')
                if not text_content:
                    print(f"Warning: Skipping item with no text content")
                    continue
                
                # Extract label (default to 1 for real news in pattern learning)
                label = item.get('label', 1)
                
                # Extract metadata
                metadata = {
                    'source': item.get('source', 'news_article_pattern'),
                    'url': item.get('url', ''),
                    'title': item.get('title', ''),
                    'training_type': item.get('training_type', 'pattern_recognition')
                }
                
                # Add to pattern cache
                self.add_pattern_to_cache(text_content, label, metadata)
                patterns_added += 1
            
            if patterns_added == 0:
                print("No valid patterns were added to cache")
                return
            
            print(f"✅ Successfully added {patterns_added} new patterns to cache")
            
            # Check if model supports incremental training for immediate improvement
            if self.is_trained and self.model is not None:
                print("🔄 Attempting immediate pattern integration...")
                
                # Check if the model is a Pipeline and supports incremental training
                if hasattr(self.model, 'named_steps'):
                    classifier = self.model.named_steps.get('classifier')
                    vectorizer = self.model.named_steps.get('tfidf')
                    
                    if classifier and vectorizer and hasattr(classifier, 'partial_fit'):
                        try:
                            # Prepare data for immediate training
                            texts = [item.get('text', '') or item.get('content', '') for item in training_data if isinstance(item, dict)]
                            labels = [item.get('label', 1) for item in training_data if isinstance(item, dict)]
                            
                            if texts:
                                # Transform using existing vectorizer
                                text_vectors = vectorizer.transform(texts)
                                
                                # Perform incremental training
                                classifier.partial_fit(text_vectors, labels)
                                print("✅ Applied immediate incremental training")
                                
                                # Save updated model
                                try:
                                    with open('models/fake_news_model.pkl', 'wb') as f:
                                        pickle.dump(self.model, f)
                                    print("✅ Model saved with new patterns")
                                except Exception as save_error:
                                    print(f"Warning: Could not save model: {save_error}")
                        except Exception as inc_error:
                            print(f"Could not apply immediate training: {inc_error}")
                            print("Patterns saved to cache for future full retraining")
                    else:
                        print("🔄 Model doesn't support incremental training")
                        print("✅ Patterns cached for future full retraining")
                else:
                    print("🔄 Model structure doesn't support incremental updates")
                    print("✅ Patterns cached for future full retraining")
            else:
                print("✅ Patterns cached - model not currently available for immediate training")
            
            print("✅ Feedback data stored successfully")
            
        except Exception as e:
            print(f"Error during feedback processing: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get_feedback_stats(self):
        """Get statistics about user feedback"""
        return get_feedback_stats(self.feedback_data, self.retrain_threshold)
    
    def load_model(self, filepath='models/fake_news_model.pkl'):
        """Load a pre-trained model from disk with Git LFS verification"""
        try:
            print(f"Attempting to load model from: {filepath}")
            
            # First verify the file is properly downloaded from Git LFS
            is_valid, message = verify_lfs_file(filepath)
            if not is_valid:
                print(f"LFS verification failed: {message}")
                
                # Try alternative locations for the model file
                alternative_paths = [
                    os.path.join(os.getcwd(), filepath),
                    os.path.join(os.path.dirname(__file__), filepath),
                    os.path.join('/tmp', filepath),  # Sometimes files might be in tmp during deployment
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        print(f"Trying alternative path: {alt_path}")
                        is_valid, message = verify_lfs_file(alt_path)
                        if is_valid:
                            filepath = alt_path
                            break
                
                if not is_valid:
                    print(f"Model file '{filepath}' not found or not properly downloaded from Git LFS.")
                    print("This might be due to:")
                    print("1. Git LFS not being properly configured")
                    print("2. Model files not being pulled during deployment")
                    print("3. Insufficient permissions to access the files")
                    return False
            
            print(f"Loading model from verified file: {filepath}")
            model_data = joblib.load(filepath)
            
            # Validate model data structure
            required_keys = ['model', 'stemmer', 'stop_words', 'accuracy']
            missing_keys = [key for key in required_keys if key not in model_data]
            if missing_keys:
                print(f"Warning: Model file missing keys: {missing_keys}")
                return False
            
            self.model = model_data['model']
            self.stemmer = model_data['stemmer']
            self.stop_words = model_data['stop_words']
            self.accuracy = model_data['accuracy']
            self.is_trained = True
            
            # Load additional metadata if available
            training_samples = model_data.get('training_samples', 'Unknown')
            feedback_samples = model_data.get('feedback_samples', 0)
            last_retrain = model_data.get('last_retrain', 'Unknown')
            
            print(f"✓ Model loaded successfully with accuracy: {self.accuracy:.4f}")
            print(f"  Training samples: {training_samples}, Feedback samples: {feedback_samples}")
            if last_retrain != 'Unknown':
                print(f"  Last retrained: {last_retrain}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            if "No such file or directory" in str(e):
                print("The model file was not found. This might indicate a Git LFS issue.")
            elif "pickle" in str(e).lower() or "joblib" in str(e).lower():
                print("The model file appears to be corrupted or incomplete.")
            return False
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        return preprocess_text(text, self.stemmer, self.stop_words)
    
    def load_and_prepare_data(self, filepath):
        """Load and prepare the dataset with Git LFS verification"""
        print(f"Loading dataset from: {filepath}")
        
        # Verify LFS file before loading
        is_valid, message = verify_lfs_file(filepath)
        if not is_valid:
            print(f"Dataset LFS verification failed: {message}")
            
            # Try alternative locations
            alternative_paths = [
                os.path.join(os.getcwd(), filepath),
                os.path.join(os.path.dirname(__file__), filepath),
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"Trying alternative dataset path: {alt_path}")
                    is_valid, message = verify_lfs_file(alt_path)
                    if is_valid:
                        filepath = alt_path
                        break
            
            if not is_valid:
                raise FileNotFoundError(f"Dataset file '{filepath}' not found or not properly downloaded from Git LFS")
        
        print(f"Loading dataset from verified file: {filepath}")
        df = pd.read_csv(filepath)
        
        print(f"✓ Dataset loaded successfully: {len(df)} rows")
        
        # Handle missing values
        df['title'] = df['title'].fillna('')
        df['text'] = df['text'].fillna('')
        
        # Combine title and text
        df['combined_text'] = df['title'] + ' ' + df['text']
        
        # Preprocess the combined text
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        print(f"✓ Dataset preprocessed: {len(df)} valid rows after cleaning")
        return df
    
    def train_best_model(self, df):
        """Train and select the best model"""
        X = df['processed_text']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Naive Bayes': MultinomialNB()
        }
        
        best_accuracy = 0
        best_model = None
        
        for name, model in models.items():
            # Create pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = pipeline
        
        self.model = best_model
        self.is_trained = True
        print(f"Best model selected with accuracy: {best_accuracy:.4f}")
        
        return best_accuracy
    
    def predict(self, text):
        """Predict if a news article is fake or real"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet!")
        
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return {
                'prediction': 'Unknown',
                'confidence': 0.5,
                'probabilities': {'Fake': 0.5, 'Real': 0.5},
                'error': 'Text is empty after preprocessing'
            }
        
        try:
            prediction = self.model.predict([processed_text])[0]
            probability = self.model.predict_proba([processed_text])[0]
            
            return {
                'prediction': 'Real' if prediction == 1 else 'Fake',
                'confidence': float(max(probability)),
                'probabilities': {
                    'Fake': float(probability[0]),
                    'Real': float(probability[1])
                }
            }
        except Exception as e:
            return {
                'prediction': 'Unknown',
                'confidence': 0.5,
                'probabilities': {'Fake': 0.5, 'Real': 0.5},
                'error': str(e)
            }
