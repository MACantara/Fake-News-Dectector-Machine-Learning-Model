from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
import requests
from bs4 import BeautifulSoup
import pickle
import joblib
import os
import json
from datetime import datetime
import threading
import time
from urllib.parse import urljoin, urlparse, parse_qs
import concurrent.futures
from collections import defaultdict
import sqlite3
import hashlib
from whoosh import index
from whoosh.fields import Schema, TEXT, DATETIME, ID, KEYWORD, NUMERIC
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import And, Or, Term, Phrase
from dateutil import parser as date_parser
from textblob import TextBlob
from fuzzywuzzy import fuzz, process
from url_news_classifier import URLNewsClassifier
from routes.news_crawler_routes import news_crawler_bp, init_news_crawler
from routes.philippine_news_search_routes import philippine_news_bp, init_philippine_search_index, get_philippine_search_index
from routes.political_news_routes import political_news_bp, init_political_detector
from modules.political_news_detector import PoliticalNewsDetector, extract_political_content_from_url
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

app = Flask(__name__)

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
        self.retrain_threshold = 10  # Retrain after collecting 10 feedback samples
        
        # Pattern learning cache for news articles
        self.pattern_cache = []
        self.pattern_cache_file = 'news_pattern_cache.json'
        self.pattern_cache_threshold = 5  # Number of patterns before considering full retraining
        
        # Reference to URL classifier for retraining
        self.url_classifier = None
        
        self.load_feedback_data()
        self.load_pattern_cache()
    
    def load_feedback_data(self):
        """Load existing feedback data"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    self.feedback_data = json.load(f)
                print(f"Loaded {len(self.feedback_data)} feedback entries")
        except Exception as e:
            print(f"Error loading feedback data: {str(e)}")
            self.feedback_data = []
    
    def save_feedback_data(self):
        """Save feedback data to file"""
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving feedback data: {str(e)}")
    
    def load_pattern_cache(self):
        """Load pattern cache for news articles"""
        try:
            if os.path.exists(self.pattern_cache_file):
                with open(self.pattern_cache_file, 'r', encoding='utf-8') as f:
                    self.pattern_cache = json.load(f)
                print(f"Loaded {len(self.pattern_cache)} cached news patterns")
        except Exception as e:
            print(f"Error loading pattern cache: {str(e)}")
            self.pattern_cache = []
    
    def save_pattern_cache(self):
        """Save pattern cache to file"""
        try:
            with open(self.pattern_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.pattern_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving pattern cache: {str(e)}")
    
    def add_pattern_to_cache(self, text, label, metadata=None):
        """Add a new pattern to the cache"""
        pattern_entry = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'label': label,
            'processed_text': self.preprocess_text(text),
            'metadata': metadata or {}
        }
        
        self.pattern_cache.append(pattern_entry)
        self.save_pattern_cache()
        
        print(f"Pattern added to cache. Total cached patterns: {len(self.pattern_cache)}")
        
        # Check if we should suggest full retraining
        if len(self.pattern_cache) >= self.pattern_cache_threshold:
            print(f"Pattern cache threshold reached ({self.pattern_cache_threshold}). Consider full model retraining for optimal performance.")
    
    def get_pattern_cache_stats(self):
        """Get statistics about the pattern cache"""
        if not self.pattern_cache:
            return {
                'total_patterns': 0,
                'cache_threshold': self.pattern_cache_threshold,
                'needs_retraining': False
            }
        
        return {
            'total_patterns': len(self.pattern_cache),
            'cache_threshold': self.pattern_cache_threshold,
            'needs_retraining': len(self.pattern_cache) >= self.pattern_cache_threshold,
            'oldest_pattern': self.pattern_cache[0]['timestamp'] if self.pattern_cache else None,
            'newest_pattern': self.pattern_cache[-1]['timestamp'] if self.pattern_cache else None
        }
    
    def retrain_with_cached_patterns(self):
        """Perform full model retraining using cached patterns"""
        try:
            if not self.pattern_cache:
                print("No cached patterns available for retraining")
                return False
            
            print(f"🚀 Starting full model retraining with {len(self.pattern_cache)} cached patterns...")
            
            # Load original dataset
            try:
                df = self.load_and_prepare_data('WELFake_Dataset.csv')
                print(f"✅ Loaded original dataset: {len(df)} samples")
            except Exception as e:
                print(f"❌ Could not load original dataset: {e}")
                print("🔄 Attempting retraining with cached patterns only...")
                df = None
            
            # Prepare cached pattern data
            cached_texts = []
            cached_labels = []
            
            for pattern in self.pattern_cache:
                text = pattern.get('text', '')
                if text:
                    cached_texts.append(text)
                    cached_labels.append(pattern.get('label', 1))
            
            if not cached_texts:
                print("❌ No valid text data in cached patterns")
                return False
            
            # Create DataFrame for cached patterns
            cached_df = pd.DataFrame({
                'processed_text': [self.preprocess_text(text) for text in cached_texts],
                'label': cached_labels
            })
            
            print(f"✅ Prepared cached patterns: {len(cached_df)} samples")
            
            # Combine with original dataset if available
            if df is not None:
                # Combine original data with cached patterns
                combined_df = pd.concat([df[['processed_text', 'label']], cached_df], ignore_index=True)
                print(f"✅ Combined dataset: {len(combined_df)} total samples")
            else:
                # Use only cached patterns
                combined_df = cached_df
                print(f"⚠️ Using cached patterns only: {len(combined_df)} samples")
            
            # Store current model performance for comparison
            old_accuracy = getattr(self, 'accuracy', None)
            
            # Retrain the model
            print("🔄 Training new model with combined dataset...")
            new_accuracy = self.train_best_model(combined_df)
            
            if new_accuracy:
                print(f"✅ Model retraining completed!")
                print(f"📊 New model accuracy: {new_accuracy:.4f}")
                if old_accuracy:
                    improvement = new_accuracy - old_accuracy
                    print(f"📈 Accuracy change: {improvement:+.4f}")
                
                # Mark cached patterns as used for training
                for pattern in self.pattern_cache:
                    pattern['used_for_training'] = True
                    pattern['training_date'] = datetime.now().isoformat()
                
                self.save_pattern_cache()
                
                # Clear the cache after successful training
                self.clear_used_patterns()
                
                print("✅ Pattern cache updated and cleaned")
                return True
            else:
                print("❌ Model retraining failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during cached pattern retraining: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def clear_used_patterns(self):
        """Clear patterns that have been used for training"""
        original_count = len(self.pattern_cache)
        self.pattern_cache = [p for p in self.pattern_cache if not p.get('used_for_training', False)]
        cleared_count = original_count - len(self.pattern_cache)
        
        if cleared_count > 0:
            print(f"🧹 Cleared {cleared_count} used patterns from cache")
            self.save_pattern_cache()
        
        return cleared_count
    
    def retrain_url_classifier_with_patterns(self):
        """Retrain URL classifier using both pattern cache and URL classifier feedback data"""
        try:
            print("🚀 Starting URL classifier retraining with cached patterns and feedback data...")
            
            # Check if URL classifier is available
            if not self.url_classifier:
                print("❌ URL classifier not available - cannot perform retraining")
                return False
            
            # Prepare training data from pattern cache (news article URLs)
            url_patterns = []
            url_labels = []
            
            print(f"📊 Processing {len(self.pattern_cache)} cached news patterns...")
            for pattern in self.pattern_cache:
                url = pattern.get('metadata', {}).get('url', '')
                if url and url.startswith(('http://', 'https://')):
                    url_patterns.append(url)
                    url_labels.append(True)  # These are confirmed news article URLs
            
            print(f"✅ Extracted {len(url_patterns)} URLs from news pattern cache")
            
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
            
            # Combine both data sources
            all_urls = url_patterns + feedback_urls
            all_labels = url_labels + feedback_labels
            
            if len(all_urls) < 5:
                print(f"❌ Insufficient training data for URL classifier: {len(all_urls)} URLs (minimum 5 required)")
                return False
            
            print(f"🔄 Training URL classifier with {len(all_urls)} URLs...")
            print(f"   📰 News URLs: {sum(all_labels)} | 🚫 Non-news URLs: {len(all_labels) - sum(all_labels)}")
            
            # Add some negative examples (non-news URLs) if we have mostly positive examples
            positive_ratio = sum(all_labels) / len(all_labels) if all_labels else 0
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
                all_urls.extend(non_news_urls)
                all_labels.extend([False] * len(non_news_urls))
                print(f"✅ Added {len(non_news_urls)} negative examples")
            
            # Create training data for URL classifier
            training_data = []
            for url, label in zip(all_urls, all_labels):
                training_data.append({
                    'url': url,
                    'actual_label': label,
                    'predicted_label': label,  # Initial assumption
                    'user_confidence': 1.0,
                    'source': 'pattern_cache_retraining'
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
            
            # Mark pattern cache entries as used for URL classifier training
            for pattern in self.pattern_cache:
                pattern['used_for_url_training'] = True
                pattern['url_training_date'] = datetime.now().isoformat()
            
            self.save_pattern_cache()
            print("✅ Pattern cache updated with URL training metadata")
            
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
            df = self.load_and_prepare_data('WELFake_Dataset.csv')
            
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
                    joblib.dump(model_data, 'fake_news_model.pkl')
                    
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
                                    with open('fake_news_model.pkl', 'wb') as f:
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
            
            # Provide guidance on next steps and trigger automatic retraining if needed
            cache_stats = self.get_pattern_cache_stats()
            print(f"📊 Pattern cache status: {cache_stats['total_patterns']}/{cache_stats['cache_threshold']} patterns")
            
            if cache_stats['needs_retraining']:
                print("🎯 Pattern cache threshold exceeded - starting automatic URL classifier retraining...")
                try:
                    # Trigger automatic URL classifier retraining with cached patterns and feedback data
                    self.retrain_url_classifier_with_patterns()
                    print("✅ Automatic URL classifier retraining completed successfully")
                except Exception as retrain_error:
                    print(f"❌ URL classifier retraining failed: {retrain_error}")
                    print("🔄 Patterns remain cached for manual retraining")
            else:
                remaining = cache_stats['cache_threshold'] - cache_stats['total_patterns']
                print(f"📈 {remaining} more patterns needed to trigger automatic URL classifier retraining")
            
            print("✅ Pattern learning completed successfully")
            
        except Exception as e:
            print(f"Error during pattern learning: {str(e)}")
            print(f"Training data type: {type(training_data)}")
            if isinstance(training_data, list) and len(training_data) > 0:
                print(f"First item type: {type(training_data[0])}")
                print(f"First item keys: {list(training_data[0].keys()) if isinstance(training_data[0], dict) else 'Not a dict'}")
            import traceback
            traceback.print_exc()
    
    def get_feedback_stats(self):
        """Get statistics about user feedback"""
        if not self.feedback_data:
            return {
                'total_feedback': 0,
                'used_for_training': 0,
                'pending_training': 0,
                'accuracy_improvement': None
            }
        
        used_count = len([f for f in self.feedback_data if f.get('used_for_training', False)])
        pending_count = len(self.feedback_data) - used_count
        
        return {
            'total_feedback': len(self.feedback_data),
            'used_for_training': used_count,
            'pending_training': pending_count,
            'retrain_threshold': self.retrain_threshold,
            'needs_retraining': pending_count >= self.retrain_threshold
        }
    
    def verify_lfs_file(self, filepath):
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

    def load_model(self, filepath='fake_news_model.pkl'):
        """Load a pre-trained model from disk with Git LFS verification"""
        try:
            print(f"Attempting to load model from: {filepath}")
            
            # First verify the file is properly downloaded from Git LFS
            is_valid, message = self.verify_lfs_file(filepath)
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
                        is_valid, message = self.verify_lfs_file(alt_path)
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
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def load_and_prepare_data(self, filepath):
        """Load and prepare the dataset with Git LFS verification"""
        print(f"Loading dataset from: {filepath}")
        
        # Verify LFS file before loading
        is_valid, message = self.verify_lfs_file(filepath)
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
                    is_valid, message = self.verify_lfs_file(alt_path)
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

# Initialize the detectors and search index
detector = FakeNewsDetector()
political_detector = PoliticalNewsDetector()
url_news_classifier = URLNewsClassifier()

# Initialize news crawler routes
init_news_crawler(url_news_classifier)

# Initialize Philippine news search routes
philippine_search_index = init_philippine_search_index()

# Initialize political news detector routes
init_political_detector(political_detector)

# Register the blueprints
app.register_blueprint(news_crawler_bp)
app.register_blueprint(philippine_news_bp)
app.register_blueprint(political_news_bp)

# Set URL classifier reference in fake news detector for retraining purposes
detector.url_classifier = url_news_classifier

@app.route('/')
def index():
    return render_template('news_analysis.html')

@app.route('/philippine-search')
def philippine_search():
    return render_template('philippine_search.html')

@app.route('/url-classifier')
def url_classifier():
    return render_template('url_classifier.html')

@app.route('/analyze-url')
def analyze_url():
    """Route for analyzing a URL directly (for external links)"""
    url = request.args.get('url')
    if not url:
        return redirect(url_for('index'))
    return render_template('news_analysis.html', prefill_url=url)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not detector.is_trained:
            return jsonify({
                'success': False,
                'error': 'Fake news model not trained yet. Please wait for training to complete.'
            }), 500
        
        input_type = data.get('type', 'text')
        analysis_type = data.get('analysis_type', 'fake_news')  # 'fake_news', 'political', or 'both'
        
        if input_type == 'text':
            text = data.get('text', '').strip()
            if not text:
                return jsonify({
                    'success': False,
                    'error': 'No text provided'
                }), 400
            
            # Perform fake news detection
            fake_result = detector.predict(text)
            result = {'fake_news': fake_result}
            
            # Perform political classification if requested
            if analysis_type in ['political', 'both'] and political_detector.is_trained:
                political_result = political_detector.predict(text)
                result['political_classification'] = political_result
            elif analysis_type in ['political', 'both']:
                result['political_classification'] = {
                    'error': 'Political classification model not available'
                }
            
            return jsonify({
                'success': True,
                'data': result
            })
        
        elif input_type == 'url':
            url = data.get('url', '').strip()
            if not url:
                return jsonify({
                    'success': False,
                    'error': 'No URL provided'
                }), 400
            
            # Extract content from URL using the political news detector's extraction function
            article_data = extract_political_content_from_url(url)
            
            if 'error' in article_data:
                return jsonify({
                    'success': False,
                    'error': article_data['error']
                }), 400
            
            combined_text = article_data['combined_text']
            if not combined_text.strip():
                return jsonify({
                    'success': False,
                    'error': 'No content could be extracted from the URL'
                }), 400
            
            # Automatically index the URL in the Philippine news search engine and database
            indexing_result = None
            try:
                print(f"🔍 Auto-indexing URL for search engine: {url}")
                indexing_result = philippine_search_index.index_article(url, force_reindex=False)
                print(f"📚 Indexing result: {indexing_result['status']}")
                
                if indexing_result['status'] == 'success':
                    print(f"✅ Successfully indexed article with relevance score: {indexing_result.get('relevance_score', 0):.3f}")
                    if indexing_result.get('locations'):
                        print(f"📍 Found locations: {', '.join(indexing_result['locations'])}")
                    if indexing_result.get('government_entities'):
                        print(f"🏛️ Found government entities: {', '.join(indexing_result['government_entities'])}")
                elif indexing_result['status'] == 'skipped':
                    print(f"⏭️ Article skipped from indexing: {indexing_result.get('message', 'Low Philippine relevance')}")
                elif indexing_result['status'] == 'already_indexed':
                    print(f"📋 Article already in search index")
                else:
                    print(f"❌ Indexing failed: {indexing_result.get('message', 'Unknown error')}")
                    
            except Exception as indexing_error:
                print(f"⚠️ Error during auto-indexing: {str(indexing_error)}")
                # Don't fail the main analysis if indexing fails
                indexing_result = {'status': 'error', 'message': str(indexing_error)}
            
            # Perform fake news detection
            fake_result = detector.predict(combined_text)
            result = {
                'fake_news': fake_result,
                'extracted_content': {
                    'title': article_data.get('title', ''),
                    'content_preview': combined_text[:500] + '...' if len(combined_text) > 500 else combined_text,
                    'combined': combined_text,
                    'word_count': len(combined_text.split()) if combined_text else 0,
                    'author': article_data.get('author', ''),
                    'publish_date': article_data.get('publish_date', ''),
                    'political_score': article_data.get('political_score', 0),
                    'is_political_source': article_data.get('is_political_source', False)
                },
                'indexing_result': indexing_result  # Include indexing status in response
            }
            
            # Perform political classification if requested
            if analysis_type in ['political', 'both'] and political_detector.is_trained:
                political_result = political_detector.predict(combined_text)
                result['political_classification'] = political_result
            elif analysis_type in ['political', 'both']:
                result['political_classification'] = {
                    'error': 'Political classification model not available'
                }
            
            # Handle retraining trigger if requested
            trigger_retraining = data.get('trigger_retraining', False)
            if trigger_retraining:
                try:
                    print(f"🔄 Automatic model retraining requested for news article URL")
                    
                    # For news articles, always attempt retraining to learn new patterns
                    # This helps the model recognize new content structures and patterns
                    print(f"📊 Initiating automatic retraining to learn new news article patterns")
                    
                    # Start retraining in a background thread to avoid blocking the response
                    import threading
                    def background_retrain():
                        try:
                            print("🚀 Starting background model retraining for news pattern recognition...")
                            
                            # Create training samples from the current article for pattern learning
                            # Format as expected by retrain_from_new_pattern: list of dicts with 'text' and 'label'
                            training_data = [{
                                'text': combined_text,
                                'label': 1,  # Label as real news (1) for pattern recognition
                                'source': 'news_article_pattern',
                                'url': url,
                                'title': article_data.get('title', ''),
                                'training_type': 'pattern_recognition'
                            }]
                            
                            # Retrain with focus on recognizing new news patterns
                            detector.retrain_from_new_pattern(training_data)
                            print("✅ Model retraining completed - new news patterns learned")
                        except Exception as retrain_error:
                            print(f"❌ Model retraining failed: {str(retrain_error)}")
                    
                    retraining_thread = threading.Thread(target=background_retrain)
                    retraining_thread.daemon = True
                    retraining_thread.start()
                    
                    result['retraining_triggered'] = {
                        'status': 'initiated',
                        'training_type': 'news_pattern_recognition',
                        'message': 'Model retraining started to learn new news article patterns. This helps improve detection of legitimate news content.'
                    }
                    print("✅ Pattern recognition retraining thread started successfully")
                        
                except Exception as retrain_error:
                    print(f"⚠️ Error initiating pattern recognition retraining: {str(retrain_error)}")
                    result['retraining_triggered'] = {
                        'status': 'error',
                        'message': f'Failed to initiate pattern recognition retraining: {str(retrain_error)}'
                    }
            
            return jsonify({
                'success': True,
                'data': result
            })
        
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid input type'
            }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint with Git LFS file status"""
    try:
        # Check Git LFS files
        lfs_files = [
            'fake_news_model.pkl',
            'political_news_classifier.pkl', 
            'WELFake_Dataset.csv',
            'News_Category_Dataset_v3.json'
        ]
        
        file_status = {}
        for file_path in lfs_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                file_status[file_path] = {
                    'exists': True,
                    'size': file_size,
                    'is_lfs_pointer': file_size < 1000
                }
            else:
                file_status[file_path] = {
                    'exists': False,
                    'size': 0,
                    'is_lfs_pointer': False
                }
        
        # Check model status
        models_status = {
            'fake_news_detector': {
                'loaded': detector.is_trained,
                'accuracy': detector.accuracy if detector.accuracy else None
            },
            'political_detector': {
                'loaded': political_detector.is_trained,
                'accuracy': political_detector.accuracy if political_detector.accuracy else None
            }
        }
        
        # Overall health
        is_healthy = detector.is_trained or political_detector.is_trained
        
        return jsonify({
            'status': 'healthy' if is_healthy else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'models': models_status,
            'lfs_files': file_status,
            'message': 'All systems operational' if is_healthy else 'Some models unavailable'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

@app.route('/model-status')
def model_status():
    try:
        # Create the expected response structure for the frontend
        status_data = {
            'fake_news': detector.is_trained,
            'political': political_detector.is_trained
        }
        
        # Additional detailed status info (optional)
        status_info = {
            'fake_news_model': {
                'is_trained': detector.is_trained,
                'status': 'Model is ready' if detector.is_trained else 'Model is training...'
            },
            'political_model': {
                'is_trained': political_detector.is_trained,
                'status': 'Model is ready' if political_detector.is_trained else 'Model not loaded'
            }
        }
        
        if detector.is_trained and detector.accuracy:
            status_info['fake_news_model']['accuracy'] = f"{detector.accuracy:.4f}"
            status_info['fake_news_model']['status'] = f"Model ready (Accuracy: {detector.accuracy:.1%})"
        
        if political_detector.is_trained and political_detector.accuracy:
            status_info['political_model']['accuracy'] = f"{political_detector.accuracy:.4f}"
            status_info['political_model']['status'] = f"Model ready (Accuracy: {political_detector.accuracy:.1%})"
        
        # Overall status
        both_ready = detector.is_trained and political_detector.is_trained
        status_info['overall_status'] = 'Both models ready' if both_ready else 'Models loading...'
        status_info['is_trained'] = detector.is_trained  # Keep for backward compatibility
        
        # Add feedback statistics
        try:
            feedback_stats = detector.get_feedback_stats()
            status_info['feedback'] = feedback_stats
        except:
            status_info['feedback'] = {'total': 0, 'correct': 0, 'incorrect': 0}
        
        # Return in the format expected by the frontend
        return jsonify({
            'success': True,
            'status': status_data,
            'details': status_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get model status: {str(e)}',
            'status': {'fake_news': False, 'political': False}
        })

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for model improvement"""
    try:
        data = request.get_json()
        
        text = data.get('text', '').strip()
        predicted_label = data.get('predicted_label', '').strip()
        actual_label = data.get('actual_label', '').strip()
        confidence = data.get('confidence', 0.0)
        user_comment = data.get('comment', '').strip()
        
        # Validate required fields
        if not text or not predicted_label or not actual_label:
            return jsonify({'error': 'Missing required fields: text, predicted_label, actual_label'}), 400
        
        if actual_label.lower() not in ['fake', 'real']:
            return jsonify({'error': 'actual_label must be either "fake" or "real"'}), 400
        
        # Add feedback to the system
        detector.add_feedback(text, predicted_label, actual_label, confidence, user_comment)
        
        # Get updated feedback stats
        feedback_stats = detector.get_feedback_stats()
        
        response = {
            'message': 'Thank you for your feedback! It will help improve the model.',
            'feedback_stats': feedback_stats
        }
        
        if feedback_stats['needs_retraining']:
            response['message'] += f' The model will be retrained soon with {feedback_stats["pending_training"]} new feedback entries.'
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/feedback-stats')
def feedback_stats():
    """Get feedback statistics"""
    try:
        stats = detector.get_feedback_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/trigger-retrain', methods=['POST'])
def trigger_retrain():
    """Manually trigger model retraining"""
    try:
        feedback_stats = detector.get_feedback_stats()
        
        if feedback_stats['pending_training'] == 0:
            return jsonify({'message': 'No new feedback available for retraining.'}), 400
        
        # Start retraining in background
        threading.Thread(target=detector.retrain_with_feedback, daemon=True).start()
        
        return jsonify({
            'message': f'Model retraining started with {feedback_stats["pending_training"]} new feedback entries.',
            'status': 'retraining_started'
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/classify-url', methods=['POST'])
def classify_url():
    """Classify if a URL is a news article using RL model"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({
                'success': False,
                'error': 'No URL provided'
            }), 400
        
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Get prediction from RL model
        result = url_news_classifier.predict_with_confidence(url)
        
        return jsonify({
            'success': True,
            'url': url,
            'is_news_article': result['is_news_article'],
            'confidence': result['confidence'],
            'probability_news': result['probability_news'],
            'probability_not_news': result['probability_not_news'],
            'model_trained': url_news_classifier.is_trained,
            'heuristic_based': result.get('heuristic_based', False)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/url-classifier-feedback', methods=['POST'])
def submit_url_classifier_feedback():
    """Submit feedback for URL classification to improve the RL model (supports single and bulk feedback)"""
    try:
        data = request.get_json()
        
        # Check if this is bulk feedback
        if 'feedback_batch' in data:
            # Handle bulk feedback
            feedback_list = data.get('feedback_batch', [])
            
            if not feedback_list:
                return jsonify({
                    'success': False,
                    'error': 'feedback_batch is empty'
                }), 400
            
            processed_count = 0
            errors = []
            
            for feedback in feedback_list:
                try:
                    url = feedback.get('url', '').strip()
                    predicted_label = feedback.get('predicted_label')
                    actual_label = feedback.get('actual_label')
                    user_confidence = feedback.get('user_confidence', 1.0)
                    
                    if not url or predicted_label is None or actual_label is None:
                        errors.append(f"Missing required fields for URL: {url}")
                        continue
                    
                    # Add feedback to RL model
                    url_news_classifier.add_feedback(
                        url=url,
                        predicted_label=predicted_label,
                        actual_label=actual_label,
                        user_confidence=user_confidence
                    )
                    processed_count += 1
                    
                except Exception as e:
                    errors.append(f"Error processing feedback for {feedback.get('url', 'unknown')}: {str(e)}")
            
            # Get updated model stats
            model_stats = url_news_classifier.get_model_stats()
            
            return jsonify({
                'success': True,
                'bulk_feedback': True,
                'processed_count': processed_count,
                'total_submitted': len(feedback_list),
                'errors': errors,
                'feedback_count': model_stats['feedback_count'],
                'is_trained': model_stats['is_trained'],
                'message': f'Successfully processed {processed_count} out of {len(feedback_list)} feedback entries'
            })
        
        else:
            # Handle single feedback (existing logic)
            url = data.get('url', '').strip()
            predicted_label = data.get('predicted_label')
            actual_label = data.get('actual_label')
            user_confidence = data.get('user_confidence', 1.0)
            
            if not url or predicted_label is None or actual_label is None:
                return jsonify({
                    'success': False,
                    'error': 'Missing required fields: url, predicted_label, actual_label'
                }), 400
            
            # Add feedback to RL model
            feedback_entry = url_news_classifier.add_feedback(
                url=url,
                predicted_label=predicted_label,
                actual_label=actual_label,
                user_confidence=user_confidence
            )
            
            # Get updated model stats
            model_stats = url_news_classifier.get_model_stats()
            
            return jsonify({
                'success': True,
                'feedback_added': True,
                'feedback_count': model_stats['feedback_count'],
            'model_accuracy': model_stats.get('feedback_accuracy', 0),
            'model_trained': model_stats['is_trained'],
            'message': 'Thank you for your feedback! The model will improve with your input.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/url-classifier-stats')
def get_url_classifier_stats():
    """Get URL classifier model statistics"""
    try:
        stats = url_news_classifier.get_model_stats()
        recent_feedback = url_news_classifier.get_recent_feedback(limit=5)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'recent_feedback': recent_feedback
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/retrain-url-classifier', methods=['POST'])
def retrain_url_classifier():
    """Manually trigger retraining of the URL classifier"""
    try:
        current_feedback_count = len(url_news_classifier.feedback_data)
        
        if current_feedback_count < 3:
            return jsonify({
                'success': False,
                'error': f'Need at least 3 feedback samples to retrain model. Currently have {current_feedback_count} samples.',
                'current_feedback_count': current_feedback_count,
                'minimum_required': 3
            })
        
        accuracy = url_news_classifier.retrain_model()
        
        if accuracy is None:
            return jsonify({
                'success': False,
                'error': f'Retraining failed - insufficient quality feedback data. Have {current_feedback_count} samples.',
                'current_feedback_count': current_feedback_count
            })
        
        return jsonify({
            'success': True,
            'message': f'Model retrained successfully with {current_feedback_count} feedback samples',
            'accuracy': accuracy,
            'feedback_count': current_feedback_count
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

def check_lfs_files():
    """Check if Git LFS files are properly available"""
    print("=== Checking Git LFS Files ===")
    
    lfs_files = [
        'fake_news_model.pkl',
        'political_news_classifier.pkl', 
        'WELFake_Dataset.csv',
        'News_Category_Dataset_v3.json'
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

def initialize_models():
    """Initialize and load both models with Git LFS verification"""
    try:
        print("=== Initializing Fake News Detector Models ===")
        
        # Check Git LFS files first
        available_files, missing_files = check_lfs_files()
        
        # Initialize fake news detection model
        print("\n=== Fake News Detection Model ===")
        if detector.load_model():
            print("✓ Using existing pre-trained fake news model.")
        else:
            print("⚠ No existing fake news model found. Attempting to train new model...")
            
            if 'WELFake_Dataset.csv' in available_files:
                print("Dataset available. Training new model...")
                print("⏳ This may take a few minutes...")
                
                try:
                    df = detector.load_and_prepare_data('WELFake_Dataset.csv')
                    accuracy = detector.train_best_model(df)
                    print(f"✓ Fake news model training completed with accuracy: {accuracy:.4f}")
                    
                    # Save the newly trained model
                    model_data = {
                        'model': detector.model,
                        'accuracy': accuracy,
                        'stemmer': detector.stemmer,
                        'stop_words': detector.stop_words,
                        'training_samples': len(df),
                        'created_date': datetime.now().isoformat()
                    }
                    joblib.dump(model_data, 'fake_news_model.pkl')
                    print("✓ Fake news model saved as 'fake_news_model.pkl'")
                    
                except Exception as e:
                    print(f"✗ Error training model: {str(e)}")
                    print("The application will run with limited functionality.")
            else:
                print("✗ Training dataset not available. Fake news detection will be unavailable.")
                print("Please ensure 'WELFake_Dataset.csv' is properly downloaded from Git LFS.")
        
        # Initialize political news classifier
        print("\n=== Political News Classification Model ===")
        if political_detector.load_model('political_news_classifier.pkl'):
            print("✓ Political news classifier loaded successfully.")
        else:
            print("⚠ Political news classifier not found. Political classification will be unavailable.")
            print("To enable political classification, ensure 'political_news_classifier.pkl' is properly downloaded from Git LFS.")
        
        # Final status
        print("\n=== Initialization Complete ===")
        fake_ready = detector.is_trained
        political_ready = political_detector.is_trained
        
        if fake_ready and political_ready:
            print("✓ Both models are ready!")
        elif fake_ready:
            print("✓ Fake news detection ready, political classification unavailable")
        elif political_ready:
            print("✓ Political classification ready, fake news detection unavailable")
        else:
            print("⚠ No models available - limited functionality")
            
    except Exception as e:
        print(f"✗ Error initializing models: {str(e)}")
        print("Please ensure Git LFS files are properly downloaded.")
        print("Traceback:", str(e))

if __name__ == '__main__':
    # Initialize models in a separate thread to avoid blocking
    model_thread = threading.Thread(target=initialize_models)
    model_thread.daemon = True
    model_thread.start()
    
    # Get port from environment variable for production deployment
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
