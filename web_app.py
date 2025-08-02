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
        self.load_feedback_data()
    
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
    
    def load_model(self, filepath='fake_news_model.pkl'):
        """Load a pre-trained model from disk"""
        try:
            if not os.path.exists(filepath):
                print(f"Model file '{filepath}' not found.")
                return False
                
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.stemmer = model_data['stemmer']
            self.stop_words = model_data['stop_words']
            self.accuracy = model_data['accuracy']
            self.is_trained = True
            
            # Load additional metadata if available
            training_samples = model_data.get('training_samples', 'Unknown')
            feedback_samples = model_data.get('feedback_samples', 0)
            last_retrain = model_data.get('last_retrain', 'Unknown')
            
            print(f"Model loaded successfully with accuracy: {self.accuracy:.4f}")
            print(f"Training samples: {training_samples}, Feedback samples: {feedback_samples}")
            if last_retrain != 'Unknown':
                print(f"Last retrained: {last_retrain}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
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
        """Load and prepare the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(filepath)
        
        # Handle missing values
        df['title'] = df['title'].fillna('')
        df['text'] = df['text'].fillna('')
        
        # Combine title and text
        df['combined_text'] = df['title'] + ' ' + df['text']
        
        # Preprocess the combined text
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
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

class PoliticalNewsDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False
        self.accuracy = None
        self.political_categories = {
            'POLITICS', 'U.S. NEWS', 'WORLD NEWS', 'CRIME'
        }
        self.political_keywords = {
            'government', 'politics', 'political', 'election', 'vote', 'voting', 'candidate',
            'president', 'senator', 'congress', 'parliament', 'minister', 'governor',
            'democrat', 'republican', 'party', 'campaign', 'policy', 'legislation',
            'bill', 'law', 'constitution', 'supreme court', 'federal', 'state',
            'mayor', 'city council', 'constituency', 'ballot', 'primary', 'debate',
            'administration', 'cabinet', 'house', 'senate', 'representative',
            'diplomatic', 'foreign policy', 'domestic policy', 'budget', 'tax',
            'reform', 'regulation', 'executive', 'judicial', 'legislative',
            'courthouse', 'trial', 'lawsuit', 'justice', 'attorney general'
        }
    
    def preprocess_text(self, text):
        """Advanced text preprocessing for political news classification"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep apostrophes for contractions
        text = re.sub(r'[^a-zA-Z\s\']', '', text)
        
        # Handle contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords and apply lemmatization
        processed_words = []
        for word in words:
            if word not in self.stop_words and len(word) > 2:
                lemmatized = self.lemmatizer.lemmatize(word)
                processed_words.append(lemmatized)
        
        return ' '.join(processed_words)
    
    def extract_political_features(self, text):
        """Extract political-specific features from text"""
        features = {}
        text_lower = text.lower()
        
        # Count political keywords
        political_count = sum(1 for keyword in self.political_keywords if keyword in text_lower)
        features['political_keyword_count'] = political_count
        features['political_keyword_ratio'] = political_count / max(len(text.split()), 1)
        
        # Check for political entities
        political_patterns = [
            r'\b(?:president|senator|governor|mayor|minister)\s+\w+',
            r'\b(?:mr\.|ms\.|mrs\.)\s+\w+',
            r'\b\w+\s+(?:administration|campaign|party)',
            r'\b(?:white house|congress|parliament|senate|house of representatives)'
        ]
        
        political_entities = 0
        for pattern in political_patterns:
            political_entities += len(re.findall(pattern, text_lower))
        
        features['political_entities'] = political_entities
        
        return features
    
    def predict(self, text):
        """Predict if text is political news"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet!")
        
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return {
                'prediction': 'Unknown',
                'confidence': 0.5,
                'probabilities': {'Non-Political': 0.5, 'Political': 0.5},
                'reasoning': 'Unable to analyze: Text is empty after preprocessing',
                'political_features': {},
                'error': 'Text is empty after preprocessing'
            }
        
        try:
            prediction = self.model.predict([processed_text])[0]
            probability = self.model.predict_proba([processed_text])[0]
            
            prediction_label = 'Political' if prediction == 1 else 'Non-Political'
            
            # Extract reasoning and features
            reasoning = self.extract_reasoning(text, prediction)
            political_features = self.extract_political_features(text)
            
            return {
                'prediction': prediction_label,
                'confidence': float(max(probability)),
                'probabilities': {
                    'Non-Political': float(probability[0]),
                    'Political': float(probability[1])
                },
                'reasoning': reasoning,
                'political_features': political_features
            }
        except Exception as e:
            return {
                'prediction': 'Unknown',
                'confidence': 0.5,
                'probabilities': {'Non-Political': 0.5, 'Political': 0.5},
                'reasoning': f'Error during prediction: {str(e)}',
                'political_features': {},
                'error': str(e)
            }
    
    def extract_reasoning(self, text, prediction):
        """Extract reasoning for the classification"""
        features = self.extract_political_features(text)
        text_lower = text.lower()
        
        reasoning_parts = []
        
        if prediction == 1:  # Political
            reasoning_parts.append(f"✓ Classified as POLITICAL news")
            
            if features['political_keyword_count'] > 0:
                reasoning_parts.append(f"• Found {features['political_keyword_count']} political keywords")
            
            if features['political_entities'] > 0:
                reasoning_parts.append(f"• Detected {features['political_entities']} political entities/references")
            
            # Check for specific political indicators
            if any(term in text_lower for term in ['election', 'vote', 'campaign']):
                reasoning_parts.append("• Contains election/voting related content")
            
            if any(term in text_lower for term in ['government', 'congress', 'senate', 'president']):
                reasoning_parts.append("• References government institutions or officials")
            
            if any(term in text_lower for term in ['policy', 'legislation', 'bill', 'law']):
                reasoning_parts.append("• Discusses policy or legislative matters")
        
        else:  # Non-Political
            reasoning_parts.append(f"✓ Classified as NON-POLITICAL news")
            
            if features['political_keyword_count'] == 0:
                reasoning_parts.append("• No political keywords detected")
            else:
                reasoning_parts.append(f"• Limited political content ({features['political_keyword_count']} keywords)")
            
            if features['political_entities'] == 0:
                reasoning_parts.append("• No political entities or references found")
            
            # Check for non-political indicators
            non_political_terms = ['sports', 'entertainment', 'technology', 'science', 'health', 'business']
            found_terms = [term for term in non_political_terms if term in text_lower]
            if found_terms:
                reasoning_parts.append(f"• Contains {', '.join(found_terms)} related content")
        
        return '\n'.join(reasoning_parts)
    
    def load_model(self, filepath='political_news_classifier.pkl'):
        """Load a pre-trained model"""
        try:
            if not os.path.exists(filepath):
                print(f"Political news model file '{filepath}' not found.")
                return False
                
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.stemmer = model_data.get('stemmer', PorterStemmer())
            self.lemmatizer = model_data.get('lemmatizer', WordNetLemmatizer())
            self.stop_words = model_data.get('stop_words', set(stopwords.words('english')))
            self.accuracy = model_data.get('accuracy', None)
            self.is_trained = True
            
            print(f"Political news model loaded successfully")
            if self.accuracy:
                print(f"Model accuracy: {self.accuracy:.4f}")
            
            return True
        except Exception as e:
            print(f"Error loading political news model: {str(e)}")
            return False

class NewsWebsiteCrawler:
    def __init__(self):
        # Enhanced article selectors with more comprehensive patterns
        self.common_article_selectors = [
            # Direct article links
            'a[href*="/article/"]',
            'a[href*="/articles/"]',
            'a[href*="/news/"]',
            'a[href*="/story/"]',
            'a[href*="/stories/"]',
            'a[href*="/posts/"]',
            'a[href*="/post/"]',
            'a[href*="/blog/"]',
            'a[href*="/blogs/"]',
            'a[href*="/content/"]',
            'a[href*="/read/"]',
            'a[href*="/feature/"]',
            'a[href*="/features/"]',
            'a[href*="/report/"]',
            'a[href*="/reports/"]',
            'a[href*="/analysis/"]',
            'a[href*="/opinion/"]',
            'a[href*="/editorial/"]',
            'a[href*="/breaking/"]',
            'a[href*="/latest/"]',
            'a[href*="/update/"]',
            'a[href*="/world/"]',
            'a[href*="/politics/"]',
            'a[href*="/business/"]',
            'a[href*="/sports/"]',
            'a[href*="/technology/"]',
            'a[href*="/tech/"]',
            'a[href*="/science/"]',
            'a[href*="/health/"]',
            'a[href*="/entertainment/"]',
            'a[href*="/lifestyle/"]',
            'a[href*="/local/"]',
            'a[href*="/national/"]',
            'a[href*="/international/"]',
            
            # Article container classes and IDs
            'article a[href]',
            '.article a[href]',
            '.article-link',
            '.article-title a',
            '.article-headline a',
            '.news-item a',
            '.news-link',
            '.news-title a',
            '.news-headline a',
            '.story-link',
            '.story-title a',
            '.story-headline a',
            '.post-link',
            '.post-title a',
            '.post-headline a',
            '.entry-title a',
            '.entry-link',
            '.content-title a',
            '.content-link',
            '.headline a',
            '.headline-link',
            '.title a',
            '.title-link',
            '.featured-article a',
            '.featured-story a',
            '.featured-news a',
            '.breaking-news a',
            '.latest-news a',
            '.top-story a',
            '.main-story a',
            '.lead-story a',
            
            # Header tags with links
            'h1 a[href]',
            'h2 a[href]',
            'h3 a[href]',
            'h4 a[href]',
            'h5 a[href]',
            'h6 a[href]',
            
            # Common news website patterns
            '.teaser a',
            '.teaser-title a',
            '.summary a',
            '.excerpt a',
            '.snippet a',
            '.preview a',
            '.card a[href]',
            '.card-title a',
            '.card-link',
            '.item a[href]',
            '.item-title a',
            '.list-item a',
            '.grid-item a',
            '.feed-item a',
            '.media a[href]',
            '.media-title a',
            '.thumbnail a',
            '.link-overlay',
            
            # News aggregator patterns
            '.article-list a',
            '.news-list a',
            '.story-list a',
            '.content-list a',
            '.feed a[href]',
            '.stream a[href]',
            '.listing a[href]',
            
            # WordPress and CMS patterns
            '.entry a[href]',
            '.post a[href]',
            '.content a[href]',
            '.wp-post a',
            '.hentry a',
            
            # Bootstrap and framework patterns
            '.list-group-item a',
            '.nav-link[href*="article"]',
            '.nav-link[href*="news"]',
            '.nav-link[href*="story"]',
            
            # Data attribute selectors
            'a[data-article]',
            'a[data-story]',
            'a[data-news]',
            'a[data-post]',
            'a[data-link-type="article"]',
            'a[data-content-type="article"]',
            'a[data-type="article"]',
            
            # Role-based selectors
            'a[role="article"]',
            'a[itemtype*="Article"]',
            
            # Specific news site patterns
            '.story-card a',
            '.article-card a',
            '.news-card a',
            '.content-card a',
            '.promo a[href]',
            '.promo-title a',
            '.module a[href]',
            '.widget a[href]',
            '.section a[href]'
        ]
        
        # Enhanced news indicators
        self.news_indicators = [
            'article', 'articles', 'news', 'story', 'stories', 'post', 'posts', 
            'blog', 'blogs', 'headline', 'headlines', 'breaking', 'update', 
            'updates', 'report', 'reports', 'analysis', 'exclusive', 'feature', 
            'features', 'content', 'read', 'editorial', 'opinion', 'commentary',
            'interview', 'review', 'preview', 'recap', 'roundup', 'digest',
            'bulletin', 'brief', 'briefing', 'alert', 'flash', 'live',
            'latest', 'recent', 'today', 'current', 'trending', 'popular',
            'world', 'politics', 'political', 'business', 'economy', 'finance',
            'sports', 'sport', 'technology', 'tech', 'science', 'health',
            'entertainment', 'lifestyle', 'culture', 'local', 'national',
            'international', 'global', 'breaking-news', 'top-story',
            'main-story', 'lead-story', 'featured', 'spotlight'
        ]
        
        # Enhanced exclude patterns
        self.exclude_patterns = [
            'mailto:', 'tel:', 'javascript:', '#', 'void(0)',
            'facebook.com', 'twitter.com', 'instagram.com', 'tiktok.com',
            'linkedin.com', 'youtube.com', 'pinterest.com', 'snapchat.com',
            'reddit.com', 'tumblr.com', 'whatsapp.com', 'telegram.me',
            'about', 'contact', 'privacy', 'terms', 'cookies', 'policy',
            'subscribe', 'newsletter', 'signup', 'login', 'register', 'account',
            'search', 'category', 'categories', 'tag', 'tags', 'author', 'authors',
            'archive', 'archives', 'sitemap', 'rss', 'feed', 'xml',
            'advertisement', 'ads', 'sponsor', 'affiliate', 'promo',
            'weather', 'horoscope', 'crossword', 'puzzle', 'game', 'games',
            'job', 'jobs', 'career', 'careers', 'classified', 'marketplace',
            'event', 'events', 'calendar', 'schedule', 'obituary', 'obituaries',
            'comment', 'comments', 'forum', 'discussion', 'chat',
            'photo', 'photos', 'gallery', 'galleries', 'video', 'videos',
            'podcast', 'podcasts', 'audio', 'multimedia',
            'subscription', 'paywall', 'premium', 'membership',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp',
            '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv'
        ]
        
        # Title extraction patterns for better article identification
        self.title_selectors = [
            'title', 'h1', 'h2', 'h3', '.title', '.headline', '.article-title',
            '.news-title', '.story-title', '.post-title', '.entry-title',
            '.content-title', '.page-title', '[data-title]'
        ]
    
    def is_news_link(self, href, link_text, link_element=None):
        """Determine if a link is likely a news article with enhanced detection"""
        if not href:
            return False
            
        # Clean and normalize URL and text
        href_lower = href.lower().strip()
        text_lower = link_text.lower().strip() if link_text else ""
        
        # Exclude unwanted patterns
        if any(pattern in href_lower for pattern in self.exclude_patterns):
            return False
        
        # Skip if link text is too short or generic
        if link_text and len(link_text.strip()) < 10:
            generic_words = ['more', 'read', 'click', 'here', 'link', 'view', 'see', 'go', 'next', 'prev']
            if text_lower in generic_words or text_lower.replace(' ', '') in ['readmore', 'clickhere', 'seemore']:
                return False
        
        # Check for news indicators in URL
        url_has_news_indicator = any(indicator in href_lower for indicator in self.news_indicators)
        
        # Check for news indicators in link text
        text_has_news_indicator = any(indicator in text_lower for indicator in self.news_indicators)
        
        # Check for date patterns in URL (very common in news sites)
        date_patterns = [
            r'/\d{4}/\d{1,2}/\d{1,2}/',     # /2023/12/25/
            r'/\d{4}-\d{1,2}-\d{1,2}/',     # /2023-12-25/
            r'/\d{4}/\d{1,2}/',             # /2023/12/
            r'/\d{8}/',                     # /20231225/
            r'/\d{6}/',                     # /202312/
            r'date=\d{4}-\d{1,2}-\d{1,2}',  # date=2023-12-25
            r'year=\d{4}',                  # year=2023
            r'month=\d{1,2}',               # month=12
            r'day=\d{1,2}'                  # day=25
        ]
        has_date_pattern = any(re.search(pattern, href) for pattern in date_patterns)
        
        # Check for article ID patterns
        id_patterns = [
            r'/\d{6,}/',           # Long numeric IDs
            r'/\d{4,}-',           # ID followed by dash
            r'id=\d+',             # id parameter
            r'article_id=\d+',     # article_id parameter
            r'story_id=\d+',       # story_id parameter
            r'post_id=\d+',        # post_id parameter
            r'/p/\d+',             # /p/123456
            r'/a/\d+',             # /a/123456
            r'/s/\d+',             # /s/123456
        ]
        has_id_pattern = any(re.search(pattern, href) for pattern in id_patterns)
        
        # Check for slug patterns (article titles in URLs)
        slug_patterns = [
            r'/[a-z0-9-]{15,}',    # Long hyphenated slugs
            r'/\w+-\w+-\w+',       # Multiple hyphenated words
            r'/how-',              # How-to articles
            r'/why-',              # Why articles
            r'/what-',             # What articles
            r'/when-',             # When articles
            r'/where-',            # Where articles
        ]
        has_slug_pattern = any(re.search(pattern, href_lower) for pattern in slug_patterns)
        
        # Check for common news URL structures
        news_url_patterns = [
            r'/breaking/',
            r'/latest/',
            r'/trending/',
            r'/featured/',
            r'/spotlight/',
            r'/exclusive/',
            r'/developing/',
            r'/update/',
            r'/alert/',
            r'/live/',
            r'/today/',
            r'/this-week/',
            r'/this-month/',
        ]
        has_news_url_pattern = any(re.search(pattern, href_lower) for pattern in news_url_patterns)
        
        # Check if link element has news-related attributes
        element_score = 0
        if link_element:
            # Check classes
            classes = link_element.get('class', [])
            if isinstance(classes, list):
                class_text = ' '.join(classes).lower()
            else:
                class_text = str(classes).lower()
            
            if any(indicator in class_text for indicator in ['article', 'news', 'story', 'post', 'headline', 'title']):
                element_score += 2
            
            # Check data attributes
            for attr in link_element.attrs:
                if attr.startswith('data-') and any(indicator in attr.lower() for indicator in ['article', 'news', 'story', 'post']):
                    element_score += 1
        
        # Check if link text looks like a news headline
        headline_indicators = [
            len(text_lower.split()) >= 4,  # At least 4 words
            any(word in text_lower for word in ['says', 'announces', 'reports', 'reveals', 'confirms', 'denies', 'claims']),
            any(char in link_text for char in ['"', "'", ':', '?', '!']),  # Punctuation common in headlines
            re.search(r'\b(new|latest|breaking|exclusive|update|first|last|next|final)\b', text_lower),
            re.search(r'\b(will|could|should|may|might|can|must)\b', text_lower),
            re.search(r'\b(after|before|during|since|while|when|where|why|how)\b', text_lower),
        ]
        headline_score = sum(1 for indicator in headline_indicators if indicator)
        
        # Calculate overall confidence score
        confidence_score = 0
        
        # URL-based scoring
        if url_has_news_indicator:
            confidence_score += 3
        if has_date_pattern:
            confidence_score += 2
        if has_id_pattern:
            confidence_score += 1
        if has_slug_pattern:
            confidence_score += 1
        if has_news_url_pattern:
            confidence_score += 2
        
        # Text-based scoring
        if text_has_news_indicator:
            confidence_score += 2
        confidence_score += headline_score
        
        # Element-based scoring
        confidence_score += element_score
        
        # Length bonus for substantial link text
        if link_text and 20 <= len(link_text) <= 200:
            confidence_score += 1
        
        # Return True if confidence score meets threshold
        return confidence_score >= 3
    
    def extract_article_links(self, url, max_links=10):
        """Extract article links from a news website with enhanced detection"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            
            article_links = []
            seen_urls = set()
            link_candidates = []
            
            print(f"Crawling website: {url}")
            print(f"Website title: {soup.title.string if soup.title else 'No title found'}")
            
            # First pass: Try specific article selectors
            for selector in self.common_article_selectors:
                try:
                    links = soup.select(selector)
                    print(f"Selector '{selector}' found {len(links)} links")
                    
                    for link in links:
                        href = link.get('href')
                        if not href:
                            continue
                        
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            href = urljoin(base_url, href)
                        elif not href.startswith('http'):
                            href = urljoin(url, href)
                        
                        # Avoid duplicates
                        if href in seen_urls:
                            continue
                        seen_urls.add(href)
                        
                        # Get link text and clean it
                        link_text = link.get_text(strip=True)
                        if not link_text:
                            # Try to get text from title attribute
                            link_text = link.get('title', '')
                        
                        # Try to get better title from surrounding elements
                        enhanced_title = self.extract_enhanced_title(link, link_text)
                        
                        if self.is_news_link(href, enhanced_title, link):
                            confidence_score = self.calculate_link_confidence(href, enhanced_title, link, selector)
                            link_candidates.append({
                                'url': href,
                                'title': enhanced_title[:150] + '...' if len(enhanced_title) > 150 else enhanced_title,
                                'selector_used': selector,
                                'confidence': confidence_score,
                                'link_element': str(link)[:200] + '...' if len(str(link)) > 200 else str(link)
                            })
                            
                except Exception as e:
                    print(f"Error with selector '{selector}': {str(e)}")
                    continue
            
            # Second pass: Broader search if we don't have enough candidates
            if len(link_candidates) < max_links:
                print("Running broader search for more articles...")
                
                # Look for articles and main content areas
                content_areas = soup.find_all(['article', 'main', 'section'], class_=re.compile(r'(content|article|news|story|post)', re.I))
                for area in content_areas:
                    area_links = area.find_all('a', href=True)
                    print(f"Found {len(area_links)} links in content area")
                    
                    for link in area_links:
                        if len(link_candidates) >= max_links * 2:  # Get extra candidates for sorting
                            break
                            
                        href = link.get('href')
                        if not href:
                            continue
                        
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            href = urljoin(base_url, href)
                        elif not href.startswith('http'):
                            href = urljoin(url, href)
                        
                        # Avoid duplicates
                        if href in seen_urls:
                            continue
                        seen_urls.add(href)
                        
                        link_text = link.get_text(strip=True)
                        enhanced_title = self.extract_enhanced_title(link, link_text)
                        
                        if self.is_news_link(href, enhanced_title, link):
                            confidence_score = self.calculate_link_confidence(href, enhanced_title, link, 'content_area_search')
                            link_candidates.append({
                                'url': href,
                                'title': enhanced_title[:150] + '...' if len(enhanced_title) > 150 else enhanced_title,
                                'selector_used': 'content_area_search',
                                'confidence': confidence_score,
                                'link_element': str(link)[:200] + '...' if len(str(link)) > 200 else str(link)
                            })
            
            # Third pass: If still not enough, try all links with strict filtering
            if len(link_candidates) < max_links // 2:
                print("Running final broad search...")
                all_links = soup.find_all('a', href=True)
                print(f"Found {len(all_links)} total links on page")
                
                for link in all_links:
                    if len(link_candidates) >= max_links * 3:  # Get many candidates for sorting
                        break
                    
                    href = link.get('href')
                    if not href:
                        continue
                    
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        href = urljoin(base_url, href)
                    elif not href.startswith('http'):
                        href = urljoin(url, href)
                    
                    # Avoid duplicates
                    if href in seen_urls:
                        continue
                    seen_urls.add(href)
                    
                    link_text = link.get_text(strip=True)
                    enhanced_title = self.extract_enhanced_title(link, link_text)
                    
                    if self.is_news_link(href, enhanced_title, link):
                        confidence_score = self.calculate_link_confidence(href, enhanced_title, link, 'broad_search')
                        link_candidates.append({
                            'url': href,
                            'title': enhanced_title[:150] + '...' if len(enhanced_title) > 150 else enhanced_title,
                            'selector_used': 'broad_search',
                            'confidence': confidence_score,
                            'link_element': str(link)[:200] + '...' if len(str(link)) > 200 else str(link)
                        })
            
            # Sort candidates by confidence score and take the best ones
            link_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            article_links = link_candidates[:max_links]
            
            print(f"Found {len(link_candidates)} total candidates, returning top {len(article_links)}")
            for i, article in enumerate(article_links[:5]):  # Log top 5
                print(f"  {i+1}. {article['title'][:50]}... (confidence: {article['confidence']})")
            
            return {
                'success': True,
                'articles': article_links,
                'total_found': len(article_links),
                'total_candidates': len(link_candidates),
                'website_title': soup.title.string if soup.title else urlparse(url).netloc
            }
            
        except requests.RequestException as e:
            print(f"Network error crawling {url}: {str(e)}")
            return {
                'success': False,
                'error': f'Network error: {str(e)}',
                'articles': []
            }
        except Exception as e:
            print(f"Parsing error crawling {url}: {str(e)}")
            return {
                'success': False,
                'error': f'Parsing error: {str(e)}',
                'articles': []
            }
    
    def extract_enhanced_title(self, link_element, fallback_text):
        """Extract enhanced title from link and surrounding elements"""
        title_text = fallback_text or ""
        
        # Try title attribute first
        if not title_text and link_element.get('title'):
            title_text = link_element.get('title').strip()
        
        # Try aria-label
        if not title_text and link_element.get('aria-label'):
            title_text = link_element.get('aria-label').strip()
        
        # Try data-title or similar attributes
        for attr in ['data-title', 'data-headline', 'data-text']:
            if not title_text and link_element.get(attr):
                title_text = link_element.get(attr).strip()
                break
        
        # Look for title in parent elements
        if not title_text or len(title_text) < 15:
            parent = link_element.parent
            for _ in range(3):  # Check up to 3 levels up
                if not parent:
                    break
                
                # Check for title in parent's text content
                parent_text = parent.get_text(strip=True)
                if parent_text and len(parent_text) > len(title_text) and len(parent_text) < 300:
                    # Make sure the parent text isn't just a container with multiple articles
                    links_in_parent = parent.find_all('a')
                    if len(links_in_parent) <= 2:  # Parent should contain mostly this article
                        title_text = parent_text
                        break
                
                # Check for specific title elements in parent
                for selector in ['.title', '.headline', '.article-title', 'h1', 'h2', 'h3', 'h4']:
                    title_elem = parent.select_one(selector)
                    if title_elem:
                        elem_text = title_elem.get_text(strip=True)
                        if elem_text and len(elem_text) > len(title_text):
                            title_text = elem_text
                            break
                
                parent = parent.parent
        
        # Look for title in child elements
        if not title_text or len(title_text) < 15:
            for selector in ['.title', '.headline', '.text', 'span', 'div']:
                child = link_element.select_one(selector)
                if child:
                    child_text = child.get_text(strip=True)
                    if child_text and len(child_text) > len(title_text):
                        title_text = child_text
                        break
        
        # Clean up the title
        if title_text:
            # Remove extra whitespace
            title_text = ' '.join(title_text.split())
            # Remove common prefixes/suffixes
            prefixes_to_remove = ['read more:', 'more:', 'full story:', 'story:', 'article:']
            for prefix in prefixes_to_remove:
                if title_text.lower().startswith(prefix):
                    title_text = title_text[len(prefix):].strip()
        
        return title_text or "Untitled Article"
    
    def calculate_link_confidence(self, href, title, link_element, selector_used):
        """Calculate confidence score for a link being a news article"""
        confidence = 0
        
        # Base score from is_news_link checks
        if self.is_news_link(href, title, link_element):
            confidence += 3
        
        # Bonus for specific selectors
        selector_bonuses = {
            'article a[href]': 5,
            '.article-title a': 5,
            '.news-title a': 5,
            '.story-title a': 5,
            'h1 a[href]': 4,
            'h2 a[href]': 4,
            'h3 a[href]': 3,
            '.headline a': 4,
            '.featured-article a': 4,
        }
        
        for pattern, bonus in selector_bonuses.items():
            if pattern in selector_used:
                confidence += bonus
                break
        
        # Title quality bonus
        if title and len(title) > 20:
            confidence += 2
        if title and any(word in title.lower() for word in ['breaking', 'exclusive', 'update', 'report']):
            confidence += 1
        
        # URL quality bonus
        href_lower = href.lower()
        if any(pattern in href_lower for pattern in ['/article/', '/news/', '/story/', '/post/']):
            confidence += 2
        
        # Date pattern bonus
        if re.search(r'/\d{4}/\d{1,2}/\d{1,2}/', href):
            confidence += 3
        
        return confidence
    
    def analyze_articles_batch(self, article_urls, analysis_type='both'):
        """Analyze multiple articles in parallel"""
        results = []
        
        def analyze_single_article(article_info):
            try:
                # Extract content from article
                content_result = extract_article_content(article_info['url'])
                
                if 'error' in content_result:
                    return {
                        'url': article_info['url'],
                        'title': article_info['title'],
                        'error': content_result['error'],
                        'status': 'failed'
                    }
                
                # Analyze the content
                text_to_analyze = content_result['combined']
                if not text_to_analyze.strip():
                    return {
                        'url': article_info['url'],
                        'title': article_info['title'],
                        'error': 'No content extracted',
                        'status': 'failed'
                    }
                
                analysis_result = {
                    'url': article_info['url'],
                    'title': article_info['title'],
                    'extracted_title': content_result.get('title', ''),
                    'content_preview': text_to_analyze[:200] + '...' if len(text_to_analyze) > 200 else text_to_analyze,
                    'status': 'success'
                }
                
                # Perform fake news detection
                if analysis_type in ['fake_news', 'both']:
                    try:
                        fake_result = detector.predict(text_to_analyze)
                        analysis_result['fake_news'] = fake_result
                    except Exception as e:
                        analysis_result['fake_news'] = {'error': str(e)}
                
                # Perform political classification
                if analysis_type in ['political', 'both']:
                    try:
                        political_result = political_detector.predict(text_to_analyze)
                        analysis_result['political_classification'] = political_result
                    except Exception as e:
                        analysis_result['political_classification'] = {'error': str(e)}
                
                return analysis_result
                
            except Exception as e:
                return {
                    'url': article_info['url'],
                    'title': article_info['title'],
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_article = {executor.submit(analyze_single_article, article): article 
                               for article in article_urls}
            
            for future in concurrent.futures.as_completed(future_to_article):
                try:
                    result = future.result(timeout=30)  # 30 second timeout per article
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    article = future_to_article[future]
                    results.append({
                        'url': article['url'],
                        'title': article['title'],
                        'error': 'Analysis timeout',
                        'status': 'timeout'
                    })
                except Exception as e:
                    article = future_to_article[future]
                    results.append({
                        'url': article['url'],
                        'title': article['title'],
                        'error': str(e),
                        'status': 'failed'
                    })
        
        return results

# Initialize the detectors and crawler
detector = FakeNewsDetector()
political_detector = PoliticalNewsDetector()
news_crawler = NewsWebsiteCrawler()

def extract_article_content(url):
    """Extract article content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find article content
        article_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.content',
            '.story-body',
            'main',
            '.entry-content'
        ]
        
        content = ""
        title = ""
        
        # Get title
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Try to find article content
        for selector in article_selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text().strip() for elem in elements])
                break
        
        # If no specific article content found, get all paragraphs
        if not content:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs])
        
        return {
            'title': title,
            'content': content,
            'combined': f"{title} {content}".strip()
        }
    
    except Exception as e:
        return {'error': f"Failed to extract content: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not detector.is_trained:
            return jsonify({'error': 'Fake news model not trained yet. Please wait for training to complete.'}), 500
        
        input_type = data.get('type', 'text')
        analysis_type = data.get('analysis_type', 'fake_news')  # 'fake_news', 'political', or 'both'
        
        if input_type == 'text':
            text = data.get('text', '').strip()
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            
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
            
            return jsonify(result)
        
        elif input_type == 'url':
            url = data.get('url', '').strip()
            if not url:
                return jsonify({'error': 'No URL provided'}), 400
            
            # Extract content from URL
            article_data = extract_article_content(url)
            
            if 'error' in article_data:
                return jsonify(article_data), 400
            
            combined_text = article_data['combined']
            if not combined_text.strip():
                return jsonify({'error': 'No content could be extracted from the URL'}), 400
            
            # Perform fake news detection
            fake_result = detector.predict(combined_text)
            result = {
                'fake_news': fake_result,
                'extracted_content': {
                    'title': article_data['title'],
                    'content_preview': combined_text[:500] + '...' if len(combined_text) > 500 else combined_text
                }
            }
            
            # Perform political classification if requested
            if analysis_type in ['political', 'both'] and political_detector.is_trained:
                political_result = political_detector.predict(combined_text)
                result['political_classification'] = political_result
            elif analysis_type in ['political', 'both']:
                result['political_classification'] = {
                    'error': 'Political classification model not available'
                }
            
            return jsonify(result)
        
        else:
            return jsonify({'error': 'Invalid input type'}), 400
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/crawl-website', methods=['POST'])
def crawl_website():
    """Crawl a news website for article links"""
    try:
        data = request.get_json()
        website_url = data.get('website_url', '').strip()
        max_articles = int(data.get('max_articles', 10))
        
        if not website_url:
            return jsonify({'error': 'Website URL is required'}), 400
        
        # Validate URL
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
        
        # Limit max articles to prevent abuse
        max_articles = min(max_articles, 20)
        
        # Crawl the website
        crawl_result = news_crawler.extract_article_links(website_url, max_articles)
        
        if not crawl_result['success']:
            return jsonify({
                'error': crawl_result['error'],
                'articles': []
            }), 400
        
        return jsonify({
            'success': True,
            'website_title': crawl_result['website_title'],
            'total_found': crawl_result['total_found'],
            'articles': crawl_result['articles']
        })
    
    except Exception as e:
        return jsonify({'error': f'Crawling failed: {str(e)}'}), 500

@app.route('/analyze-website', methods=['POST'])
def analyze_website():
    """Crawl and analyze articles from a news website"""
    try:
        data = request.get_json()
        website_url = data.get('website_url', '').strip()
        max_articles = int(data.get('max_articles', 5))
        analysis_type = data.get('analysis_type', 'both')
        
        if not website_url:
            return jsonify({'error': 'Website URL is required'}), 400
        
        # Validate URL
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
        
        # Limit max articles to prevent long processing times
        max_articles = min(max_articles, 10)
        
        # First crawl the website to get article links
        crawl_result = news_crawler.extract_article_links(website_url, max_articles)
        
        if not crawl_result['success']:
            return jsonify({
                'error': f"Failed to crawl website: {crawl_result['error']}",
                'results': []
            }), 400
        
        if not crawl_result['articles']:
            return jsonify({
                'error': 'No articles found on the website',
                'results': []
            }), 400
        
        # Analyze the found articles
        analysis_results = news_crawler.analyze_articles_batch(
            crawl_result['articles'], 
            analysis_type
        )
        
        # Calculate summary statistics
        successful_analyses = [r for r in analysis_results if r['status'] == 'success']
        failed_analyses = [r for r in analysis_results if r['status'] != 'success']
        
        summary = {
            'total_articles': len(crawl_result['articles']),
            'successful_analyses': len(successful_analyses),
            'failed_analyses': len(failed_analyses),
            'website_title': crawl_result['website_title']
        }
        
        # Add aggregated statistics for successful analyses
        if successful_analyses and analysis_type in ['fake_news', 'both']:
            fake_predictions = []
            for result in successful_analyses:
                if 'fake_news' in result and 'error' not in result['fake_news']:
                    fake_predictions.append(result['fake_news']['prediction'])
            
            if fake_predictions:
                fake_count = fake_predictions.count('Fake')
                real_count = fake_predictions.count('Real')
                summary['fake_news_summary'] = {
                    'fake_articles': fake_count,
                    'real_articles': real_count,
                    'fake_percentage': (fake_count / len(fake_predictions)) * 100 if fake_predictions else 0
                }
        
        if successful_analyses and analysis_type in ['political', 'both']:
            political_predictions = []
            for result in successful_analyses:
                if 'political_classification' in result and 'error' not in result['political_classification']:
                    political_predictions.append(result['political_classification']['prediction'])
            
            if political_predictions:
                political_count = political_predictions.count('Political')
                non_political_count = political_predictions.count('Non-Political')
                summary['political_summary'] = {
                    'political_articles': political_count,
                    'non_political_articles': non_political_count,
                    'political_percentage': (political_count / len(political_predictions)) * 100 if political_predictions else 0
                }
        
        return jsonify({
            'success': True,
            'summary': summary,
            'results': analysis_results
        })
    
    except Exception as e:
        return jsonify({'error': f'Website analysis failed: {str(e)}'}), 500

@app.route('/model-status')
def model_status():
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
    feedback_stats = detector.get_feedback_stats()
    status_info['feedback'] = feedback_stats
    
    return jsonify(status_info)

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

def initialize_models():
    """Initialize and load both models"""
    try:
        print("Initializing fake news detection model...")
        
        # First try to load existing fake news model
        if detector.load_model():
            print("Using existing pre-trained fake news model.")
        else:
            print("No existing fake news model found. Training new model...")
            print("This may take a few minutes...")
            
            # If no existing model, train a new one
            df = detector.load_and_prepare_data('WELFake_Dataset.csv')
            accuracy = detector.train_best_model(df)
            print(f"Fake news model training completed with accuracy: {accuracy:.4f}")
            
            # Save the newly trained model
            model_data = {
                'model': detector.model,
                'accuracy': accuracy,
                'stemmer': detector.stemmer,
                'stop_words': detector.stop_words
            }
            joblib.dump(model_data, 'fake_news_model.pkl')
            print("Fake news model saved as 'fake_news_model.pkl'")
        
        # Try to load political news classifier
        print("Initializing political news classification model...")
        if political_detector.load_model('political_news_classifier.pkl'):
            print("Political news classifier loaded successfully.")
        else:
            print("Political news classifier not found. Political classification will be unavailable.")
            print("To enable political classification, ensure 'political_news_classifier.pkl' is available.")
        
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        print("Please ensure 'WELFake_Dataset.csv' is in the project directory.")

if __name__ == '__main__':
    # Initialize models in a separate thread to avoid blocking
    import threading
    model_thread = threading.Thread(target=initialize_models)
    model_thread.daemon = True
    model_thread.start()
    
    app.run(debug=True, port=5000)
