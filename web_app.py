from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
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
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

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

# Initialize the detector
detector = FakeNewsDetector()

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
            return jsonify({'error': 'Model not trained yet. Please wait for training to complete.'}), 500
        
        input_type = data.get('type', 'text')
        
        if input_type == 'text':
            text = data.get('text', '').strip()
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            
            result = detector.predict(text)
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
            
            result = detector.predict(combined_text)
            result['extracted_content'] = {
                'title': article_data['title'],
                'content_preview': combined_text[:500] + '...' if len(combined_text) > 500 else combined_text
            }
            
            return jsonify(result)
        
        else:
            return jsonify({'error': 'Invalid input type'}), 400
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/model-status')
def model_status():
    status_info = {
        'is_trained': detector.is_trained,
        'status': 'Model is ready' if detector.is_trained else 'Model is training...'
    }
    
    if detector.is_trained and detector.accuracy:
        status_info['accuracy'] = f"{detector.accuracy:.4f}"
        status_info['status'] = f"Model ready (Accuracy: {detector.accuracy:.1%})"
    
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

def initialize_model():
    """Initialize and train the model"""
    try:
        print("Initializing fake news detection model...")
        
        # First try to load existing model
        if detector.load_model():
            print("Using existing pre-trained model.")
            return
        
        print("No existing model found. Training new model...")
        print("This may take a few minutes...")
        
        # If no existing model, train a new one
        df = detector.load_and_prepare_data('WELFake_Dataset.csv')
        accuracy = detector.train_best_model(df)
        print(f"Model training completed with accuracy: {accuracy:.4f}")
        
        # Save the newly trained model
        model_data = {
            'model': detector.model,
            'accuracy': accuracy,
            'stemmer': detector.stemmer,
            'stop_words': detector.stop_words
        }
        joblib.dump(model_data, 'fake_news_model.pkl')
        print("Model saved as 'fake_news_model.pkl'")
        
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        print("Please ensure 'WELFake_Dataset.csv' is in the project directory.")

if __name__ == '__main__':
    # Initialize model in a separate thread to avoid blocking
    import threading
    model_thread = threading.Thread(target=initialize_model)
    model_thread.daemon = True
    model_thread.start()
    
    app.run(debug=True, port=5000)
