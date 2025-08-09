"""
Political News Detector Module
Advanced classifier for identifying political news content
"""

import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from .utils import extract_political_keywords


class PoliticalNewsDetector:
    """
    Advanced classifier for identifying political news content
    Uses machine learning and keyword analysis to classify news articles
    """
    
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
        # Use centralized political keywords from utils
        self.political_keywords = extract_political_keywords()
    
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

    def load_model(self, filepath='political_news_classifier.pkl'):
        """Load a pre-trained model with Git LFS verification"""
        try:
            print(f"Attempting to load political model from: {filepath}")
            
            # First verify the file is properly downloaded from Git LFS
            is_valid, message = self.verify_lfs_file(filepath)
            if not is_valid:
                print(f"LFS verification failed: {message}")
                
                # Try alternative locations for the model file
                alternative_paths = [
                    os.path.join(os.getcwd(), filepath),
                    os.path.join(os.path.dirname(__file__), filepath),
                    os.path.join('/tmp', filepath),
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        print(f"Trying alternative path: {alt_path}")
                        is_valid, message = self.verify_lfs_file(alt_path)
                        if is_valid:
                            filepath = alt_path
                            break
                
                if not is_valid:
                    print(f"Political news model file '{filepath}' not found or not properly downloaded from Git LFS.")
                    return False
            
            print(f"Loading political model from verified file: {filepath}")
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.stemmer = model_data.get('stemmer', PorterStemmer())
            self.lemmatizer = model_data.get('lemmatizer', WordNetLemmatizer())
            self.stop_words = model_data.get('stop_words', set(stopwords.words('english')))
            self.accuracy = model_data.get('accuracy', None)
            self.is_trained = True
            
            print(f"✓ Political news model loaded successfully")
            if self.accuracy:
                print(f"  Model accuracy: {self.accuracy:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error loading political news model: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            return False
    
    def train_model(self, df):
        """Train the political news classifier"""
        try:
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
            self.accuracy = best_accuracy
            self.is_trained = True
            print(f"Best political news model selected with accuracy: {best_accuracy:.4f}")
            
            return best_accuracy
            
        except Exception as e:
            print(f"Error training political news model: {str(e)}")
            return None
    
    def save_model(self, filepath='political_news_classifier.pkl'):
        """Save the trained model to disk"""
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("No trained model to save")
            
            model_data = {
                'model': self.model,
                'stemmer': self.stemmer,
                'lemmatizer': self.lemmatizer,
                'stop_words': self.stop_words,
                'accuracy': self.accuracy,
                'political_keywords': self.political_keywords,
                'political_categories': self.political_categories
            }
            
            joblib.dump(model_data, filepath)
            print(f"✓ Political news model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving political news model: {str(e)}")
            return False
