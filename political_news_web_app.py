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
            'POLITICS', 'U.S. NEWS', 'WORLD NEWS', 'CRIME'  # Crime often has political elements
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
        
        # Check for political time references
        date_patterns = [
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}',
            r'\b\d{4}\s+election',
            r'\b(?:20\d{2}|19\d{2})\b'
        ]
        
        date_count = 0
        for pattern in date_patterns:
            date_count += len(re.findall(pattern, text_lower))
        
        features['date_references'] = date_count
        
        return features
    
    def load_news_category_dataset(self, filepath='News_Category_Dataset_v3.json', max_samples=8000):
        """Load and process the News Category Dataset v3 for web app"""
        print(f"Loading News Category Dataset from {filepath}...")
        
        data = []
        political_count = 0
        non_political_count = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if len(data) >= max_samples:
                        break
                    
                    try:
                        record = json.loads(line.strip())
                        
                        category = record.get('category', '').upper()
                        headline = record.get('headline', '')
                        short_description = record.get('short_description', '')
                        
                        combined_text = f"{headline} {short_description}".strip()
                        
                        if len(combined_text) < 20:
                            continue
                        
                        is_political = 1 if category in self.political_categories else 0
                        
                        # Balance the dataset
                        if is_political and political_count >= max_samples // 2:
                            continue
                        elif not is_political and non_political_count >= max_samples // 2:
                            continue
                        
                        data.append({
                            'text': combined_text,
                            'category': category,
                            'is_political': is_political
                        })
                        
                        if is_political:
                            political_count += 1
                        else:
                            non_political_count += 1
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue
        
        except FileNotFoundError:
            return self.create_sample_dataset()
        
        df = pd.DataFrame(data)
        print(f"Dataset loaded: {len(df)} samples ({political_count} political, {non_political_count} non-political)")
        
        return df

    def create_sample_dataset(self):
        """Create a sample dataset with political and non-political news"""
        political_samples = [
            "President announces new economic policy for the upcoming fiscal year",
            "Senate votes on healthcare reform bill after heated debate",
            "Governor's campaign faces criticism over education funding proposal",
            "Congressional hearing reveals concerns about foreign interference",
            "Mayor implements new housing policies to address urban development",
            "Supreme Court decision impacts voting rights legislation",
            "Democratic candidate leads in latest election polls",
            "Republican party unveils tax reform proposal",
            "White House responds to international diplomatic crisis",
            "Local representative introduces environmental protection bill",
            "Federal budget allocation sparks political controversy",
            "Prime Minister discusses trade agreements with foreign delegates",
            "Election results show significant voter turnout increase",
            "Parliamentary session addresses constitutional amendments",
            "City council approves new zoning regulations",
            "Political analyst predicts close race in upcoming primary",
            "Administration's foreign policy faces bipartisan criticism",
            "State legislature passes controversial immigration bill",
            "Campaign finance reports reveal major donor contributions",
            "Judicial committee reviews Supreme Court nomination",
            "Vice President visits swing states before midterm elections",
            "Congressional committee investigates executive branch actions",
            "Local politician announces bid for higher office",
            "International summit addresses global political tensions",
            "Voting rights activists rally at state capitol building"
        ]
        
        non_political_samples = [
            "Scientists discover new species in Amazon rainforest",
            "Tech company launches revolutionary smartphone with AI features",
            "Local restaurant wins national culinary competition",
            "Weather forecast predicts severe storms this weekend",
            "Professional sports team signs star player for record deal",
            "Medical researchers develop breakthrough cancer treatment",
            "University announces expansion of engineering program",
            "Stock market reaches new highs amid economic growth",
            "Celebrity couple announces engagement on social media",
            "Archaeological team uncovers ancient civilization artifacts",
            "Startup company revolutionizes renewable energy technology",
            "Fashion week showcases latest trends from top designers",
            "Movie premiere attracts thousands of fans downtown",
            "Hospital performs first robotic surgery in the region",
            "Art museum opens new exhibition featuring contemporary artists",
            "Music festival brings international performers to the city",
            "Space agency launches satellite for climate monitoring",
            "Chef opens new restaurant featuring fusion cuisine",
            "Technology conference highlights artificial intelligence advances",
            "Marathon event raises funds for local charity organization",
            "Wildlife conservation efforts protect endangered species",
            "Automotive industry unveils electric vehicle innovations",
            "Educational institution receives grant for research project",
            "Entertainment industry celebrates award show winners",
            "Sports championship draws record television audience"
        ]
        
        # Create DataFrame
        data = []
        for text in political_samples:
            data.append({'text': text, 'category': 'POLITICS', 'is_political': 1})
        for text in non_political_samples:
            data.append({'text': text, 'category': 'ENTERTAINMENT', 'is_political': 0})
        
        return pd.DataFrame(data)

    def prepare_and_train_model(self):
        """Prepare data and train the model using real dataset"""
        print("Loading News Category Dataset...")
        try:
            df = self.load_news_category_dataset('News_Category_Dataset_v3.json', max_samples=8000)
        except:
            print("Failed to load JSON dataset, using sample data...")
            df = self.create_sample_dataset()
        
        print("Preprocessing text data...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        df = df[df['processed_text'].str.len() > 0]
        
        # Split data
        X = df['processed_text']
        y = df['is_political']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Test multiple models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        best_accuracy = 0
        best_model = None
        
        for name, model in models.items():
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))),
                ('classifier', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = pipeline
        
        self.model = best_model
        self.accuracy = best_accuracy
        self.is_trained = True
        
        print(f"Best model selected with accuracy: {best_accuracy:.4f}")
        return best_accuracy
    
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
            reasoning_parts.append("This text was classified as POLITICAL news based on:")
            
            if features['political_keyword_count'] > 0:
                found_keywords = [kw for kw in self.political_keywords if kw in text_lower]
                reasoning_parts.append(f"🏛️ Political keywords detected: {', '.join(found_keywords[:5])}")
            
            if features['political_entities'] > 0:
                reasoning_parts.append(f"👤 Political entities/references found: {features['political_entities']}")
            
            if features['date_references'] > 0:
                reasoning_parts.append(f"📅 Time-sensitive political references: {features['date_references']}")
            
            # Specific indicators
            if any(word in text_lower for word in ['election', 'vote', 'campaign']):
                reasoning_parts.append("🗳️ Election/voting-related content detected")
            
            if any(word in text_lower for word in ['president', 'senator', 'governor', 'mayor']):
                reasoning_parts.append("🏢 Government officials mentioned")
            
            if any(word in text_lower for word in ['policy', 'legislation', 'bill', 'law']):
                reasoning_parts.append("📜 Policy/legislative content identified")
        
        else:  # Non-Political
            reasoning_parts.append("This text was classified as NON-POLITICAL news based on:")
            
            if features['political_keyword_count'] == 0:
                reasoning_parts.append("❌ No political keywords detected")
            
            non_political_keywords = [
                'sports', 'entertainment', 'technology', 'science', 'health',
                'business', 'weather', 'travel', 'food', 'fashion', 'art'
            ]
            
            found_non_political = [kw for kw in non_political_keywords if kw in text_lower]
            if found_non_political:
                reasoning_parts.append(f"📰 Non-political content indicators: {', '.join(found_non_political[:3])}")
            
            if features['political_entities'] == 0:
                reasoning_parts.append("🚫 No political entities or officials mentioned")
        
        return '\n'.join(reasoning_parts)
    
    def load_model(self, filepath='political_news_model.pkl'):
        """Load a pre-trained model"""
        try:
            if not os.path.exists(filepath):
                return False
            
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.accuracy = model_data['accuracy']
            self.stemmer = model_data['stemmer']
            self.lemmatizer = model_data['lemmatizer']
            self.stop_words = model_data['stop_words']
            self.political_keywords = model_data['political_keywords']
            self.is_trained = True
            
            print(f"Model loaded successfully with accuracy: {self.accuracy:.4f}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

# Initialize the detector
detector = PoliticalNewsDetector()

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
            'article', '.article-content', '.post-content', '.content',
            '.story-body', 'main', '.entry-content'
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
    return render_template('political_news_index.html')

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
                'content_preview': combined_text
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
    
    return jsonify(status_info)

def initialize_model():
    """Initialize and train the model"""
    try:
        print("Initializing political news classification model...")
        
        # First try to load existing model
        if detector.load_model():
            print("Using existing pre-trained model.")
            return
        
        print("No existing model found. Training new model...")
        print("This may take a few minutes...")
        
        # Train a new model
        accuracy = detector.prepare_and_train_model()
        print(f"Model training completed with accuracy: {accuracy:.4f}")
        
        # Save the newly trained model
        model_data = {
            'model': detector.model,
            'accuracy': accuracy,
            'stemmer': detector.stemmer,
            'lemmatizer': detector.lemmatizer,
            'stop_words': detector.stop_words,
            'political_keywords': detector.political_keywords,
            'timestamp': datetime.now().isoformat()
        }
        joblib.dump(model_data, 'political_news_model.pkl')
        print("Model saved as 'political_news_model.pkl'")
        
    except Exception as e:
        print(f"Error initializing model: {str(e)}")

if __name__ == '__main__':
    # Initialize model in a separate thread to avoid blocking
    model_thread = threading.Thread(target=initialize_model)
    model_thread.daemon = True
    model_thread.start()
    
    app.run(debug=True, port=5001)
