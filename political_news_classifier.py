import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
import pickle
import joblib
import os
from datetime import datetime
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

class PoliticalNewsClassifier:
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
        
        # Remove stopwords and apply stemming/lemmatization
        processed_words = []
        for word in words:
            if word not in self.stop_words and len(word) > 2:
                # Use lemmatization for better word forms
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
        
        # Check for political entities (basic patterns)
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
        
        # Check for dates (elections, terms, etc.)
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
    
    def load_news_category_dataset(self, filepath='News_Category_Dataset_v3.json'):
        """Load and process the News Category Dataset v3"""
        print(f"Loading News Category Dataset from {filepath}...")
        
        data = []
        political_count = 0
        non_political_count = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    
                    try:
                        record = json.loads(line.strip())
                        
                        # Extract relevant fields
                        category = record.get('category', '').upper()
                        headline = record.get('headline', '')
                        short_description = record.get('short_description', '')
                        authors = record.get('authors', '')
                        date = record.get('date', '')
                        link = record.get('link', '')
                        
                        # Combine headline and short description for text analysis
                        combined_text = f"{headline} {short_description}".strip()
                        
                        # Skip if text is too short
                        if len(combined_text) < 20:
                            continue
                        
                        # Determine if political news
                        is_political = 1 if category in self.political_categories else 0
                        
                        data.append({
                            'text': combined_text,
                            'category': category,
                            'is_political': is_political,
                            'headline': headline,
                            'short_description': short_description,
                            'authors': authors,
                            'date': date,
                            'link': link
                        })
                        
                        if is_political:
                            political_count += 1
                        else:
                            non_political_count += 1
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error processing line {i}: {e}")
                        continue
        
        except FileNotFoundError:
            print(f"Dataset file {filepath} not found. Creating sample dataset instead.")
            return self.create_sample_dataset()
        
        df = pd.DataFrame(data)
        
        print(f"Dataset loaded with {len(df)} samples")
        print(f"Political news: {political_count} samples")
        print(f"Non-political news: {non_political_count} samples")
        
        # Show category distribution
        print("\nCategory distribution:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.head(10).items():
            status = "✓ Political" if category in self.political_categories else "• Non-Political"
            print(f"  {category}: {count} ({status})")
        
        return df
    
    def create_sample_dataset(self):
        """Create a sample dataset with political and non-political news (fallback)"""
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
            "Local representative introduces environmental protection bill"
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
            "Archaeological team uncovers ancient civilization artifacts"
        ]
        
        # Create DataFrame
        data = []
        for text in political_samples:
            data.append({'text': text, 'category': 'POLITICS', 'is_political': 1})
        for text in non_political_samples:
            data.append({'text': text, 'category': 'ENTERTAINMENT', 'is_political': 0})
        
        return pd.DataFrame(data)

    def load_or_create_data(self, filepath=None):
        """Load data from News Category Dataset or create sample data"""
        if filepath and filepath.endswith('.json'):
            # Load JSON dataset
            return self.load_news_category_dataset(filepath)
        elif filepath and os.path.exists(filepath):
            # Load CSV dataset
            print(f"Loading data from {filepath}...")
            df = pd.read_csv(filepath)
            if 'text' not in df.columns:
                raise ValueError("Dataset must contain 'text' column")
            if 'is_political' not in df.columns and 'category' in df.columns:
                df['is_political'] = df['category'].apply(lambda x: 1 if 'political' in x.lower() else 0)
            elif 'is_political' not in df.columns:
                raise ValueError("Dataset must contain 'is_political' or 'category' column")
        else:
            # Try to load the default JSON dataset, fallback to sample data
            try:
                return self.load_news_category_dataset('News_Category_Dataset_v3.json')
            except:
                print("Using sample dataset...")
                df = self.create_sample_dataset()
        
        print(f"Dataset loaded with {len(df)} samples")
        if 'is_political' in df.columns:
            print(f"Political news: {sum(df['is_political'])} samples")
            print(f"Non-political news: {len(df) - sum(df['is_political'])} samples")
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        print("Preprocessing text data...")
        
        # Handle missing values
        df['text'] = df['text'].fillna('')
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        # Extract additional features
        print("Extracting political features...")
        political_features = df['text'].apply(self.extract_political_features)
        feature_df = pd.DataFrame(political_features.tolist())
        
        # Combine text and additional features (for advanced models)
        df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
        
        print(f"Data preparation complete. Final dataset: {len(df)} samples")
        return df
    
    def train_models(self, X_train, X_test, y_train, y_test, additional_features_train=None, additional_features_test=None):
        """Train multiple models and select the best one"""
        print("Training multiple models...")
        
        # Models to test
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB()
        }
        
        # Vectorizers to test
        vectorizers = {
            'TF-IDF': TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2)),
            'Count': CountVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        }
        
        results = {}
        best_score = 0
        best_model = None
        best_model_name = ""
        
        for vec_name, vectorizer in vectorizers.items():
            for model_name, model in models.items():
                print(f"Training {model_name} with {vec_name}...")
                
                # Create pipeline
                pipeline = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', model)
                ])
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                # Cross-validation score
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                
                # Store results
                model_key = f"{model_name} + {vec_name}"
                results[model_key] = {
                    'accuracy': accuracy,
                    'auc': auc_score,
                    'cv_score': cv_mean,
                    'pipeline': pipeline
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  AUC: {auc_score:.4f}")
                print(f"  CV Score: {cv_mean:.4f}")
                
                # Update best model
                if cv_mean > best_score:
                    best_score = cv_mean
                    best_model = pipeline
                    best_model_name = model_key
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best CV score: {best_score:.4f}")
        
        self.model = best_model
        self.accuracy = best_score
        self.is_trained = True
        
        # Print detailed results for best model
        print(f"\nDetailed results for best model:")
        y_pred = best_model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=['Non-Political', 'Political']))
        
        return results
    
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
                'error': 'Text is empty after preprocessing'
            }
        
        try:
            prediction = self.model.predict([processed_text])[0]
            probability = self.model.predict_proba([processed_text])[0]
            
            # Extract reasoning
            reasoning = self.extract_reasoning(text, prediction)
            
            return {
                'prediction': 'Political' if prediction == 1 else 'Non-Political',
                'confidence': float(max(probability)),
                'probabilities': {
                    'Non-Political': float(probability[0]),
                    'Political': float(probability[1])
                },
                'reasoning': reasoning,
                'political_features': self.extract_political_features(text)
            }
        except Exception as e:
            return {
                'prediction': 'Unknown',
                'confidence': 0.5,
                'probabilities': {'Non-Political': 0.5, 'Political': 0.5},
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
                reasoning_parts.append(f"• Political keywords detected: {', '.join(found_keywords[:5])}")
            
            if features['political_entities'] > 0:
                reasoning_parts.append(f"• Political entities/references found: {features['political_entities']}")
            
            if features['date_references'] > 0:
                reasoning_parts.append(f"• Time-sensitive political references: {features['date_references']}")
            
            # Check for specific political indicators
            if any(word in text_lower for word in ['election', 'vote', 'campaign']):
                reasoning_parts.append("• Election/voting-related content detected")
            
            if any(word in text_lower for word in ['president', 'senator', 'governor', 'mayor']):
                reasoning_parts.append("• Government officials mentioned")
            
            if any(word in text_lower for word in ['policy', 'legislation', 'bill', 'law']):
                reasoning_parts.append("• Policy/legislative content identified")
        
        else:  # Non-Political
            reasoning_parts.append("This text was classified as NON-POLITICAL news based on:")
            
            if features['political_keyword_count'] == 0:
                reasoning_parts.append("• No political keywords detected")
            
            # Check for non-political indicators
            non_political_keywords = [
                'sports', 'entertainment', 'technology', 'science', 'health',
                'business', 'weather', 'travel', 'food', 'fashion', 'art'
            ]
            
            found_non_political = [kw for kw in non_political_keywords if kw in text_lower]
            if found_non_political:
                reasoning_parts.append(f"• Non-political content indicators: {', '.join(found_non_political[:3])}")
            
            if features['political_entities'] == 0:
                reasoning_parts.append("• No political entities or officials mentioned")
        
        return '\n'.join(reasoning_parts)
    
    def save_model(self, filepath='political_news_classifier.pkl'):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save!")
        
        model_data = {
            'model': self.model,
            'accuracy': self.accuracy,
            'stemmer': self.stemmer,
            'lemmatizer': self.lemmatizer,
            'stop_words': self.stop_words,
            'political_keywords': self.political_keywords,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='political_news_classifier.pkl'):
        """Load a pre-trained model"""
        try:
            if not os.path.exists(filepath):
                print(f"Model file '{filepath}' not found.")
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

def main():
    # Initialize classifier
    classifier = PoliticalNewsClassifier()
    
    # Try to load existing model
    if classifier.load_model():
        print("Using existing trained model.")
    else:
        print("Training new model...")
        
        # Load or create data
        df = classifier.load_or_create_data()  # Load from JSON dataset
        
        # Prepare data
        df = classifier.prepare_data(df)
        
        # Split data
        X = df['processed_text']
        y = df['is_political']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train models
        results = classifier.train_models(X_train, X_test, y_train, y_test)
        
        # Save the model
        classifier.save_model()
    
    # Test with sample texts
    test_texts = [
        "President Biden announces new infrastructure spending plan",
        "Local restaurant wins award for best pizza in the city",
        "Senate hearing on climate change legislation begins tomorrow",
        "Scientists discover cure for rare genetic disease",
        "Governor's approval ratings drop after tax increase proposal",
        "Technology company releases innovative electric vehicle",
        "Congressional committee investigates campaign finance violations",
        "Professional basketball team trades star player",
        "Supreme Court to hear case on voting rights",
        "Weather forecast shows sunny skies for the weekend"
    ]
    
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    
    for text in test_texts:
        result = classifier.predict(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: Non-Political={result['probabilities']['Non-Political']:.4f}, Political={result['probabilities']['Political']:.4f}")
        if 'reasoning' in result:
            print(f"Reasoning: {result['reasoning']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
