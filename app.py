import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
import pickle
import joblib
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
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
    
    def load_and_explore_data(self, filepath):
        """Load and explore the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(filepath)
        
        print(f"Dataset shape: {df.shape}")
        print(f"\nDataset info:")
        print(df.info())
        
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
        
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
        # Visualize label distribution
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        df['label'].value_counts().plot(kind='bar')
        plt.title('Label Distribution')
        plt.xlabel('Label (0=Fake, 1=Real)')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        df['label'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Label Distribution (Percentage)')
        plt.ylabel('')
        
        plt.tight_layout()
        plt.show()
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        print("Preprocessing data...")
        
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
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare performance"""
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Naive Bayes': MultinomialNB()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
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
            
            # Store results
            results[name] = {
                'model': pipeline,
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            print(f"\nClassification Report for {name}:")
            print(classification_report(y_test, y_pred))
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        self.model = results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
        
        # Save the best model
        self.save_model(results[best_model_name]['accuracy'])
        
        return results
    
    def save_model(self, accuracy):
        """Save the trained model to disk"""
        if self.model is not None:
            model_data = {
                'model': self.model,
                'accuracy': accuracy,
                'stemmer': self.stemmer,
                'stop_words': self.stop_words
            }
            
            # Save using joblib for better scikit-learn compatibility
            joblib.dump(model_data, 'fake_news_model.pkl')
            print(f"Model saved as 'fake_news_model.pkl' with accuracy: {accuracy:.4f}")
    
    def load_model(self, filepath='fake_news_model.pkl'):
        """Load a pre-trained model from disk"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.stemmer = model_data['stemmer']
            self.stop_words = model_data['stop_words']
            accuracy = model_data['accuracy']
            print(f"Model loaded successfully with accuracy: {accuracy:.4f}")
            return True
        except FileNotFoundError:
            print(f"Model file '{filepath}' not found.")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, text):
        """Predict if a news article is fake or real"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        processed_text = self.preprocess_text(text)
        prediction = self.model.predict([processed_text])[0]
        probability = self.model.predict_proba([processed_text])[0]
        
        return {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': max(probability),
            'probabilities': {
                'Fake': probability[0],
                'Real': probability[1]
            }
        }

def main():
    # Initialize the detector
    detector = FakeNewsDetector()
    
    # Check if a pre-trained model exists
    print("Checking for existing trained model...")
    if detector.load_model():
        print("Using existing trained model.")
        
        # Test the loaded model with sample predictions
        sample_texts = [
            "Breaking: Scientists discover new planet with potential for life",
            "URGENT: Government plans secret alien cover-up exposed by whistleblower",
            "Stock market reaches new highs amid economic recovery"
        ]
        
        print("\n" + "="*50)
        print("SAMPLE PREDICTIONS (LOADED MODEL)")
        print("="*50)
        
        for text in sample_texts:
            result = detector.predict(text)
            print(f"\nText: {text}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities: Fake={result['probabilities']['Fake']:.4f}, Real={result['probabilities']['Real']:.4f}")
        
        return
    
    print("No existing model found. Training new model...")
    
    # Load and explore data
    df = detector.load_and_explore_data('WELFake_Dataset.csv')
    
    # Prepare data
    df = detector.prepare_data(df)
    
    # Split data
    X = df['processed_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train models
    results = detector.train_models(X_train, X_test, y_train, y_test)
    
    # Example predictions
    sample_texts = [
        "Breaking: Scientists discover new planet with potential for life",
        "URGENT: Government plans secret alien cover-up exposed by whistleblower",
        "Stock market reaches new highs amid economic recovery"
    ]
    
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    for text in sample_texts:
        result = detector.predict(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: Fake={result['probabilities']['Fake']:.4f}, Real={result['probabilities']['Real']:.4f}")

if __name__ == "__main__":
    main()