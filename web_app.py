import os
import threading
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Import modular components
from modules.fake_news_detector import FakeNewsDetector
from modules.political_news_detector import PoliticalNewsDetector
from modules.philippine_news_search_index import PhilippineNewsSearchIndex

# Import routes
from routes.news_crawler_routes import news_crawler_bp
from routes.philippine_news_search_routes import philippine_news_bp, init_philippine_search_index
from routes.political_news_routes import political_news_bp, init_political_detector
from routes.fake_news_routes import fake_news_bp, init_fake_news_detector
from routes.url_classifier_routes import url_classifier_bp, init_url_classifier

# Import utility functions from political news detector
from modules.political_news_detector.utils import extract_political_content_from_url

app = Flask(__name__)

# Initialize detectors
detector = FakeNewsDetector()
political_detector = PoliticalNewsDetector()

@app.route('/')
def index():
    """Main page"""
    return render_template('news_analysis.html')

@app.route('/philippine-search')
def philippine_search():
    """Philippine news search page"""
    return render_template('philippine_search.html')

@app.route('/news-analysis')
def news_analysis():
    """News analysis page with optional URL prefill"""
    url = request.args.get('url', '')
    if url and not url.startswith(('http://', 'https://')):
        # Redirect invalid URLs back to the main page
        return redirect(url_for('index'))
    return render_template('news_analysis.html', prefill_url=url)

@app.route('/health')
def health_check():
    """Health check endpoint with Git LFS file status"""
    try:
        # Check Git LFS files
        lfs_files = [
            'fake_news_model.pkl',
            'political_news_classifier.pkl', 
            'datasets/WELFake_Dataset.csv',
            'datasets/News_Category_Dataset_v3.json'
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

def initialize_models():
    """Initialize all models and components"""
    try:
        print("=== Model Initialization Started ===")
        
        # Initialize Philippine news search index
        print("\n=== Philippine News Search Index ===")
        try:
            from routes.philippine_news_search_routes import get_philippine_search_index
            search_index = get_philippine_search_index()
            search_index.initialize()
            print("✓ Philippine news search index initialized successfully.")
        except Exception as e:
            print(f"⚠ Philippine news search index initialization failed: {str(e)}")
        
        # Initialize fake news detector
        print("\n=== Fake News Detection Model ===")
        if detector.load_model('fake_news_model.pkl'):
            print("✓ Fake news model loaded successfully.")
            if detector.accuracy:
                print(f"  Model accuracy: {detector.accuracy:.1%}")
        else:
            print("⚠ Pre-trained fake news model not found. Training from scratch...")
            # Check if training data exists
            if os.path.exists('datasets/WELFake_Dataset.csv'):
                try:
                    print("📊 Loading training dataset...")
                    
                    # Check if it's a Git LFS pointer file
                    with open('datasets/WELFake_Dataset.csv', 'r', encoding='utf-8') as f:
                        first_line = f.readline()
                        if 'version https://git-lfs.github.com' in first_line:
                            print("❌ WELFake_Dataset.csv is a Git LFS pointer file")
                            print("Please run 'git lfs pull' to download the actual dataset")
                            return
                    
                    # Load and prepare data
                    data = pd.read_csv('datasets/WELFake_Dataset.csv')
                    print(f"✓ Dataset loaded: {len(data)} samples")
                    
                    # Train the model
                    print("🚀 Training fake news detection model...")
                    best_model, best_accuracy, best_vectorizer = detector.train_best_model(data)
                    print(f"✓ Model trained with accuracy: {best_accuracy:.1%}")
                    
                    # Save the model
                    model_data = {
                        'model': best_model,
                        'vectorizer': best_vectorizer,
                        'accuracy': best_accuracy,
                        'timestamp': datetime.now().isoformat()
                    }
                    joblib.dump(model_data, 'fake_news_model.pkl')
                    print("✓ Fake news model saved as 'fake_news_model.pkl'")
                    
                except Exception as e:
                    print(f"✗ Error training model: {str(e)}")
                    print("The application will run with limited functionality.")
            else:
                print("✗ Training dataset not available. Fake news detection will be unavailable.")
                print("Please ensure 'datasets/WELFake_Dataset.csv' is properly downloaded from Git LFS.")
        
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

# Initialize module routes
philippine_search_index = init_philippine_search_index()
init_political_detector(political_detector)
init_fake_news_detector(detector, political_detector, philippine_search_index)
url_classifier = init_url_classifier()

# Register the blueprints
app.register_blueprint(news_crawler_bp)
app.register_blueprint(philippine_news_bp)
app.register_blueprint(political_news_bp)
app.register_blueprint(fake_news_bp)
app.register_blueprint(url_classifier_bp)

if __name__ == '__main__':
    # Initialize models in a separate thread to avoid blocking
    model_thread = threading.Thread(target=initialize_models)
    model_thread.daemon = True
    model_thread.start()
    
    # Get port from environment variable for production deployment
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
