from flask import Flask, render_template, request, jsonify
import os
import threading
from datetime import datetime
from url_news_classifier import URLNewsClassifier
from routes.news_crawler_routes import news_crawler_bp, init_news_crawler
from routes.philippine_news_search_routes import philippine_news_bp, init_philippine_search_index, get_philippine_search_index
from routes.political_news_routes import political_news_bp, init_political_detector
from routes.fake_news_routes import fake_news_bp, init_fake_news_detector
from modules.political_news_detector import PoliticalNewsDetector
from modules.fake_news_detector import FakeNewsDetector, check_lfs_files
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


# Initialize the detectors and search index
detector = FakeNewsDetector()
political_detector = PoliticalNewsDetector()
url_news_classifier = URLNewsClassifier()

# Initialize news crawler routes
init_news_crawler(url_news_classifier)

# Initialize Philippine news search routes
philippine_search_index = init_philippine_search_index()

# Initialize fake news detector routes
init_fake_news_detector(detector, political_detector, philippine_search_index)

# Register the blueprints
app.register_blueprint(news_crawler_bp)
app.register_blueprint(philippine_news_bp)
app.register_blueprint(political_news_bp)
app.register_blueprint(fake_news_bp)

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
