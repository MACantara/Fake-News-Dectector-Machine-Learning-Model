"""
Routes for Fake News Detection functionality
Handles fake news detection, feedback submission, and model retraining
"""

import sys
import os
import threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, request, jsonify
from modules.political_news_detector import extract_political_content_from_url

# Create blueprint for fake news detection routes
fake_news_bp = Blueprint('fake_news', __name__)

# Initialize the fake news detector (will be set by main app)
fake_news_detector = None
political_detector = None
philippine_search_index = None


def init_fake_news_detector(detector, political_det=None, search_index=None):
    """Initialize the fake news detector and dependencies"""
    global fake_news_detector, political_detector, philippine_search_index
    fake_news_detector = detector
    political_detector = political_det
    philippine_search_index = search_index


@fake_news_bp.route('/predict', methods=['POST'])
def predict():
    """Predict if news content is fake or real"""
    try:
        data = request.get_json()
        
        if not fake_news_detector.is_trained:
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
            fake_result = fake_news_detector.predict(text)
            result = {'fake_news': fake_result}
            
            # Perform political classification if requested
            if analysis_type in ['political', 'both'] and political_detector and political_detector.is_trained:
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
                if philippine_search_index:
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
                else:
                    print("⚠️ Philippine search index not available for auto-indexing")
                    
            except Exception as indexing_error:
                print(f"⚠️ Error during auto-indexing: {str(indexing_error)}")
                # Don't fail the main analysis if indexing fails
                indexing_result = {'status': 'error', 'message': str(indexing_error)}
            
            # Perform fake news detection
            fake_result = fake_news_detector.predict(combined_text)
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
            if analysis_type in ['political', 'both'] and political_detector and political_detector.is_trained:
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
                            fake_news_detector.retrain_from_new_pattern(training_data)
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


@fake_news_bp.route('/submit-feedback', methods=['POST'])
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
        fake_news_detector.add_feedback(text, predicted_label, actual_label, confidence, user_comment)
        
        # Get updated feedback stats
        feedback_stats = fake_news_detector.get_feedback_stats()
        
        response = {
            'message': 'Thank you for your feedback! It will help improve the model.',
            'feedback_stats': feedback_stats
        }
        
        if feedback_stats['needs_retraining']:
            response['message'] += f' The model will be retrained soon with {feedback_stats["pending_training"]} new feedback entries.'
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@fake_news_bp.route('/feedback-stats')
def feedback_stats():
    """Get feedback statistics"""
    try:
        stats = fake_news_detector.get_feedback_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@fake_news_bp.route('/trigger-retrain', methods=['POST'])
def trigger_retrain():
    """Manually trigger model retraining"""
    try:
        feedback_stats = fake_news_detector.get_feedback_stats()
        
        if feedback_stats['pending_training'] == 0:
            return jsonify({'message': 'No new feedback available for retraining.'}), 400
        
        # Start retraining in background
        threading.Thread(target=fake_news_detector.retrain_with_feedback, daemon=True).start()
        
        return jsonify({
            'message': f'Model retraining started with {feedback_stats["pending_training"]} new feedback entries.',
            'status': 'retraining_started'
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@fake_news_bp.route('/model-status')
def model_status():
    """Get detailed model status information"""
    try:
        # Create the expected response structure for the frontend
        status_data = {
            'fake_news': fake_news_detector.is_trained,
            'political': political_detector.is_trained if political_detector else False
        }
        
        # Additional detailed status info (optional)
        status_info = {
            'fake_news_model': {
                'is_trained': fake_news_detector.is_trained,
                'status': 'Model is ready' if fake_news_detector.is_trained else 'Model is training...'
            },
            'political_model': {
                'is_trained': political_detector.is_trained if political_detector else False,
                'status': 'Model is ready' if (political_detector and political_detector.is_trained) else 'Model not loaded'
            }
        }
        
        if fake_news_detector.is_trained and fake_news_detector.accuracy:
            status_info['fake_news_model']['accuracy'] = f"{fake_news_detector.accuracy:.4f}"
            status_info['fake_news_model']['status'] = f"Model ready (Accuracy: {fake_news_detector.accuracy:.1%})"
        
        if political_detector and political_detector.is_trained and political_detector.accuracy:
            status_info['political_model']['accuracy'] = f"{political_detector.accuracy:.4f}"
            status_info['political_model']['status'] = f"Model ready (Accuracy: {political_detector.accuracy:.1%})"
        
        # Overall status
        both_ready = fake_news_detector.is_trained and (political_detector and political_detector.is_trained)
        status_info['overall_status'] = 'Both models ready' if both_ready else 'Models loading...'
        status_info['is_trained'] = fake_news_detector.is_trained  # Keep for backward compatibility
        
        # Add feedback statistics
        try:
            feedback_stats = fake_news_detector.get_feedback_stats()
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


@fake_news_bp.route('/pattern-cache-stats')
def pattern_cache_stats():
    """Get pattern cache statistics"""
    try:
        stats = fake_news_detector.get_pattern_cache_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        })


@fake_news_bp.route('/retrain-with-cached-patterns', methods=['POST'])
def retrain_with_cached_patterns():
    """Manually trigger retraining using cached patterns"""
    try:
        cache_stats = fake_news_detector.get_pattern_cache_stats()
        
        if cache_stats['total_patterns'] == 0:
            return jsonify({
                'success': False,
                'message': 'No cached patterns available for retraining.'
            }), 400
        
        # Start retraining in background
        def background_retrain():
            try:
                result = fake_news_detector.retrain_with_cached_patterns()
                print(f"✅ Cached pattern retraining completed: {'Success' if result else 'Failed'}")
            except Exception as e:
                print(f"❌ Cached pattern retraining failed: {str(e)}")
        
        threading.Thread(target=background_retrain, daemon=True).start()
        
        return jsonify({
            'success': True,
            'message': f'Model retraining started with {cache_stats["total_patterns"]} cached patterns.',
            'status': 'retraining_started'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        })
