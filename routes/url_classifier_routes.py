"""
Routes for URL News Classifier functionality
Handles URL classification, feedback submission, model retraining, and statistics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, request, jsonify, render_template
from modules.url_news_classifier import URLNewsClassifier
from datetime import datetime

# Create blueprint for URL classifier routes
url_classifier_bp = Blueprint('url_classifier', __name__)

# Global URL classifier instance
url_classifier = None

def init_url_classifier():
    """Initialize the URL classifier"""
    global url_classifier
    url_classifier = URLNewsClassifier()
    return url_classifier

def get_url_classifier():
    """Get the initialized URL classifier instance"""
    return url_classifier

@url_classifier_bp.route('/url-classifier')
def url_classifier_page():
    """URL classifier page"""
    return render_template('url_classifier.html')

@url_classifier_bp.route('/classify-url', methods=['POST'])
def classify_url():
    """Classify a single URL as news or non-news"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({
                'success': False,
                'error': 'URL is required'
            }), 400
        
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            return jsonify({
                'success': False,
                'error': 'URL must start with http:// or https://'
            }), 400
        
        # Get prediction from URL classifier
        result = url_classifier.predict_with_confidence(url)
        
        return jsonify({
            'success': True,
            'url': url,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'is_news_article': result['is_news_article'],
            'probability_news': result['probability_news'],
            'probability_not_news': result['probability_not_news'],
            'model_trained': url_classifier.is_trained,
            'heuristic_based': result.get('heuristic_based', False)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Classification failed: {str(e)}'
        }), 500

@url_classifier_bp.route('/classify-batch', methods=['POST'])
def classify_batch():
    """Classify multiple URLs efficiently"""
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        confidence_threshold = data.get('confidence_threshold', 0.7)
        
        if not urls:
            return jsonify({
                'success': False,
                'error': 'URLs list is required'
            }), 400
        
        if len(urls) > 100:
            return jsonify({
                'success': False,
                'error': 'Maximum 100 URLs allowed per batch'
            }), 400
        
        # Validate URLs
        valid_urls = []
        invalid_urls = []
        for url in urls:
            if isinstance(url, str) and url.strip().startswith(('http://', 'https://')):
                valid_urls.append(url.strip())
            else:
                invalid_urls.append(url)
        
        if not valid_urls:
            return jsonify({
                'success': False,
                'error': 'No valid URLs provided'
            }), 400
        
        # Analyze URLs
        analysis = url_classifier.analyze_urls_batch(valid_urls, confidence_threshold)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'invalid_urls': invalid_urls,
            'model_trained': url_classifier.is_trained
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Batch classification failed: {str(e)}'
        }), 500

@url_classifier_bp.route('/url-classifier-feedback', methods=['POST'])
def submit_url_classifier_feedback():
    """Submit feedback for URL classification to improve the RL model"""
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
            
            # Process bulk feedback
            result = url_classifier.add_batch_feedback(feedback_list)
            
            return jsonify({
                'success': True,
                'message': f'Bulk feedback processed successfully',
                'processed_count': result['processed_count'],
                'errors': result['errors'],
                'total_feedback': result['total_feedback'],
                'model_stats': url_classifier.get_model_stats()
            })
        else:
            # Handle single feedback
            url = data.get('url', '').strip()
            predicted_label = data.get('predicted_label')
            actual_label = data.get('actual_label', '').strip()
            user_confidence = data.get('user_confidence', 1.0)
            user_comment = data.get('comment', '').strip()
            
            # Validate required fields
            if not url:
                return jsonify({
                    'success': False,
                    'error': 'URL is required'
                }), 400
            
            if predicted_label is None:
                return jsonify({
                    'success': False,
                    'error': 'predicted_label is required'
                }), 400
            
            if actual_label.lower() not in ['news', 'non-news', 'true', 'false', '1', '0']:
                return jsonify({
                    'success': False,
                    'error': 'actual_label must be "news", "non-news", "true", "false", "1", or "0"'
                }), 400
            
            # Convert actual_label to boolean
            if actual_label.lower() in ['news', 'true', '1']:
                actual_bool = True
            else:
                actual_bool = False
            
            # Add feedback
            feedback_entry = url_classifier.add_feedback(
                url=url,
                predicted_label=bool(predicted_label),
                actual_label=actual_bool,
                user_confidence=float(user_confidence)
            )
            
            # Get updated model stats
            model_stats = url_classifier.get_model_stats()
            
            return jsonify({
                'success': True,
                'message': 'Feedback submitted successfully',
                'feedback_entry': feedback_entry,
                'model_stats': model_stats,
                'retraining_triggered': len(url_classifier.feedback_data) % 100 == 0 and len(url_classifier.feedback_data) >= 100
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to submit feedback: {str(e)}'
        }), 500

@url_classifier_bp.route('/url-classifier-stats')
def url_classifier_stats():
    """Get URL classifier statistics and performance metrics"""
    try:
        stats = url_classifier.get_performance_stats()
        recent_feedback = url_classifier.get_recent_feedback(10)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'recent_feedback': recent_feedback
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get stats: {str(e)}'
        }), 500

@url_classifier_bp.route('/retrain-url-classifier', methods=['POST'])
def retrain_url_classifier():
    """Retrain URL classifier with feedback data"""
    try:
        data = request.get_json() if request.is_json else {}
        force_retrain = data.get('force_retrain', False)
        
        feedback_count = len(url_classifier.feedback_data)
        
        if not force_retrain and feedback_count < 100:
            return jsonify({
                'success': False,
                'error': f'Need at least 10 feedback entries for retraining, have {feedback_count}'
            }), 400
        
        # Start retraining
        print(f"🚀 Starting URL classifier retraining with {feedback_count} feedback entries...")
        
        accuracy = url_classifier.retrain_model()
        
        if accuracy is None:
            return jsonify({
                'success': False,
                'error': 'Retraining failed - insufficient data'
            }), 400
        
        return jsonify({
            'success': True,
            'message': f'URL classifier retrained successfully',
            'accuracy': accuracy,
            'feedback_entries_used': feedback_count,
            'model_stats': url_classifier.get_model_stats()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to retrain classifier: {str(e)}'
        }), 500

@url_classifier_bp.route('/url-classifier-model-status')
def url_classifier_model_status():
    """Get URL classifier model status"""
    try:
        stats = url_classifier.get_model_stats()
        
        return jsonify({
            'success': True,
            'is_trained': url_classifier.is_trained,
            'model_path': url_classifier.model_path,
            'feedback_count': len(url_classifier.feedback_data),
            'training_iterations': len(url_classifier.training_history),
            'last_accuracy': stats.get('last_accuracy'),
            'overall_accuracy': stats.get('accuracy', 0.0),
            'feature_weights': url_classifier.feature_weights
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get model status: {str(e)}'
        }), 500

@url_classifier_bp.route('/export-url-classifier-feedback')
def export_url_classifier_feedback():
    """Export URL classifier feedback data"""
    try:
        feedback_data = url_classifier.feedback_data
        
        return jsonify({
            'success': True,
            'feedback_count': len(feedback_data),
            'feedback_data': feedback_data,
            'export_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to export feedback: {str(e)}'
        }), 500

@url_classifier_bp.route('/import-url-classifier-feedback', methods=['POST'])
def import_url_classifier_feedback():
    """Import URL classifier feedback data"""
    try:
        data = request.get_json()
        feedback_data = data.get('feedback_data', [])
        
        if not feedback_data:
            return jsonify({
                'success': False,
                'error': 'No feedback data provided'
            }), 400
        
        # Validate and add feedback
        valid_count = 0
        errors = []
        
        for entry in feedback_data:
            try:
                required_fields = ['url', 'predicted_label', 'actual_label']
                if all(field in entry for field in required_fields):
                    url_classifier.feedback_data.append(entry)
                    valid_count += 1
                else:
                    errors.append(f"Missing required fields in entry: {entry}")
            except Exception as e:
                errors.append(f"Error processing entry {entry}: {str(e)}")
        
        # Save updated feedback
        url_classifier.save_feedback()
        
        return jsonify({
            'success': True,
            'message': f'Imported {valid_count} feedback entries',
            'imported_count': valid_count,
            'errors': errors,
            'total_feedback': len(url_classifier.feedback_data)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to import feedback: {str(e)}'
        }), 500

@url_classifier_bp.route('/clear-url-classifier-feedback', methods=['POST'])
def clear_url_classifier_feedback():
    """Clear all URL classifier feedback data (use with caution)"""
    try:
        data = request.get_json() if request.is_json else {}
        confirm = data.get('confirm', False)
        
        if not confirm:
            return jsonify({
                'success': False,
                'error': 'Must set confirm=true to clear feedback data'
            }), 400
        
        feedback_count = len(url_classifier.feedback_data)
        url_classifier.feedback_data = []
        url_classifier.save_feedback()
        
        return jsonify({
            'success': True,
            'message': f'Cleared {feedback_count} feedback entries',
            'cleared_count': feedback_count
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to clear feedback: {str(e)}'
        }), 500
