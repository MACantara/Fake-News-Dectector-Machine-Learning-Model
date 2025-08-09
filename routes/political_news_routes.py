"""
Routes for Political News Detector functionality
Handles political news classification and analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, request, jsonify
from modules.political_news_detector import (
    PoliticalNewsDetector,
    extract_political_content_from_url,
    validate_political_classification,
    format_political_analysis_result,
    get_political_news_categories,
    extract_political_keywords
)

# Create blueprint for political news detector routes
political_news_bp = Blueprint('political_news', __name__)

# Initialize the political news detector (will be set by main app)
political_detector = None


def init_political_detector(detector_instance=None):
    """Initialize the political news detector"""
    global political_detector
    if detector_instance:
        political_detector = detector_instance
    else:
        political_detector = PoliticalNewsDetector()


@political_news_bp.route('/classify-political-text', methods=['POST'])
def classify_political_text():
    """Classify text as political or non-political news"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if len(text) < 10:
            return jsonify({'error': 'Text is too short for reliable classification'}), 400
        
        # Perform classification
        result = political_detector.predict(text)
        
        # Validate the classification
        validation = validate_political_classification(
            result['prediction'], 
            result['confidence'], 
            text
        )
        
        # Format the result
        formatted_result = format_political_analysis_result(result)
        formatted_result['validation'] = validation
        formatted_result['input_text_length'] = len(text.split())
        
        return jsonify({
            'success': True,
            'classification': formatted_result,
            'model_info': {
                'is_trained': political_detector.is_trained,
                'accuracy': political_detector.accuracy
            }
        })
        
    except ValueError as e:
        return jsonify({'error': f'Classification error: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@political_news_bp.route('/analyze-political-url', methods=['POST'])
def analyze_political_url():
    """Extract content from URL and classify as political news"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Extract content from URL
        content_data = extract_political_content_from_url(url)
        
        if 'error' in content_data:
            return jsonify({
                'error': f'Failed to extract content: {content_data["error"]}',
                'url': url
            }), 400
        
        # Classify the extracted content
        if content_data['combined_text']:
            classification_result = political_detector.predict(content_data['combined_text'])
            
            # Validate the classification
            validation = validate_political_classification(
                classification_result['prediction'],
                classification_result['confidence'],
                content_data['combined_text']
            )
            
            # Format the result
            formatted_result = format_political_analysis_result(classification_result)
            formatted_result['validation'] = validation
        else:
            formatted_result = {
                'classification': 'Unknown',
                'confidence': 0.0,
                'analysis': {'error': 'No content could be extracted'},
                'validation': {'is_valid': False, 'warnings': ['No content extracted']}
            }
        
        return jsonify({
            'success': True,
            'url': url,
            'content_info': {
                'title': content_data.get('title', ''),
                'author': content_data.get('author', ''),
                'publish_date': content_data.get('publish_date', ''),
                'content_length': content_data.get('content_length', 0),
                'political_score': content_data.get('political_score', 0),
                'is_political_source': content_data.get('is_political_source', False)
            },
            'classification': formatted_result,
            'political_entities': content_data.get('political_entities', {}),
            'model_info': {
                'is_trained': political_detector.is_trained,
                'accuracy': political_detector.accuracy
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@political_news_bp.route('/political-detector-status')
def get_political_detector_status():
    """Get the status and information about the political news detector"""
    try:
        return jsonify({
            'detector_info': {
                'is_trained': political_detector.is_trained,
                'accuracy': political_detector.accuracy,
                'model_available': political_detector.model is not None
            },
            'categories': get_political_news_categories(),
            'features': {
                'keyword_analysis': True,
                'entity_extraction': True,
                'confidence_scoring': True,
                'reasoning_extraction': True
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@political_news_bp.route('/political-keywords')
def get_political_keywords():
    """Get the list of political keywords used for classification"""
    try:
        keywords = list(extract_political_keywords())
        
        return jsonify({
            'keywords': sorted(keywords),
            'total_count': len(keywords),
            'categories': {
                'government': ['government', 'federal', 'state', 'administration'],
                'elections': ['election', 'vote', 'voting', 'campaign', 'candidate'],
                'officials': ['president', 'senator', 'governor', 'mayor', 'minister'],
                'institutions': ['congress', 'senate', 'house', 'supreme court'],
                'legislation': ['bill', 'law', 'policy', 'legislation', 'reform']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@political_news_bp.route('/batch-classify-political', methods=['POST'])
def batch_classify_political():
    """Classify multiple texts for political content"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Array of texts is required'}), 400
        
        if len(texts) > 50:  # Limit batch size
            return jsonify({'error': 'Maximum 50 texts allowed per batch'}), 400
        
        results = []
        
        for i, text in enumerate(texts):
            try:
                if not text or not text.strip():
                    results.append({
                        'index': i,
                        'text_preview': '',
                        'classification': 'Unknown',
                        'error': 'Empty text'
                    })
                    continue
                
                # Perform classification
                classification_result = political_detector.predict(text.strip())
                
                # Format result
                formatted_result = format_political_analysis_result(classification_result)
                
                results.append({
                    'index': i,
                    'text_preview': text[:100] + '...' if len(text) > 100 else text,
                    'classification': formatted_result['classification'],
                    'confidence': formatted_result['confidence'],
                    'confidence_percentage': formatted_result['confidence_percentage'],
                    'reasoning': formatted_result['analysis']['reasoning']
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'text_preview': text[:100] + '...' if len(text) > 100 else text,
                    'classification': 'Error',
                    'error': str(e)
                })
        
        # Summary statistics
        successful_classifications = [r for r in results if 'error' not in r]
        political_count = len([r for r in successful_classifications if r['classification'] == 'Political'])
        non_political_count = len([r for r in successful_classifications if r['classification'] == 'Non-Political'])
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total_texts': len(texts),
                'successful_classifications': len(successful_classifications),
                'political_count': political_count,
                'non_political_count': non_political_count,
                'errors': len(results) - len(successful_classifications)
            },
            'model_info': {
                'is_trained': political_detector.is_trained,
                'accuracy': political_detector.accuracy
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@political_news_bp.route('/train-political-model', methods=['POST'])
def train_political_model():
    """Train or retrain the political news classification model"""
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path', 'datasets/News_Category_Dataset_v3.json')
        
        # This would typically load training data and retrain the model
        # For now, return a placeholder response
        return jsonify({
            'message': 'Model training initiated',
            'status': 'Training not implemented in this endpoint',
            'suggestion': 'Use the model training script directly for retraining'
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
