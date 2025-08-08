#!/usr/bin/env python3
"""
Test script to verify that model retraining works with existing feedback data
"""

from url_news_classifier import URLNewsClassifier
import json

def test_retrain():
    # Initialize classifier
    classifier = URLNewsClassifier()
    
    print(f"Initial state:")
    print(f"  - Is trained: {classifier.is_trained}")
    print(f"  - Feedback count: {len(classifier.feedback_data)}")
    print(f"  - Training iterations: {len(classifier.training_history)}")
    
    # Check feedback data structure
    if classifier.feedback_data:
        first_feedback = classifier.feedback_data[0]
        print(f"\nFirst feedback entry:")
        print(f"  - URL: {first_feedback.get('url', 'N/A')}")
        print(f"  - Has feature_vector: {bool(first_feedback.get('feature_vector'))}")
        print(f"  - Feature vector length: {len(first_feedback.get('feature_vector', []))}")
        print(f"  - Actual label: {first_feedback.get('actual_label')}")
        print(f"  - Was correct: {first_feedback.get('was_correct')}")
    
    # Count feedback entries with and without feature vectors
    with_vectors = sum(1 for f in classifier.feedback_data if f.get('feature_vector'))
    without_vectors = len(classifier.feedback_data) - with_vectors
    
    print(f"\nFeedback analysis:")
    print(f"  - With feature vectors: {with_vectors}")
    print(f"  - Without feature vectors: {without_vectors}")
    
    # Attempt to retrain
    print(f"\nAttempting to retrain model...")
    try:
        result = classifier.retrain_model()
        
        print(f"Retraining completed!")
        print(f"  - Is trained now: {classifier.is_trained}")
        print(f"  - Training iterations: {len(classifier.training_history)}")
        
        if classifier.training_history:
            last_training = classifier.training_history[-1]
            print(f"  - Last training accuracy: {last_training.get('accuracy', 'N/A')}")
            print(f"  - Samples used: {last_training.get('samples_count', 'N/A')}")
        
        # Test a prediction to see if model works
        test_url = "https://example.com/news/breaking-news-story"
        prediction = classifier.predict_with_confidence(test_url)
        print(f"\nTest prediction for '{test_url}':")
        print(f"  - Prediction: {prediction['prediction']}")
        print(f"  - Confidence: {prediction['confidence']:.3f}")
        print(f"  - Is news article: {prediction['is_news_article']}")
        
    except Exception as e:
        print(f"Error during retraining: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retrain()
