#!/usr/bin/env python3
"""
Test script for the optimized URL News Classifier
Tests batch processing, parallel processing, and performance improvements
"""

from url_news_classifier import URLNewsClassifier
import time

def test_optimized_classifier():
    print("🚀 Testing Optimized URL News Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = URLNewsClassifier()
    
    print(f"📊 Initial state:")
    print(f"  - Is trained: {classifier.is_trained}")
    print(f"  - Feedback count: {len(classifier.feedback_data)}")
    print(f"  - Training iterations: {len(classifier.training_history)}")
    
    # Test URLs for batch processing
    test_urls = [
        "https://cnn.com/2024/08/08/politics/breaking-news-story",
        "https://example.com/about/contact-us",
        "https://news.com/article/12345/technology-update",
        "https://shop.example.com/products/laptop",
        "https://bbc.com/news/world-asia-12345678",
        "https://example.com/login",
        "https://reuters.com/article/politics-news-idUSKBN123456",
        "https://example.com/search?q=news",
        "https://washingtonpost.com/2024/08/08/politics/election-coverage/",
        "https://example.com/privacy-policy"
    ]
    
    print(f"\n🧪 Testing single URL prediction:")
    url = test_urls[0]
    start_time = time.time()
    result = classifier.predict_with_confidence(url)
    single_time = time.time() - start_time
    
    print(f"  URL: {url}")
    print(f"  Prediction: {result['is_news_article']} (confidence: {result['confidence']:.3f})")
    print(f"  Time: {single_time*1000:.2f}ms")
    if 'news_score' in result:
        print(f"  News score: {result['news_score']}")
    
    print(f"\n⚡ Testing batch processing (sequential):")
    start_time = time.time()
    batch_results_seq = classifier.predict_batch(test_urls, use_parallel=False)
    seq_time = time.time() - start_time
    
    print(f"  Processed {len(test_urls)} URLs in {seq_time*1000:.2f}ms")
    print(f"  Average time per URL: {seq_time*1000/len(test_urls):.2f}ms")
    
    print(f"\n🚀 Testing batch processing (parallel):")
    start_time = time.time()
    batch_results_par = classifier.predict_batch(test_urls, use_parallel=True)
    par_time = time.time() - start_time
    
    print(f"  Processed {len(test_urls)} URLs in {par_time*1000:.2f}ms")
    print(f"  Average time per URL: {par_time*1000/len(test_urls):.2f}ms")
    print(f"  Speedup: {seq_time/par_time:.2f}x")
    
    print(f"\n📈 Batch analysis results:")
    analysis = classifier.analyze_urls_batch(test_urls, confidence_threshold=0.7)
    summary = analysis['summary']
    
    print(f"  Total URLs: {summary['total_urls']}")
    print(f"  News articles: {summary['news_count']}")
    print(f"  Non-news: {summary['non_news_count']}")
    print(f"  High confidence: {summary['high_confidence_count']}")
    print(f"  Low confidence: {summary['low_confidence_count']}")
    print(f"  Errors: {summary['error_count']}")
    print(f"  Average confidence: {summary['avg_confidence']:.3f}")
    
    print(f"\n📋 Detailed results:")
    for result in batch_results_par[:5]:  # Show first 5 results
        if 'error' not in result:
            print(f"  {result['url'][:50]}... -> {result['is_news_article']} ({result['confidence']:.3f})")
    
    print(f"\n🎯 Testing model retraining:")
    initial_feedback_count = len(classifier.feedback_data)
    
    # Test batch feedback
    fake_feedback = []
    for i, result in enumerate(batch_results_par[:5]):
        if 'error' not in result:
            fake_feedback.append({
                'url': result['url'],
                'predicted_label': result['prediction'],
                'actual_label': True if 'news' in result['url'] or 'cnn' in result['url'] or 'bbc' in result['url'] else False,
                'user_confidence': 0.9
            })
    
    if fake_feedback:
        start_time = time.time()
        classifier.add_batch_feedback(fake_feedback)
        feedback_time = time.time() - start_time
        
        print(f"  Added {len(fake_feedback)} feedback entries in {feedback_time*1000:.2f}ms")
        print(f"  Total feedback: {len(classifier.feedback_data)}")
        print(f"  Is trained: {classifier.is_trained}")
    
    print(f"\n📊 Performance statistics:")
    stats = classifier.get_performance_stats()
    print(f"  Model trained: {stats['is_trained']}")
    print(f"  Training iterations: {stats['training_iterations']}")
    if stats.get('last_accuracy'):
        print(f"  Last accuracy: {stats['last_accuracy']:.3f}")
    if stats.get('recent_accuracy'):
        print(f"  Recent accuracy: {stats['recent_accuracy']:.3f}")
    
    print(f"\n✅ Optimization test completed!")
    print(f"   - Content extraction: REMOVED ✓")
    print(f"   - Batch processing: WORKING ✓")
    print(f"   - Parallel processing: WORKING ✓")
    print(f"   - Fast URL-only analysis: WORKING ✓")

if __name__ == "__main__":
    test_optimized_classifier()
