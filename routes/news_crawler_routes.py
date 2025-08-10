"""
Routes for News Website Crawler functionality
Handles crawling and analyzing news websites
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, request, jsonify
from modules.news_website_crawler import NewsWebsiteCrawler

# Create blueprint for news crawler routes
news_crawler_bp = Blueprint('news_crawler', __name__)

# Initialize the news crawler (will be set by main app)
news_crawler = None


def init_news_crawler(url_classifier):
    """Initialize the news crawler with URL classifier"""
    global news_crawler
    news_crawler = NewsWebsiteCrawler(url_classifier=url_classifier)


def get_news_crawler():
    """Get the shared news crawler instance for direct method calls"""
    return news_crawler


@news_crawler_bp.route('/crawl-website', methods=['POST'])
def crawl_website():
    """Crawl a news website for article links with optional URL filtering"""
    try:
        data = request.get_json()
        website_url = data.get('website_url', '').strip()
        max_articles = int(data.get('max_articles', 10))
        enable_filtering = data.get('enable_filtering', True)  # New parameter for URL filtering
        confidence_threshold = float(data.get('confidence_threshold', 0.6))  # Filtering threshold
        
        if not website_url:
            return jsonify({'error': 'Website URL is required'}), 400
        
        # Validate URL
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
        
        # Configure the crawler's filtering settings
        news_crawler.set_filtering_mode(enable_filtering, confidence_threshold)
        
        # Crawl the website with URL filtering
        crawl_result = news_crawler.extract_article_links(
            website_url, 
            enable_filtering=enable_filtering
        )
        
        if not crawl_result['success']:
            return jsonify({
                'error': crawl_result['error'],
                'articles': []
            }), 400
        
        # Normalize article data structure - handle both filtered and unfiltered results
        articles = crawl_result['articles']
        normalized_articles = []
        
        if enable_filtering and articles:
            # With filtering enabled, articles are objects with metadata
            for article in articles:
                if isinstance(article, dict):
                    normalized_articles.append({
                        'url': article.get('url', ''),
                        'title': article.get('text', '') or article.get('title', ''),
                        'link_text': article.get('text', ''),
                        'confidence': article.get('classification', {}).get('confidence', 0.0),
                        'is_news_prediction': article.get('classification', {}).get('is_news_article', True),
                        'probability_news': article.get('classification', {}).get('probability_news', 0.0),
                        'probability_not_news': article.get('classification', {}).get('probability_not_news', 0.0)
                    })
                else:
                    # Fallback for string URLs
                    normalized_articles.append({
                        'url': str(article),
                        'title': '',
                        'link_text': '',
                        'confidence': 1.0,
                        'is_news_prediction': True,
                        'probability_news': 1.0,
                        'probability_not_news': 0.0
                    })
        else:
            # Without filtering, articles are simple URLs
            for article in articles:
                url = article if isinstance(article, str) else article.get('url', str(article))
                normalized_articles.append({
                    'url': url,
                    'title': '',
                    'link_text': '',
                    'confidence': 1.0,
                    'is_news_prediction': True,
                    'probability_news': 1.0,
                    'probability_not_news': 0.0
                })
        
        # Apply max_articles limit to normalized articles
        if max_articles and len(normalized_articles) > max_articles:
            normalized_articles = normalized_articles[:max_articles]
            print(f"📋 Limited to top {max_articles} articles")
        
        # Prepare comprehensive response
        response_data = {
            'success': True,
            'website_title': crawl_result.get('website_title', 'Unknown'),
            'total_found': len(normalized_articles),
            'articles': normalized_articles,
            'filtering_enabled': enable_filtering,
            'classification_method': crawl_result.get('classification_method', 'Basic URL patterns'),
            'crawler_stats': {
                'total_candidates': crawl_result.get('total_candidates', len(normalized_articles)),
                'filtered_articles': len(normalized_articles),
                'filtered_out': crawl_result.get('filtered_out', 0),
                'confidence_threshold': confidence_threshold if enable_filtering else None
            }
        }
        
        # Add detailed filtering statistics if available
        if enable_filtering and 'classification_stats' in crawl_result:
            response_data['classification_stats'] = crawl_result['classification_stats']
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"❌ Crawling error: {str(e)}")
        return jsonify({'error': f'Crawling failed: {str(e)}'}), 500


@news_crawler_bp.route('/analyze-website', methods=['POST'])
def analyze_website():
    """Crawl and analyze articles from a news website with enhanced URL filtering"""
    try:
        data = request.get_json()
        website_url = data.get('website_url', '').strip()
        max_articles = int(data.get('max_articles', 5))
        analysis_type = data.get('analysis_type', 'both')
        enable_filtering = data.get('enable_filtering', True)  # New parameter for URL filtering
        confidence_threshold = float(data.get('confidence_threshold', 0.6))  # Filtering threshold
        
        if not website_url:
            return jsonify({'error': 'Website URL is required'}), 400
        
        # Validate URL
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
        
        print(f"🔍 Starting website analysis for: {website_url}")
        print(f"📊 Analysis settings: max_articles={max_articles}, type={analysis_type}, filtering={'enabled' if enable_filtering else 'disabled'}")
        
        # Configure the crawler's filtering settings
        news_crawler.set_filtering_mode(enable_filtering, confidence_threshold)
        
        # First crawl the website to get article links with intelligent filtering
        crawl_result = news_crawler.extract_article_links(
            website_url, 
            enable_filtering=enable_filtering
        )
        
        if not crawl_result['success']:
            return jsonify({
                'error': f"Failed to crawl website: {crawl_result['error']}",
                'results': []
            }), 400
        
        if not crawl_result['articles']:
            return jsonify({
                'error': 'No articles found on the website',
                'results': [],
                'crawler_stats': {
                    'total_candidates': crawl_result.get('total_candidates', 0),
                    'filtered_articles': 0,
                    'filtered_out': crawl_result.get('filtered_out', 0)
                }
            }), 400
        
        # Normalize and prepare articles for analysis
        articles = crawl_result['articles']
        articles_to_analyze = []
        
        # Handle different article formats from crawler
        for article in articles:
            if isinstance(article, dict):
                # Enhanced article object with metadata
                article_data = {
                    'url': article.get('url', ''),
                    'title': article.get('text', '') or article.get('title', ''),
                    'link_text': article.get('text', ''),
                    'classification_confidence': article.get('classification', {}).get('confidence', 0.0),
                    'is_news_prediction': article.get('classification', {}).get('is_news_article', True)
                }
            else:
                # Simple URL string
                article_data = {
                    'url': str(article),
                    'title': '',
                    'link_text': '',
                    'classification_confidence': 1.0,
                    'is_news_prediction': True
                }
            
            articles_to_analyze.append(article_data)
        
        # Apply max_articles limit with intelligent sorting
        if max_articles and len(articles_to_analyze) > max_articles:
            # Sort by classification confidence (highest first) if available
            if enable_filtering:
                articles_to_analyze.sort(
                    key=lambda x: x.get('classification_confidence', 0.0), 
                    reverse=True
                )
            
            articles_to_analyze = articles_to_analyze[:max_articles]
            print(f"📋 Limited analysis to top {max_articles} articles (sorted by confidence)")
        
        print(f"🚀 Analyzing {len(articles_to_analyze)} articles using batch processing...")
        
        # Use the crawler's enhanced batch analysis with proper article format
        analysis_results = news_crawler.analyze_articles_batch(
            articles_to_analyze,  # Pass the normalized article data
            analysis_type
        )
        
        # Enhance analysis results with crawler metadata
        enhanced_results = []
        for i, result in enumerate(analysis_results):
            enhanced_result = result.copy()
            
            # Add original crawler classification data if available
            if i < len(articles_to_analyze):
                original_article = articles_to_analyze[i]
                enhanced_result['crawler_classification'] = {
                    'confidence': original_article.get('classification_confidence', 0.0),
                    'is_news_prediction': original_article.get('is_news_prediction', True),
                    'link_text': original_article.get('link_text', '')
                }
            
            enhanced_results.append(enhanced_result)
        
        # Calculate comprehensive statistics
        successful_analyses = [r for r in enhanced_results if r.get('status') == 'success']
        failed_analyses = [r for r in enhanced_results if r.get('status') != 'success']
        
        # Calculate analysis type specific stats
        if successful_analyses:
            fake_news_count = sum(1 for r in successful_analyses 
                                if r.get('fake_news_analysis', {}).get('prediction') == 'FAKE')
            political_count = sum(1 for r in successful_analyses 
                                if r.get('political_analysis', {}).get('is_political', False))
        else:
            fake_news_count = 0
            political_count = 0
        
        # Prepare comprehensive summary
        summary = {
            'website_info': {
                'url': website_url,
                'title': crawl_result.get('website_title', 'Unknown'),
                'crawl_timestamp': crawl_result.get('crawl_timestamp')
            },
            'crawler_stats': {
                'total_candidates': crawl_result.get('total_candidates', len(articles_to_analyze)),
                'filtered_articles': len(articles_to_analyze),
                'filtered_out': crawl_result.get('filtered_out', 0),
                'filtering_enabled': enable_filtering,
                'classification_method': crawl_result.get('classification_method', 'Basic patterns'),
                'confidence_threshold': confidence_threshold if enable_filtering else None
            },
            'analysis_stats': {
                'total_articles': len(articles_to_analyze),
                'successful_analyses': len(successful_analyses),
                'failed_analyses': len(failed_analyses),
                'analysis_type': analysis_type,
                'success_rate': (len(successful_analyses) / len(articles_to_analyze) * 100) if articles_to_analyze else 0
            },
            'content_stats': {
                'fake_news_detected': fake_news_count,
                'political_articles': political_count,
                'real_news_articles': len(successful_analyses) - fake_news_count
            }
        }
        
        # Add detailed filtering statistics if available
        if enable_filtering and 'classification_stats' in crawl_result:
            summary['classification_stats'] = crawl_result['classification_stats']
        
        print(f"✅ Analysis complete: {len(successful_analyses)}/{len(articles_to_analyze)} successful")
        
        return jsonify({
            'success': True,
            'summary': summary,
            'results': enhanced_results
        })
    
    except Exception as e:
        print(f"❌ Website analysis error: {str(e)}")
        return jsonify({
            'error': f'Website analysis failed: {str(e)}',
            'results': []
        }), 500


@news_crawler_bp.route('/crawler-status', methods=['GET'])
def get_crawler_status():
    """Get current crawler configuration and statistics"""
    try:
        if not news_crawler:
            return jsonify({
                'error': 'News crawler not initialized',
                'available': False
            }), 503
        
        # Get current configuration
        config = {
            'filtering_enabled': news_crawler.enable_filtering,
            'confidence_threshold': news_crawler.confidence_threshold,
            'url_classifier_available': news_crawler.url_classifier is not None,
            'crawler_version': '2.0',
            'supported_analysis_types': ['both', 'fake_news', 'political']
        }
        
        # Get classification statistics
        stats = news_crawler.get_classification_stats()
        
        # Calculate performance metrics
        total_classified = stats.get('total_urls', 0)
        accuracy_rate = 0
        if total_classified > 0:
            successful_classifications = total_classified - stats.get('classification_errors', 0)
            accuracy_rate = (successful_classifications / total_classified) * 100
        
        performance = {
            'total_urls_classified': total_classified,
            'classification_accuracy_rate': round(accuracy_rate, 2),
            'classification_errors': stats.get('classification_errors', 0),
            'news_articles_found': stats.get('classified_as_news', 0),
            'non_news_filtered': stats.get('classified_as_non_news', 0)
        }
        
        return jsonify({
            'available': True,
            'configuration': config,
            'statistics': stats,
            'performance': performance,
            'status': 'operational'
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Failed to get crawler status: {str(e)}',
            'available': False
        }), 500


@news_crawler_bp.route('/configure-crawler', methods=['POST'])
def configure_crawler():
    """Configure crawler filtering settings"""
    try:
        if not news_crawler:
            return jsonify({
                'error': 'News crawler not initialized',
                'success': False
            }), 503
        
        data = request.get_json()
        enable_filtering = data.get('enable_filtering', True)
        confidence_threshold = float(data.get('confidence_threshold', 0.6))
        
        # Validate threshold
        if not 0.0 <= confidence_threshold <= 1.0:
            return jsonify({
                'error': 'Confidence threshold must be between 0.0 and 1.0',
                'success': False
            }), 400
        
        # Update crawler configuration
        news_crawler.set_filtering_mode(enable_filtering, confidence_threshold)
        
        # Reset statistics for new configuration
        news_crawler.reset_classification_stats()
        
        return jsonify({
            'success': True,
            'message': 'Crawler configuration updated',
            'configuration': {
                'filtering_enabled': enable_filtering,
                'confidence_threshold': confidence_threshold
            }
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Failed to configure crawler: {str(e)}',
            'success': False
        }), 500


@news_crawler_bp.route('/reset-crawler-stats', methods=['POST'])
def reset_crawler_stats():
    """Reset crawler classification statistics"""
    try:
        if not news_crawler:
            return jsonify({
                'error': 'News crawler not initialized',
                'success': False
            }), 503
        
        news_crawler.reset_classification_stats()
        
        return jsonify({
            'success': True,
            'message': 'Crawler statistics reset',
            'statistics': news_crawler.get_classification_stats()
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Failed to reset crawler stats: {str(e)}',
            'success': False
        }), 500


@news_crawler_bp.route('/crawler-info', methods=['GET'])
def get_crawler_info():
    """Get comprehensive crawler information and capabilities"""
    try:
        if not news_crawler:
            return jsonify({
                'error': 'News crawler not initialized',
                'available': False
            }), 503
        
        # Get comprehensive crawler information
        crawler_info = news_crawler.get_crawler_info()
        
        # Add additional runtime information
        crawler_info['runtime'] = {
            'initialized': True,
            'url_classifier_integration': news_crawler.url_classifier is not None,
            'last_crawl_stats': news_crawler.get_classification_stats()
        }
        
        return jsonify({
            'success': True,
            'crawler_info': crawler_info
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Failed to get crawler info: {str(e)}',
            'success': False
        }), 500
