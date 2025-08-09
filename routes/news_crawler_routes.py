"""
Routes for News Website Crawler functionality
Handles crawling and analyzing news websites
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, request, jsonify
from modules.news_website_crawler.crawler import NewsWebsiteCrawler

# Create blueprint for news crawler routes
news_crawler_bp = Blueprint('news_crawler', __name__)

# Initialize the news crawler (will be set by main app)
news_crawler = None


def init_news_crawler(url_classifier):
    """Initialize the news crawler with URL classifier"""
    global news_crawler
    news_crawler = NewsWebsiteCrawler(url_classifier=url_classifier)


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
        
        # Apply max_articles limit at the Flask route level
        articles = crawl_result['articles']
        if max_articles and len(articles) > max_articles:
            articles = articles[:max_articles]
            print(f"📋 Limited to top {max_articles} articles")
        
        # Prepare response with filtering information
        response_data = {
            'success': True,
            'website_title': crawl_result['website_title'],
            'total_found': len(articles),  # Use limited count
            'articles': articles,  # Use limited articles
            'filtering_enabled': crawl_result.get('filtering_enabled', False),
            'classification_method': crawl_result.get('classification_method', 'Unknown'),
        }
        
        # Add filtering statistics if available
        if enable_filtering and 'classification_stats' in crawl_result:
            response_data.update({
                'total_candidates': crawl_result['total_candidates'],
                'filtered_out': crawl_result.get('filtered_out', 0),
                'classification_stats': crawl_result['classification_stats'],
                'confidence_threshold': confidence_threshold
            })
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': f'Crawling failed: {str(e)}'}), 500


@news_crawler_bp.route('/analyze-website', methods=['POST'])
def analyze_website():
    """Crawl and analyze articles from a news website with URL filtering"""
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
        
        # Configure the crawler's filtering settings
        news_crawler.set_filtering_mode(enable_filtering, confidence_threshold)
        
        # First crawl the website to get article links with filtering
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
                'results': []
            }), 400
        
        # Apply max_articles limit at the Flask route level
        articles_to_analyze = crawl_result['articles']
        if max_articles and len(articles_to_analyze) > max_articles:
            articles_to_analyze = articles_to_analyze[:max_articles]
            print(f"📋 Limited analysis to top {max_articles} articles")
        
        # Analyze the found articles
        analysis_results = news_crawler.analyze_articles_batch(
            articles_to_analyze, 
            analysis_type
        )
        
        # Calculate summary statistics
        successful_analyses = [r for r in analysis_results if r['status'] == 'success']
        failed_analyses = [r for r in analysis_results if r['status'] != 'success']
        
        summary = {
            'total_articles': len(articles_to_analyze),  # Use limited count
            'successful_analyses': len(successful_analyses),
            'failed_analyses': len(failed_analyses),
            'website_title': crawl_result['website_title'],
            'filtering_enabled': crawl_result.get('filtering_enabled', False),
            'classification_method': crawl_result.get('classification_method', 'Unknown')
        }
        
        # Add filtering statistics if available
        if enable_filtering and 'classification_stats' in crawl_result:
            summary.update({
                'total_candidates': crawl_result['total_candidates'],
                'filtered_out': crawl_result.get('filtered_out', 0),
                'classification_stats': crawl_result['classification_stats'],
                'confidence_threshold': confidence_threshold
            })
        
        return jsonify({
            'success': True,
            'summary': summary,
            'results': analysis_results
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Website analysis failed: {str(e)}',
            'results': []
        }), 500
