"""
Routes for Philippine News Search Index functionality
Handles Philippine news article indexing, searching, and analytics
"""

import sqlite3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, request, jsonify
from urllib.parse import urlparse
from modules.philippine_news_search_index import (
    PhilippineNewsSearchIndex,
    get_philippine_news_categories,
    get_philippine_news_sources
)

# Create blueprint for Philippine news search routes
philippine_news_bp = Blueprint('philippine_news', __name__)

# Initialize the Philippine news search index (will be set by main app)
philippine_search_index = None

def init_philippine_search_index():
    """Initialize the Philippine news search index"""
    global philippine_search_index
    philippine_search_index = PhilippineNewsSearchIndex()
    return philippine_search_index

def get_philippine_search_index():
    """Get the initialized Philippine search index instance"""
    return philippine_search_index

@philippine_news_bp.route('/index-philippine-article', methods=['POST'])
def index_philippine_article():
    """Index a Philippine news article URL into the search database"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        force_reindex = data.get('force_reindex', False)
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Validate URL format
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Index the article
        result = philippine_search_index.index_article(url, force_reindex)
        
        if result['status'] == 'success':
            return jsonify({
                'message': 'Article indexed successfully',
                'article_id': result['article_id'],
                'relevance_score': result['relevance_score'],
                'locations_found': result['locations'],
                'government_entities_found': result['government_entities'],
                'url': url
            })
        elif result['status'] == 'already_indexed':
            return jsonify({
                'message': 'Article already indexed',
                'url': url
            }), 200
        elif result['status'] == 'skipped':
            return jsonify({
                'message': 'Article skipped - not relevant to Philippine news',
                'url': url
            }), 200
        else:
            return jsonify({
                'error': f'Indexing failed: {result.get("message", "Unknown error")}',
                'url': url
            }), 500
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@philippine_news_bp.route('/search-philippine-news', methods=['GET', 'POST'])
def search_philippine_news():
    """Search Philippine news articles"""
    try:
        if request.method == 'POST':
            data = request.get_json()
            query = data.get('query', '').strip()
            category = data.get('category')
            source = data.get('source')
            limit = min(int(data.get('limit', 20)), 100)  # Max 100 results
        else:
            query = request.args.get('q', '').strip()
            category = request.args.get('category')
            source = request.args.get('source')
            limit = min(int(request.args.get('limit', 20)), 100)
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        # Perform search
        search_results = philippine_search_index.search_articles(
            query=query,
            limit=limit,
            category=category,
            source=source
        )
        
        return jsonify({
            'query': query,
            'results': search_results['results'],
            'total_count': search_results['count'],
            'response_time': search_results['response_time'],
            'filters': {
                'category': category,
                'source': source,
                'limit': limit
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@philippine_news_bp.route('/philippine-news-analytics')
def philippine_news_analytics():
    """Get analytics and statistics for the Philippine news index"""
    try:
        analytics = philippine_search_index.get_search_analytics()
        return jsonify(analytics)
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@philippine_news_bp.route('/find-similar-content', methods=['POST'])
def find_similar_content():
    """Find similar articles in the Philippine news database based on content similarity"""
    try:
        data = request.get_json()
        content_text = data.get('content', '').strip()
        limit = min(int(data.get('limit', 10)), 20)  # Max 20 results
        minimum_similarity = float(data.get('minimum_similarity', 0.1))
        
        if not content_text:
            return jsonify({'error': 'Content text is required'}), 400
        
        if len(content_text) < 50:
            return jsonify({'error': 'Content text must be at least 50 characters long'}), 400
        
        # Find similar content using the search index
        similar_results = philippine_search_index.find_similar_content(
            content_text=content_text,
            limit=limit,
            minimum_similarity=minimum_similarity
        )
        
        return jsonify({
            'success': True,
            'content_length': len(content_text),
            'query_summary': similar_results.get('query_summary', ''),
            'top_keywords': similar_results.get('top_keywords', []),
            'results': similar_results['results'],
            'total_count': similar_results['count'],
            'response_time': similar_results['response_time'],
            'minimum_similarity': minimum_similarity,
            'search_type': 'content_similarity'
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@philippine_news_bp.route('/philippine-article/<int:article_id>')
def get_philippine_article(article_id):
    """Get full details of a specific Philippine news article"""
    try:
        article = philippine_search_index.get_article_by_id(article_id)
        
        if not article:
            return jsonify({'error': 'Article not found'}), 404
        
        return jsonify(article)
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@philippine_news_bp.route('/batch-index-philippine-articles', methods=['POST'])
def batch_index_philippine_articles():
    """Index multiple Philippine news article URLs in batch"""
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        force_reindex = data.get('force_reindex', False)
        
        if not urls or not isinstance(urls, list):
            return jsonify({'error': 'URLs array is required'}), 400
        
        if len(urls) > 50:  # Limit batch size
            return jsonify({'error': 'Maximum 50 URLs allowed per batch'}), 400
        
        # Process URLs in background thread for better performance
        def process_batch():
            results = []
            for url in urls:
                try:
                    result = philippine_search_index.index_article(url.strip(), force_reindex)
                    results.append({
                        'url': url,
                        'status': result['status'],
                        'message': result.get('message', ''),
                        'article_id': result.get('article_id'),
                        'relevance_score': result.get('relevance_score', 0)
                    })
                except Exception as e:
                    results.append({
                        'url': url,
                        'status': 'error',
                        'message': str(e)
                    })
            
            return results
        
        # For now, process synchronously (can be made async later)
        batch_results = process_batch()
        
        # Summary statistics
        success_count = len([r for r in batch_results if r['status'] == 'success'])
        skipped_count = len([r for r in batch_results if r['status'] == 'skipped'])
        error_count = len([r for r in batch_results if r['status'] == 'error'])
        already_indexed_count = len([r for r in batch_results if r['status'] == 'already_indexed'])
        
        return jsonify({
            'message': f'Batch indexing completed',
            'summary': {
                'total_urls': len(urls),
                'successfully_indexed': success_count,
                'skipped': skipped_count,
                'errors': error_count,
                'already_indexed': already_indexed_count
            },
            'results': batch_results
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@philippine_news_bp.route('/crawl-and-index-website', methods=['POST'])
def crawl_and_index_website():
    """Crawl a news website and automatically index all found Philippine news articles"""
    try:
        data = request.get_json()
        website_url = data.get('website_url', '').strip()
        max_articles = int(data.get('max_articles', 20))
        force_reindex = data.get('force_reindex', False)
        
        if not website_url:
            return jsonify({'error': 'Website URL is required'}), 400
        
        # Validate URL format
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
        
        parsed_url = urlparse(website_url)
        if not parsed_url.netloc:
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Perform crawling and indexing
        result = philippine_search_index.crawl_and_index_website(
            website_url, 
            max_articles, 
            force_reindex
        )
        
        if result['status'] == 'error':
            return jsonify({
                'error': result['message'],
                'website_url': website_url
            }), 500
        
        return jsonify({
            'success': True,
            'message': result['message'],
            'website_url': result['website_url'],
            'website_title': result.get('website_title', ''),
            'summary': result['summary'],
            'results': result['results']
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@philippine_news_bp.route('/get-crawl-history')
def get_crawl_history():
    """Get history of website crawling and indexing tasks"""
    try:
        limit = int(request.args.get('limit', 50))
        limit = min(limit, 100)  # Maximum 100 records
        
        history = philippine_search_index.get_crawl_history(limit)
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@philippine_news_bp.route('/philippine-news-categories')
def get_philippine_news_categories_route():
    """Get available categories in the Philippine news index"""
    try:
        conn = sqlite3.connect(philippine_search_index.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT category, COUNT(*) as count 
            FROM philippine_articles 
            WHERE category IS NOT NULL AND category != ''
            GROUP BY category 
            ORDER BY count DESC
        ''')
        
        categories = [{'category': row[0], 'count': row[1]} for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({'categories': categories})
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@philippine_news_bp.route('/philippine-news-sources')
def get_philippine_news_sources_route():
    """Get available sources in the Philippine news index"""
    try:
        conn = sqlite3.connect(philippine_search_index.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT source_domain, COUNT(*) as count 
            FROM philippine_articles 
            GROUP BY source_domain 
            ORDER BY count DESC
        ''')
        
        sources = [{'source': row[0], 'count': row[1]} for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({'sources': sources})
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
