"""
Routes for Philippine News Search Index functionality
Handles Philippine news article indexing, searching, and analytics
"""

import sqlite3
import sys
import os
import time
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
    """Search Philippine news articles with pagination and performance optimizations"""
    try:
        start_time = time.time()
        
        if request.method == 'POST':
            data = request.get_json()
            query = data.get('query', '').strip()
            source = data.get('source')
            page = max(int(data.get('page', 1)), 1)  # Page number (1-based)
            per_page = min(max(int(data.get('per_page', 20)), 1), 100)  # Results per page (1-100)
        else:
            query = request.args.get('q', '').strip()
            source = request.args.get('source')
            page = max(int(request.args.get('page', 1)), 1)
            per_page = min(max(int(request.args.get('per_page', 20)), 1), 100)
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        # Calculate offset for pagination
        offset = (page - 1) * per_page
        
        # Perform optimized search with pagination
        search_results = philippine_search_index.search_articles(
            query=query,
            limit=per_page,
            offset=offset,
            source=source
        )
        
        end_time = time.time()
        total_response_time = round((end_time - start_time) * 1000, 2)
        
        # Calculate pagination metadata
        total_results = search_results.get('total_count', search_results['count'])
        total_pages = max(1, (total_results + per_page - 1) // per_page)  # Ceiling division
        has_next = page < total_pages
        has_prev = page > 1
        
        return jsonify({
            'query': query,
            'results': search_results['results'],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_results': total_results,
                'total_pages': total_pages,
                'has_next': has_next,
                'has_prev': has_prev,
                'next_page': page + 1 if has_next else None,
                'prev_page': page - 1 if has_prev else None,
                'showing_from': offset + 1 if total_results > 0 else 0,
                'showing_to': min(offset + len(search_results['results']), total_results)
            },
            'response_time': total_response_time,  # Total route processing time in ms
            'search_engine_time': round(search_results.get('response_time', 0) * 1000, 2),  # Search engine time in ms
            'filters': {
                'source': source
            },
            'performance_stats': {
                'total_response_time_ms': total_response_time,
                'search_engine_time_ms': round(search_results.get('response_time', 0) * 1000, 2),
                'search_optimization': 'Connection pooling, indexed queries, and pagination'
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
    """Index multiple Philippine news article URLs in batch with atomic transactions and performance optimizations"""
    try:
        start_time = time.time()
        
        data = request.get_json()
        urls = data.get('urls', [])
        force_reindex = data.get('force_reindex', False)
        
        if not urls or not isinstance(urls, list):
            return jsonify({'error': 'URLs array is required'}), 400
        
        if len(urls) > 50:  # Limit batch size for performance
            return jsonify({'error': 'Maximum 50 URLs allowed per batch'}), 400
        
        # Remove duplicates for efficiency
        unique_urls = list(set(urls))
        
        # Use the optimized batch indexing method with atomic transactions
        batch_results = philippine_search_index.batch_index_articles(unique_urls, force_reindex)
        
        # Calculate performance statistics
        success_count = len([r for r in batch_results if r['status'] == 'success'])
        skipped_count = len([r for r in batch_results if r['status'] == 'skipped'])
        error_count = len([r for r in batch_results if r['status'] == 'error'])
        already_indexed_count = len([r for r in batch_results if r['status'] == 'already_indexed'])
        
        end_time = time.time()
        total_response_time = round((end_time - start_time) * 1000, 2)
        
        return jsonify({
            'success': True,
            'message': f'Batch indexing completed with atomic transactions in {total_response_time}ms',
            'summary': {
                'total_urls_submitted': len(urls),
                'unique_urls_processed': len(unique_urls),
                'successfully_indexed': success_count,
                'skipped': skipped_count,
                'errors': error_count,
                'already_indexed': already_indexed_count
            },
            'performance_stats': {
                'total_response_time_ms': total_response_time,
                'avg_time_per_url_ms': round(total_response_time / len(unique_urls), 2) if unique_urls else 0,
                'optimization_features': [
                    'Atomic database transactions',
                    'Connection pooling',
                    'Batch processing',
                    'Duplicate URL removal'
                ]
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
    """Get available categories in the Philippine news index with optimized caching"""
    try:
        # Use connection pooling if available, otherwise fallback to direct connection
        if hasattr(philippine_search_index, 'db_pool'):
            with philippine_search_index.db_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT category, COUNT(*) as count 
                    FROM philippine_articles 
                    WHERE category IS NOT NULL AND category != ''
                    GROUP BY category 
                    ORDER BY count DESC
                    LIMIT 50
                ''')
                
                categories = [{'category': row[0], 'count': row[1]} for row in cursor.fetchall()]
        else:
            conn = sqlite3.connect(philippine_search_index.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT category, COUNT(*) as count 
                FROM philippine_articles 
                WHERE category IS NOT NULL AND category != ''
                GROUP BY category 
                ORDER BY count DESC
                LIMIT 50
            ''')
            
            categories = [{'category': row[0], 'count': row[1]} for row in cursor.fetchall()]
            conn.close()
        
        return jsonify({
            'success': True,
            'categories': categories,
            'total_categories': len(categories)
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@philippine_news_bp.route('/philippine-news-sources')
def get_philippine_news_sources_route():
    """Get list of available Philippine news sources from tracked websites"""
    try:
        # Import here to avoid circular imports
        import sqlite3
        from urllib.parse import urlparse
        
        # Get sources from tracked websites database
        sources = []
        try:
            conn = sqlite3.connect('news_tracker.db')
            cursor = conn.cursor()
            
            # Get distinct root domains from tracked websites
            cursor.execute('''
                SELECT DISTINCT url, name 
                FROM tracked_websites 
                WHERE status = 'active'
                ORDER BY name
            ''')
            
            websites = cursor.fetchall()
            conn.close()
            
            # Extract root domains and create source list
            domain_map = {}
            for url, name in websites:
                try:
                    parsed = urlparse(url)
                    domain = parsed.netloc.lower()
                    
                    # Remove www. prefix if present
                    if domain.startswith('www.'):
                        domain = domain[4:]
                    
                    # Use the website name as display name, domain as value
                    if domain not in domain_map:
                        domain_map[domain] = {
                            'source': domain,
                            'name': name,
                            'url': url,
                            'count': 1
                        }
                    else:
                        domain_map[domain]['count'] += 1
                
                except Exception as e:
                    continue
            
            # Convert to list format expected by frontend
            sources = list(domain_map.values())
            
            # Sort by name
            sources.sort(key=lambda x: x['name'])
            
        except Exception as e:
            print(f"Error getting tracked websites: {e}")
            # Fallback to Philippine news index sources if database query fails
            if hasattr(philippine_search_index, 'db_pool'):
                with philippine_search_index.db_pool.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        SELECT source_domain, COUNT(*) as count 
                        FROM philippine_articles 
                        GROUP BY source_domain 
                        ORDER BY count DESC
                        LIMIT 100
                    ''')
                    
                    sources = [{'source': row[0], 'name': row[0], 'count': row[1]} for row in cursor.fetchall()]
            else:
                # Final fallback to hardcoded sources
                sources = [
                    {'source': 'abs-cbn.com', 'name': 'ABS-CBN', 'count': 0},
                    {'source': 'gmanetwork.com', 'name': 'GMA News', 'count': 0},
                    {'source': 'inquirer.net', 'name': 'Philippine Daily Inquirer', 'count': 0},
                    {'source': 'rappler.com', 'name': 'Rappler', 'count': 0},
                    {'source': 'philstar.com', 'name': 'Philippine Star', 'count': 0},
                    {'source': 'manilabulletin.com.ph', 'name': 'Manila Bulletin', 'count': 0},
                    {'source': 'cnnphilippines.com', 'name': 'CNN Philippines', 'count': 0},
                    {'source': 'pna.gov.ph', 'name': 'Philippine News Agency', 'count': 0}
                ]
        
        return jsonify({
            'success': True,
            'sources': sources,
            'total_sources': len(sources)
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
