"""
News Tracker Routes
Handles website tracking, article fetching, and verification
Uses the sophisticated news crawler for intelligent article extraction

INTEGRATION FEATURES:
- Uses /crawl-website endpoint for intelligent article discovery
- Stores crawler metadata (confidence, predictions) for feedback
- Contributes to ML model improvement through user verifications
- Automatically indexes verified news articles into Philippine news search system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, render_template, request, jsonify, session
import json
import sqlite3
import requests
from datetime import datetime, timedelta
from urllib.parse import urlparse
import time
import threading
import hashlib
import logging
import uuid
from modules.url_news_classifier import URLNewsClassifier
from routes.url_classifier_routes import get_url_classifier
from routes.philippine_news_search_routes import get_philippine_search_index
from routes.news_crawler_routes import get_news_crawler

news_tracker_bp = Blueprint('news_tracker', __name__)

# Configuration
DATABASE_FILE = 'news_tracker.db'

# Initialize database
def init_news_tracker_db():
    """Initialize the news tracker database"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tracked_websites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            fetch_interval INTEGER DEFAULT 60,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_fetch TIMESTAMP,
            status TEXT DEFAULT 'active',
            user_session TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS article_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT,
            description TEXT,
            content TEXT,
            site_name TEXT,
            website_id INTEGER,
            found_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            verified BOOLEAN DEFAULT FALSE,
            is_news BOOLEAN,
            verified_at TIMESTAMP,
            user_session TEXT,
            confidence REAL DEFAULT 0.0,
            is_news_prediction BOOLEAN DEFAULT TRUE,
            probability_news REAL DEFAULT 0.0,
            FOREIGN KEY (website_id) REFERENCES tracked_websites (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fetch_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            website_id INTEGER,
            fetch_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            articles_found INTEGER DEFAULT 0,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            FOREIGN KEY (website_id) REFERENCES tracked_websites (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    
    # Migrate existing databases to add new crawler metadata columns
    migrate_database_schema()

def migrate_database_schema():
    """Add new columns to existing database if they don't exist"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Check if new columns exist and add them if they don't
        cursor.execute("PRAGMA table_info(article_queue)")
        columns = [column[1] for column in cursor.fetchall()]
        
        new_columns = [
            ('confidence', 'REAL DEFAULT 0.0'),
            ('is_news_prediction', 'BOOLEAN DEFAULT TRUE'),
            ('probability_news', 'REAL DEFAULT 0.0')
        ]
        
        for column_name, column_definition in new_columns:
            if column_name not in columns:
                cursor.execute(f'ALTER TABLE article_queue ADD COLUMN {column_name} {column_definition}')
                logging.info(f"✅ Added column {column_name} to article_queue table")
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logging.error(f"❌ Database migration error: {str(e)}")

# Initialize database on import
init_news_tracker_db()

@news_tracker_bp.route('/news-tracker')
def news_tracker():
    """Render the news tracker page"""
    # Ensure user has a session
    if 'session_id' not in session:
        import uuid
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('news_tracker.html')

@news_tracker_bp.route('/api/news-tracker/add-website', methods=['POST'])
def add_website():
    """Add a website to track"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        name = data.get('name', '').strip()
        interval = int(data.get('interval', 60))
        
        if not url or not name:
            return jsonify({'success': False, 'error': 'URL and name are required'})
        
        # Validate URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return jsonify({'success': False, 'error': 'Invalid URL format'})
        except Exception:
            return jsonify({'success': False, 'error': 'Invalid URL format'})
        
        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        user_session = session.get('session_id', 'default')
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO tracked_websites (url, name, fetch_interval, user_session)
                VALUES (?, ?, ?, ?)
            ''', (url, name, interval, user_session))
            
            website_id = cursor.lastrowid
            conn.commit()
            
            return jsonify({
                'success': True,
                'id': website_id,
                'message': f'Successfully added {name} to tracking list'
            })
            
        except sqlite3.IntegrityError:
            return jsonify({'success': False, 'error': 'This website is already being tracked'})
        
        finally:
            conn.close()
            
    except Exception as e:
        logging.error(f"Error adding website: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'})

@news_tracker_bp.route('/api/news-tracker/remove-website/<int:website_id>', methods=['DELETE'])
def remove_website(website_id):
    """Remove a website from tracking"""
    try:
        user_session = session.get('session_id', 'default')
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Check if website exists and belongs to user
        cursor.execute('''
            SELECT id FROM tracked_websites 
            WHERE id = ? AND user_session = ?
        ''', (website_id, user_session))
        
        if not cursor.fetchone():
            return jsonify({'success': False, 'error': 'Website not found'})
        
        # Remove website and its articles
        cursor.execute('DELETE FROM article_queue WHERE website_id = ?', (website_id,))
        cursor.execute('DELETE FROM fetch_logs WHERE website_id = ?', (website_id,))
        cursor.execute('DELETE FROM tracked_websites WHERE id = ?', (website_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Website removed successfully'})
        
    except Exception as e:
        logging.error(f"Error removing website: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'})

@news_tracker_bp.route('/api/news-tracker/fetch-articles', methods=['POST'])
def fetch_articles():
    """Fetch articles from all tracked websites"""
    try:
        user_session = session.get('session_id', 'default')
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Get tracked websites
        cursor.execute('''
            SELECT id, url, name FROM tracked_websites 
            WHERE status = 'active' AND user_session = ?
        ''', (user_session,))
        
        websites = cursor.fetchall()
        all_articles = []
        
        for website_id, url, name in websites:
            try:
                articles = fetch_articles_from_crawler(url, name, website_id)
                all_articles.extend(articles)
                
                # Update last fetch time
                cursor.execute('''
                    UPDATE tracked_websites 
                    SET last_fetch = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (website_id,))
                
                # Log fetch attempt
                cursor.execute('''
                    INSERT INTO fetch_logs (website_id, articles_found, success)
                    VALUES (?, ?, ?)
                ''', (website_id, len(articles), True))
                
            except Exception as e:
                logging.error(f"Error fetching from {url}: {str(e)}")
                cursor.execute('''
                    INSERT INTO fetch_logs (website_id, articles_found, success, error_message)
                    VALUES (?, ?, ?, ?)
                ''', (website_id, 0, False, str(e)))
        
        # Save articles to database
        new_articles = []
        for article in all_articles:
            try:
                cursor.execute('''
                    INSERT INTO article_queue (url, title, description, content, site_name, website_id, user_session, confidence, is_news_prediction, probability_news)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article['url'], article['title'], article['description'],
                    article['content'], article['site_name'], article['website_id'],
                    user_session, article.get('confidence', 0.0), 
                    article.get('is_news_prediction', True), article.get('probability_news', 0.0)
                ))
                
                new_articles.append({
                    'id': cursor.lastrowid,
                    **article
                })
                
            except sqlite3.IntegrityError:
                # Article already exists
                pass
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'articles': new_articles,
            'message': f'Found {len(new_articles)} new articles'
        })
        
    except Exception as e:
        logging.error(f"Error fetching articles: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'})

@news_tracker_bp.route('/api/news-tracker/verify-article', methods=['POST'])
def verify_article():
    """Verify if an article is news or not"""
    try:
        data = request.get_json()
        article_id = data.get('articleId')
        is_news = data.get('isNews')
        url = data.get('url')
        
        user_session = session.get('session_id', 'default')
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Get article data including crawler metadata before updating
        cursor.execute('''
            SELECT url, confidence, is_news_prediction, probability_news
            FROM article_queue 
            WHERE id = ? AND user_session = ?
        ''', (article_id, user_session))
        
        article_data = cursor.fetchone()
        
        # Update article verification
        cursor.execute('''
            UPDATE article_queue 
            SET verified = TRUE, is_news = ?, verified_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_session = ?
        ''', (is_news, article_id, user_session))
        
        conn.commit()
        conn.close()
        
        # Send feedback to URL classifier for model improvement
        send_url_classifier_feedback(
            url=url,
            article_data=article_data,
            user_verification=is_news
        )
        
        # If article is verified as news, index it in Philippine news search system using batch method
        philippine_indexing_success = False
        if is_news:
            # Use batch method for consistency even for single articles
            batch_results = batch_index_philippine_news_articles([url])
            philippine_indexing_success = len(batch_results) > 0 and batch_results[0].get('success', False)
        
        return jsonify({
            'success': True,
            'message': f'Article marked as {"news" if is_news else "not news"}',
            'url_classifier_feedback_sent': True,  # Indicate feedback was attempted
            'philippine_indexing_attempted': is_news,  # Indicate if indexing was attempted
            'philippine_indexing_success': philippine_indexing_success  # Indicate if indexing succeeded
        })
        
    except Exception as e:
        logging.error(f"Error verifying article: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'})

@news_tracker_bp.route('/api/news-tracker/batch-verify-articles', methods=['POST'])
def batch_verify_articles():
    """Verify multiple articles at once for efficiency"""
    try:
        data = request.get_json()
        articles = data.get('articles', [])
        
        if not articles or not isinstance(articles, list):
            return jsonify({'success': False, 'error': 'Articles array is required'})
        
        if len(articles) > 10:  # Max 10 articles per batch
            return jsonify({'success': False, 'error': 'Maximum 10 articles allowed per batch'})
        
        user_session = session.get('session_id', 'default')
        
        # Process each article in the batch
        results = []
        news_articles_for_indexing = []
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        for article in articles:
            try:
                article_id = article.get('articleId')
                is_news = article.get('isNews')
                url = article.get('url')
                
                if not article_id or is_news is None:
                    results.append({
                        'articleId': article_id,
                        'success': False,
                        'error': 'Missing articleId or isNews'
                    })
                    continue
                
                # Get article data including crawler metadata before updating
                cursor.execute('''
                    SELECT url, confidence, is_news_prediction, probability_news
                    FROM article_queue 
                    WHERE id = ? AND user_session = ?
                ''', (article_id, user_session))
                
                article_data = cursor.fetchone()
                
                if not article_data:
                    results.append({
                        'articleId': article_id,
                        'success': False,
                        'error': 'Article not found'
                    })
                    continue
                
                # Update article verification
                cursor.execute('''
                    UPDATE article_queue 
                    SET verified = TRUE, is_news = ?, verified_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND user_session = ?
                ''', (is_news, article_id, user_session))
                
                # Send feedback to URL classifier for model improvement
                send_url_classifier_feedback(
                    url=url or article_data[0],
                    article_data=article_data,
                    user_verification=is_news
                )
                
                # Collect news articles for batch Philippine indexing
                if is_news:
                    news_articles_for_indexing.append(url or article_data[0])
                
                results.append({
                    'articleId': article_id,
                    'success': True,
                    'message': f'Article marked as {"news" if is_news else "not news"}',
                    'url_classifier_feedback_sent': True,
                    'philippine_indexing_queued': is_news
                })
                
            except Exception as e:
                logging.error(f"Error verifying article {article.get('articleId')}: {str(e)}")
                results.append({
                    'articleId': article.get('articleId'),
                    'success': False,
                    'error': str(e)
                })
        
        conn.commit()
        conn.close()
        
        # Batch index news articles into Philippine news system using optimized batch method
        philippine_indexing_results = []
        if news_articles_for_indexing:
            logging.info(f"🚀 Batch indexing {len(news_articles_for_indexing)} verified news articles into Philippine system")
            philippine_indexing_results = batch_index_philippine_news_articles(news_articles_for_indexing)
        
        # Calculate summary statistics (enhanced to match philippine_news_search_routes pattern)
        successful_verifications = len([r for r in results if r['success']])
        failed_verifications = len([r for r in results if not r['success']])
        news_count = len([r for r in results if r.get('philippine_indexing_queued')])
        
        # Enhanced Philippine indexing statistics
        if philippine_indexing_results:
            successful_indexing = len([r for r in philippine_indexing_results if r['status'] == 'success'])
            skipped_indexing = len([r for r in philippine_indexing_results if r['status'] == 'skipped'])
            error_indexing = len([r for r in philippine_indexing_results if r['status'] == 'error'])
            already_indexed = len([r for r in philippine_indexing_results if r['status'] == 'already_indexed'])
        else:
            successful_indexing = skipped_indexing = error_indexing = already_indexed = 0
        
        return jsonify({
            'success': True,
            'message': f'Batch verification completed: {successful_verifications} successful, {failed_verifications} failed',
            'summary': {
                'total_articles': len(articles),
                'successful_verifications': successful_verifications,
                'failed_verifications': failed_verifications,
                'news_articles_found': news_count,
                'philippine_indexing_attempts': len(philippine_indexing_results),
                'successful_philippine_indexing': successful_indexing,
                'philippine_indexing_details': {
                    'successfully_indexed': successful_indexing,
                    'already_indexed': already_indexed,
                    'skipped': skipped_indexing,
                    'errors': error_indexing
                }
            },
            'results': results,
            'philippine_indexing_results': philippine_indexing_results
        })
        
    except Exception as e:
        logging.error(f"Error in batch verification: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'})

@news_tracker_bp.route('/api/news-tracker/get-data')
def get_tracker_data():
    """Get all tracker data for the frontend"""
    try:
        user_session = session.get('session_id', 'default')
        
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get tracked websites with article counts
        cursor.execute('''
            SELECT w.*, 
                   COUNT(CASE WHEN a.id IS NOT NULL THEN 1 END) as article_count,
                   MAX(a.found_at) as last_article,
                   COUNT(CASE WHEN a.verified = 1 THEN 1 END) as verified_count
            FROM tracked_websites w
            LEFT JOIN article_queue a ON w.id = a.website_id
            WHERE w.user_session = ?
            GROUP BY w.id
            ORDER BY w.added_at DESC
        ''', (user_session,))
        
        websites = []
        for row in cursor.fetchall():
            website = dict(row)
            # Convert timestamps to ISO format for JavaScript
            if website['added_at']:
                website['addedAt'] = website['added_at']
            if website['last_fetch']:
                website['lastFetch'] = website['last_fetch']
            if website['last_article']:
                website['lastArticle'] = website['last_article']
            
            websites.append(website)
        
        # Get article queue
        cursor.execute('''
            SELECT a.*, w.name as site_name
            FROM article_queue a
            LEFT JOIN tracked_websites w ON a.website_id = w.id
            WHERE a.user_session = ?
            ORDER BY a.found_at DESC
        ''', (user_session,))
        
        articles = []
        for row in cursor.fetchall():
            article = dict(row)
            # Convert timestamps and booleans for JavaScript
            if article['found_at']:
                article['foundAt'] = article['found_at']
            if article['verified_at']:
                article['verifiedAt'] = article['verified_at']
            
            # Convert SQLite boolean (0/1) to actual boolean
            article['verified'] = bool(article['verified'])
            if article['is_news'] is not None:
                article['isNews'] = bool(article['is_news'])
            
            articles.append(article)
        
        conn.close()
        
        return jsonify({
            'success': True,
            'websites': websites,
            'articles': articles
        })
        
    except Exception as e:
        logging.error(f"Error getting tracker data: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'})

def fetch_articles_from_crawler(url, site_name, website_id):
    """Fetch articles from a website using direct crawler method calls for better performance"""
    articles = []
    
    try:
        # Get the shared crawler instance (much faster than HTTP)
        news_crawler = get_news_crawler()
        
        if news_crawler is None:
            logging.warning("⚠️ News crawler not available, skipping article fetch")
            return articles
        
        # Prepare crawler parameters (same as HTTP endpoint)
        crawler_params = {
            'website_url': url,
            'enable_filtering': True,  # Use intelligent filtering
            'confidence_threshold': 0.6  # Only get likely news articles
        }
        
        # Use direct method call (eliminates HTTP overhead: ~7-33ms saved per request)
        start_time = time.time()
        
        # Configure the crawler's filtering settings
        news_crawler.set_filtering_mode(
            crawler_params['enable_filtering'], 
            crawler_params['confidence_threshold']
        )
        
        # Extract article links using the same method as HTTP endpoint
        crawl_result = news_crawler.extract_article_links(
            crawler_params['website_url'], 
            enable_filtering=crawler_params['enable_filtering']
        )
        
        if not crawl_result['success']:
            logging.error(f"❌ Direct crawler failed for {url}: {crawl_result.get('error', 'Unknown error')}")
            return articles
        
        # Normalize article data structure (same as HTTP endpoint)
        raw_articles = crawl_result['articles']
        normalized_articles = []
        
        if crawler_params['enable_filtering'] and raw_articles:
            # With filtering enabled, articles are objects with metadata
            for article in raw_articles:
                if isinstance(article, dict):
                    normalized_articles.append({
                        'url': article.get('url', ''),
                        'title': article.get('title', ''),
                        'link_text': article.get('link_text', ''),
                        'confidence': article.get('confidence', 0.0),
                        'is_news_prediction': article.get('is_news_prediction', True),
                        'probability_news': article.get('probability_news', 0.0),
                        'probability_not_news': article.get('probability_not_news', 1.0)
                    })
        else:
            # Without filtering, articles are just URLs
            for article_url in raw_articles:
                normalized_articles.append({
                    'url': article_url,
                    'title': '',
                    'link_text': '',
                    'confidence': 1.0,
                    'is_news_prediction': True,
                    'probability_news': 1.0,
                    'probability_not_news': 0.0
                })
        
        
        # Build crawler result in same format as HTTP endpoint
        crawler_result = {
            'success': True,
            'website_title': crawl_result.get('website_title', 'Unknown'),
            'total_found': len(normalized_articles),
            'articles': normalized_articles,
            'filtering_enabled': crawler_params['enable_filtering'],
            'classification_method': crawl_result.get('classification_method', 'Basic URL patterns'),
            'crawler_stats': {
                'total_candidates': crawl_result.get('total_candidates', len(normalized_articles)),
                'filtered_articles': len(normalized_articles),
                'filtered_out': crawl_result.get('filtered_out', 0),
                'confidence_threshold': crawler_params['confidence_threshold']
            }
        }
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        if crawler_result.get('success') and crawler_result.get('articles'):
            # Convert crawler results to news tracker format
            for article in crawler_result['articles']:
                # Extract data from crawler's normalized format
                article_data = {
                    'url': article.get('url', ''),
                    'title': article.get('title', '') or article.get('link_text', ''),
                    'description': '',  # Crawler doesn't provide descriptions yet
                    'content': '',
                    'site_name': site_name,
                    'website_id': website_id,
                    'found_at': datetime.now().isoformat(),
                    # Add crawler-specific metadata
                    'confidence': article.get('confidence', 0.0),
                    'is_news_prediction': article.get('is_news_prediction', True),
                    'probability_news': article.get('probability_news', 0.0)
                }
                
                # Only include articles with valid URLs
                if article_data['url']:
                    articles.append(article_data)
            
            logging.info(f"✅ Direct crawler found {len(articles)} articles from {url} (took {duration_ms:.1f}ms)")
            logging.info(f"📊 Crawler stats: {crawler_result.get('crawler_stats', {})}")
        else:
            logging.warning(f"⚠️ Direct crawler returned no articles for {url}")
            
    except Exception as e:
        logging.error(f"❌ Error with direct crawler call for {url}: {str(e)}")
        
        # Fallback to HTTP request if direct method fails
        logging.info(f"🔄 Falling back to HTTP crawler for {url}")
        try:
            return fetch_articles_from_crawler_http(url, site_name, website_id)
        except Exception as fallback_error:
            logging.error(f"❌ HTTP fallback also failed for {url}: {str(fallback_error)}")
    
    return articles

def fetch_articles_from_crawler_http(url, site_name, website_id):
    """Fallback HTTP-based crawler method (kept for reliability)"""
    articles = []
    
    try:
        # Prepare request to crawler endpoint
        crawler_data = {
            'website_url': url,
            'enable_filtering': True,
            'confidence_threshold': 0.6
        }
        
        # Make request to crawler endpoint
        response = requests.post(
            'http://127.0.0.1:5000/crawl-website',
            json=crawler_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            crawler_result = response.json()
            
            if crawler_result.get('success') and crawler_result.get('articles'):
                # Convert crawler results to news tracker format
                for article in crawler_result['articles']:
                    # Extract data from crawler's normalized format
                    article_data = {
                        'url': article.get('url', ''),
                        'title': article.get('title', '') or article.get('link_text', ''),
                        'description': '',
                        'content': '',
                        'site_name': site_name,
                        'website_id': website_id,
                        'found_at': datetime.now().isoformat(),
                        'confidence': article.get('confidence', 0.0),
                        'is_news_prediction': article.get('is_news_prediction', True),
                        'probability_news': article.get('probability_news', 0.0)
                    }
                    
                    if article_data['url']:
                        articles.append(article_data)
                
                logging.info(f"✅ HTTP fallback crawler found {len(articles)} articles from {url}")
            else:
                logging.warning(f"⚠️ HTTP fallback crawler returned no articles for {url}")
        else:
            logging.error(f"❌ HTTP fallback crawler request failed for {url}: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Network error with HTTP fallback crawler for {url}: {str(e)}")
    except Exception as e:
        logging.error(f"❌ Error with HTTP fallback crawler for {url}: {str(e)}")
    
    return articles

def send_url_classifier_feedback(url, article_data, user_verification):
    """Send feedback to URL classifier to improve the model using direct method calls"""
    try:
        if not url or article_data is None:
            logging.warning("⚠️ Incomplete data for URL classifier feedback")
            return
        
        # Extract crawler prediction data
        article_url = article_data[0] if article_data else url
        confidence = article_data[1] if len(article_data) > 1 else None
        predicted_is_news = article_data[2] if len(article_data) > 2 else True  # Default to True if no prediction
        probability_news = article_data[3] if len(article_data) > 3 else None
        
        # Get the shared URL classifier instance
        url_classifier = get_url_classifier()
        
        if url_classifier is None:
            logging.warning("⚠️ URL classifier not available for feedback")
            return
        
        # Prepare feedback data (same format as HTTP endpoint)
        feedback_data = {
            'url': article_url,
            'predicted_label': bool(predicted_is_news),  # What the crawler/model predicted
            'actual_label': 'news' if user_verification else 'non-news',  # What user verified
            'user_confidence': 1.0,  # High confidence since it's human verification
            'comment': f'News Tracker verification - Confidence: {confidence}, Prob: {probability_news}'
        }
        
        # Use direct method call instead of HTTP request
        start_time = time.time()
        
        # Add feedback directly to the classifier
        success = url_classifier.add_feedback(
            url=feedback_data['url'],
            predicted_label=feedback_data['predicted_label'],
            actual_label=feedback_data['actual_label'],
            user_confidence=feedback_data['user_confidence']
        )
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        if success:
            logging.info(f"✅ URL classifier feedback added successfully for {url} (took {duration_ms:.1f}ms)")
            
            # Get current model stats
            stats = url_classifier.get_model_stats()
            logging.info(f"📊 Model stats: {stats}")
            
            # Check if retraining should be triggered (every 10 feedback entries)
            feedback_count = stats.get('total_feedback', 0)
            if feedback_count > 0 and feedback_count % 10 == 0:
                logging.info("🚀 URL classifier retraining recommended (10 new feedback entries)")
        else:
            logging.warning(f"⚠️ URL classifier feedback failed for {url}")
            
    except Exception as e:
        logging.warning(f"⚠️ Error sending URL classifier feedback: {str(e)}")
    # Don't raise exceptions - feedback is optional and shouldn't break verification

def batch_index_philippine_news_articles(urls, force_reindex=False):
    """
    Batch index multiple news articles into the Philippine news search system using direct method calls
    
    This function uses the same optimized approach as /batch-index-philippine-articles endpoint
    from philippine_news_search_routes.py for better performance and consistency.
    
    Args:
        urls (list): List of article URLs to index
        force_reindex (bool): Whether to force reindexing of existing articles
        
    Returns:
        list: List of indexing results with status, message, and metadata for each URL
    """
    try:
        if not urls or not isinstance(urls, list):
            logging.warning("⚠️ No URLs provided for batch Philippine news indexing")
            return []
            
        # Get the Philippine search index instance
        philippine_search_index = get_philippine_search_index()
        
        if philippine_search_index is None:
            logging.warning("⚠️ Philippine search index not available for batch indexing")
            return []
        
        # Process URLs in batch similar to philippine_news_search_routes.py
        def process_batch():
            results = []
            for url in urls:
                try:
                    # Validate URL format
                    from urllib.parse import urlparse
                    parsed_url = urlparse(url.strip())
                    if not parsed_url.scheme or not parsed_url.netloc:
                        results.append({
                            'url': url,
                            'status': 'error',
                            'message': 'Invalid URL format',
                            'success': False
                        })
                        continue
                    
                    # Index the article using direct method call (same as batch-index-philippine-articles)
                    result = philippine_search_index.index_article(url.strip(), force_reindex)
                    
                    # Convert result to consistent format
                    results.append({
                        'url': url,
                        'status': result['status'],
                        'message': result.get('message', ''),
                        'article_id': result.get('article_id'),
                        'relevance_score': result.get('relevance_score', 0),
                        'success': result['status'] in ['success', 'already_indexed', 'skipped']
                    })
                    
                except Exception as e:
                    results.append({
                        'url': url,
                        'status': 'error',
                        'message': str(e),
                        'success': False
                    })
            
            return results
        
        # Process synchronously for reliability (can be made async later)
        start_time = time.time()
        batch_results = process_batch()
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Calculate summary statistics
        success_count = len([r for r in batch_results if r['status'] == 'success'])
        skipped_count = len([r for r in batch_results if r['status'] == 'skipped'])
        error_count = len([r for r in batch_results if r['status'] == 'error'])
        already_indexed_count = len([r for r in batch_results if r['status'] == 'already_indexed'])
        
        logging.info(f"✅ Batch Philippine indexing completed in {duration_ms:.1f}ms:")
        logging.info(f"   📊 Successfully indexed: {success_count}")
        logging.info(f"   ℹ️ Already indexed: {already_indexed_count}")
        logging.info(f"   ⏭️ Skipped (not relevant): {skipped_count}")
        logging.info(f"   ❌ Errors: {error_count}")
        
        return batch_results
        
    except Exception as e:
        logging.warning(f"⚠️ Error in batch Philippine news indexing: {str(e)}")
        return []

def index_philippine_news_article(url, force_reindex=False):
    """Index a verified news article into the Philippine news search system using direct method calls"""
    try:
        if not url:
            logging.warning("⚠️ No URL provided for Philippine news indexing")
            return False
            
        # Get the Philippine search index instance
        philippine_search_index = get_philippine_search_index()
        
        if philippine_search_index is None:
            logging.warning("⚠️ Philippine search index not available for indexing")
            return False
        
        # Validate URL format (same as in philippine_news_search_routes.py)
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logging.warning(f"⚠️ Invalid URL format for Philippine indexing: {url}")
            return False
        
        # Index the article using direct method call
        start_time = time.time()
        
        result = philippine_search_index.index_article(url, force_reindex)
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        if result['status'] == 'success':
            logging.info(f"✅ Article indexed successfully in Philippine news system: {url} (took {duration_ms:.1f}ms)")
            logging.info(f"📊 Philippine indexing details: ID={result.get('article_id')}, Score={result.get('relevance_score')}")
            return True
        elif result['status'] == 'already_indexed':
            logging.info(f"ℹ️ Article already indexed in Philippine news system: {url}")
            return True
        elif result['status'] == 'skipped':
            logging.info(f"⏭️ Article skipped - not relevant to Philippine news: {url}")
            return True
        else:
            logging.warning(f"⚠️ Philippine news indexing failed for {url}: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        logging.warning(f"⚠️ Error indexing article into Philippine news system: {str(e)}")
        return False
    # Don't raise exceptions - indexing is optional and shouldn't break verification

# Auto-fetch functionality (would run in background)
def start_auto_fetch():
    """Start background auto-fetch process"""
    def auto_fetch_worker():
        while True:
            try:
                conn = sqlite3.connect(DATABASE_FILE)
                cursor = conn.cursor()
                
                # Get websites that need fetching
                cursor.execute('''
                    SELECT id, url, name, fetch_interval, last_fetch
                    FROM tracked_websites
                    WHERE status = 'active'
                ''')
                
                websites = cursor.fetchall()
                current_time = datetime.now()
                
                for website_id, url, name, interval, last_fetch in websites:
                    try:
                        if last_fetch:
                            last_fetch_time = datetime.fromisoformat(last_fetch)
                            if current_time - last_fetch_time < timedelta(minutes=interval):
                                continue
                        
                        # Fetch articles using crawler
                        articles = fetch_articles_from_crawler(url, name, website_id)
                        
                        # Save new articles
                        for article in articles:
                            try:
                                cursor.execute('''
                                    INSERT INTO article_queue (url, title, description, content, site_name, website_id, user_session, confidence, is_news_prediction, probability_news)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    article['url'], article['title'], article['description'],
                                    article['content'], article['site_name'], article['website_id'], 'default',
                                    article.get('confidence', 0.0), article.get('is_news_prediction', True), 
                                    article.get('probability_news', 0.0)
                                ))
                            except sqlite3.IntegrityError:
                                pass  # Article already exists
                        
                        # Update last fetch time
                        cursor.execute('''
                            UPDATE tracked_websites 
                            SET last_fetch = CURRENT_TIMESTAMP 
                            WHERE id = ?
                        ''', (website_id,))
                        
                        conn.commit()
                        
                    except Exception as e:
                        logging.error(f"Auto-fetch error for {url}: {str(e)}")
                
                conn.close()
                
            except Exception as e:
                logging.error(f"Auto-fetch worker error: {str(e)}")
            
            time.sleep(60)  # Check every minute
    
    # Start background thread
    thread = threading.Thread(target=auto_fetch_worker, daemon=True)
    thread.start()

# Uncomment to enable auto-fetch
# start_auto_fetch()
