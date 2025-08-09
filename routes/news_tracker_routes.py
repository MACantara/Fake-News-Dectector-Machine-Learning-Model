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
        
        # If article is verified as news, index it in Philippine news search system
        philippine_indexing_success = False
        if is_news:
            philippine_indexing_success = index_philippine_news_article(url)
        
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
    except Exception as e:
        logging.error(f"Error verifying article: {str(e)}")
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
    """Fetch articles from a website using the news crawler endpoint"""
    articles = []
    
    try:
        # Prepare request to crawler endpoint
        crawler_data = {
            'website_url': url,
            'max_articles': 15,  # Get more articles than before
            'enable_filtering': True,  # Use intelligent filtering
            'confidence_threshold': 0.6  # Only get likely news articles
        }
        
        # Make request to crawler endpoint
        # Use 127.0.0.1 to avoid potential localhost resolution issues
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
                
                logging.info(f"✅ Crawler found {len(articles)} articles from {url}")
                logging.info(f"📊 Crawler stats: {crawler_result.get('crawler_stats', {})}")
            else:
                logging.warning(f"⚠️ Crawler returned no articles for {url}")
        else:
            logging.error(f"❌ Crawler request failed for {url}: {response.status_code}")
            # Fallback to empty list - could implement basic scraping here if needed
            
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Network error calling crawler for {url}: {str(e)}")
    except Exception as e:
        logging.error(f"❌ Error fetching from crawler for {url}: {str(e)}")
    
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
