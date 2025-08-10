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

# Optimized imports - grouped and minimal
from flask import Blueprint, render_template, request, jsonify, session
import sqlite3
import requests
from datetime import datetime, timedelta
from urllib.parse import urlparse
import time
import threading
import uuid
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from modules.url_news_classifier import URLNewsClassifier
from routes.url_classifier_routes import get_url_classifier
from routes.philippine_news_search_routes import get_philippine_search_index
from routes.news_crawler_routes import get_news_crawler

# Performance optimization: Connection pooling for database operations
import queue
from contextlib import contextmanager

news_tracker_bp = Blueprint('news_tracker', __name__)

# Configuration
DATABASE_FILE = 'news_tracker.db'

# Performance optimization: Database connection pool
class DatabasePool:
    def __init__(self, db_path, pool_size=5):
        self.db_path = db_path
        self.pool = queue.Queue(maxsize=pool_size)
        self._init_pool(pool_size)
    
    def _init_pool(self, pool_size):
        for _ in range(pool_size):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute('PRAGMA journal_mode=WAL')  # Performance optimization
            conn.execute('PRAGMA synchronous=NORMAL')  # Performance optimization
            conn.execute('PRAGMA cache_size=10000')  # Increase cache size
            conn.execute('PRAGMA temp_store=MEMORY')  # Use memory for temp storage
            self.pool.put(conn)
    
    @contextmanager
    def get_connection(self):
        conn = self.pool.get()
        try:
            yield conn
        finally:
            self.pool.put(conn)

# Initialize connection pool
db_pool = DatabasePool(DATABASE_FILE)

# Initialize database
def init_news_tracker_db():
    """Initialize the news tracker database with optimized settings"""
    with db_pool.get_connection() as conn:
        cursor = conn.cursor()
        
        # Create tables with optimized schema
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
        
        # Create optimized indexes for performance
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_url ON tracked_websites(url)',
            'CREATE INDEX IF NOT EXISTS idx_user_session ON tracked_websites(user_session)',
            'CREATE INDEX IF NOT EXISTS idx_status ON tracked_websites(status)',
            'CREATE INDEX IF NOT EXISTS idx_article_url ON article_queue(url)',
            'CREATE INDEX IF NOT EXISTS idx_article_website ON article_queue(website_id)',
            'CREATE INDEX IF NOT EXISTS idx_article_session ON article_queue(user_session)',
            'CREATE INDEX IF NOT EXISTS idx_article_verified ON article_queue(verified)',
            'CREATE INDEX IF NOT EXISTS idx_article_found_at ON article_queue(found_at)',
            'CREATE INDEX IF NOT EXISTS idx_fetch_website ON fetch_logs(website_id)',
            'CREATE INDEX IF NOT EXISTS idx_fetch_time ON fetch_logs(fetch_time)'
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
    
    # Migrate existing databases to add new crawler metadata columns
    migrate_database_schema()

def migrate_database_schema():
    """Add new columns to existing database if they don't exist"""
    try:
        with db_pool.get_connection() as conn:
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
            
            conn.commit()
        
    except Exception as e:
        print(f"❌ Database migration error: {str(e)}")

# Initialize database on import
init_news_tracker_db()

@news_tracker_bp.route('/news-tracker')
def news_tracker():
    """Render the news tracker page"""
    # Ensure user has a session (optimized)
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('news_tracker.html')

@news_tracker_bp.route('/api/news-tracker/add-website', methods=['POST'])
def add_website():
    """Add a website to track with optimized validation"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
        
        # Optimized URL validation
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return jsonify({'success': False, 'error': 'Invalid URL format'}), 400
        except Exception:
            return jsonify({'success': False, 'error': 'Invalid URL format'}), 400
        
        # Auto-generate name from URL
        name = get_domain_display_name(extract_root_domain(url))
        
        user_session = session.get('session_id', 'default')
        
        with db_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO tracked_websites (url, name, user_session)
                    VALUES (?, ?, ?)
                ''', (url, name, user_session))
                
                website_id = cursor.lastrowid
                conn.commit()
                
                return jsonify({
                    'success': True,
                    'id': website_id,
                    'message': f'Successfully added {name} to tracking list'
                })
                
            except sqlite3.IntegrityError:
                return jsonify({'success': False, 'error': 'This website is already being tracked'}), 409
            
    except Exception as e:
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@news_tracker_bp.route('/api/news-tracker/remove-website/<int:website_id>', methods=['DELETE'])
def remove_website(website_id):
    """Remove a website from tracking with optimized batch deletion"""
    try:
        user_session = session.get('session_id', 'default')
        
        with db_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if website exists and belongs to user
            cursor.execute('''
                SELECT id FROM tracked_websites 
                WHERE id = ? AND user_session = ?
            ''', (website_id, user_session))
            
            if not cursor.fetchone():
                return jsonify({'success': False, 'error': 'Website not found'}), 404
            
            # Optimized batch deletion using single transaction
            delete_operations = [
                ('DELETE FROM article_queue WHERE website_id = ?', (website_id,)),
                ('DELETE FROM fetch_logs WHERE website_id = ?', (website_id,)),
                ('DELETE FROM tracked_websites WHERE id = ?', (website_id,))
            ]
            
            for sql, params in delete_operations:
                cursor.execute(sql, params)
            
            conn.commit()
            
            return jsonify({'success': True, 'message': 'Website removed successfully'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@news_tracker_bp.route('/api/news-tracker/fetch-articles', methods=['POST'])
def fetch_articles():
    """Fetch articles from all tracked websites with maximum optimization"""
    try:
        user_session = session.get('session_id', 'default')
        start_time = time.time()
        
        # Phase 1: Single read-only query to get all tracked websites
        with db_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, url, name FROM tracked_websites 
                WHERE status = 'active' AND user_session = ?
            ''', (user_session,))
            websites = cursor.fetchall()
        
        if not websites:
            return jsonify({'success': True, 'articles': [], 'message': 'No websites being tracked'})
        
        # Phase 2: Parallel article collection with optimized data structures
        all_articles = []
        batch_operations = {
            'website_updates': [],
            'fetch_logs': [],
            'article_inserts': []
        }
        
        # Process all websites and collect data efficiently
        for website_id, url, name in websites:
            try:
                articles = fetch_articles_from_crawler(url, name, website_id)
                all_articles.extend(articles)
                
                # Prepare optimized batch operations
                batch_operations['website_updates'].append((website_id,))
                batch_operations['fetch_logs'].append((website_id, len(articles), True, None))
                
            except Exception as e:
                batch_operations['fetch_logs'].append((website_id, 0, False, str(e)))
        
        # Phase 3: Optimize article data preparation
        valid_articles = []
        if all_articles:
            # Pre-allocate list for better memory performance
            batch_operations['article_inserts'] = []
            batch_operations['article_inserts'].extend([
                (
                    article['url'], article['title'], article['description'],
                    article['content'], article['site_name'], article['website_id'],
                    user_session, article.get('confidence', 0.0), 
                    article.get('is_news_prediction', True), 
                    article.get('probability_news', 0.0)
                )
                for article in all_articles
            ])
            valid_articles = all_articles
        
        # Phase 4: Single atomic database transaction with optimized batch inserts
        new_articles = []
        with db_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Execute all operations in single optimized transaction
                if batch_operations['website_updates']:
                    cursor.executemany('''
                        UPDATE tracked_websites 
                        SET last_fetch = CURRENT_TIMESTAMP 
                        WHERE id = ?
                    ''', batch_operations['website_updates'])
                
                if batch_operations['fetch_logs']:
                    cursor.executemany('''
                        INSERT INTO fetch_logs (website_id, articles_found, success, error_message)
                        VALUES (?, ?, ?, ?)
                    ''', batch_operations['fetch_logs'])
                
                if batch_operations['article_inserts']:
                    cursor.executemany('''
                        INSERT OR IGNORE INTO article_queue 
                        (url, title, description, content, site_name, website_id, user_session, confidence, is_news_prediction, probability_news)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', batch_operations['article_inserts'])
                    
                    # Optimized retrieval of newly inserted articles
                    if valid_articles:
                        urls_to_check = [article['url'] for article in valid_articles]
                        placeholders = ','.join(['?' for _ in urls_to_check])
                        cursor.execute(f'''
                            SELECT id, url FROM article_queue 
                            WHERE url IN ({placeholders}) AND user_session = ?
                        ''', urls_to_check + [user_session])
                        
                        # Create efficient lookup dict
                        url_to_id = {row[1]: row[0] for row in cursor.fetchall()}
                        
                        # Build new articles list efficiently
                        new_articles = [
                            {'id': url_to_id[article['url']], **article}
                            for article in valid_articles
                            if article['url'] in url_to_id
                        ]
                
                conn.commit()
                
            except Exception as e:
                conn.rollback()
                return jsonify({'success': False, 'error': f'Database transaction failed: {str(e)}'}), 500
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        return jsonify({
            'success': True,
            'articles': new_articles,
            'message': f'Found {len(new_articles)} new articles in {duration_ms:.1f}ms',
            'performance_stats': {
                'total_duration_ms': round(duration_ms, 1),
                'websites_processed': len(websites),
                'articles_collected': len(all_articles),
                'new_articles_inserted': len(new_articles),
                'avg_time_per_website': round(duration_ms / len(websites), 1) if websites else 0
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

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
        return jsonify({'success': False, 'error': 'Internal server error'})

@news_tracker_bp.route('/api/news-tracker/batch-verify-articles', methods=['POST'])
def batch_verify_articles():
    """Verify multiple articles at once for efficiency"""
    try:
        data = request.get_json()
        articles = data.get('articles', [])
        
        if not articles or not isinstance(articles, list):
            return jsonify({'success': False, 'error': 'Articles array is required'})
        
        if len(articles) > 100:  # Max 100 articles per batch
            return jsonify({'success': False, 'error': 'Maximum 100 articles allowed per batch'})
        
        user_session = session.get('session_id', 'default')
        
        # Phase 1: Data Collection and Validation (NO database writes)
        
        results = []
        news_articles_for_indexing = []
        batch_operations = {
            'article_updates': [],    # verification updates
            'feedback_data': [],      # URL classifier feedback
            'philippine_urls': []     # URLs for Philippine indexing
        }
        
        # Get existing article data first (single read operation)
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        valid_articles = []
        article_data_cache = {}
        
        # Validate and fetch all article data upfront
        for i, article in enumerate(articles):
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
                
                # Get article data including crawler metadata
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
                
                # Cache article data and prepare for batch operations
                article_data_cache[article_id] = article_data
                valid_articles.append({
                    'articleId': article_id,
                    'isNews': is_news,
                    'url': url,
                    'article_data': article_data
                })
                
            except Exception as e:
                results.append({
                    'articleId': article.get('articleId'),
                    'success': False,
                    'error': str(e)
                })
        
        conn.close()
        
        # Phase 2: Prepare All Database Operations (NO database writes yet)
        
        for i, article in enumerate(valid_articles):
            article_id = article['articleId']
            is_news = article['isNews']
            url = article['url']
            article_data = article['article_data']
            
            # Prepare database update operation
            batch_operations['article_updates'].append((
                is_news,  # is_news
                article_id,  # id
                user_session  # user_session
            ))
            
            # Prepare URL classifier feedback data
            batch_operations['feedback_data'].append({
                'url': url or article_data[0],
                'article_data': article_data,
                'user_verification': is_news
            })
            
            # Collect news articles for batch Philippine indexing
            if is_news:
                final_url = url or article_data[0]
                news_articles_for_indexing.append(final_url)
                batch_operations['philippine_urls'].append(final_url)
            else:
                pass
            
            # Prepare success result
            results.append({
                'articleId': article_id,
                'success': True,
                'message': f'Article marked as {"news" if is_news else "not news"}',
                'url_classifier_feedback_sent': True,
                'philippine_indexing_queued': is_news
            })
        
        # Phase 3: Single Atomic Database Transaction
        
        if batch_operations['article_updates']:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            
            try:
                # Execute batch article verification updates
                cursor.executemany('''
                    UPDATE article_queue 
                    SET verified = TRUE, is_news = ?, verified_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND user_session = ?
                ''', batch_operations['article_updates'])
                
                
                # Commit database transaction
                conn.commit()
                
            except Exception as e:
                conn.rollback()
                
                # Update all results to show failure
                for result in results:
                    if result.get('success'):
                        result['success'] = False
                        result['error'] = f'Database batch update failed: {str(e)}'
                
                conn.close()
                return jsonify({'success': False, 'error': f'Database transaction failed: {str(e)}'})
            
            finally:
                conn.close()
        
        # Phase 4: Post-Database Operations (URL Classifier Feedback & Philippine Indexing)
        
        # Send URL classifier feedback (external system - done after DB success)
        for feedback_item in batch_operations['feedback_data']:
            try:
                send_url_classifier_feedback(
                    url=feedback_item['url'],
                    article_data=feedback_item['article_data'],
                    user_verification=feedback_item['user_verification']
                )
            except Exception as e:
                print(f"⚠️ Failed to send URL classifier feedback: {str(e)}")
        
        # Log summary statistics
        total_articles = len(articles)
        news_articles_count = len(news_articles_for_indexing)
        non_news_count = total_articles - news_articles_count
        
        
        # Batch index news articles into Philippine news system using optimized batch method
        philippine_indexing_results = []
        if news_articles_for_indexing:
            philippine_indexing_results = batch_index_philippine_news_articles(news_articles_for_indexing)
        else:
            print("ℹ️ DEBUG: No news articles to index into Philippine system")
        
        # Calculate summary statistics
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
            'philippine_indexing_results': philippine_indexing_results,
            'batch_summary': {
                'valid_articles_processed': len(valid_articles),
                'database_operations': {
                    'article_updates': len(batch_operations['article_updates']),
                    'url_classifier_feedback': len(batch_operations['feedback_data']),
                    'philippine_urls_queued': len(batch_operations['philippine_urls'])
                }
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': 'Internal server error'})

def calculate_comprehensive_metrics(user_session='default'):
    """Calculate comprehensive prediction evaluation metrics"""
    try:
        with db_pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all verified articles with predictions
            cursor.execute('''
                SELECT 
                    is_news_prediction,
                    is_news as actual_result,
                    confidence,
                    probability_news,
                    verified_at
                FROM article_queue 
                WHERE verified = TRUE 
                AND user_session = ?
                AND is_news_prediction IS NOT NULL
                AND is_news IS NOT NULL
            ''', (user_session,))
            
            results = cursor.fetchall()
            
            if len(results) < 2:
                return {
                    'status': 'insufficient_data',
                    'message': 'Need at least 2 verified articles for meaningful metrics',
                    'total_verified': len(results)
                }
            
            # Prepare data for metrics calculation
            predictions = [bool(row[0]) for row in results]  # Predicted labels
            actuals = [bool(row[1]) for row in results]      # Actual labels
            confidences = [float(row[2]) if row[2] is not None else 0.5 for row in results]
            probabilities = [float(row[3]) if row[3] is not None else 0.5 for row in results]
            
            # Convert boolean to int for sklearn
            y_true = [1 if actual else 0 for actual in actuals]
            y_pred = [1 if pred else 0 for pred in predictions]
            y_prob = probabilities
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate basic metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate advanced metrics
            try:
                mcc = matthews_corrcoef(y_true, y_pred)
            except:
                mcc = 0
            
            try:
                auc_roc = roc_auc_score(y_true, y_prob)
            except:
                auc_roc = 0.5
            
            # Calculate rates
            tpr = recall  # True Positive Rate (Sensitivity)
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
            
            # Calculate prediction calibration (how well confidence matches accuracy)
            confidence_bins = np.linspace(0, 1, 11)  # 10 bins
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for i in range(len(confidence_bins) - 1):
                lower, upper = confidence_bins[i], confidence_bins[i + 1]
                mask = [(lower <= conf < upper) for conf in confidences]
                
                if any(mask):
                    bin_predictions = [y_pred[j] for j, m in enumerate(mask) if m]
                    bin_actuals = [y_true[j] for j, m in enumerate(mask) if m]
                    bin_confs = [confidences[j] for j, m in enumerate(mask) if m]
                    
                    bin_accuracy = sum([p == a for p, a in zip(bin_predictions, bin_actuals)]) / len(bin_predictions)
                    bin_avg_conf = sum(bin_confs) / len(bin_confs)
                    
                    bin_accuracies.append(bin_accuracy)
                    bin_confidences.append(bin_avg_conf)
                    bin_counts.append(len(bin_predictions))
                else:
                    bin_accuracies.append(0)
                    bin_confidences.append((lower + upper) / 2)
                    bin_counts.append(0)
            
            # Calculate Expected Calibration Error (ECE)
            total_samples = len(results)
            ece = sum([
                (count / total_samples) * abs(acc - conf) 
                for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
                if count > 0
            ]) if total_samples > 0 else 0
            
            # Calculate prediction consistency over time
            if len(results) >= 10:
                # Group by week and calculate weekly accuracy
                weekly_accuracies = []
                current_week_results = []
                current_week = None
                
                for i, row in enumerate(results):
                    verified_at = row[4]
                    if verified_at:
                        week = datetime.fromisoformat(verified_at).isocalendar()[1]
                        
                        if current_week is None:
                            current_week = week
                        
                        if week == current_week:
                            current_week_results.append((y_pred[i], y_true[i]))
                        else:
                            if current_week_results:
                                week_accuracy = sum([p == a for p, a in current_week_results]) / len(current_week_results)
                                weekly_accuracies.append(week_accuracy)
                            current_week_results = [(y_pred[i], y_true[i])]
                            current_week = week
                
                # Add last week
                if current_week_results:
                    week_accuracy = sum([p == a for p, a in current_week_results]) / len(current_week_results)
                    weekly_accuracies.append(week_accuracy)
                
                consistency_variance = np.var(weekly_accuracies) if len(weekly_accuracies) > 1 else 0
            else:
                weekly_accuracies = []
                consistency_variance = 0
            
            return {
                'status': 'success',
                'total_verified_articles': len(results),
                'confusion_matrix': {
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn)
                },
                'basic_metrics': {
                    'accuracy': round(accuracy, 4),
                    'precision': round(precision, 4),
                    'recall': round(recall, 4),
                    'f1_score': round(f1, 4)
                },
                'advanced_metrics': {
                    'matthews_correlation_coefficient': round(mcc, 4),
                    'auc_roc': round(auc_roc, 4),
                    'true_positive_rate': round(tpr, 4),
                    'true_negative_rate': round(tnr, 4),
                    'false_positive_rate': round(fpr, 4),
                    'false_negative_rate': round(fnr, 4)
                },
                'calibration_metrics': {
                    'expected_calibration_error': round(ece, 4),
                    'calibration_bins': [
                        {
                            'confidence_range': f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}",
                            'avg_confidence': round(bin_confidences[i], 3),
                            'accuracy': round(bin_accuracies[i], 3),
                            'count': bin_counts[i]
                        }
                        for i in range(len(bin_accuracies))
                        if bin_counts[i] > 0
                    ]
                },
                'temporal_consistency': {
                    'weekly_accuracies': [round(acc, 3) for acc in weekly_accuracies],
                    'consistency_variance': round(consistency_variance, 4),
                    'is_consistent': consistency_variance < 0.05  # Low variance indicates consistency
                },
                'prediction_quality': {
                    'high_confidence_correct': sum([
                        1 for i, (pred, actual, conf) in enumerate(zip(y_pred, y_true, confidences))
                        if conf > 0.8 and pred == actual
                    ]),
                    'high_confidence_total': sum([1 for conf in confidences if conf > 0.8]),
                    'low_confidence_correct': sum([
                        1 for i, (pred, actual, conf) in enumerate(zip(y_pred, y_true, confidences))
                        if conf < 0.6 and pred == actual
                    ]),
                    'low_confidence_total': sum([1 for conf in confidences if conf < 0.6])
                }
            }
    
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Error calculating metrics: {str(e)}'
        }

@news_tracker_bp.route('/api/news-tracker/prediction-metrics')
def get_prediction_metrics():
    """Get comprehensive prediction evaluation metrics"""
    try:
        user_session = session.get('session_id', 'default')
        metrics = calculate_comprehensive_metrics(user_session)
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@news_tracker_bp.route('/api/news-tracker/get-data')
def get_tracker_data():
    """Get all tracker data for the frontend with optimized queries"""
    try:
        user_session = session.get('session_id', 'default')
        
        with db_pool.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Optimized single query to get websites with aggregated stats
            cursor.execute('''
                SELECT w.*, 
                       COALESCE(COUNT(a.id), 0) as article_count,
                       MAX(a.found_at) as last_article,
                       COALESCE(SUM(CASE WHEN a.verified = 1 THEN 1 ELSE 0 END), 0) as verified_count
                FROM tracked_websites w
                LEFT JOIN article_queue a ON w.id = a.website_id
                WHERE w.user_session = ?
                GROUP BY w.id
                ORDER BY w.added_at DESC
            ''', (user_session,))
            
            # Efficiently process website data with domain extraction
            websites = []
            for row in cursor.fetchall():
                website = dict(row)
                # Only process timestamps if they exist (optimize for null values)
                if website['added_at']:
                    website['addedAt'] = website['added_at']
                if website['last_fetch']:
                    website['lastFetch'] = website['last_fetch']
                if website['last_article']:
                    website['lastArticle'] = website['last_article']
                
                # Add domain information for grouping
                website['rootDomain'] = extract_root_domain(website['url'])
                website['displayName'] = get_domain_display_name(website['rootDomain'])
                
                websites.append(website)
            
            # Optimized query for articles with left join
            cursor.execute('''
                SELECT a.*, w.name as site_name
                FROM article_queue a
                LEFT JOIN tracked_websites w ON a.website_id = w.id
                WHERE a.user_session = ?
                ORDER BY a.found_at DESC
                LIMIT 1000
            ''', (user_session,))
            
            # Efficiently process article data
            articles = []
            for row in cursor.fetchall():
                article = dict(row)
                # Optimize timestamp conversion
                if article['found_at']:
                    article['foundAt'] = article['found_at']
                if article['verified_at']:
                    article['verifiedAt'] = article['verified_at']
                
                # Optimize boolean conversion (use bitwise operations)
                article['verified'] = bool(article['verified'])
                if article['is_news'] is not None:
                    article['isNews'] = bool(article['is_news'])
                
                articles.append(article)
        
        # Calculate basic prediction metrics for inclusion in response
        basic_metrics = calculate_comprehensive_metrics(user_session)
        
        # Group websites by domain for enhanced organization
        domain_groups = group_websites_by_domain(websites)
        
        return jsonify({
            'success': True,
            'websites': websites,
            'domain_groups': domain_groups,
            'articles': articles,
            'stats': {
                'total_websites': len(websites),
                'total_domains': len(domain_groups),
                'total_articles': len(articles),
                'verified_articles': sum(1 for a in articles if a['verified'])
            },
            'prediction_metrics': basic_metrics if basic_metrics.get('status') == 'success' else None
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

def extract_root_domain(url):
    """Extract root domain from URL for grouping purposes with improved ccTLD handling"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        hostname = parsed.hostname.replace('www.', '') if parsed.hostname else 'unknown'
        
        # Handle special domain mappings first
        special_domains = get_special_domain_mappings()
        if hostname in special_domains:
            return special_domains[hostname]
        
        # Check if any special domain is a subdomain of this domain
        for key, value in special_domains.items():
            if hostname.endswith('.' + key) or hostname == key:
                return value
        
        parts = hostname.split('.')
        
        # Handle country code TLDs (ccTLDs) and second-level domains
        if is_country_code_tld(parts):
            return extract_domain_with_cctld(parts)
        
        # Handle standard domains
        if len(parts) >= 2:
            # For domains like news.bbc.com, check if subdomain should be preserved
            if len(parts) >= 3 and should_preserve_subdomain(parts):
                return '.'.join(parts[-3:])
            return '.'.join(parts[-2:])
        
        return hostname
    except Exception:
        return 'unknown'

def get_special_domain_mappings():
    """Map specific domains to their canonical grouping domain"""
    return {
        'mb.com.ph': 'manilabulletin.com.ph',
        'businessmirror.com.ph': 'businessmirror.com.ph',
        'news.abs-cbn.com': 'abs-cbn.com',
        'news.gma.network': 'gmanetwork.com',
        'cnnphilippines.com': 'cnn.com',
        'news.yahoo.com': 'yahoo.com',
        'abscbn.com': 'abs-cbn.com',
        'gmanews.tv': 'gmanetwork.com',
        'manilastandard.net': 'manilastandardtoday.com',
        'tribune.net.ph': 'tribuneonline.org',
        'bworldonline.com': 'businessworld.com.ph',
        'pna.gov.ph': 'pna.gov.ph',
        'philstar.com': 'philstar.com'
    }

def is_country_code_tld(parts):
    """Check if domain uses a country code TLD"""
    if len(parts) < 3:
        return False
    
    cctlds = [
        'com.ph', 'net.ph', 'org.ph', 'gov.ph', 'edu.ph',
        'co.uk', 'org.uk', 'ac.uk', 'gov.uk',
        'com.au', 'net.au', 'org.au', 'gov.au', 'edu.au',
        'co.jp', 'or.jp', 'ne.jp', 'go.jp', 'ac.jp',
        'com.sg', 'net.sg', 'org.sg', 'gov.sg', 'edu.sg',
        'com.my', 'net.my', 'org.my', 'gov.my', 'edu.my'
    ]
    
    last_two_parts = '.'.join(parts[-2:])
    return last_two_parts in cctlds

def extract_domain_with_cctld(parts):
    """Extract domain with country code TLD properly"""
    if len(parts) == 3:
        # domain.com.ph
        return '.'.join(parts)
    elif len(parts) >= 4:
        # subdomain.domain.com.ph
        # Check if subdomain should be preserved
        subdomain = parts[0]
        if should_preserve_subdomain_for_cctld(subdomain):
            return '.'.join(parts[-4:])
        return '.'.join(parts[-3:])
    
    return '.'.join(parts)

def should_preserve_subdomain(parts):
    """Determine if subdomain should be preserved for standard domains"""
    if len(parts) < 3:
        return False
    
    subdomain = parts[0]
    preserve_subdomains = [
        'news', 'www', 'm', 'mobile', 'edition', 
        'international', 'cnn', 'bbc', 'sports'
    ]
    
    # Don't preserve common news subdomains unless it's a major organization
    if subdomain in preserve_subdomains:
        domain = '.'.join(parts[-2:])
        major_orgs = [
            'abs-cbn.com', 'gmanetwork.com', 'rappler.com',
            'cnn.com', 'bbc.com', 'reuters.com', 'nytimes.com'
        ]
        return domain in major_orgs
    
    return False

def should_preserve_subdomain_for_cctld(subdomain):
    """For ccTLD domains, be more conservative about preserving subdomains"""
    preserve_subdomains = ['news', 'www']
    return subdomain in preserve_subdomains

def get_domain_display_name(domain):
    """Convert domain to friendly display name"""
    domain_map = {
        'cnn.com': 'CNN',
        'bbc.com': 'BBC',
        'reuters.com': 'Reuters',
        'nytimes.com': 'New York Times',
        'washingtonpost.com': 'Washington Post',
        'theguardian.com': 'The Guardian',
        'abc.net.au': 'ABC News',
        'yahoo.com': 'Yahoo News',
        'foxnews.com': 'Fox News',
        'nbcnews.com': 'NBC News',
        'cbsnews.com': 'CBS News',
        'npr.org': 'NPR',
        'apnews.com': 'Associated Press',
        'usatoday.com': 'USA Today',
        'wsj.com': 'Wall Street Journal',
        'bloomberg.com': 'Bloomberg',
        'time.com': 'Time',
        'newsweek.com': 'Newsweek',
        'huffpost.com': 'HuffPost',
        'politico.com': 'Politico',
        
        # Philippine news organizations
        'rappler.com': 'Rappler',
        'abs-cbn.com': 'ABS-CBN',
        'gmanetwork.com': 'GMA News',
        'inquirer.net': 'Philippine Daily Inquirer',
        'philstar.com': 'Philippine Star',
        'manilabulletin.com.ph': 'Manila Bulletin',
        'businessworld.com.ph': 'BusinessWorld',
        'businessmirror.com.ph': 'BusinessMirror',
        'manilastandard.net': 'Manila Standard',
        'manilastandardtoday.com': 'Manila Standard Today',
        'tribune.net.ph': 'Tribune',
        'tribuneonline.org': 'Tribune',
        'sunstar.com.ph': 'SunStar',
        'pna.gov.ph': 'Philippine News Agency',
        'malaya.com.ph': 'Malaya Business Insight',
        'remate.ph': 'Remate',
        'tempo.com.ph': 'Tempo',
        'journal.com.ph': 'The Journal',
        'manilatimes.net': 'Manila Times',
        'journal.ph': 'The Journal Online'
    }
    
    if domain in domain_map:
        return domain_map[domain]
    
    # Handle special patterns for Philippine domains
    if domain.endswith('.com.ph') or domain.endswith('.net.ph') or domain.endswith('.org.ph'):
        base_name = domain.split('.')[0]
        return ' '.join(word.capitalize() for word in base_name.replace('-', ' ').replace('_', ' ').split())
    
    # Generate display name from domain
    name = domain.replace('.com', '').replace('.org', '').replace('.net', '').replace('.ph', '').replace('.co.uk', '').replace('.com.au', '')
    return ' '.join(word.capitalize() for word in name.replace('-', ' ').replace('_', ' ').replace('.', ' ').split())

def group_websites_by_domain(websites):
    """Group websites by their root domain"""
    groups = {}
    
    for website in websites:
        root_domain = website.get('rootDomain', 'unknown')
        
        if root_domain not in groups:
            groups[root_domain] = {
                'domain': root_domain,
                'display_name': website.get('displayName', root_domain),
                'websites': [],
                'total_articles': 0,
                'active_count': 0,
                'total_count': 0
            }
        
        group = groups[root_domain]
        group['websites'].append(website)
        group['total_articles'] += website.get('article_count', 0)
        group['total_count'] += 1
        
        if website.get('status') == 'active':
            group['active_count'] += 1
    
    # Sort websites within each group by article count
    for group in groups.values():
        group['websites'].sort(key=lambda x: x.get('article_count', 0), reverse=True)
    
    # Sort groups by total articles (descending) then by domain name
    sorted_groups = sorted(groups.values(), key=lambda x: (-x['total_articles'], x['domain']))
    
    return sorted_groups

def fetch_articles_from_crawler(url, site_name, website_id):
    """Fetch articles from a website using direct crawler method calls for better performance"""
    articles = []
    
    try:
        # Get the shared crawler instance (much faster than HTTP)
        news_crawler = get_news_crawler()
        
        if news_crawler is None:
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
            
        else:
            pass
            
    except Exception as e:
        
        # Fallback to HTTP request if direct method fails
        try:
            return fetch_articles_from_crawler_http(url, site_name, website_id)
        except Exception as fallback_error:
            print(f"❌ Fallback error: {str(fallback_error)}")
    
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
                
            else:
                pass
        else:
            pass
            
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP request error: {str(e)}")
    except Exception as e:
        print(f"❌ HTTP error: {str(e)}")
    
    return articles

def send_url_classifier_feedback(url, article_data, user_verification):
    """Send feedback to URL classifier to improve the model using direct method calls"""
    try:
        if not url or article_data is None:
            return
        
        # Extract crawler prediction data
        article_url = article_data[0] if article_data else url
        confidence = article_data[1] if len(article_data) > 1 else None
        predicted_is_news = article_data[2] if len(article_data) > 2 else True  # Default to True if no prediction
        probability_news = article_data[3] if len(article_data) > 3 else None
        
        # Get the shared URL classifier instance
        url_classifier = get_url_classifier()
        
        if url_classifier is None:
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
            
            # Get current model stats
            stats = url_classifier.get_model_stats()
            
            # Check if retraining should be triggered (every 100 feedback entries)
            feedback_count = stats.get('total_feedback', 0)
            if feedback_count > 0 and feedback_count % 100 == 0:
                print(f"Triggering URL classifier retraining with {feedback_count} feedback entries...")
        else:
            pass
            
    except Exception as e:
        print(f"⚠️ URL classifier feedback failed: {str(e)}")
        
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
            return []
        
        print(f"🚀 Batch Philippine indexing: {len(urls)} URLs")
        
        # Check for duplicate URLs
        unique_urls = list(set(urls))
        urls_to_process = unique_urls
            
        # Get the Philippine search index instance
        philippine_search_index = get_philippine_search_index()
        
        if philippine_search_index is None:
            print("❌ Philippine search index not available")
            return []
        
        # Phase 1: Data Collection and Validation (NO indexing operations yet)
        valid_urls = []
        batch_operations = {
            'indexing_operations': [],  # URLs ready for indexing
            'validation_errors': []     # URLs with validation errors
        }
        
        # Validate all URLs first (no indexing yet)
        for i, url in enumerate(urls_to_process):
            try:
                
                # Validate URL format
                from urllib.parse import urlparse
                parsed_url = urlparse(url.strip())
                if not parsed_url.scheme or not parsed_url.netloc:
                    batch_operations['validation_errors'].append({
                        'url': url,
                        'status': 'error',
                        'message': 'Invalid URL format',
                        'success': False
                    })
                    continue
                
                # Add to valid URLs for batch processing
                cleaned_url = url.strip()
                valid_urls.append(cleaned_url)
                batch_operations['indexing_operations'].append({
                    'url': cleaned_url,
                    'original_index': i,
                    'force_reindex': force_reindex
                })
                
                
            except Exception as e:
                batch_operations['validation_errors'].append({
                    'url': url,
                    'status': 'error',
                    'message': str(e),
                    'success': False
                })
        
        # Phase 2: Batch Philippine News Indexing Operations using new atomic method
        
        indexing_results = []
        start_time = time.time()
        
        # Process all valid URLs using new batch indexing with atomic transactions
        if batch_operations['indexing_operations']:
            try:
                # Extract URLs for batch processing
                urls_for_batch = [op['url'] for op in batch_operations['indexing_operations']]
                force_reindex_flag = batch_operations['indexing_operations'][0]['force_reindex'] if batch_operations['indexing_operations'] else False
                
                
                # Use the new batch indexing method with atomic transactions
                batch_results = philippine_search_index.batch_index_articles(urls_for_batch, force_reindex_flag)
                
                # Convert results to consistent format
                for i, result in enumerate(batch_results):
                    
                    indexing_results.append({
                        'url': result['url'],
                        'status': result['status'],
                        'message': result.get('message', ''),
                        'article_id': result.get('article_id'),
                        'relevance_score': result.get('relevance_score', 0),
                        'locations': result.get('locations', []),
                        'government_entities': result.get('government_entities', []),
                        'success': result['status'] in ['success', 'already_indexed', 'skipped']
                    })
                
                
            except Exception as e:
                
                # Add error results for failed batch
                for operation in batch_operations['indexing_operations']:
                    indexing_results.append({
                        'url': operation['url'],
                        'status': 'error',
                        'message': f'Atomic batch indexing failed: {str(e)}',
                        'success': False
                    })
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Phase 3: Combine Results and Generate Final Statistics
        
        # Combine validation errors and indexing results
        batch_results = []
        batch_results.extend(batch_operations['validation_errors'])
        batch_results.extend(indexing_results)
        
        # Calculate summary statistics
        success_count = len([r for r in batch_results if r['status'] == 'success'])
        skipped_count = len([r for r in batch_results if r['status'] == 'skipped'])
        error_count = len([r for r in batch_results if r['status'] == 'error'])
        already_indexed_count = len([r for r in batch_results if r['status'] == 'already_indexed'])
        
        print(f"✅ Batch completed in {duration_ms:.0f}ms: {success_count} success, {already_indexed_count} existing, {skipped_count} skipped, {error_count} errors")
        
        return batch_results
        
    except Exception as e:
        print(f"❌ Batch Philippine indexing failed: {str(e)}")
        return []

def index_philippine_news_article(url, force_reindex=False):
    """Index a verified news article into the Philippine news search system using direct method calls"""
    try:
        if not url:
            return False
            
        # Get the Philippine search index instance
        philippine_search_index = get_philippine_search_index()
        
        if philippine_search_index is None:
            return False
        
        # Validate URL format (same as in philippine_news_search_routes.py)
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return False
        
        # Index the article using direct method call
        start_time = time.time()
        
        result = philippine_search_index.index_article(url, force_reindex)
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        if result['status'] == 'success':
            return True
        elif result['status'] == 'already_indexed':
            return True
        elif result['status'] == 'skipped':
            return True
        else:
            return False
            
    except Exception as e:
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
                
                # Phase 1: Collect data from all websites (no database writes)
                website_batch_data = []
                
                for website_id, url, name, interval, last_fetch in websites:
                    try:
                        if last_fetch:
                            last_fetch_time = datetime.fromisoformat(last_fetch)
                            if current_time - last_fetch_time < timedelta(minutes=interval):
                                continue
                        
                        # Fetch articles using crawler
                        articles = fetch_articles_from_crawler(url, name, website_id)
                        
                        # Collect articles for batch processing (don't insert yet)
                        website_batch_data.append({
                            'website_id': website_id,
                            'url': url,
                            'name': name,
                            'articles': articles,
                            'success': True,
                            'error': None
                        })
                        
                    except Exception as e:
                        
                        # Collect error for batch processing
                        website_batch_data.append({
                            'website_id': website_id,
                            'url': url,
                            'name': name,
                            'articles': [],
                            'success': False,
                            'error': str(e)
                        })
                
                # Phase 2: Prepare All Database Operations
                batch_operations = {
                    'website_updates': [],   # last_fetch updates
                    'article_inserts': [],   # new articles
                    'fetch_logs': []        # log entries
                }
                
                for website_data in website_batch_data:
                    website_id = website_data['website_id']
                    articles = website_data['articles']
                    success = website_data['success']
                    error = website_data['error']
                    
                    # Prepare website update
                    batch_operations['website_updates'].append((website_id,))
                    
                    # Prepare fetch log
                    batch_operations['fetch_logs'].append((
                        website_id, len(articles), success, error
                    ))
                    
                    # Prepare article inserts
                    if articles:
                        for article in articles:
                            try:
                                batch_operations['article_inserts'].append((
                                    article['url'], article['title'], article['description'],
                                    article['content'], article['site_name'], article['website_id'], 'default',
                                    article.get('confidence', 0.0), article.get('is_news_prediction', True), 
                                    article.get('probability_news', 0.0)
                                ))
                            except KeyError as ke:
                                print(f"❌ Missing key in article data: {str(ke)}")
                
                # Phase 3: Single Atomic Transaction
                if (batch_operations['website_updates'] or 
                    batch_operations['article_inserts'] or 
                    batch_operations['fetch_logs']):
                    
                    try:
                        # Execute all operations in single transaction
                        if batch_operations['website_updates']:
                            cursor.executemany('''
                                UPDATE tracked_websites 
                                SET last_fetch = CURRENT_TIMESTAMP 
                                WHERE id = ?
                            ''', batch_operations['website_updates'])
                        
                        if batch_operations['fetch_logs']:
                            cursor.executemany('''
                                INSERT INTO fetch_logs (website_id, articles_found, success, error_message)
                                VALUES (?, ?, ?, ?)
                            ''', batch_operations['fetch_logs'])
                        
                        if batch_operations['article_inserts']:
                            cursor.executemany('''
                                INSERT OR IGNORE INTO article_queue (url, title, description, content, site_name, website_id, user_session, confidence, is_news_prediction, probability_news)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', batch_operations['article_inserts'])
                        
                        # Commit entire auto-fetch transaction
                        conn.commit()
                        
                        
                    except Exception as e:
                        conn.rollback()
                
                conn.close()
                
            except Exception as e:
                print(f"❌ Auto-fetch error: {str(e)}")
            
            time.sleep(60)  # Check every minute
    
    # Start background thread
    thread = threading.Thread(target=auto_fetch_worker, daemon=True)
    thread.start()

# Uncomment to enable auto-fetch
# start_auto_fetch()
