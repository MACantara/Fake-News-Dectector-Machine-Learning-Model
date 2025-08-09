"""
News Tracker Routes
Handles website tracking, article fetching, and verification
"""

from flask import Blueprint, render_template, request, jsonify, session
import json
import sqlite3
import requests
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import time
import threading
from bs4 import BeautifulSoup
import feedparser
import hashlib
import logging

news_tracker_bp = Blueprint('news_tracker', __name__)

# Configuration
DATABASE_FILE = 'news_tracker.db'
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

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
                articles = fetch_articles_from_website(url, name, website_id)
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
                    INSERT INTO article_queue (url, title, description, content, site_name, website_id, user_session)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article['url'], article['title'], article['description'],
                    article['content'], article['site_name'], article['website_id'],
                    user_session
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
        
        # Update article verification
        cursor.execute('''
            UPDATE article_queue 
            SET verified = TRUE, is_news = ?, verified_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_session = ?
        ''', (is_news, article_id, user_session))
        
        conn.commit()
        conn.close()
        
        # Also send to URL classifier for learning (if available)
        try:
            feedback_data = {
                'url': url,
                'classification': 'news' if is_news else 'not_news'
            }
            # This would integrate with your existing URL classifier
            # requests.post('/api/url-classifier/feedback', json=feedback_data)
        except Exception:
            pass  # Ignore errors in feedback submission
        
        return jsonify({
            'success': True,
            'message': f'Article marked as {"news" if is_news else "not news"}'
        })
        
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

def fetch_articles_from_website(url, site_name, website_id):
    """Fetch articles from a website using multiple methods"""
    articles = []
    
    try:
        # Method 1: Try RSS/Atom feeds
        rss_articles = fetch_from_rss(url, site_name, website_id)
        articles.extend(rss_articles)
        
        if not articles:
            # Method 2: Web scraping
            scraped_articles = fetch_from_scraping(url, site_name, website_id)
            articles.extend(scraped_articles)
        
    except Exception as e:
        logging.error(f"Error fetching from {url}: {str(e)}")
    
    return articles

def fetch_from_rss(base_url, site_name, website_id):
    """Try to find and parse RSS feeds"""
    articles = []
    
    # Common RSS feed URLs to try
    rss_urls = [
        urljoin(base_url, '/rss'),
        urljoin(base_url, '/feed'),
        urljoin(base_url, '/rss.xml'),
        urljoin(base_url, '/feed.xml'),
        urljoin(base_url, '/atom.xml'),
        urljoin(base_url, '/news/rss'),
        urljoin(base_url, '/news/feed'),
    ]
    
    for rss_url in rss_urls:
        try:
            feed = feedparser.parse(rss_url)
            
            if feed.entries:
                for entry in feed.entries[:10]:  # Limit to 10 articles
                    article = {
                        'url': entry.get('link', ''),
                        'title': entry.get('title', ''),
                        'description': entry.get('summary', ''),
                        'content': entry.get('content', [{}])[0].get('value', '') if entry.get('content') else '',
                        'site_name': site_name,
                        'website_id': website_id,
                        'found_at': datetime.now().isoformat()
                    }
                    
                    if article['url']:
                        articles.append(article)
                
                break  # Stop after finding a working feed
                
        except Exception as e:
            logging.debug(f"RSS feed {rss_url} failed: {str(e)}")
            continue
    
    return articles

def fetch_from_scraping(url, site_name, website_id):
    """Scrape articles from website"""
    articles = []
    
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Common article link selectors
        article_selectors = [
            'article a[href]',
            '.article a[href]',
            '.post a[href]',
            '.news-item a[href]',
            'h1 a[href], h2 a[href], h3 a[href]',
            'a[href*="article"]',
            'a[href*="news"]',
            'a[href*="post"]',
        ]
        
        found_links = set()
        
        for selector in article_selectors:
            links = soup.select(selector)
            
            for link in links[:5]:  # Limit per selector
                href = link.get('href')
                if not href:
                    continue
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    href = urljoin(url, href)
                elif not href.startswith(('http://', 'https://')):
                    continue
                
                # Skip duplicate URLs
                if href in found_links:
                    continue
                found_links.add(href)
                
                # Extract article info
                title = link.get_text(strip=True) or link.get('title', '')
                description = ''
                
                # Try to find description from parent elements
                parent = link.find_parent(['article', 'div', 'section'])
                if parent:
                    desc_elem = parent.find(['p', '.description', '.summary', '.excerpt'])
                    if desc_elem:
                        description = desc_elem.get_text(strip=True)
                
                article = {
                    'url': href,
                    'title': title,
                    'description': description,
                    'content': '',
                    'site_name': site_name,
                    'website_id': website_id,
                    'found_at': datetime.now().isoformat()
                }
                
                articles.append(article)
                
                if len(articles) >= 10:  # Limit total articles
                    break
            
            if len(articles) >= 10:
                break
    
    except Exception as e:
        logging.error(f"Error scraping {url}: {str(e)}")
    
    return articles

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
                        
                        # Fetch articles
                        articles = fetch_articles_from_website(url, name, website_id)
                        
                        # Save new articles
                        for article in articles:
                            try:
                                cursor.execute('''
                                    INSERT INTO article_queue (url, title, description, content, site_name, website_id, user_session)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    article['url'], article['title'], article['description'],
                                    article['content'], article['site_name'], article['website_id'], 'default'
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
