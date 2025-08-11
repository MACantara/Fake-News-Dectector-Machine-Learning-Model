"""
RSS Feed Routes for News Analysis Application
Handles RSS feed fetching, parsing, and article extraction
"""

from flask import Blueprint, request, jsonify, render_template
import feedparser
import requests
from datetime import datetime, timedelta
import logging
from urllib.parse import urljoin, urlparse
import time
import html
import re
import sqlite3
import os

# Create blueprint
rss_feed_bp = Blueprint('rss_feed', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Database file path
RSS_DATABASE_FILE = 'rss_feeds.db'

# Philippine RSS feeds - comprehensive list
DEFAULT_RSS_FEEDS = [
    {
        'name': 'Philippine News Agency',
        'url': 'https://syndication.pna.gov.ph/rss',
        'category': 'Government',
        'description': 'Official news agency of the Philippines',
        'active': True
    },
    {
        'name': 'Philippine Daily Inquirer',
        'url': 'https://newsinfo.inquirer.net/rss',
        'category': 'News',
        'description': 'Leading Filipino-language daily newspaper',
        'active': True
    },
    {
        'name': 'Rappler',
        'url': 'https://www.rappler.com/rss',
        'category': 'News',
        'description': 'Digital news platform',
        'active': True
    },
    {
        'name': 'Philippine Star - Headlines',
        'url': 'https://www.philstar.com/rss/headlines',
        'category': 'Headlines',
        'description': 'Philippine Star headlines feed',
        'active': True
    },
    {
        'name': 'Philippine Star - Nation',
        'url': 'https://www.philstar.com/rss/nation',
        'category': 'Politics',
        'description': 'Philippine Star nation news',
        'active': False
    },
    {
        'name': 'ABS-CBN News',
        'url': 'https://news.abs-cbn.com/rss',
        'category': 'News',
        'description': 'ABS-CBN News RSS feed',
        'active': True
    },
    {
        'name': 'GMA News',
        'url': 'https://www.gmanetwork.com/news/rss/',
        'category': 'News',
        'description': 'GMA Network news RSS feed',
        'active': True
    },
    {
        'name': 'Manila Bulletin',
        'url': 'https://mb.com.ph/rss',
        'category': 'News',
        'description': 'Manila Bulletin RSS feed',
        'active': True
    },
    {
        'name': 'BusinessWorld',
        'url': 'https://www.bworldonline.com/feed/',
        'category': 'Business',
        'description': 'BusinessWorld RSS feed',
        'active': False
    },
    {
        'name': 'Philippine Star - Business',
        'url': 'https://www.philstar.com/rss/business',
        'category': 'Business',
        'description': 'Philippine Star business news',
        'active': False
    },
    {
        'name': 'Malaya Business Insight',
        'url': 'https://malaya.com.ph/feed/',
        'category': 'Business',
        'description': 'Malaya Business Insight RSS feed',
        'active': False
    },
    {
        'name': 'Interaksyon',
        'url': 'https://interaksyon.philstar.com/feed/',
        'category': 'News',
        'description': 'Interaksyon news RSS feed',
        'active': False
    },
    {
        'name': 'Sunstar Philippines',
        'url': 'https://www.sunstar.com.ph/api/v1/collections/home.rss',
        'category': 'Regional',
        'description': 'Sunstar regional news',
        'active': False
    },
    {
        'name': 'One News PH',
        'url': 'https://www.onenews.ph/rss',
        'category': 'News',
        'description': 'One News Philippines RSS feed',
        'active': False
    }
]

class RSSFeedDatabase:
    """SQLite database manager for RSS feeds"""
    
    def __init__(self, db_path=RSS_DATABASE_FILE):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the RSS feeds database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create RSS feeds table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rss_feeds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    url TEXT UNIQUE NOT NULL,
                    category TEXT DEFAULT 'News',
                    description TEXT,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_fetched TIMESTAMP,
                    fetch_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    last_error TEXT
                )
            ''')
            
            # Create RSS articles table for caching
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rss_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feed_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT UNIQUE NOT NULL,
                    description TEXT,
                    published_date TIMESTAMP,
                    author TEXT,
                    categories TEXT,
                    images TEXT,
                    source_domain TEXT,
                    word_count INTEGER DEFAULT 0,
                    content_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (feed_id) REFERENCES rss_feeds (id)
                )
            ''')
            
            # Create RSS fetch logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rss_fetch_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feed_id INTEGER NOT NULL,
                    fetch_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    articles_found INTEGER DEFAULT 0,
                    success BOOLEAN DEFAULT TRUE,
                    processing_time REAL DEFAULT 0.0,
                    error_message TEXT,
                    FOREIGN KEY (feed_id) REFERENCES rss_feeds (id)
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rss_feeds_active ON rss_feeds(active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rss_feeds_category ON rss_feeds(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rss_articles_feed ON rss_articles(feed_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rss_articles_url ON rss_articles(url)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rss_articles_published ON rss_articles(published_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rss_fetch_logs_feed ON rss_fetch_logs(feed_id)')
            
            conn.commit()
            
            # Insert default feeds if table is empty
            cursor.execute('SELECT COUNT(*) FROM rss_feeds')
            if cursor.fetchone()[0] == 0:
                self._populate_default_feeds(cursor)
                conn.commit()
            
            conn.close()
            logger.info("RSS feeds database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RSS feeds database: {e}")
            raise
    
    def _populate_default_feeds(self, cursor):
        """Populate database with default RSS feeds"""
        for feed in DEFAULT_RSS_FEEDS:
            cursor.execute('''
                INSERT INTO rss_feeds (name, url, category, description, active)
                VALUES (?, ?, ?, ?, ?)
            ''', (feed['name'], feed['url'], feed['category'], 
                  feed['description'], feed['active']))
    
    def get_all_feeds(self, active_only=False):
        """Get all RSS feeds from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if active_only:
                cursor.execute('SELECT * FROM rss_feeds WHERE active = 1 ORDER BY name')
            else:
                cursor.execute('SELECT * FROM rss_feeds ORDER BY name')
            
            feeds = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return feeds
            
        except Exception as e:
            logger.error(f"Error getting RSS feeds: {e}")
            return []
    
    def get_feed_by_id(self, feed_id):
        """Get RSS feed by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM rss_feeds WHERE id = ?', (feed_id,))
            feed = cursor.fetchone()
            conn.close()
            
            return dict(feed) if feed else None
            
        except Exception as e:
            logger.error(f"Error getting RSS feed by ID {feed_id}: {e}")
            return None
    
    def add_feed(self, name, url, category='News', description='', active=True):
        """Add new RSS feed to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO rss_feeds (name, url, category, description, active)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, url, category, description, active))
            
            feed_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return feed_id
            
        except sqlite3.IntegrityError:
            logger.warning(f"RSS feed URL already exists: {url}")
            return None
        except Exception as e:
            logger.error(f"Error adding RSS feed: {e}")
            return None
    
    def update_feed(self, feed_id, **kwargs):
        """Update RSS feed in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build dynamic update query
            set_clauses = []
            values = []
            
            for key, value in kwargs.items():
                if key in ['name', 'url', 'category', 'description', 'active']:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
            
            if set_clauses:
                set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                values.append(feed_id)
                
                query = f"UPDATE rss_feeds SET {', '.join(set_clauses)} WHERE id = ?"
                cursor.execute(query, values)
                
                conn.commit()
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error updating RSS feed {feed_id}: {e}")
            return False
    
    def delete_feed(self, feed_id):
        """Delete RSS feed and related data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete related articles and logs
            cursor.execute('DELETE FROM rss_articles WHERE feed_id = ?', (feed_id,))
            cursor.execute('DELETE FROM rss_fetch_logs WHERE feed_id = ?', (feed_id,))
            cursor.execute('DELETE FROM rss_feeds WHERE id = ?', (feed_id,))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting RSS feed {feed_id}: {e}")
            return False
    
    def log_fetch(self, feed_id, articles_found, success, processing_time, error_message=None):
        """Log RSS fetch attempt"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO rss_fetch_logs (feed_id, articles_found, success, processing_time, error_message)
                VALUES (?, ?, ?, ?, ?)
            ''', (feed_id, articles_found, success, processing_time, error_message))
            
            # Update feed stats
            if success:
                cursor.execute('''
                    UPDATE rss_feeds 
                    SET last_fetched = CURRENT_TIMESTAMP, 
                        fetch_count = fetch_count + 1,
                        last_error = NULL
                    WHERE id = ?
                ''', (feed_id,))
            else:
                cursor.execute('''
                    UPDATE rss_feeds 
                    SET error_count = error_count + 1,
                        last_error = ?
                    WHERE id = ?
                ''', (error_message, feed_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging RSS fetch for feed {feed_id}: {e}")

# Initialize database
rss_db = RSSFeedDatabase()

@rss_feed_bp.route('/rss-feeds')
def rss_feeds_page():
    """Render the RSS feeds management page"""
    return render_template('rss_feeds.html')

@rss_feed_bp.route('/api/rss-feeds/stats', methods=['GET'])
def get_rss_feeds_stats():
    """Get RSS feeds statistics"""
    try:
        conn = sqlite3.connect(rss_db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get basic stats
        cursor.execute('SELECT COUNT(*) as total_feeds FROM rss_feeds')
        total_feeds = cursor.fetchone()['total_feeds']
        
        cursor.execute('SELECT COUNT(*) as active_feeds FROM rss_feeds WHERE active = 1')
        active_feeds = cursor.fetchone()['active_feeds']
        
        cursor.execute('SELECT COUNT(*) as total_articles FROM rss_articles')
        total_articles = cursor.fetchone()['total_articles']
        
        # Get recent fetch statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_fetches,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_fetches,
                AVG(processing_time) as avg_processing_time,
                SUM(articles_found) as total_articles_found
            FROM rss_fetch_logs 
            WHERE fetch_time > datetime('now', '-24 hours')
        ''')
        recent_stats = dict(cursor.fetchone())
        
        # Get top performing feeds
        cursor.execute('''
            SELECT 
                f.name,
                f.url,
                COUNT(l.id) as fetch_count,
                AVG(l.processing_time) as avg_time,
                SUM(l.articles_found) as total_articles
            FROM rss_feeds f
            LEFT JOIN rss_fetch_logs l ON f.id = l.feed_id
            WHERE l.fetch_time > datetime('now', '-7 days')
            GROUP BY f.id, f.name, f.url
            ORDER BY total_articles DESC
            LIMIT 5
        ''')
        top_feeds = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_feeds': total_feeds,
                'active_feeds': active_feeds,
                'total_articles': total_articles,
                'recent_24h': recent_stats,
                'top_feeds': top_feeds
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting RSS feeds stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@rss_feed_bp.route('/api/rss-feeds/<int:feed_id>/logs', methods=['GET'])
def get_feed_logs(feed_id):
    """Get fetch logs for a specific RSS feed"""
    try:
        limit = request.args.get('limit', 50, type=int)
        
        conn = sqlite3.connect(rss_db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM rss_fetch_logs 
            WHERE feed_id = ? 
            ORDER BY fetch_time DESC 
            LIMIT ?
        ''', (feed_id, limit))
        
        logs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({
            'success': True,
            'logs': logs
        })
        
    except Exception as e:
        logger.error(f"Error getting RSS feed logs for feed {feed_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@rss_feed_bp.route('/api/rss-feeds', methods=['GET'])
def get_rss_feeds():
    """Get list of available RSS feeds"""
    try:
        active_only = request.args.get('active_only', 'false').lower() == 'true'
        feeds = rss_db.get_all_feeds(active_only=active_only)
        
        return jsonify({
            'success': True,
            'feeds': feeds,
            'total_count': len(feeds)
        })
    except Exception as e:
        logger.error(f"Error getting RSS feeds: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@rss_feed_bp.route('/api/rss-feeds', methods=['POST'])
def add_rss_feed():
    """Add new RSS feed"""
    try:
        data = request.get_json()
        name = data.get('name')
        url = data.get('url')
        category = data.get('category', 'News')
        description = data.get('description', '')
        active = data.get('active', True)
        
        if not name or not url:
            return jsonify({
                'success': False,
                'error': 'Name and URL are required'
            }), 400
        
        feed_id = rss_db.add_feed(name, url, category, description, active)
        
        if feed_id:
            return jsonify({
                'success': True,
                'message': 'RSS feed added successfully',
                'feed_id': feed_id
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to add RSS feed (URL may already exist)'
            }), 400
            
    except Exception as e:
        logger.error(f"Error adding RSS feed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@rss_feed_bp.route('/api/rss-feeds/<int:feed_id>', methods=['PUT'])
def update_rss_feed(feed_id):
    """Update RSS feed"""
    try:
        data = request.get_json()
        
        # Remove id and timestamps from update data
        update_data = {k: v for k, v in data.items() 
                      if k not in ['id', 'created_at', 'updated_at', 'last_fetched']}
        
        if rss_db.update_feed(feed_id, **update_data):
            return jsonify({
                'success': True,
                'message': 'RSS feed updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to update RSS feed'
            }), 400
            
    except Exception as e:
        logger.error(f"Error updating RSS feed {feed_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@rss_feed_bp.route('/api/rss-feeds/<int:feed_id>', methods=['DELETE'])
def delete_rss_feed(feed_id):
    """Delete RSS feed"""
    try:
        if rss_db.delete_feed(feed_id):
            return jsonify({
                'success': True,
                'message': 'RSS feed deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to delete RSS feed'
            }), 400
            
    except Exception as e:
        logger.error(f"Error deleting RSS feed {feed_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@rss_feed_bp.route('/api/rss-feed/articles', methods=['POST'])
def fetch_rss_articles():
    """Fetch articles from RSS feed"""
    try:
        data = request.get_json()
        feed_url = data.get('feed_url')
        feed_id = data.get('feed_id')
        limit = data.get('limit', 20)
        hours_back = data.get('hours_back', 24)
        
        # Get feed info from database if feed_id provided
        if feed_id:
            feed_info = rss_db.get_feed_by_id(feed_id)
            if not feed_info:
                return jsonify({
                    'success': False,
                    'error': 'RSS feed not found in database'
                }), 404
            feed_url = feed_info['url']
        elif not feed_url:
            return jsonify({
                'success': False,
                'error': 'RSS feed URL or feed ID is required'
            }), 400
        
        # Start timing
        start_time = time.time()
        
        # Parse RSS feed
        feed_data = parse_rss_feed(feed_url)
        processing_time = time.time() - start_time
        
        # Log the fetch attempt if we have a feed_id
        if feed_id:
            rss_db.log_fetch(
                feed_id=feed_id,
                articles_found=len(feed_data.get('articles', [])) if feed_data['success'] else 0,
                success=feed_data['success'],
                processing_time=processing_time,
                error_message=feed_data.get('error') if not feed_data['success'] else None
            )
        
        if not feed_data['success']:
            return jsonify(feed_data), 400
        
        # Filter articles by time if specified
        if hours_back:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            filtered_articles = []
            
            for article in feed_data['articles']:
                if article.get('published_date'):
                    try:
                        pub_date = datetime.fromisoformat(article['published_date'].replace('Z', '+00:00'))
                        if pub_date.replace(tzinfo=None) > cutoff_time:
                            filtered_articles.append(article)
                    except:
                        # If date parsing fails, include the article
                        filtered_articles.append(article)
                else:
                    # If no date, include the article
                    filtered_articles.append(article)
            
            feed_data['articles'] = filtered_articles
        
        # Limit number of articles
        if limit and len(feed_data['articles']) > limit:
            feed_data['articles'] = feed_data['articles'][:limit]
        
        # Add performance metrics
        feed_data['performance'] = {
            'processing_time': processing_time,
            'articles_fetched': len(feed_data['articles']),
            'feed_title': feed_data.get('feed_info', {}).get('title', 'Unknown'),
            'feed_url': feed_url,
            'feed_id': feed_id
        }
        
        return jsonify(feed_data)
        
    except Exception as e:
        logger.error(f"Error fetching RSS articles: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to fetch RSS articles: {str(e)}'
        }), 500

@rss_feed_bp.route('/api/rss-feed/batch-articles', methods=['POST'])
def fetch_batch_rss_articles():
    """Fetch articles from multiple RSS feeds"""
    try:
        data = request.get_json()
        feed_ids = data.get('feed_ids', [])
        limit_per_feed = data.get('limit_per_feed', 10)
        hours_back = data.get('hours_back', 24)
        
        if not feed_ids:
            # Get all active feeds
            active_feeds = rss_db.get_all_feeds(active_only=True)
            feed_ids = [feed['id'] for feed in active_feeds]
        
        all_articles = []
        performance_data = {
            'total_processing_time': 0,
            'feed_sources': 0,
            'articles_per_second': 0,
            'successful_feeds': 0,
            'failed_feeds': 0
        }
        
        for feed_id in feed_ids:
            try:
                # Fetch articles for this feed
                response = fetch_rss_articles()
                if response[1] == 200:  # Success
                    feed_data = response[0].get_json()
                    if feed_data['success']:
                        all_articles.extend(feed_data['articles'])
                        performance_data['total_processing_time'] += feed_data['performance']['processing_time']
                        performance_data['successful_feeds'] += 1
                    else:
                        performance_data['failed_feeds'] += 1
                else:
                    performance_data['failed_feeds'] += 1
                    
            except Exception as e:
                logger.error(f"Error fetching from feed {feed_id}: {e}")
                performance_data['failed_feeds'] += 1
        
        performance_data['feed_sources'] = len(feed_ids)
        if performance_data['total_processing_time'] > 0:
            performance_data['articles_per_second'] = len(all_articles) / performance_data['total_processing_time']
        
        return jsonify({
            'success': True,
            'articles': all_articles,
            'total_articles': len(all_articles),
            'performance': performance_data
        })
        
    except Exception as e:
        logger.error(f"Error fetching batch RSS articles: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to fetch batch RSS articles: {str(e)}'
        }), 500

@rss_feed_bp.route('/api/rss-feed/parse', methods=['POST'])
def parse_rss_endpoint():
    """Parse RSS feed and return basic info"""
    try:
        data = request.get_json()
        feed_url = data.get('feed_url')
        
        if not feed_url:
            return jsonify({
                'success': False,
                'error': 'RSS feed URL is required'
            }), 400
        
        result = parse_rss_feed(feed_url, articles_limit=5)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error parsing RSS feed: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to parse RSS feed: {str(e)}'
        }), 500

def parse_rss_feed(feed_url, articles_limit=None):
    """Parse RSS feed and extract articles"""
    try:
        # Set user agent to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the RSS feed with timeout
        response = requests.get(feed_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse the RSS feed
        feed = feedparser.parse(response.content)
        
        if feed.bozo and feed.bozo_exception:
            logger.warning(f"RSS feed parsing warning: {feed.bozo_exception}")
        
        # Extract feed information
        feed_info = {
            'title': getattr(feed.feed, 'title', 'Unknown Feed'),
            'description': getattr(feed.feed, 'description', ''),
            'link': getattr(feed.feed, 'link', feed_url),
            'language': getattr(feed.feed, 'language', 'en'),
            'updated': getattr(feed.feed, 'updated', ''),
            'total_entries': len(feed.entries)
        }
        
        # Extract articles
        articles = []
        entries = feed.entries[:articles_limit] if articles_limit else feed.entries
        
        for entry in entries:
            article = extract_article_from_entry(entry, feed_url)
            if article:
                articles.append(article)
        
        return {
            'success': True,
            'feed_info': feed_info,
            'articles': articles,
            'total_found': len(articles)
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching RSS feed {feed_url}: {e}")
        return {
            'success': False,
            'error': f'Network error: {str(e)}'
        }
    except Exception as e:
        logger.error(f"Error parsing RSS feed {feed_url}: {e}")
        return {
            'success': False,
            'error': f'Parsing error: {str(e)}'
        }

def extract_article_from_entry(entry, feed_url):
    """Extract article information from RSS entry"""
    try:
        # Get article URL
        article_url = getattr(entry, 'link', '')
        if not article_url:
            return None
        
        # Make URL absolute if it's relative
        if article_url.startswith('/'):
            base_url = f"{urlparse(feed_url).scheme}://{urlparse(feed_url).netloc}"
            article_url = urljoin(base_url, article_url)
        
        # Extract title
        title = getattr(entry, 'title', 'No Title')
        title = html.unescape(title).strip()
        
        # Extract description/summary
        description = ''
        if hasattr(entry, 'summary'):
            description = entry.summary
        elif hasattr(entry, 'description'):
            description = entry.description
        
        # Clean HTML from description
        if description:
            description = clean_html(description)
            description = html.unescape(description).strip()
        
        # Extract publication date
        published_date = ''
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                pub_time = time.mktime(entry.published_parsed)
                published_date = datetime.fromtimestamp(pub_time).isoformat()
            except:
                pass
        elif hasattr(entry, 'published'):
            published_date = entry.published
        
        # Extract author
        author = ''
        if hasattr(entry, 'author'):
            author = entry.author
        elif hasattr(entry, 'authors') and entry.authors:
            author = ', '.join([a.get('name', '') for a in entry.authors if a.get('name')])
        
        # Extract categories/tags
        categories = []
        if hasattr(entry, 'tags'):
            categories = [tag.get('term', '') for tag in entry.tags if tag.get('term')]
        elif hasattr(entry, 'category'):
            categories = [entry.category]
        
        # Extract media/images
        images = []
        if hasattr(entry, 'media_content'):
            for media in entry.media_content:
                if media.get('type', '').startswith('image/'):
                    images.append(media.get('url', ''))
        
        # Try to find images in description
        if not images and description:
            img_pattern = re.compile(r'<img[^>]+src="([^"]+)"', re.IGNORECASE)
            img_matches = img_pattern.findall(description)
            images.extend(img_matches)
        
        return {
            'title': title,
            'url': article_url,
            'description': description,
            'published_date': published_date,
            'author': author,
            'categories': categories,
            'images': images[:3],  # Limit to first 3 images
            'source_domain': urlparse(article_url).netloc,
            'word_count': len(description.split()) if description else 0
        }
        
    except Exception as e:
        logger.error(f"Error extracting article from RSS entry: {e}")
        return None

def clean_html(text):
    """Remove HTML tags from text"""
    if not text:
        return ''
    
    # Remove HTML tags
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

@rss_feed_bp.route('/api/rss-feed/analyze', methods=['POST'])
def analyze_rss_articles():
    """Analyze articles from RSS feed using fake news and political classifiers"""
    try:
        data = request.get_json()
        articles = data.get('articles', [])
        analysis_type = data.get('analysis_type', 'both')
        
        if not articles:
            return jsonify({
                'success': False,
                'error': 'No articles provided for analysis'
            }), 400
        
        # Import analysis modules
        from modules.fake_news_detector.detector import FakeNewsDetector
        from modules.political_news_detector.detector import PoliticalNewsDetector
        
        analyzed_articles = []
        total_articles = len(articles)
        successful_analyses = 0
        failed_analyses = 0
        
        # Initialize detectors based on analysis type
        fake_detector = None
        political_detector = None
        
        if analysis_type in ['fake_news', 'both']:
            try:
                fake_detector = FakeNewsDetector()
            except Exception as e:
                logger.error(f"Failed to initialize fake news detector: {e}")
        
        if analysis_type in ['political', 'both']:
            try:
                political_detector = PoliticalNewsDetector()
            except Exception as e:
                logger.error(f"Failed to initialize political detector: {e}")
        
        # Analyze each article
        for i, article in enumerate(articles):
            try:
                analyzed_article = article.copy()
                
                # Use title and description for analysis
                content_for_analysis = f"{article.get('title', '')} {article.get('description', '')}"
                
                if not content_for_analysis.strip():
                    analyzed_article['analysis_error'] = 'No content available for analysis'
                    failed_analyses += 1
                    analyzed_articles.append(analyzed_article)
                    continue
                
                # Fake news analysis
                if fake_detector:
                    try:
                        fake_result = fake_detector.predict(content_for_analysis)
                        analyzed_article['fake_news'] = fake_result
                    except Exception as e:
                        analyzed_article['fake_news'] = {'error': str(e)}
                
                # Political analysis
                if political_detector:
                    try:
                        political_result = political_detector.predict(content_for_analysis)
                        analyzed_article['political_classification'] = political_result
                    except Exception as e:
                        analyzed_article['political_classification'] = {'error': str(e)}
                
                successful_analyses += 1
                analyzed_articles.append(analyzed_article)
                
            except Exception as e:
                logger.error(f"Error analyzing article {i}: {e}")
                analyzed_article = article.copy()
                analyzed_article['analysis_error'] = str(e)
                failed_analyses += 1
                analyzed_articles.append(analyzed_article)
        
        # Calculate summary statistics
        summary = {
            'total_articles': total_articles,
            'successful_analyses': successful_analyses,
            'failed_analyses': failed_analyses
        }
        
        # Add analysis-specific summaries
        if analysis_type in ['fake_news', 'both']:
            fake_count = sum(1 for a in analyzed_articles 
                           if a.get('fake_news', {}).get('prediction') == 'Fake')
            real_count = successful_analyses - fake_count
            
            summary['fake_news_summary'] = {
                'fake_count': fake_count,
                'real_count': real_count,
                'fake_percentage': (fake_count / successful_analyses * 100) if successful_analyses > 0 else 0
            }
        
        if analysis_type in ['political', 'both']:
            political_count = sum(1 for a in analyzed_articles 
                                if a.get('political_classification', {}).get('prediction') == 'Political')
            non_political_count = successful_analyses - political_count
            
            summary['political_summary'] = {
                'political_count': political_count,
                'non_political_count': non_political_count,
                'political_percentage': (political_count / successful_analyses * 100) if successful_analyses > 0 else 0
            }
        
        return jsonify({
            'success': True,
            'results': analyzed_articles,
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error analyzing RSS articles: {e}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500
