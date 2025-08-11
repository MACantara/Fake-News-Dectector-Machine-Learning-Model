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

# Create blueprint
rss_feed_bp = Blueprint('rss_feed', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Default RSS feeds
DEFAULT_RSS_FEEDS = {
    'pna': {
        'name': 'Philippine News Agency',
        'url': 'https://syndication.pna.gov.ph/rss',
        'category': 'Government',
        'active': True
    },
    'inquirer': {
        'name': 'Philippine Daily Inquirer',
        'url': 'https://newsinfo.inquirer.net/rss',
        'category': 'News',
        'active': True
    },
    'rappler': {
        'name': 'Rappler',
        'url': 'https://www.rappler.com/rss',
        'category': 'News',
        'active': True
    },
    'philstar': {
        'name': 'Philippine Star',
        'url': 'https://www.philstar.com/rss/headlines',
        'category': 'News',
        'active': True
    }
}

@rss_feed_bp.route('/rss-feeds')
def rss_feeds_page():
    """Render the RSS feeds management page"""
    return render_template('rss_feeds.html')

@rss_feed_bp.route('/api/rss-feeds', methods=['GET'])
def get_rss_feeds():
    """Get list of available RSS feeds"""
    try:
        return jsonify({
            'success': True,
            'feeds': DEFAULT_RSS_FEEDS
        })
    except Exception as e:
        logger.error(f"Error getting RSS feeds: {e}")
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
        limit = data.get('limit', 20)
        hours_back = data.get('hours_back', 24)
        
        if not feed_url:
            return jsonify({
                'success': False,
                'error': 'RSS feed URL is required'
            }), 400
        
        # Start timing
        start_time = time.time()
        
        # Parse RSS feed
        feed_data = parse_rss_feed(feed_url)
        
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
        processing_time = time.time() - start_time
        feed_data['performance'] = {
            'processing_time': processing_time,
            'articles_fetched': len(feed_data['articles']),
            'feed_title': feed_data.get('feed_info', {}).get('title', 'Unknown'),
            'feed_url': feed_url
        }
        
        return jsonify(feed_data)
        
    except Exception as e:
        logger.error(f"Error fetching RSS articles: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to fetch RSS articles: {str(e)}'
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
