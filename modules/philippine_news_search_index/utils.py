"""
Philippine News Search Index - Utility Functions
Content extraction and processing utilities for Philippine news articles
"""

import requests
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dateutil import parser as date_parser


def extract_advanced_content(url):
    """Enhanced content extraction with Philippine news focus"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()
        
        # Extract title with multiple fallbacks
        title = ""
        title_selectors = [
            'h1.entry-title', 'h1.article-title', 'h1.post-title',
            'h1', 'title', '.headline', '.article-headline',
            '[property="og:title"]', '[name="twitter:title"]'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True) if selector not in ['[property="og:title"]', '[name="twitter:title"]'] else element.get('content', '')
                if title and len(title) > 10:
                    break
        
        # Extract main content
        content = ""
        content_selectors = [
            'article', '.article-content', '.post-content', '.entry-content',
            '.story-body', '.article-body', '.news-content', '.content',
            'main', '.main-content', '.page-content'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text(strip=True) for elem in elements])
                if len(content) > 200:
                    break
        
        # If no specific content found, get all paragraphs
        if not content or len(content) < 200:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30])
        
        # Extract metadata
        author = ""
        author_selectors = [
            '[rel="author"]', '.author', '.byline', '.by',
            '[name="author"]', '[property="article:author"]'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                author = element.get_text(strip=True) if 'name' not in selector and 'property' not in selector else element.get('content', '')
                if author:
                    break
        
        # Extract publish date
        publish_date = None
        date_selectors = [
            '[property="article:published_time"]', '[name="publishdate"]',
            '[name="date"]', '.publish-date', '.date', '.post-date',
            '.article-date', 'time[datetime]', '.timestamp'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                date_text = element.get('content') or element.get('datetime') or element.get_text(strip=True)
                if date_text:
                    try:
                        publish_date = date_parser.parse(date_text)
                        break
                    except:
                        continue
        
        # Generate summary
        summary = ""
        if content:
            # Try to find meta description first
            meta_desc = soup.find('meta', {'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                summary = meta_desc['content']
            else:
                # Generate summary from first few sentences
                sentences = re.split(r'[.!?]+', content)
                summary = '. '.join(sentences[:3]) + '.' if len(sentences) >= 3 else content[:300] + '...'
        
        # Extract category from URL or meta tags
        category = ""
        parsed_url = urlparse(url)
        path_parts = [part for part in parsed_url.path.split('/') if part]
        
        # Common news categories
        news_categories = ['politics', 'business', 'sports', 'technology', 'entertainment', 
                         'world', 'local', 'national', 'opinion', 'lifestyle', 'health']
        
        for part in path_parts:
            if part.lower() in news_categories:
                category = part.lower()
                break
        
        return {
            'title': title or 'Untitled',
            'content': content or '',
            'summary': summary or '',
            'author': author or '',
            'publish_date': publish_date,
            'category': category or 'general'
        }
        
    except Exception as e:
        print(f"Content extraction error for {url}: {e}")
        return None


def get_philippine_news_categories():
    """Get list of available news categories"""
    return [
        {'id': 'politics', 'name': 'Politics', 'description': 'Political news and government affairs'},
        {'id': 'business', 'name': 'Business', 'description': 'Business and economic news'},
        {'id': 'sports', 'name': 'Sports', 'description': 'Sports news and events'},
        {'id': 'technology', 'name': 'Technology', 'description': 'Technology and innovation news'},
        {'id': 'entertainment', 'name': 'Entertainment', 'description': 'Entertainment and celebrity news'},
        {'id': 'world', 'name': 'World', 'description': 'International news'},
        {'id': 'local', 'name': 'Local', 'description': 'Local and regional news'},
        {'id': 'national', 'name': 'National', 'description': 'National news'},
        {'id': 'opinion', 'name': 'Opinion', 'description': 'Opinion pieces and editorials'},
        {'id': 'lifestyle', 'name': 'Lifestyle', 'description': 'Lifestyle and culture news'},
        {'id': 'health', 'name': 'Health', 'description': 'Health and medical news'}
    ]


def get_philippine_news_sources():
    """Get list of Philippine news sources"""
    return [
        {'domain': 'abs-cbn.com', 'name': 'ABS-CBN', 'type': 'broadcast'},
        {'domain': 'gma.tv', 'name': 'GMA Network', 'type': 'broadcast'},
        {'domain': 'inquirer.net', 'name': 'Philippine Daily Inquirer', 'type': 'newspaper'},
        {'domain': 'rappler.com', 'name': 'Rappler', 'type': 'online'},
        {'domain': 'philstar.com', 'name': 'Philippine Star', 'type': 'newspaper'},
        {'domain': 'manilabulletin.ph', 'name': 'Manila Bulletin', 'type': 'newspaper'},
        {'domain': 'businessworld.com.ph', 'name': 'BusinessWorld', 'type': 'business'},
        {'domain': 'pna.gov.ph', 'name': 'Philippine News Agency', 'type': 'government'},
        {'domain': 'cnnphilippines.com', 'name': 'CNN Philippines', 'type': 'broadcast'},
        {'domain': 'philnews.ph', 'name': 'Philippine News', 'type': 'online'},
        {'domain': 'sunstar.com.ph', 'name': 'SunStar', 'type': 'newspaper'},
        {'domain': 'manilatimes.net', 'name': 'Manila Times', 'type': 'newspaper'},
        {'domain': 'tribune.net.ph', 'name': 'Tribune', 'type': 'newspaper'},
        {'domain': 'remate.ph', 'name': 'Remate', 'type': 'tabloid'},
        {'domain': 'tempo.com.ph', 'name': 'Tempo', 'type': 'newspaper'},
        {'domain': 'journal.com.ph', 'name': 'Manila Journal', 'type': 'newspaper'},
        {'domain': 'malaya.com.ph', 'name': 'Malaya', 'type': 'newspaper'},
        {'domain': 'mindanews.com', 'name': 'MindaNews', 'type': 'regional'},
        {'domain': 'cebudailynews.inquirer.net', 'name': 'Cebu Daily News', 'type': 'regional'},
        {'domain': 'bworldonline.com', 'name': 'BusinessWorld Online', 'type': 'business'}
    ]


def validate_philippine_url(url):
    """Validate if a URL is from a Philippine news source"""
    philippine_domains = [
        'abs-cbn.com', 'gma.tv', 'inquirer.net', 'rappler.com', 'philstar.com',
        'manilabulletin.ph', 'businessworld.com.ph', 'pna.gov.ph', 'cnnphilippines.com',
        'philnews.ph', 'sunstar.com.ph', 'manilatimes.net', 'tribune.net.ph',
        'remate.ph', 'tempo.com.ph', 'journal.com.ph', 'malaya.com.ph',
        'mindanews.com', 'cebudailynews.inquirer.net', 'bworldonline.com'
    ]
    
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Check if domain is in the Philippine news sources list
        for ph_domain in philippine_domains:
            if ph_domain in domain:
                return True, f"Recognized Philippine news source: {ph_domain}"
        
        # Check for .ph domains as secondary validation
        if domain.endswith('.ph'):
            return True, "Philippine domain (.ph)"
        
        return False, "Not recognized as Philippine news source"
        
    except Exception as e:
        return False, f"URL validation error: {str(e)}"


def extract_philippine_keywords_from_content(content):
    """Extract Philippine-specific keywords and entities from content"""
    philippine_keywords = {
        'government': ['doh', 'dof', 'dilg', 'dnd', 'deped', 'dot', 'dtr', 'dswd', 'da', 'denr', 'doe', 'dost'],
        'places': ['manila', 'cebu', 'davao', 'quezon city', 'makati', 'taguig', 'pasig', 'marikina', 'antipolo',
                  'caloocan', 'las pinas', 'paranaque', 'muntinlupa', 'valenzuela', 'pasay', 'malabon',
                  'navotas', 'san juan', 'mandaluyong', 'pateros', 'luzon', 'visayas', 'mindanao',
                  'ncr', 'metro manila', 'baguio', 'palawan', 'boracay', 'bohol', 'iloilo', 'bacolod',
                  'cagayan de oro', 'zamboanga', 'butuan', 'tacloban', 'legazpi', 'naga'],
        'officials': ['president', 'vice president', 'senator', 'congressman', 'mayor', 'governor',
                     'secretary', 'undersecretary', 'assistant secretary', 'spokesperson'],
        'politics': ['malacañang', 'malacanang', 'palace', 'senate', 'congress', 'house of representatives',
                    'supreme court', 'comelec', 'ombudsman', 'pcoo', 'pcij', 'neda', 'nsc', 'ndrrmc'],
        'economy': ['bsp', 'bangko sentral', 'psei', 'philippine stock exchange', 'peso', 'gdp', 'inflation',
                   'ofw', 'remittances', 'bpo', 'pogo', 'dti', 'neda'],
        'culture': ['filipino', 'pilipino', 'tagalog', 'cebuano', 'ilocano', 'bicolano', 'waray', 'hiligaynon',
                   'kapampangan', 'pangasinense', 'maranao', 'tausug', 'maguindanao'],
        'events': ['holy week', 'undas', 'christmas', 'new year', 'rizal day', 'independence day',
                  'edsa revolution', 'people power', 'martial law', 'sona', 'bayanihan']
    }
    
    content_lower = content.lower()
    found_keywords = {}
    
    for category, keywords in philippine_keywords.items():
        found_in_category = []
        for keyword in keywords:
            if keyword in content_lower:
                found_in_category.append(keyword)
        
        if found_in_category:
            found_keywords[category] = found_in_category
    
    return found_keywords


def calculate_content_quality_score(content_data):
    """Calculate a quality score for extracted content"""
    score = 0.0
    
    if not content_data:
        return 0.0
    
    # Title presence and length
    title = content_data.get('title', '')
    if title and len(title) > 10:
        score += 0.2
        if len(title) > 30:
            score += 0.1
    
    # Content presence and length
    content = content_data.get('content', '')
    if content:
        content_length = len(content)
        if content_length > 200:
            score += 0.3
        if content_length > 500:
            score += 0.1
        if content_length > 1000:
            score += 0.1
    
    # Summary presence
    summary = content_data.get('summary', '')
    if summary and len(summary) > 20:
        score += 0.1
    
    # Author presence
    author = content_data.get('author', '')
    if author and len(author) > 2:
        score += 0.1
    
    # Date presence
    publish_date = content_data.get('publish_date')
    if publish_date:
        score += 0.1
    
    return min(1.0, score)


def clean_extracted_text(text):
    """Clean and normalize extracted text content"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Normalize spacing around punctuation
    text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
    
    return text.strip()
