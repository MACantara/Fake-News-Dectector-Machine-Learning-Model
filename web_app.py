from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
import requests
from bs4 import BeautifulSoup
import pickle
import joblib
import os
import json
from datetime import datetime
import threading
import time
from urllib.parse import urljoin, urlparse, parse_qs
import concurrent.futures
from collections import defaultdict
import sqlite3
import hashlib
from whoosh import index
from whoosh.fields import Schema, TEXT, DATETIME, ID, KEYWORD, NUMERIC
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import And, Or, Term, Phrase
from dateutil import parser as date_parser
from textblob import TextBlob
from fuzzywuzzy import fuzz, process
from url_news_classifier import URLNewsClassifier
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

app = Flask(__name__)

class PhilippineNewsSearchIndex:
    """
    Specialized search engine index for Philippine news articles with SQLite database integration
    """
    def __init__(self, db_path='philippine_news_index.db', index_dir='whoosh_index'):
        self.db_path = db_path
        self.index_dir = index_dir
        self.stemmer = PorterStemmer()
        
        # Philippine-specific keywords and entities
        self.philippine_keywords = {
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
        
        # Philippine news sources patterns
        self.philippine_news_domains = [
            'abs-cbn.com', 'gma.tv', 'inquirer.net', 'rappler.com', 'philstar.com',
            'manilabulletin.ph', 'businessworld.com.ph', 'pna.gov.ph', 'cnnphilippines.com',
            'philnews.ph', 'sunstar.com.ph', 'manilatimes.net', 'tribune.net.ph',
            'remate.ph', 'tempo.com.ph', 'journal.com.ph', 'malaya.com.ph',
            'mindanews.com', 'cebudailynews.inquirer.net', 'bworldonline.com'
        ]
        
        # Initialize database and search index
        self.init_database()
        self.init_search_index()
    
    def init_database(self):
        """Initialize SQLite database for storing Philippine news articles"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create main articles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS philippine_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT,
                    author TEXT,
                    publish_date DATETIME,
                    source_domain TEXT,
                    category TEXT,
                    tags TEXT,
                    philippine_relevance_score REAL,
                    location_mentions TEXT,
                    government_entities TEXT,
                    language TEXT DEFAULT 'en',
                    sentiment_score REAL,
                    content_hash TEXT,
                    indexed_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    fake_news_prediction REAL,
                    political_prediction REAL,
                    view_count INTEGER DEFAULT 0,
                    search_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create search queries table for analytics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    results_count INTEGER,
                    query_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_ip TEXT,
                    response_time REAL
                )
            ''')
            
            # Create indexing tasks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS indexing_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    completed_date DATETIME
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_url ON philippine_articles(url)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_publish_date ON philippine_articles(publish_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_domain ON philippine_articles(source_domain)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON philippine_articles(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_philippine_relevance ON philippine_articles(philippine_relevance_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_indexed_date ON philippine_articles(indexed_date)')
            
            conn.commit()
            conn.close()
            print("✓ Philippine News Database initialized successfully")
            
        except Exception as e:
            print(f"✗ Database initialization error: {e}")
    
    def init_search_index(self):
        """Initialize Whoosh search index for full-text search"""
        try:
            # Create index directory if it doesn't exist
            if not os.path.exists(self.index_dir):
                os.makedirs(self.index_dir)
            
            # Define schema for Philippine news articles
            schema = Schema(
                id=ID(stored=True, unique=True),
                url=ID(stored=True),
                title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
                content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
                summary=TEXT(stored=True, analyzer=StemmingAnalyzer()),
                author=TEXT(stored=True),
                publish_date=DATETIME(stored=True),
                source_domain=KEYWORD(stored=True),
                category=KEYWORD(stored=True),
                tags=KEYWORD(stored=True, commas=True),
                location_mentions=KEYWORD(stored=True, commas=True),
                government_entities=KEYWORD(stored=True, commas=True),
                language=KEYWORD(stored=True),
                philippine_relevance_score=NUMERIC(stored=True),
                sentiment_score=NUMERIC(stored=True)
            )
            
            # Create or open the index
            if index.exists_in(self.index_dir):
                self.search_index = index.open_dir(self.index_dir)
            else:
                self.search_index = index.create_in(self.index_dir, schema)
            
            print("✓ Philippine News Search Index initialized successfully")
            
        except Exception as e:
            print(f"✗ Search index initialization error: {e}")
            self.search_index = None
    
    def calculate_philippine_relevance(self, title, content, url):
        """Calculate relevance score for Philippine news (0-1 scale)"""
        score = 0.0
        text_combined = f"{title} {content}".lower()
        
        # Check URL domain
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # High score for known Philippine news domains
        for ph_domain in self.philippine_news_domains:
            if ph_domain in domain:
                score += 0.4
                break
        
        # Check for Philippine keywords
        for category, keywords in self.philippine_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_combined)
            if matches > 0:
                # Weight different categories
                weight = {
                    'places': 0.15,
                    'officials': 0.12,
                    'government': 0.12,
                    'politics': 0.10,
                    'economy': 0.08,
                    'culture': 0.05,
                    'events': 0.05
                }.get(category, 0.05)
                
                # Diminishing returns for multiple matches
                category_score = min(weight, weight * (1 + 0.1 * (matches - 1)))
                score += category_score
        
        # Check for "Philippines" or "Filipino" mentions
        philippines_mentions = len(re.findall(r'\b(?:philippines?|filipino?s?|pilipino?s?)\b', text_combined))
        if philippines_mentions > 0:
            score += min(0.2, 0.05 * philippines_mentions)
        
        # Bonus for Philippine peso currency mentions
        if re.search(r'\b(?:php|peso|pesos|₱)\b', text_combined):
            score += 0.05
        
        # Normalize score to 0-1 range
        return min(1.0, score)
    
    def extract_philippine_entities(self, text):
        """Extract Philippine-specific entities and locations from text"""
        text_lower = text.lower()
        
        locations = []
        government_entities = []
        
        # Extract locations
        for location in self.philippine_keywords['places']:
            if location in text_lower:
                locations.append(location.title())
        
        # Extract government entities
        for entity in self.philippine_keywords['government']:
            if entity in text_lower:
                government_entities.append(entity.upper())
        
        # Extract officials (with context)
        officials_pattern = r'\b(?:president|senator|congressman|mayor|governor|secretary)\s+(\w+(?:\s+\w+)?)\b'
        officials = re.findall(officials_pattern, text_lower)
        government_entities.extend([f"{match}" for match in officials])
        
        return list(set(locations)), list(set(government_entities))
    
    def extract_advanced_content(self, url):
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
    
    def index_article(self, url, force_reindex=False):
        """Index a single Philippine news article"""
        try:
            # Check if already indexed
            if not force_reindex:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM philippine_articles WHERE url = ?', (url,))
                if cursor.fetchone():
                    conn.close()
                    return {'status': 'already_indexed', 'url': url}
                conn.close()
            
            # Add to indexing tasks
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('INSERT INTO indexing_tasks (url, status) VALUES (?, ?)', (url, 'processing'))
            task_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Extract content
            content_data = self.extract_advanced_content(url)
            if not content_data:
                self.update_indexing_task(task_id, 'failed', 'Content extraction failed')
                return {'status': 'error', 'message': 'Content extraction failed'}
            
            # Calculate Philippine relevance
            relevance_score = self.calculate_philippine_relevance(
                content_data['title'], 
                content_data['content'], 
                url
            )
            
            # Skip if not relevant to Philippines (threshold: 0.1)
            if relevance_score < 0.1:
                self.update_indexing_task(task_id, 'skipped', 'Low Philippine relevance score')
                return {'status': 'skipped', 'message': 'Not relevant to Philippine news'}
            
            # Extract Philippine entities
            locations, government_entities = self.extract_philippine_entities(
                f"{content_data['title']} {content_data['content']}"
            )
            
            # Calculate sentiment
            sentiment_score = 0.0
            try:
                blob = TextBlob(content_data['content'][:1000])  # First 1000 chars for performance
                sentiment_score = blob.sentiment.polarity
            except:
                pass
            
            # Generate content hash
            content_hash = hashlib.md5(
                f"{content_data['title']}{content_data['content']}".encode('utf-8')
            ).hexdigest()
            
            # Get source domain
            source_domain = urlparse(url).netloc
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO philippine_articles 
                (url, title, content, summary, author, publish_date, source_domain, category, 
                 tags, philippine_relevance_score, location_mentions, government_entities, 
                 sentiment_score, content_hash, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                url, content_data['title'], content_data['content'], content_data['summary'],
                content_data['author'], content_data['publish_date'], source_domain,
                content_data['category'], ','.join(locations + government_entities),
                relevance_score, ','.join(locations), ','.join(government_entities),
                sentiment_score, content_hash
            ))
            
            article_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Add to search index
            if self.search_index:
                writer = self.search_index.writer()
                writer.add_document(
                    id=str(article_id),
                    url=url,
                    title=content_data['title'],
                    content=content_data['content'],
                    summary=content_data['summary'],
                    author=content_data['author'],
                    publish_date=content_data['publish_date'],
                    source_domain=source_domain,
                    category=content_data['category'],
                    tags=','.join(locations + government_entities),
                    location_mentions=','.join(locations),
                    government_entities=','.join(government_entities),
                    philippine_relevance_score=relevance_score,
                    sentiment_score=sentiment_score
                )
                writer.commit()
            
            # Update indexing task
            self.update_indexing_task(task_id, 'completed', None)
            
            return {
                'status': 'success',
                'article_id': article_id,
                'relevance_score': relevance_score,
                'locations': locations,
                'government_entities': government_entities
            }
            
        except Exception as e:
            self.update_indexing_task(task_id if 'task_id' in locals() else None, 'failed', str(e))
            return {'status': 'error', 'message': str(e)}
    
    def update_indexing_task(self, task_id, status, error_message=None):
        """Update indexing task status"""
        if not task_id:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE indexing_tasks 
                SET status = ?, error_message = ?, completed_date = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (status, error_message, task_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error updating indexing task {task_id}: {e}")
    
    def search_articles(self, query, limit=20, category=None, date_range=None, source=None):
        """Search Philippine news articles with advanced filtering"""
        start_time = time.time()
        
        try:
            results = []
            
            if self.search_index:
                # Use Whoosh for full-text search
                searcher = self.search_index.searcher()
                
                # Build query
                query_parts = []
                
                # Main text search
                if query:
                    parser = MultifieldParser(['title', 'content', 'summary'], self.search_index.schema)
                    text_query = parser.parse(query)
                    query_parts.append(text_query)
                
                # Category filter
                if category:
                    category_query = Term('category', category)
                    query_parts.append(category_query)
                
                # Source filter
                if source:
                    source_query = Term('source_domain', source)
                    query_parts.append(source_query)
                
                # Combine queries
                if query_parts:
                    if len(query_parts) == 1:
                        final_query = query_parts[0]
                    else:
                        final_query = And(query_parts)
                else:
                    final_query = None
                
                # Execute search
                if final_query:
                    search_results = searcher.search(final_query, limit=limit)
                    
                    for result in search_results:
                        results.append({
                            'id': result['id'],
                            'url': result['url'],
                            'title': result['title'],
                            'summary': result['summary'],
                            'author': result.get('author', ''),
                            'publish_date': result.get('publish_date'),
                            'source_domain': result['source_domain'],
                            'category': result.get('category', ''),
                            'relevance_score': result.get('philippine_relevance_score', 0),
                            'score': result.score
                        })
                
                searcher.close()
            
            # Fallback to SQL search if Whoosh fails or no results
            if not results:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                sql_parts = ['SELECT * FROM philippine_articles WHERE 1=1']
                params = []
                
                if query:
                    sql_parts.append('AND (title LIKE ? OR content LIKE ? OR summary LIKE ?)')
                    search_term = f'%{query}%'
                    params.extend([search_term, search_term, search_term])
                
                if category:
                    sql_parts.append('AND category = ?')
                    params.append(category)
                
                if source:
                    sql_parts.append('AND source_domain LIKE ?')
                    params.append(f'%{source}%')
                
                sql_parts.append('ORDER BY philippine_relevance_score DESC, indexed_date DESC')
                sql_parts.append('LIMIT ?')
                params.append(limit)
                
                cursor.execute(' '.join(sql_parts), params)
                rows = cursor.fetchall()
                
                # Convert to dict format
                columns = [description[0] for description in cursor.description]
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    results.append(row_dict)
                
                conn.close()
            
            # Log search query
            response_time = time.time() - start_time
            self.log_search_query(query, len(results), response_time)
            
            return {
                'results': results,
                'count': len(results),
                'query': query,
                'response_time': response_time
            }
            
        except Exception as e:
            print(f"Search error: {e}")
            return {'results': [], 'count': 0, 'error': str(e)}
    
    def find_similar_content(self, content_text, limit=10, minimum_similarity=0.1):
        """Find similar articles based on content similarity using TF-IDF and keyword matching"""
        import re
        from collections import Counter
        
        start_time = time.time()
        
        try:
            results = []
            
            # Clean and prepare input text
            content_text = content_text.strip()
            if len(content_text) < 50:  # Too short for meaningful comparison
                return {
                    'results': [],
                    'count': 0,
                    'query_summary': 'Content too short for similarity matching',
                    'response_time': time.time() - start_time
                }
            
            # Extract meaningful keywords from content
            # Remove common words and extract significant terms
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
                'his', 'her', 'its', 'our', 'their', 'said', 'says', 'according', 'also', 'from', 'as', 'ng',
                'ang', 'sa', 'na', 'kay', 'ni', 'para', 'mga', 'nang', 'kung', 'hindi', 'siya', 'ako', 'tayo'
            }
            
            # Extract words and clean them
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content_text.lower())
            meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Get word frequency for importance weighting
            word_freq = Counter(meaningful_words)
            top_keywords = [word for word, count in word_freq.most_common(20)]
            
            if not top_keywords:
                return {
                    'results': [],
                    'count': 0,
                    'query_summary': 'No meaningful keywords found',
                    'response_time': time.time() - start_time
                }
            
            # Search using the Whoosh index first (if available)
            if self.search_index:
                try:
                    searcher = self.search_index.searcher()
                    
                    # Create a query using the top keywords
                    from whoosh.query import Or, Term, And
                    from whoosh.qparser import MultifieldParser
                    
                    # Build keyword-based search query
                    keyword_queries = []
                    for keyword in top_keywords[:10]:  # Use top 10 keywords
                        keyword_queries.append(Term('content', keyword))
                        keyword_queries.append(Term('title', keyword))
                        keyword_queries.append(Term('summary', keyword))
                    
                    if keyword_queries:
                        # Use OR to find articles containing any of the keywords
                        search_query = Or(keyword_queries)
                        search_results = searcher.search(search_query, limit=limit * 2)  # Get more results for filtering
                        
                        for result in search_results:
                            # Calculate basic similarity score
                            result_content = (result.get('content', '') + ' ' + 
                                            result.get('title', '') + ' ' + 
                                            result.get('summary', '')).lower()
                            
                            # Count matching keywords
                            matches = sum(1 for keyword in top_keywords if keyword in result_content)
                            similarity_score = matches / len(top_keywords) if top_keywords else 0
                            
                            if similarity_score >= minimum_similarity:
                                results.append({
                                    'id': result['id'],
                                    'url': result['url'],
                                    'title': result['title'],
                                    'summary': result.get('summary', ''),
                                    'author': result.get('author', ''),
                                    'publish_date': result.get('publish_date'),
                                    'source_domain': result['source_domain'],
                                    'category': result.get('category', ''),
                                    'relevance_score': result.get('philippine_relevance_score', 0),
                                    'similarity_score': round(similarity_score, 3),
                                    'matching_keywords': [kw for kw in top_keywords if kw in result_content]
                                })
                    
                    searcher.close()
                except Exception as search_error:
                    print(f"Whoosh search error: {search_error}")
            
            # Fallback to SQL-based similarity search if Whoosh failed or no results
            if not results:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Build SQL query for keyword matching
                keyword_conditions = []
                params = []
                
                for keyword in top_keywords[:8]:  # Use top 8 keywords for SQL
                    keyword_conditions.append('(title LIKE ? OR content LIKE ? OR summary LIKE ?)')
                    search_term = f'%{keyword}%'
                    params.extend([search_term, search_term, search_term])
                
                if keyword_conditions:
                    sql_query = f'''
                        SELECT *, 
                               (CASE 
                                   {" + ".join([f"WHEN (title LIKE ? OR content LIKE ? OR summary LIKE ?) THEN 1" 
                                              for _ in range(len(top_keywords[:8]))])}
                                   ELSE 0 
                               END) as match_count
                        FROM philippine_articles 
                        WHERE ({" OR ".join(keyword_conditions)})
                        ORDER BY match_count DESC, philippine_relevance_score DESC
                        LIMIT ?
                    '''
                    
                    # Double the params for the CASE statement
                    case_params = []
                    for keyword in top_keywords[:8]:
                        search_term = f'%{keyword}%'
                        case_params.extend([search_term, search_term, search_term])
                    
                    all_params = case_params + params + [limit]
                    
                    cursor.execute(sql_query, all_params)
                    rows = cursor.fetchall()
                    
                    # Convert to dict format and calculate similarity
                    columns = [description[0] for description in cursor.description]
                    for row in rows:
                        row_dict = dict(zip(columns, row))
                        
                        # Calculate similarity score
                        result_content = (row_dict.get('content', '') + ' ' + 
                                        row_dict.get('title', '') + ' ' + 
                                        row_dict.get('summary', '')).lower()
                        
                        matches = sum(1 for keyword in top_keywords if keyword in result_content)
                        similarity_score = matches / len(top_keywords) if top_keywords else 0
                        
                        if similarity_score >= minimum_similarity:
                            row_dict['similarity_score'] = round(similarity_score, 3)
                            row_dict['matching_keywords'] = [kw for kw in top_keywords if kw in result_content]
                            results.append(row_dict)
                
                conn.close()
            
            # Sort by similarity score
            results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            # Limit final results
            if len(results) > limit:
                results = results[:limit]
            
            # Log the similarity search
            response_time = time.time() - start_time
            query_summary = f"Keywords: {', '.join(top_keywords[:5])}"
            self.log_search_query(f"SIMILARITY: {query_summary}", len(results), response_time)
            
            return {
                'results': results,
                'count': len(results),
                'query_summary': query_summary,
                'top_keywords': top_keywords[:10],
                'response_time': response_time,
                'minimum_similarity': minimum_similarity
            }
            
        except Exception as e:
            print(f"Similarity search error: {e}")
            return {
                'results': [], 
                'count': 0, 
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def log_search_query(self, query, results_count, response_time):
        """Log search query for analytics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO search_queries (query, results_count, response_time)
                VALUES (?, ?, ?)
            ''', (query, results_count, response_time))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging search query: {e}")
    
    def get_search_analytics(self):
        """Get search analytics and statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total articles count
            cursor.execute('SELECT COUNT(*) FROM philippine_articles')
            total_articles = cursor.fetchone()[0]
            
            # Articles by source
            cursor.execute('''
                SELECT source_domain, COUNT(*) as count 
                FROM philippine_articles 
                GROUP BY source_domain 
                ORDER BY count DESC 
                LIMIT 10
            ''')
            sources = [{'domain': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
            # Articles by category
            cursor.execute('''
                SELECT category, COUNT(*) as count 
                FROM philippine_articles 
                GROUP BY category 
                ORDER BY count DESC
            ''')
            categories = [{'category': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
            # Recent indexing activity
            cursor.execute('''
                SELECT DATE(indexed_date) as date, COUNT(*) as count 
                FROM philippine_articles 
                WHERE indexed_date > datetime('now', '-30 days')
                GROUP BY DATE(indexed_date) 
                ORDER BY date DESC
            ''')
            recent_activity = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
            # Top search queries
            cursor.execute('''
                SELECT query, COUNT(*) as frequency, AVG(results_count) as avg_results
                FROM search_queries 
                WHERE query_date > datetime('now', '-7 days')
                GROUP BY query 
                ORDER BY frequency DESC 
                LIMIT 10
            ''')
            top_queries = [{'query': row[0], 'frequency': row[1], 'avg_results': row[2]} for row in cursor.fetchall()]
            
            # Average relevance score
            cursor.execute('SELECT AVG(philippine_relevance_score) FROM philippine_articles')
            avg_relevance = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'total_articles': total_articles,
                'sources': sources,
                'categories': categories,
                'recent_activity': recent_activity,
                'top_queries': top_queries,
                'average_relevance_score': round(avg_relevance, 3)
            }
            
        except Exception as e:
            print(f"Analytics error: {e}")
            return {}
    
    def get_article_by_id(self, article_id):
        """Get full article details by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM philippine_articles WHERE id = ?', (article_id,))
            row = cursor.fetchone()
            
            if row:
                columns = [description[0] for description in cursor.description]
                article = dict(zip(columns, row))
                
                # Update view count
                cursor.execute('UPDATE philippine_articles SET view_count = view_count + 1 WHERE id = ?', (article_id,))
                conn.commit()
                
                conn.close()
                return article
            
            conn.close()
            return None
            
        except Exception as e:
            print(f"Error getting article {article_id}: {e}")
            return None
    
    def crawl_and_index_website(self, website_url, max_articles=20, force_reindex=False):
        """Crawl a news website and index all found articles"""
        try:
            print(f"Starting website crawl and index: {website_url}")
            
            # Initialize news crawler if not exists
            if not hasattr(self, 'news_crawler'):
                self.news_crawler = NewsWebsiteCrawler()
            
            # Crawl the website for article links
            crawl_result = self.news_crawler.extract_article_links(website_url)
            
            if not crawl_result['success']:
                return {
                    'status': 'error',
                    'message': f"Failed to crawl website: {crawl_result['error']}",
                    'website_url': website_url,
                    'results': []
                }
            
            if not crawl_result['articles']:
                return {
                    'status': 'completed',
                    'message': 'No articles found on the website',
                    'website_url': website_url,
                    'website_title': crawl_result.get('website_title', ''),
                    'results': []
                }
            
            # Apply max_articles limit for indexing
            articles_to_index = crawl_result['articles']
            if max_articles and len(articles_to_index) > max_articles:
                articles_to_index = articles_to_index[:max_articles]
                print(f"📋 Limited indexing to top {max_articles} articles")
            
            print(f"Found {len(articles_to_index)} articles to index")
            
            # Index each found article
            indexing_results = []
            successful_indexes = 0
            skipped_indexes = 0
            error_indexes = 0
            already_indexed_count = 0
            
            for article in articles_to_index:
                try:
                    article_url = article['url']
                    article_title = article['title']
                    
                    print(f"Indexing: {article_title[:50]}...")
                    
                    # Index the article
                    index_result = self.index_article(article_url, force_reindex)
                    
                    result_entry = {
                        'url': article_url,
                        'title': article_title,
                        'status': index_result['status'],
                        'confidence': article.get('confidence', 0),
                        'selector_used': article.get('selector_used', '')
                    }
                    
                    if index_result['status'] == 'success':
                        successful_indexes += 1
                        result_entry.update({
                            'article_id': index_result['article_id'],
                            'relevance_score': index_result['relevance_score'],
                            'locations_found': index_result['locations'],
                            'government_entities_found': index_result['government_entities']
                        })
                    elif index_result['status'] == 'skipped':
                        skipped_indexes += 1
                        result_entry['message'] = index_result.get('message', 'Low Philippine relevance')
                    elif index_result['status'] == 'already_indexed':
                        already_indexed_count += 1
                        result_entry['message'] = 'Article already in index'
                    else:
                        error_indexes += 1
                        result_entry['error'] = index_result.get('message', 'Unknown error')
                    
                    indexing_results.append(result_entry)
                    
                except Exception as e:
                    error_indexes += 1
                    indexing_results.append({
                        'url': article.get('url', ''),
                        'title': article.get('title', ''),
                        'status': 'error',
                        'error': str(e)
                    })
            
            # Log crawling task completion
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO indexing_tasks (url, status, error_message, completed_date)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    website_url, 
                    'website_crawl_completed',
                    f"Found {len(crawl_result['articles'])} articles, indexed {successful_indexes}"
                ))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Error logging crawl task: {e}")
            
            return {
                'status': 'completed',
                'message': f'Website crawl and indexing completed',
                'website_url': website_url,
                'website_title': crawl_result.get('website_title', ''),
                'summary': {
                    'total_articles_found': len(crawl_result['articles']),
                    'articles_processed': len(articles_to_index),  # Add this to show limited count
                    'successfully_indexed': successful_indexes,
                    'skipped': skipped_indexes,
                    'errors': error_indexes,
                    'already_indexed': already_indexed_count
                },
                'results': indexing_results
            }
            
        except Exception as e:
            print(f"Error during website crawl and index: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'website_url': website_url,
                'results': []
            }
    
    def get_crawl_history(self, limit=50):
        """Get history of website crawling tasks"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT url, status, error_message, created_date, completed_date
                FROM indexing_tasks 
                WHERE status LIKE '%website%' OR status = 'website_crawl_completed'
                ORDER BY created_date DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    'url': row[0],
                    'status': row[1],
                    'message': row[2] or '',
                    'started_date': row[3],
                    'completed_date': row[4]
                })
            
            return history
            
        except Exception as e:
            print(f"Error getting crawl history: {e}")
            return []

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False
        self.accuracy = None
        self.feedback_file = 'user_feedback.json'
        self.feedback_data = []
        self.retrain_threshold = 10  # Retrain after collecting 10 feedback samples
        self.load_feedback_data()
    
    def load_feedback_data(self):
        """Load existing feedback data"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    self.feedback_data = json.load(f)
                print(f"Loaded {len(self.feedback_data)} feedback entries")
        except Exception as e:
            print(f"Error loading feedback data: {str(e)}")
            self.feedback_data = []
    
    def save_feedback_data(self):
        """Save feedback data to file"""
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving feedback data: {str(e)}")
    
    def add_feedback(self, text, predicted_label, actual_label, confidence, user_comment=None):
        """Add user feedback for model improvement"""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'predicted_label': predicted_label,
            'actual_label': actual_label,
            'confidence': confidence,
            'user_comment': user_comment,
            'processed_text': self.preprocess_text(text)
        }
        
        self.feedback_data.append(feedback_entry)
        self.save_feedback_data()
        
        print(f"Feedback added. Total feedback entries: {len(self.feedback_data)}")
        
        # Check if we should retrain the model
        unprocessed_feedback = [f for f in self.feedback_data if not f.get('used_for_training', False)]
        if len(unprocessed_feedback) >= self.retrain_threshold:
            print(f"Threshold reached ({self.retrain_threshold} new feedback entries). Scheduling model retraining...")
            # Schedule retraining in a separate thread to avoid blocking
            threading.Thread(target=self.retrain_with_feedback, daemon=True).start()
    
    def retrain_with_feedback(self):
        """Retrain the model incorporating user feedback"""
        try:
            print("Starting model retraining with user feedback...")
            
            # Load original dataset
            df = self.load_and_prepare_data('WELFake_Dataset.csv')
            
            # Add feedback data to training set
            unprocessed_feedback = [f for f in self.feedback_data if not f.get('used_for_training', False)]
            
            if unprocessed_feedback:
                feedback_df = pd.DataFrame([
                    {
                        'processed_text': f['processed_text'],
                        'label': 1 if f['actual_label'].lower() == 'real' else 0
                    }
                    for f in unprocessed_feedback
                    if f['processed_text'].strip()  # Only include non-empty processed text
                ])
                
                if not feedback_df.empty:
                    # Combine original data with feedback data
                    combined_df = pd.concat([
                        df[['processed_text', 'label']],
                        feedback_df
                    ], ignore_index=True)
                    
                    print(f"Training with {len(df)} original samples + {len(feedback_df)} feedback samples")
                    
                    # Retrain the model
                    old_accuracy = self.accuracy
                    new_accuracy = self.train_best_model(combined_df)
                    
                    # Save the updated model
                    model_data = {
                        'model': self.model,
                        'accuracy': new_accuracy,
                        'stemmer': self.stemmer,
                        'stop_words': self.stop_words,
                        'training_samples': len(combined_df),
                        'feedback_samples': len(feedback_df),
                        'last_retrain': datetime.now().isoformat()
                    }
                    joblib.dump(model_data, 'fake_news_model.pkl')
                    
                    # Mark feedback as used for training
                    for feedback in unprocessed_feedback:
                        feedback['used_for_training'] = True
                        feedback['training_date'] = datetime.now().isoformat()
                    
                    self.save_feedback_data()
                    
                    print(f"Model retrained successfully!")
                    print(f"Previous accuracy: {old_accuracy:.4f}")
                    print(f"New accuracy: {new_accuracy:.4f}")
                    print(f"Improvement: {(new_accuracy - old_accuracy):.4f}")
                    
        except Exception as e:
            print(f"Error during retraining: {str(e)}")
    
    def get_feedback_stats(self):
        """Get statistics about user feedback"""
        if not self.feedback_data:
            return {
                'total_feedback': 0,
                'used_for_training': 0,
                'pending_training': 0,
                'accuracy_improvement': None
            }
        
        used_count = len([f for f in self.feedback_data if f.get('used_for_training', False)])
        pending_count = len(self.feedback_data) - used_count
        
        return {
            'total_feedback': len(self.feedback_data),
            'used_for_training': used_count,
            'pending_training': pending_count,
            'retrain_threshold': self.retrain_threshold,
            'needs_retraining': pending_count >= self.retrain_threshold
        }
    
    def verify_lfs_file(self, filepath):
        """Verify if a file is properly downloaded from Git LFS"""
        try:
            if not os.path.exists(filepath):
                return False, f"File {filepath} does not exist"
            
            # Check file size - LFS pointer files are very small (~100 bytes)
            file_size = os.path.getsize(filepath)
            if file_size < 1000:  # Less than 1KB might be an LFS pointer
                print(f"Warning: {filepath} is very small ({file_size} bytes), might be an LFS pointer file")
                return False, f"File appears to be an LFS pointer (size: {file_size} bytes)"
            
            # Try to read the beginning of the file to check for LFS pointer content
            with open(filepath, 'rb') as f:
                first_bytes = f.read(100)
                if b'version https://git-lfs.github.com/spec/' in first_bytes:
                    return False, "File is a Git LFS pointer, not the actual file"
            
            print(f"✓ {filepath} verified as proper file (size: {file_size} bytes)")
            return True, "File verified successfully"
            
        except Exception as e:
            return False, f"Error verifying file: {str(e)}"

    def load_model(self, filepath='fake_news_model.pkl'):
        """Load a pre-trained model from disk with Git LFS verification"""
        try:
            print(f"Attempting to load model from: {filepath}")
            
            # First verify the file is properly downloaded from Git LFS
            is_valid, message = self.verify_lfs_file(filepath)
            if not is_valid:
                print(f"LFS verification failed: {message}")
                
                # Try alternative locations for the model file
                alternative_paths = [
                    os.path.join(os.getcwd(), filepath),
                    os.path.join(os.path.dirname(__file__), filepath),
                    os.path.join('/tmp', filepath),  # Sometimes files might be in tmp during deployment
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        print(f"Trying alternative path: {alt_path}")
                        is_valid, message = self.verify_lfs_file(alt_path)
                        if is_valid:
                            filepath = alt_path
                            break
                
                if not is_valid:
                    print(f"Model file '{filepath}' not found or not properly downloaded from Git LFS.")
                    print("This might be due to:")
                    print("1. Git LFS not being properly configured")
                    print("2. Model files not being pulled during deployment")
                    print("3. Insufficient permissions to access the files")
                    return False
            
            print(f"Loading model from verified file: {filepath}")
            model_data = joblib.load(filepath)
            
            # Validate model data structure
            required_keys = ['model', 'stemmer', 'stop_words', 'accuracy']
            missing_keys = [key for key in required_keys if key not in model_data]
            if missing_keys:
                print(f"Warning: Model file missing keys: {missing_keys}")
                return False
            
            self.model = model_data['model']
            self.stemmer = model_data['stemmer']
            self.stop_words = model_data['stop_words']
            self.accuracy = model_data['accuracy']
            self.is_trained = True
            
            # Load additional metadata if available
            training_samples = model_data.get('training_samples', 'Unknown')
            feedback_samples = model_data.get('feedback_samples', 0)
            last_retrain = model_data.get('last_retrain', 'Unknown')
            
            print(f"✓ Model loaded successfully with accuracy: {self.accuracy:.4f}")
            print(f"  Training samples: {training_samples}, Feedback samples: {feedback_samples}")
            if last_retrain != 'Unknown':
                print(f"  Last retrained: {last_retrain}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            if "No such file or directory" in str(e):
                print("The model file was not found. This might indicate a Git LFS issue.")
            elif "pickle" in str(e).lower() or "joblib" in str(e).lower():
                print("The model file appears to be corrupted or incomplete.")
            return False
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def load_and_prepare_data(self, filepath):
        """Load and prepare the dataset with Git LFS verification"""
        print(f"Loading dataset from: {filepath}")
        
        # Verify LFS file before loading
        is_valid, message = self.verify_lfs_file(filepath)
        if not is_valid:
            print(f"Dataset LFS verification failed: {message}")
            
            # Try alternative locations
            alternative_paths = [
                os.path.join(os.getcwd(), filepath),
                os.path.join(os.path.dirname(__file__), filepath),
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"Trying alternative dataset path: {alt_path}")
                    is_valid, message = self.verify_lfs_file(alt_path)
                    if is_valid:
                        filepath = alt_path
                        break
            
            if not is_valid:
                raise FileNotFoundError(f"Dataset file '{filepath}' not found or not properly downloaded from Git LFS")
        
        print(f"Loading dataset from verified file: {filepath}")
        df = pd.read_csv(filepath)
        
        print(f"✓ Dataset loaded successfully: {len(df)} rows")
        
        # Handle missing values
        df['title'] = df['title'].fillna('')
        df['text'] = df['text'].fillna('')
        
        # Combine title and text
        df['combined_text'] = df['title'] + ' ' + df['text']
        
        # Preprocess the combined text
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        print(f"✓ Dataset preprocessed: {len(df)} valid rows after cleaning")
        return df
    
    def train_best_model(self, df):
        """Train and select the best model"""
        X = df['processed_text']
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Naive Bayes': MultinomialNB()
        }
        
        best_accuracy = 0
        best_model = None
        
        for name, model in models.items():
            # Create pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = pipeline
        
        self.model = best_model
        self.is_trained = True
        print(f"Best model selected with accuracy: {best_accuracy:.4f}")
        
        return best_accuracy
    
    def predict(self, text):
        """Predict if a news article is fake or real"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet!")
        
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return {
                'prediction': 'Unknown',
                'confidence': 0.5,
                'probabilities': {'Fake': 0.5, 'Real': 0.5},
                'error': 'Text is empty after preprocessing'
            }
        
        try:
            prediction = self.model.predict([processed_text])[0]
            probability = self.model.predict_proba([processed_text])[0]
            
            return {
                'prediction': 'Real' if prediction == 1 else 'Fake',
                'confidence': float(max(probability)),
                'probabilities': {
                    'Fake': float(probability[0]),
                    'Real': float(probability[1])
                }
            }
        except Exception as e:
            return {
                'prediction': 'Unknown',
                'confidence': 0.5,
                'probabilities': {'Fake': 0.5, 'Real': 0.5},
                'error': str(e)
            }

class PoliticalNewsDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False
        self.accuracy = None
        self.political_categories = {
            'POLITICS', 'U.S. NEWS', 'WORLD NEWS', 'CRIME'
        }
        self.political_keywords = {
            'government', 'politics', 'political', 'election', 'vote', 'voting', 'candidate',
            'president', 'senator', 'congress', 'parliament', 'minister', 'governor',
            'democrat', 'republican', 'party', 'campaign', 'policy', 'legislation',
            'bill', 'law', 'constitution', 'supreme court', 'federal', 'state',
            'mayor', 'city council', 'constituency', 'ballot', 'primary', 'debate',
            'administration', 'cabinet', 'house', 'senate', 'representative',
            'diplomatic', 'foreign policy', 'domestic policy', 'budget', 'tax',
            'reform', 'regulation', 'executive', 'judicial', 'legislative',
            'courthouse', 'trial', 'lawsuit', 'justice', 'attorney general'
        }
    
    def preprocess_text(self, text):
        """Advanced text preprocessing for political news classification"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep apostrophes for contractions
        text = re.sub(r'[^a-zA-Z\s\']', '', text)
        
        # Handle contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords and apply lemmatization
        processed_words = []
        for word in words:
            if word not in self.stop_words and len(word) > 2:
                lemmatized = self.lemmatizer.lemmatize(word)
                processed_words.append(lemmatized)
        
        return ' '.join(processed_words)
    
    def extract_political_features(self, text):
        """Extract political-specific features from text"""
        features = {}
        text_lower = text.lower()
        
        # Count political keywords
        political_count = sum(1 for keyword in self.political_keywords if keyword in text_lower)
        features['political_keyword_count'] = political_count
        features['political_keyword_ratio'] = political_count / max(len(text.split()), 1)
        
        # Check for political entities
        political_patterns = [
            r'\b(?:president|senator|governor|mayor|minister)\s+\w+',
            r'\b(?:mr\.|ms\.|mrs\.)\s+\w+',
            r'\b\w+\s+(?:administration|campaign|party)',
            r'\b(?:white house|congress|parliament|senate|house of representatives)'
        ]
        
        political_entities = 0
        for pattern in political_patterns:
            political_entities += len(re.findall(pattern, text_lower))
        
        features['political_entities'] = political_entities
        
        return features
    
    def predict(self, text):
        """Predict if text is political news"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet!")
        
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return {
                'prediction': 'Unknown',
                'confidence': 0.5,
                'probabilities': {'Non-Political': 0.5, 'Political': 0.5},
                'reasoning': 'Unable to analyze: Text is empty after preprocessing',
                'political_features': {},
                'error': 'Text is empty after preprocessing'
            }
        
        try:
            prediction = self.model.predict([processed_text])[0]
            probability = self.model.predict_proba([processed_text])[0]
            
            prediction_label = 'Political' if prediction == 1 else 'Non-Political'
            
            # Extract reasoning and features
            reasoning = self.extract_reasoning(text, prediction)
            political_features = self.extract_political_features(text)
            
            return {
                'prediction': prediction_label,
                'confidence': float(max(probability)),
                'probabilities': {
                    'Non-Political': float(probability[0]),
                    'Political': float(probability[1])
                },
                'reasoning': reasoning,
                'political_features': political_features
            }
        except Exception as e:
            return {
                'prediction': 'Unknown',
                'confidence': 0.5,
                'probabilities': {'Non-Political': 0.5, 'Political': 0.5},
                'reasoning': f'Error during prediction: {str(e)}',
                'political_features': {},
                'error': str(e)
            }
    
    def extract_reasoning(self, text, prediction):
        """Extract reasoning for the classification"""
        features = self.extract_political_features(text)
        text_lower = text.lower()
        
        reasoning_parts = []
        
        if prediction == 1:  # Political
            reasoning_parts.append(f"✓ Classified as POLITICAL news")
            
            if features['political_keyword_count'] > 0:
                reasoning_parts.append(f"• Found {features['political_keyword_count']} political keywords")
            
            if features['political_entities'] > 0:
                reasoning_parts.append(f"• Detected {features['political_entities']} political entities/references")
            
            # Check for specific political indicators
            if any(term in text_lower for term in ['election', 'vote', 'campaign']):
                reasoning_parts.append("• Contains election/voting related content")
            
            if any(term in text_lower for term in ['government', 'congress', 'senate', 'president']):
                reasoning_parts.append("• References government institutions or officials")
            
            if any(term in text_lower for term in ['policy', 'legislation', 'bill', 'law']):
                reasoning_parts.append("• Discusses policy or legislative matters")
        
        else:  # Non-Political
            reasoning_parts.append(f"✓ Classified as NON-POLITICAL news")
            
            if features['political_keyword_count'] == 0:
                reasoning_parts.append("• No political keywords detected")
            else:
                reasoning_parts.append(f"• Limited political content ({features['political_keyword_count']} keywords)")
            
            if features['political_entities'] == 0:
                reasoning_parts.append("• No political entities or references found")
            
            # Check for non-political indicators
            non_political_terms = ['sports', 'entertainment', 'technology', 'science', 'health', 'business']
            found_terms = [term for term in non_political_terms if term in text_lower]
            if found_terms:
                reasoning_parts.append(f"• Contains {', '.join(found_terms)} related content")
        
        return '\n'.join(reasoning_parts)
    
    def verify_lfs_file(self, filepath):
        """Verify if a file is properly downloaded from Git LFS"""
        try:
            if not os.path.exists(filepath):
                return False, f"File {filepath} does not exist"
            
            # Check file size - LFS pointer files are very small (~100 bytes)
            file_size = os.path.getsize(filepath)
            if file_size < 1000:  # Less than 1KB might be an LFS pointer
                print(f"Warning: {filepath} is very small ({file_size} bytes), might be an LFS pointer file")
                return False, f"File appears to be an LFS pointer (size: {file_size} bytes)"
            
            # Try to read the beginning of the file to check for LFS pointer content
            with open(filepath, 'rb') as f:
                first_bytes = f.read(100)
                if b'version https://git-lfs.github.com/spec/' in first_bytes:
                    return False, "File is a Git LFS pointer, not the actual file"
            
            print(f"✓ {filepath} verified as proper file (size: {file_size} bytes)")
            return True, "File verified successfully"
            
        except Exception as e:
            return False, f"Error verifying file: {str(e)}"

    def load_model(self, filepath='political_news_classifier.pkl'):
        """Load a pre-trained model with Git LFS verification"""
        try:
            print(f"Attempting to load political model from: {filepath}")
            
            # First verify the file is properly downloaded from Git LFS
            is_valid, message = self.verify_lfs_file(filepath)
            if not is_valid:
                print(f"LFS verification failed: {message}")
                
                # Try alternative locations for the model file
                alternative_paths = [
                    os.path.join(os.getcwd(), filepath),
                    os.path.join(os.path.dirname(__file__), filepath),
                    os.path.join('/tmp', filepath),
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        print(f"Trying alternative path: {alt_path}")
                        is_valid, message = self.verify_lfs_file(alt_path)
                        if is_valid:
                            filepath = alt_path
                            break
                
                if not is_valid:
                    print(f"Political news model file '{filepath}' not found or not properly downloaded from Git LFS.")
                    return False
            
            print(f"Loading political model from verified file: {filepath}")
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.stemmer = model_data.get('stemmer', PorterStemmer())
            self.lemmatizer = model_data.get('lemmatizer', WordNetLemmatizer())
            self.stop_words = model_data.get('stop_words', set(stopwords.words('english')))
            self.accuracy = model_data.get('accuracy', None)
            self.is_trained = True
            
            print(f"✓ Political news model loaded successfully")
            if self.accuracy:
                print(f"  Model accuracy: {self.accuracy:.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error loading political news model: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            return False

class NewsWebsiteCrawler:
    def __init__(self, url_classifier=None):
        """Initialize the crawler with URL classifier for intelligent filtering"""
        self.url_classifier = url_classifier or url_news_classifier
        self.confidence_threshold = 0.7  # Threshold for URL classification
        self.enable_filtering = True  # Can be disabled for backward compatibility
        self.classification_stats = {
            'total_urls': 0,
            'classified_as_news': 0,
            'classified_as_non_news': 0,
            'classification_errors': 0
        }
    
    def is_news_link(self, href, link_text=None, link_element=None, use_classifier=True):
        """Enhanced news link detection using URL classifier"""
        if not href:
            return False
            
        if not self.enable_filtering or not use_classifier:
            return True
            
        try:
            # Use URL classifier to predict if this is a news article URL
            prediction_result = self.url_classifier.predict_with_confidence(href)
            
            is_news = prediction_result.get('is_news_article', False)
            confidence = prediction_result.get('confidence', 0.0)
            
            # Update classification stats
            self.classification_stats['total_urls'] += 1
            if is_news:
                self.classification_stats['classified_as_news'] += 1
            else:
                self.classification_stats['classified_as_non_news'] += 1
            
            # Return True if classified as news and confidence is above threshold
            return is_news and confidence >= self.confidence_threshold
            
        except Exception as e:
            print(f"⚠️ URL classification error for {href}: {e}")
            self.classification_stats['classification_errors'] += 1
            # Fallback to basic URL pattern matching
            return self._basic_news_url_check(href, link_text)
    
    def _basic_news_url_check(self, href, link_text=None):
        """Fallback basic news URL detection using simple patterns"""
        href_lower = href.lower()
        
        # Basic news patterns
        news_patterns = [
            r'/news/', r'/article/', r'/story/', r'/post/', r'/blog/',
            r'/breaking/', r'/latest/', r'/update/', r'/report/', r'/press/',
            r'\d{4}/\d{2}/\d{2}/', r'\d{4}-\d{2}-\d{2}',
            r'/politics/', r'/sports/', r'/business/', r'/tech/', r'/health/',
            r'/entertainment/', r'/opinion/', r'/world/', r'/local/'
        ]
        
        # Non-news patterns (should be excluded)
        non_news_patterns = [
            r'/about/', r'/contact/', r'/privacy/', r'/terms/', r'/help/',
            r'/login/', r'/register/', r'/profile/', r'/settings/', r'/account/',
            r'/shop/', r'/buy/', r'/cart/', r'/checkout/', r'/product/',
            r'/search/', r'/category/', r'/tag/', r'/archive/', r'/sitemap/',
            r'/api/', r'/admin/', r'/dashboard/', r'/upload/', r'/download/'
        ]
        
        # Check for non-news patterns first (exclude these)
        import re
        for pattern in non_news_patterns:
            if re.search(pattern, href_lower):
                return False
        
        # Check for news patterns
        for pattern in news_patterns:
            if re.search(pattern, href_lower):
                return True
        
        # Check link text for news indicators
        if link_text:
            text_lower = link_text.lower()
            news_keywords = ['article', 'story', 'news', 'report', 'breaking', 'latest', 'update']
            if any(keyword in text_lower for keyword in news_keywords):
                return True
        
        # Default to True for unknown patterns (conservative approach)
        return True
    
    def extract_article_links(self, url, enable_filtering=True):
        """Enhanced method to extract and filter news article links from a website using URL classifier"""
        try:
            # Reset classification stats for this crawl
            self.classification_stats = {
                'total_urls': 0,
                'classified_as_news': 0,
                'classified_as_non_news': 0,
                'classification_errors': 0
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            
            all_links = []
            news_articles = []
            non_news_links = []
            seen_urls = set()
            
            print(f"🔍 Crawling website: {url}")
            print(f"📄 Website title: {soup.title.string if soup.title else 'No title found'}")
            print(f"🤖 URL classification {'enabled' if enable_filtering else 'disabled'}")
            
            # Extract all links first
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if not href:
                    continue
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    href = urljoin(base_url, href)
                elif not href.startswith(('http://', 'https://')):
                    href = urljoin(url, href)
                
                # Skip duplicates and invalid URLs
                if not href.startswith(('http://', 'https://')) or href in seen_urls:
                    continue
                
                seen_urls.add(href)
                
                # Get link text for additional context
                link_text = link.get_text(strip=True)
                
                # Store all links
                all_links.append({
                    'url': href,
                    'text': link_text,
                    'title': link.get('title', '')
                })
            
            print(f"🔗 Found {len(all_links)} total URLs")
            
            # Apply URL classification filtering if enabled
            if enable_filtering and self.url_classifier:
                print(f"🎯 Applying URL classification filtering...")
                
                # Batch classify URLs for efficiency
                urls_to_classify = [link['url'] for link in all_links]
                
                try:
                    # Use batch prediction for better performance
                    classification_results = self.url_classifier.predict_batch(urls_to_classify)
                    
                    for i, (link_info, classification) in enumerate(zip(all_links, classification_results)):
                        if classification.get('error'):
                            print(f"⚠️ Classification error for {link_info['url']}: {classification['error']}")
                            # Use fallback classification
                            if self._basic_news_url_check(link_info['url'], link_info['text']):
                                news_articles.append(link_info)
                            else:
                                non_news_links.append(link_info)
                        else:
                            is_news = classification.get('is_news_article', False)
                            confidence = classification.get('confidence', 0.0)
                            
                            # Add classification info to link
                            link_info['classification'] = {
                                'is_news_article': is_news,
                                'confidence': confidence,
                                'prediction': classification.get('prediction', False),
                                'probability_news': classification.get('probability_news', 0.0),
                                'probability_not_news': classification.get('probability_not_news', 0.0)
                            }
                            
                            # Filter based on classification and confidence
                            if is_news and confidence >= self.confidence_threshold:
                                news_articles.append(link_info)
                            else:
                                non_news_links.append(link_info)
                
                except Exception as e:
                    print(f"⚠️ Batch classification failed: {e}")
                    print("🔄 Falling back to individual classification...")
                    
                    # Fallback to individual classification
                    for link_info in all_links:
                        if self.is_news_link(link_info['url'], link_info['text'], use_classifier=True):
                            news_articles.append(link_info)
                        else:
                            non_news_links.append(link_info)
                
                # Update classification stats
                self.classification_stats['total_urls'] = len(all_links)
                self.classification_stats['classified_as_news'] = len(news_articles)
                self.classification_stats['classified_as_non_news'] = len(non_news_links)
                
                print(f"📊 Classification results:")
                print(f"   📰 News articles: {len(news_articles)}")
                print(f"   🚫 Non-news links: {len(non_news_links)}")
                print(f"   ⚡ Accuracy rate: {((len(news_articles) + len(non_news_links)) / len(all_links) * 100):.1f}%")
                
                # Return filtered news articles with classification info
                return {
                    'success': True,
                    'articles': news_articles,
                    'total_found': len(news_articles),
                    'total_candidates': len(all_links),
                    'filtered_out': len(non_news_links),
                    'website_title': soup.title.string if soup.title else urlparse(url).netloc,
                    'classification_method': f'URL Classifier (threshold: {self.confidence_threshold})',
                    'classification_stats': self.classification_stats.copy(),
                    'filtering_enabled': True
                }
            
            else:
                # No filtering - return all links (backward compatibility)
                print(f"🔄 URL filtering disabled - returning all {len(all_links)} URLs")
                
                # Convert to simple URL list for backward compatibility
                simple_urls = [link['url'] for link in all_links]
                
                return {
                    'success': True,
                    'articles': simple_urls,  # Simple list of URLs for backward compatibility
                    'total_found': len(simple_urls),
                    'total_candidates': len(all_links),
                    'website_title': soup.title.string if soup.title else urlparse(url).netloc,
                    'classification_method': 'No filtering - All URLs returned',
                    'filtering_enabled': False
                }
            
        except requests.RequestException as e:
            print(f"Network error crawling {url}: {str(e)}")
            return {
                'success': False,
                'error': f'Network error: {str(e)}',
                'articles': []
            }
        except Exception as e:
            print(f"Parsing error crawling {url}: {str(e)}")
            return {
                'success': False,
                'error': f'Parsing error: {str(e)}',
                'articles': []
            }
    
    def extract_enhanced_title(self, link_element, fallback_text):
        """Enhanced title extraction with multiple fallbacks"""
        title = fallback_text or ''
        
        if link_element:
            # Try different attributes for title
            title = (link_element.get('title') or 
                    link_element.get('aria-label') or 
                    link_element.get_text(strip=True) or 
                    title)
        
        return title
    
    def calculate_link_confidence(self, href, title, link_element, selector_used, classification_result=None):
        """Calculate confidence score for a link being a news article"""
        if classification_result:
            # Use URL classifier confidence if available
            return classification_result.get('confidence', 0.5)
        
        # Fallback confidence calculation based on URL patterns
        confidence = 0.5  # Base confidence
        
        href_lower = href.lower()
        
        # Boost confidence for known news patterns
        news_indicators = [
            ('news', 0.3), ('article', 0.2), ('story', 0.2), ('post', 0.1),
            ('breaking', 0.2), ('latest', 0.1), ('update', 0.1), ('report', 0.1)
        ]
        
        for indicator, boost in news_indicators:
            if indicator in href_lower:
                confidence += boost
        
        # Boost for date patterns
        import re
        if re.search(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', href):
            confidence += 0.2
        
        # Reduce confidence for non-news patterns
        non_news_indicators = ['login', 'register', 'about', 'contact', 'shop', 'cart']
        for indicator in non_news_indicators:
            if indicator in href_lower:
                confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def set_filtering_mode(self, enable_filtering=True, confidence_threshold=0.6):
        """Configure URL filtering settings"""
        self.enable_filtering = enable_filtering
        self.confidence_threshold = confidence_threshold
        print(f"🎯 URL filtering {'enabled' if enable_filtering else 'disabled'}")
        if enable_filtering:
            print(f"📊 Confidence threshold set to: {confidence_threshold}")
    
    def get_classification_stats(self):
        """Get current classification statistics"""
        return self.classification_stats.copy()
    
    def reset_classification_stats(self):
        """Reset classification statistics"""
        self.classification_stats = {
            'total_urls': 0,
            'classified_as_news': 0,
            'classified_as_non_news': 0,
            'classification_errors': 0
        }
    
    def analyze_articles_batch(self, article_urls, analysis_type='both'):
        """Analyze multiple articles in parallel - handles both URL strings and article objects"""
        results = []
        
        def analyze_single_article(article_info):
            try:
                # Handle both URL strings and article objects with classification info
                if isinstance(article_info, str):
                    # Simple URL string (backward compatibility)
                    article_url = article_info
                    article_title = ''
                    classification_info = None
                elif isinstance(article_info, dict):
                    # Article object with metadata
                    article_url = article_info.get('url', '')
                    article_title = article_info.get('text', '') or article_info.get('title', '')
                    classification_info = article_info.get('classification')
                else:
                    return {
                        'url': str(article_info),
                        'title': '',
                        'error': 'Invalid article format',
                        'status': 'failed'
                    }
                
                if not article_url:
                    return {
                        'url': '',
                        'title': article_title,
                        'error': 'No URL provided',
                        'status': 'failed'
                    }
                
                # Extract content from article
                content_result = extract_article_content(article_url)
                
                if 'error' in content_result:
                    return {
                        'url': article_url,
                        'title': article_title,
                        'error': content_result['error'],
                        'status': 'failed',
                        'classification_info': classification_info
                    }
                
                # Analyze the content
                text_to_analyze = content_result['combined']
                if not text_to_analyze.strip():
                    return {
                        'url': article_url,
                        'title': article_title,
                        'error': 'No content extracted',
                        'status': 'failed',
                        'classification_info': classification_info
                    }
                
                analysis_result = {
                    'url': article_url,
                    'title': article_title,
                    'extracted_title': content_result.get('title', ''),
                    'content_preview': text_to_analyze[:200] + '...' if len(text_to_analyze) > 200 else text_to_analyze,
                    'status': 'success',
                    'classification_info': classification_info
                }
                
                # Automatically index the article in Philippine news database (background)
                try:
                    index_result = philippine_search_index.index_article(article_url, force_reindex=False)
                    analysis_result['indexing_status'] = index_result['status']
                    if index_result['status'] == 'success':
                        analysis_result['relevance_score'] = index_result.get('relevance_score', 0)
                        analysis_result['locations_found'] = index_result.get('locations', [])
                        analysis_result['government_entities_found'] = index_result.get('government_entities', [])
                except Exception as e:
                    analysis_result['indexing_status'] = 'failed'
                    analysis_result['indexing_error'] = str(e)
                
                # Perform fake news detection
                if analysis_type in ['fake_news', 'both']:
                    try:
                        fake_result = detector.predict(text_to_analyze)
                        analysis_result['fake_news'] = fake_result
                    except Exception as e:
                        analysis_result['fake_news'] = {'error': str(e)}
                
                # Perform political classification
                if analysis_type in ['political', 'both']:
                    try:
                        political_result = political_detector.predict(text_to_analyze)
                        analysis_result['political_classification'] = political_result
                    except Exception as e:
                        analysis_result['political_classification'] = {'error': str(e)}
                
                return analysis_result
                
            except Exception as e:
                # Handle the article info properly in error case
                if isinstance(article_info, str):
                    url, title = article_info, ''
                elif isinstance(article_info, dict):
                    url = article_info.get('url', str(article_info))
                    title = article_info.get('text', '') or article_info.get('title', '')
                else:
                    url, title = str(article_info), ''
                
                return {
                    'url': url,
                    'title': title,
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Use ThreadPoolExecutor for parallel processing
        # Dynamic worker count based on the number of articles
        max_workers = min(20, len(article_urls))  # Dynamic worker count, max 20
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_article = {executor.submit(analyze_single_article, article): article 
                               for article in article_urls}
            
            for future in concurrent.futures.as_completed(future_to_article):
                try:
                    result = future.result(timeout=30)  # 30 second timeout per article
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    article = future_to_article[future]
                    # Handle timeout case properly
                    if isinstance(article, str):
                        url, title = article, ''
                    elif isinstance(article, dict):
                        url = article.get('url', str(article))
                        title = article.get('text', '') or article.get('title', '')
                    else:
                        url, title = str(article), ''
                    
                    results.append({
                        'url': url,
                        'title': title,
                        'error': 'Analysis timeout',
                        'status': 'timeout'
                    })
                except Exception as e:
                    article = future_to_article[future]
                    # Handle exception case properly
                    if isinstance(article, str):
                        url, title = article, ''
                    elif isinstance(article, dict):
                        url = article.get('url', str(article))
                        title = article.get('text', '') or article.get('title', '')
                    else:
                        url, title = str(article), ''
                    
                    results.append({
                        'url': url,
                        'title': title,
                        'error': str(e),
                        'status': 'failed'
                    })
        
        return results

# Initialize the detectors and crawler
detector = FakeNewsDetector()
political_detector = PoliticalNewsDetector()
url_news_classifier = URLNewsClassifier()
news_crawler = NewsWebsiteCrawler(url_classifier=url_news_classifier)
philippine_search_index = PhilippineNewsSearchIndex()

def extract_article_content(url):
    """Extract article content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find article content
        article_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.content',
            '.story-body',
            'main',
            '.entry-content'
        ]
        
        content = ""
        title = ""
        
        # Get title
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Try to find article content
        for selector in article_selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text().strip() for elem in elements])
                break
        
        # If no specific article content found, get all paragraphs
        if not content:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs])
        
        return {
            'title': title,
            'content': content,
            'combined': f"{title} {content}".strip()
        }
    
    except Exception as e:
        return {'error': f"Failed to extract content: {str(e)}"}

@app.route('/')
def index():
    return render_template('news_analysis.html')

@app.route('/philippine-search')
def philippine_search():
    return render_template('philippine_search.html')

@app.route('/url-classifier')
def url_classifier():
    return render_template('url_classifier.html')

@app.route('/analyze-url')
def analyze_url():
    """Route for analyzing a URL directly (for external links)"""
    url = request.args.get('url')
    if not url:
        return redirect(url_for('index'))
    return render_template('news_analysis.html', prefill_url=url)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not detector.is_trained:
            return jsonify({
                'success': False,
                'error': 'Fake news model not trained yet. Please wait for training to complete.'
            }), 500
        
        input_type = data.get('type', 'text')
        analysis_type = data.get('analysis_type', 'fake_news')  # 'fake_news', 'political', or 'both'
        
        if input_type == 'text':
            text = data.get('text', '').strip()
            if not text:
                return jsonify({
                    'success': False,
                    'error': 'No text provided'
                }), 400
            
            # Perform fake news detection
            fake_result = detector.predict(text)
            result = {'fake_news': fake_result}
            
            # Perform political classification if requested
            if analysis_type in ['political', 'both'] and political_detector.is_trained:
                political_result = political_detector.predict(text)
                result['political_classification'] = political_result
            elif analysis_type in ['political', 'both']:
                result['political_classification'] = {
                    'error': 'Political classification model not available'
                }
            
            return jsonify({
                'success': True,
                'data': result
            })
        
        elif input_type == 'url':
            url = data.get('url', '').strip()
            if not url:
                return jsonify({
                    'success': False,
                    'error': 'No URL provided'
                }), 400
            
            # Extract content from URL
            article_data = extract_article_content(url)
            
            if 'error' in article_data:
                return jsonify({
                    'success': False,
                    'error': article_data['error']
                }), 400
            
            combined_text = article_data['combined']
            if not combined_text.strip():
                return jsonify({
                    'success': False,
                    'error': 'No content could be extracted from the URL'
                }), 400
            
            # Automatically index the URL in the Philippine news search engine and database
            indexing_result = None
            try:
                print(f"🔍 Auto-indexing URL for search engine: {url}")
                indexing_result = philippine_search_index.index_article(url, force_reindex=False)
                print(f"📚 Indexing result: {indexing_result['status']}")
                
                if indexing_result['status'] == 'success':
                    print(f"✅ Successfully indexed article with relevance score: {indexing_result.get('relevance_score', 0):.3f}")
                    if indexing_result.get('locations'):
                        print(f"📍 Found locations: {', '.join(indexing_result['locations'])}")
                    if indexing_result.get('government_entities'):
                        print(f"🏛️ Found government entities: {', '.join(indexing_result['government_entities'])}")
                elif indexing_result['status'] == 'skipped':
                    print(f"⏭️ Article skipped from indexing: {indexing_result.get('message', 'Low Philippine relevance')}")
                elif indexing_result['status'] == 'already_indexed':
                    print(f"📋 Article already in search index")
                else:
                    print(f"❌ Indexing failed: {indexing_result.get('message', 'Unknown error')}")
                    
            except Exception as indexing_error:
                print(f"⚠️ Error during auto-indexing: {str(indexing_error)}")
                # Don't fail the main analysis if indexing fails
                indexing_result = {'status': 'error', 'message': str(indexing_error)}
            
            # Perform fake news detection
            fake_result = detector.predict(combined_text)
            result = {
                'fake_news': fake_result,
                'extracted_content': {
                    'title': article_data['title'],
                    'content_preview': combined_text[:500] + '...' if len(combined_text) > 500 else combined_text,
                    'combined': combined_text,
                    'word_count': len(combined_text.split()) if combined_text else 0
                },
                'indexing_result': indexing_result  # Include indexing status in response
            }
            
            # Perform political classification if requested
            if analysis_type in ['political', 'both'] and political_detector.is_trained:
                political_result = political_detector.predict(combined_text)
                result['political_classification'] = political_result
            elif analysis_type in ['political', 'both']:
                result['political_classification'] = {
                    'error': 'Political classification model not available'
                }
            
            # Handle retraining trigger if requested
            trigger_retraining = data.get('trigger_retraining', False)
            if trigger_retraining:
                try:
                    print(f"🔄 Model retraining requested by user")
                    
                    # Check if there's enough feedback data for retraining
                    feedback_stats = detector.get_feedback_stats()
                    
                    if feedback_stats['pending_training'] > 0:
                        print(f"📊 Initiating retraining with {feedback_stats['pending_training']} feedback entries")
                        
                        # Start retraining in a background thread to avoid blocking the response
                        import threading
                        def background_retrain():
                            try:
                                print("🚀 Starting background model retraining...")
                                detector.retrain_from_feedback()
                                print("✅ Model retraining completed successfully")
                            except Exception as retrain_error:
                                print(f"❌ Model retraining failed: {str(retrain_error)}")
                        
                        retraining_thread = threading.Thread(target=background_retrain)
                        retraining_thread.daemon = True
                        retraining_thread.start()
                        
                        result['retraining_triggered'] = {
                            'status': 'initiated',
                            'feedback_count': feedback_stats['pending_training'],
                            'message': 'Model retraining started in background. This may take several minutes.'
                        }
                        print("✅ Retraining thread started successfully")
                    else:
                        result['retraining_triggered'] = {
                            'status': 'skipped',
                            'feedback_count': 0,
                            'message': 'No pending feedback data available for retraining.'
                        }
                        print("⏭️ Retraining skipped - no pending feedback data")
                        
                except Exception as retrain_error:
                    print(f"⚠️ Error initiating retraining: {str(retrain_error)}")
                    result['retraining_triggered'] = {
                        'status': 'error',
                        'message': f'Failed to initiate retraining: {str(retrain_error)}'
                    }
            
            return jsonify({
                'success': True,
                'data': result
            })
        
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid input type'
            }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/crawl-website', methods=['POST'])
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

@app.route('/analyze-website', methods=['POST'])
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
            'filtering_enabled': enable_filtering,
            'confidence_threshold': confidence_threshold if enable_filtering else None
        }
        
        # Add filtering statistics if available
        if enable_filtering and 'classification_stats' in crawl_result:
            summary.update({
                'total_candidates': crawl_result['total_candidates'],
                'filtered_out': crawl_result.get('filtered_out', 0),
                'classification_stats': crawl_result['classification_stats'],
                'classification_method': crawl_result.get('classification_method', 'URL Classifier')
            })
        
        # Add aggregated statistics for successful analyses
        if successful_analyses and analysis_type in ['fake_news', 'both']:
            fake_predictions = []
            for result in successful_analyses:
                if 'fake_news' in result and 'error' not in result['fake_news']:
                    fake_predictions.append(result['fake_news']['prediction'])
            
            if fake_predictions:
                fake_count = fake_predictions.count('Fake')
                real_count = fake_predictions.count('Real')
                summary['fake_news_summary'] = {
                    'fake_articles': fake_count,
                    'real_articles': real_count,
                    'fake_percentage': (fake_count / len(fake_predictions)) * 100 if fake_predictions else 0
                }
        
        if successful_analyses and analysis_type in ['political', 'both']:
            political_predictions = []
            for result in successful_analyses:
                if 'political_classification' in result and 'error' not in result['political_classification']:
                    political_predictions.append(result['political_classification']['prediction'])
            
            if political_predictions:
                political_count = political_predictions.count('Political')
                non_political_count = political_predictions.count('Non-Political')
                summary['political_summary'] = {
                    'political_articles': political_count,
                    'non_political_articles': non_political_count,
                    'political_percentage': (political_count / len(political_predictions)) * 100 if political_predictions else 0
                }
        
        return jsonify({
            'success': True,
            'summary': summary,
            'results': analysis_results
        })
    
    except Exception as e:
        return jsonify({'error': f'Website analysis failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint with Git LFS file status"""
    try:
        # Check Git LFS files
        lfs_files = [
            'fake_news_model.pkl',
            'political_news_classifier.pkl', 
            'WELFake_Dataset.csv',
            'News_Category_Dataset_v3.json'
        ]
        
        file_status = {}
        for file_path in lfs_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                file_status[file_path] = {
                    'exists': True,
                    'size': file_size,
                    'is_lfs_pointer': file_size < 1000
                }
            else:
                file_status[file_path] = {
                    'exists': False,
                    'size': 0,
                    'is_lfs_pointer': False
                }
        
        # Check model status
        models_status = {
            'fake_news_detector': {
                'loaded': detector.is_trained,
                'accuracy': detector.accuracy if detector.accuracy else None
            },
            'political_detector': {
                'loaded': political_detector.is_trained,
                'accuracy': political_detector.accuracy if political_detector.accuracy else None
            }
        }
        
        # Overall health
        is_healthy = detector.is_trained or political_detector.is_trained
        
        return jsonify({
            'status': 'healthy' if is_healthy else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'models': models_status,
            'lfs_files': file_status,
            'message': 'All systems operational' if is_healthy else 'Some models unavailable'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

@app.route('/model-status')
def model_status():
    try:
        # Create the expected response structure for the frontend
        status_data = {
            'fake_news': detector.is_trained,
            'political': political_detector.is_trained
        }
        
        # Additional detailed status info (optional)
        status_info = {
            'fake_news_model': {
                'is_trained': detector.is_trained,
                'status': 'Model is ready' if detector.is_trained else 'Model is training...'
            },
            'political_model': {
                'is_trained': political_detector.is_trained,
                'status': 'Model is ready' if political_detector.is_trained else 'Model not loaded'
            }
        }
        
        if detector.is_trained and detector.accuracy:
            status_info['fake_news_model']['accuracy'] = f"{detector.accuracy:.4f}"
            status_info['fake_news_model']['status'] = f"Model ready (Accuracy: {detector.accuracy:.1%})"
        
        if political_detector.is_trained and political_detector.accuracy:
            status_info['political_model']['accuracy'] = f"{political_detector.accuracy:.4f}"
            status_info['political_model']['status'] = f"Model ready (Accuracy: {political_detector.accuracy:.1%})"
        
        # Overall status
        both_ready = detector.is_trained and political_detector.is_trained
        status_info['overall_status'] = 'Both models ready' if both_ready else 'Models loading...'
        status_info['is_trained'] = detector.is_trained  # Keep for backward compatibility
        
        # Add feedback statistics
        try:
            feedback_stats = detector.get_feedback_stats()
            status_info['feedback'] = feedback_stats
        except:
            status_info['feedback'] = {'total': 0, 'correct': 0, 'incorrect': 0}
        
        # Return in the format expected by the frontend
        return jsonify({
            'success': True,
            'status': status_data,
            'details': status_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get model status: {str(e)}',
            'status': {'fake_news': False, 'political': False}
        })

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for model improvement"""
    try:
        data = request.get_json()
        
        text = data.get('text', '').strip()
        predicted_label = data.get('predicted_label', '').strip()
        actual_label = data.get('actual_label', '').strip()
        confidence = data.get('confidence', 0.0)
        user_comment = data.get('comment', '').strip()
        
        # Validate required fields
        if not text or not predicted_label or not actual_label:
            return jsonify({'error': 'Missing required fields: text, predicted_label, actual_label'}), 400
        
        if actual_label.lower() not in ['fake', 'real']:
            return jsonify({'error': 'actual_label must be either "fake" or "real"'}), 400
        
        # Add feedback to the system
        detector.add_feedback(text, predicted_label, actual_label, confidence, user_comment)
        
        # Get updated feedback stats
        feedback_stats = detector.get_feedback_stats()
        
        response = {
            'message': 'Thank you for your feedback! It will help improve the model.',
            'feedback_stats': feedback_stats
        }
        
        if feedback_stats['needs_retraining']:
            response['message'] += f' The model will be retrained soon with {feedback_stats["pending_training"]} new feedback entries.'
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/feedback-stats')
def feedback_stats():
    """Get feedback statistics"""
    try:
        stats = detector.get_feedback_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/trigger-retrain', methods=['POST'])
def trigger_retrain():
    """Manually trigger model retraining"""
    try:
        feedback_stats = detector.get_feedback_stats()
        
        if feedback_stats['pending_training'] == 0:
            return jsonify({'message': 'No new feedback available for retraining.'}), 400
        
        # Start retraining in background
        threading.Thread(target=detector.retrain_with_feedback, daemon=True).start()
        
        return jsonify({
            'message': f'Model retraining started with {feedback_stats["pending_training"]} new feedback entries.',
            'status': 'retraining_started'
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/index-philippine-article', methods=['POST'])
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

@app.route('/search-philippine-news', methods=['GET', 'POST'])
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

@app.route('/philippine-news-analytics')
def philippine_news_analytics():
    """Get analytics and statistics for the Philippine news index"""
    try:
        analytics = philippine_search_index.get_search_analytics()
        return jsonify(analytics)
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/find-similar-content', methods=['POST'])
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

@app.route('/philippine-article/<int:article_id>')
def get_philippine_article(article_id):
    """Get full details of a specific Philippine news article"""
    try:
        article = philippine_search_index.get_article_by_id(article_id)
        
        if not article:
            return jsonify({'error': 'Article not found'}), 404
        
        return jsonify(article)
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/batch-index-philippine-articles', methods=['POST'])
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

@app.route('/crawl-and-index-website', methods=['POST'])
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
        
        # No limit - allow testing system to its limits
        # max_articles = min(max_articles, 50)  # REMOVED
        
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

@app.route('/get-crawl-history')
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

@app.route('/philippine-news-categories')
def get_philippine_news_categories():
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

@app.route('/philippine-news-sources')
def get_philippine_news_sources():
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

@app.route('/classify-url', methods=['POST'])
def classify_url():
    """Classify if a URL is a news article using RL model"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({
                'success': False,
                'error': 'No URL provided'
            }), 400
        
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Get prediction from RL model
        result = url_news_classifier.predict_with_confidence(url)
        
        return jsonify({
            'success': True,
            'url': url,
            'is_news_article': result['is_news_article'],
            'confidence': result['confidence'],
            'probability_news': result['probability_news'],
            'probability_not_news': result['probability_not_news'],
            'model_trained': url_news_classifier.is_trained,
            'heuristic_based': result.get('heuristic_based', False)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/url-classifier-feedback', methods=['POST'])
def submit_url_classifier_feedback():
    """Submit feedback for URL classification to improve the RL model (supports single and bulk feedback)"""
    try:
        data = request.get_json()
        
        # Check if this is bulk feedback
        if 'feedback_batch' in data:
            # Handle bulk feedback
            feedback_list = data.get('feedback_batch', [])
            
            if not feedback_list:
                return jsonify({
                    'success': False,
                    'error': 'feedback_batch is empty'
                }), 400
            
            processed_count = 0
            errors = []
            
            for feedback in feedback_list:
                try:
                    url = feedback.get('url', '').strip()
                    predicted_label = feedback.get('predicted_label')
                    actual_label = feedback.get('actual_label')
                    user_confidence = feedback.get('user_confidence', 1.0)
                    
                    if not url or predicted_label is None or actual_label is None:
                        errors.append(f"Missing required fields for URL: {url}")
                        continue
                    
                    # Add feedback to RL model
                    url_news_classifier.add_feedback(
                        url=url,
                        predicted_label=predicted_label,
                        actual_label=actual_label,
                        user_confidence=user_confidence
                    )
                    processed_count += 1
                    
                except Exception as e:
                    errors.append(f"Error processing feedback for {feedback.get('url', 'unknown')}: {str(e)}")
            
            # Get updated model stats
            model_stats = url_news_classifier.get_model_stats()
            
            return jsonify({
                'success': True,
                'bulk_feedback': True,
                'processed_count': processed_count,
                'total_submitted': len(feedback_list),
                'errors': errors,
                'feedback_count': model_stats['feedback_count'],
                'is_trained': model_stats['is_trained'],
                'message': f'Successfully processed {processed_count} out of {len(feedback_list)} feedback entries'
            })
        
        else:
            # Handle single feedback (existing logic)
            url = data.get('url', '').strip()
            predicted_label = data.get('predicted_label')
            actual_label = data.get('actual_label')
            user_confidence = data.get('user_confidence', 1.0)
            
            if not url or predicted_label is None or actual_label is None:
                return jsonify({
                    'success': False,
                    'error': 'Missing required fields: url, predicted_label, actual_label'
                }), 400
            
            # Add feedback to RL model
            feedback_entry = url_news_classifier.add_feedback(
                url=url,
                predicted_label=predicted_label,
                actual_label=actual_label,
                user_confidence=user_confidence
            )
            
            # Get updated model stats
            model_stats = url_news_classifier.get_model_stats()
            
            return jsonify({
                'success': True,
                'feedback_added': True,
                'feedback_count': model_stats['feedback_count'],
            'model_accuracy': model_stats.get('feedback_accuracy', 0),
            'model_trained': model_stats['is_trained'],
            'message': 'Thank you for your feedback! The model will improve with your input.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/url-classifier-stats')
def get_url_classifier_stats():
    """Get URL classifier model statistics"""
    try:
        stats = url_news_classifier.get_model_stats()
        recent_feedback = url_news_classifier.get_recent_feedback(limit=5)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'recent_feedback': recent_feedback
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/retrain-url-classifier', methods=['POST'])
def retrain_url_classifier():
    """Manually trigger retraining of the URL classifier"""
    try:
        current_feedback_count = len(url_news_classifier.feedback_data)
        
        if current_feedback_count < 3:
            return jsonify({
                'success': False,
                'error': f'Need at least 3 feedback samples to retrain model. Currently have {current_feedback_count} samples.',
                'current_feedback_count': current_feedback_count,
                'minimum_required': 3
            })
        
        accuracy = url_news_classifier.retrain_model()
        
        if accuracy is None:
            return jsonify({
                'success': False,
                'error': f'Retraining failed - insufficient quality feedback data. Have {current_feedback_count} samples.',
                'current_feedback_count': current_feedback_count
            })
        
        return jsonify({
            'success': True,
            'message': f'Model retrained successfully with {current_feedback_count} feedback samples',
            'accuracy': accuracy,
            'feedback_count': current_feedback_count
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

def check_lfs_files():
    """Check if Git LFS files are properly available"""
    print("=== Checking Git LFS Files ===")
    
    lfs_files = [
        'fake_news_model.pkl',
        'political_news_classifier.pkl', 
        'WELFake_Dataset.csv',
        'News_Category_Dataset_v3.json'
    ]
    
    available_files = []
    missing_files = []
    
    for file_path in lfs_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > 1000:  # Larger than typical LFS pointer
                print(f"✓ {file_path} - Available ({file_size:,} bytes)")
                available_files.append(file_path)
            else:
                print(f"⚠ {file_path} - Possibly LFS pointer ({file_size} bytes)")
                missing_files.append(file_path)
        else:
            print(f"✗ {file_path} - Not found")
            missing_files.append(file_path)
    
    print(f"\nSummary: {len(available_files)} available, {len(missing_files)} missing/incomplete")
    
    if missing_files:
        print("\nMissing or incomplete files:", missing_files)
        print("\nThis might indicate:")
        print("1. Git LFS is not properly configured")
        print("2. Files were not pulled during deployment")
        print("3. Repository access issues")
        print("\nThe application will attempt to train models from available data.")
    
    return available_files, missing_files

def initialize_models():
    """Initialize and load both models with Git LFS verification"""
    try:
        print("=== Initializing Fake News Detector Models ===")
        
        # Check Git LFS files first
        available_files, missing_files = check_lfs_files()
        
        # Initialize fake news detection model
        print("\n=== Fake News Detection Model ===")
        if detector.load_model():
            print("✓ Using existing pre-trained fake news model.")
        else:
            print("⚠ No existing fake news model found. Attempting to train new model...")
            
            if 'WELFake_Dataset.csv' in available_files:
                print("Dataset available. Training new model...")
                print("⏳ This may take a few minutes...")
                
                try:
                    df = detector.load_and_prepare_data('WELFake_Dataset.csv')
                    accuracy = detector.train_best_model(df)
                    print(f"✓ Fake news model training completed with accuracy: {accuracy:.4f}")
                    
                    # Save the newly trained model
                    model_data = {
                        'model': detector.model,
                        'accuracy': accuracy,
                        'stemmer': detector.stemmer,
                        'stop_words': detector.stop_words,
                        'training_samples': len(df),
                        'created_date': datetime.now().isoformat()
                    }
                    joblib.dump(model_data, 'fake_news_model.pkl')
                    print("✓ Fake news model saved as 'fake_news_model.pkl'")
                    
                except Exception as e:
                    print(f"✗ Error training model: {str(e)}")
                    print("The application will run with limited functionality.")
            else:
                print("✗ Training dataset not available. Fake news detection will be unavailable.")
                print("Please ensure 'WELFake_Dataset.csv' is properly downloaded from Git LFS.")
        
        # Initialize political news classifier
        print("\n=== Political News Classification Model ===")
        if political_detector.load_model('political_news_classifier.pkl'):
            print("✓ Political news classifier loaded successfully.")
        else:
            print("⚠ Political news classifier not found. Political classification will be unavailable.")
            print("To enable political classification, ensure 'political_news_classifier.pkl' is properly downloaded from Git LFS.")
        
        # Final status
        print("\n=== Initialization Complete ===")
        fake_ready = detector.is_trained
        political_ready = political_detector.is_trained
        
        if fake_ready and political_ready:
            print("✓ Both models are ready!")
        elif fake_ready:
            print("✓ Fake news detection ready, political classification unavailable")
        elif political_ready:
            print("✓ Political classification ready, fake news detection unavailable")
        else:
            print("⚠ No models available - limited functionality")
            
    except Exception as e:
        print(f"✗ Error initializing models: {str(e)}")
        print("Please ensure Git LFS files are properly downloaded.")
        print("Traceback:", str(e))

if __name__ == '__main__':
    # Initialize models in a separate thread to avoid blocking
    model_thread = threading.Thread(target=initialize_models)
    model_thread.daemon = True
    model_thread.start()
    
    # Get port from environment variable for production deployment
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
