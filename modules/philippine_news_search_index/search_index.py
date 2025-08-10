"""
Philippine News Search Index - Main Search Engine Class
Specialized search engine for Philippine news articles with SQLite and Whoosh integration
"""

import sqlite3
import os
import time
import re
import hashlib
from collections import Counter
from urllib.parse import urlparse
from datetime import datetime
from nltk.stem import PorterStemmer
from whoosh import index
from whoosh.fields import Schema, TEXT, DATETIME, ID, KEYWORD, NUMERIC
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import And, Or, Term, Phrase
from textblob import TextBlob
from .utils import extract_advanced_content


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
            content_data = extract_advanced_content(url)
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
    
    def crawl_and_index_website(self, website_url, force_reindex=False):
        """Crawl a news website and index all found articles"""
        try:
            print(f"Starting website crawl and index: {website_url}")
            
            # Import here to avoid circular imports
            from modules.news_website_crawler import NewsWebsiteCrawler
            
            # Initialize news crawler
            news_crawler = NewsWebsiteCrawler()
            
            # Crawl the website for article links
            crawl_result = news_crawler.extract_article_links(website_url)
            
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
            
            # Index all articles found without any limits
            articles_to_index = crawl_result['articles']
            
            print(f"Found {len(articles_to_index)} articles to index")
            
            # Index each found article
            indexing_results = []
            successful_indexes = 0
            skipped_indexes = 0
            error_indexes = 0
            already_indexed_count = 0
            
            for article in articles_to_index:
                try:
                    if isinstance(article, dict):
                        article_url = article.get('url', '')
                        article_title = article.get('text', '') or article.get('title', '')
                    else:
                        article_url = str(article)
                        article_title = ''
                    
                    print(f"Indexing: {article_title[:50]}...")
                    
                    # Index the article
                    index_result = self.index_article(article_url, force_reindex)
                    
                    result_entry = {
                        'url': article_url,
                        'title': article_title,
                        'status': index_result['status']
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
                        'url': article.get('url', '') if isinstance(article, dict) else str(article),
                        'title': article.get('title', '') if isinstance(article, dict) else '',
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
                    'articles_processed': len(articles_to_index),
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
