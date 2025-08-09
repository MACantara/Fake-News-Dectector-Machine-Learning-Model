"""
News Website Crawler Class
Main crawler implementation for extracting and analyzing news articles from websites
"""

import re
import requests
import concurrent.futures
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from .utils import analyze_single_article


class NewsWebsiteCrawler:
    def __init__(self, url_classifier=None):
        """Initialize the crawler with URL classifier for intelligent filtering"""
        self.url_classifier = url_classifier
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
        
        # Use ThreadPoolExecutor for parallel processing
        # Dynamic worker count based on the number of articles
        max_workers = min(20, len(article_urls))  # Dynamic worker count, max 20
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_article = {executor.submit(analyze_single_article, article, analysis_type): article 
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
