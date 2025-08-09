"""
Utility functions for the News Website Crawler module
Includes content extraction and analysis functions
"""

import requests
from bs4 import BeautifulSoup


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
            element = soup.select_one(selector)
            if element:
                content = element.get_text().strip()
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


def analyze_single_article(article_info, analysis_type='both'):
    """Analyze a single article with fake news detection and political classification"""
    try:
        # Import here to avoid circular imports
        from app import detector, political_detector, philippine_search_index
        
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
