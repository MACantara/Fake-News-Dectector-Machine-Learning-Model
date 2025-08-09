"""
Utility functions for Political News Detection
Helper functions for text processing and analysis
"""

import re
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime


def extract_political_keywords():
    """Get the standard set of political keywords used for classification"""
    return {
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


def is_political_domain(url):
    """Check if a URL is from a known political news source"""
    political_domains = {
        'politico.com', 'thehill.com', 'rollcall.com', 'washingtonpost.com',
        'nytimes.com', 'cnn.com', 'foxnews.com', 'msnbc.com', 'reuters.com',
        'apnews.com', 'npr.org', 'bbc.com', 'wsj.com', 'usatoday.com',
        'abcnews.go.com', 'cbsnews.com', 'nbcnews.com', 'pbs.org',
        'whitehouse.gov', 'senate.gov', 'house.gov', 'supremecourt.gov'
    }
    
    try:
        domain = urlparse(url).netloc.lower()
        domain = domain.replace('www.', '')
        return domain in political_domains
    except:
        return False


def extract_political_entities(text):
    """Extract political entities and references from text"""
    entities = {
        'officials': [],
        'institutions': [],
        'locations': [],
        'parties': []
    }
    
    text_lower = text.lower()
    
    # Political officials patterns
    official_patterns = [
        r'\b(?:president|senator|governor|mayor|minister|congressman|representative)\s+([a-z]+)',
        r'\b(?:mr\.|ms\.|mrs\.)\s+([a-z]+)',
        r'\b([a-z]+)\s+(?:administration|campaign)'
    ]
    
    for pattern in official_patterns:
        matches = re.findall(pattern, text_lower)
        entities['officials'].extend(matches)
    
    # Political institutions
    institution_keywords = [
        'white house', 'congress', 'senate', 'house of representatives',
        'supreme court', 'department of', 'pentagon', 'cia', 'fbi',
        'homeland security', 'treasury', 'state department'
    ]
    
    for keyword in institution_keywords:
        if keyword in text_lower:
            entities['institutions'].append(keyword)
    
    # Political parties
    party_keywords = ['democratic', 'republican', 'libertarian', 'green party']
    for party in party_keywords:
        if party in text_lower:
            entities['parties'].append(party)
    
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities


def calculate_political_score(text):
    """Calculate a political relevance score for the text"""
    text_lower = text.lower()
    political_keywords = extract_political_keywords()
    
    # Count keyword occurrences
    keyword_score = sum(1 for keyword in political_keywords if keyword in text_lower)
    
    # Extract entities
    entities = extract_political_entities(text)
    entity_score = sum(len(entities[key]) for key in entities)
    
    # Calculate weights
    text_length = len(text.split())
    if text_length == 0:
        return 0
    
    # Normalize scores
    keyword_ratio = keyword_score / text_length
    entity_ratio = entity_score / text_length
    
    # Final score (0-1 scale)
    political_score = min(1.0, (keyword_ratio * 50) + (entity_ratio * 30))
    
    return political_score


def extract_political_content_from_url(url):
    """Extract and analyze political content from a news URL"""
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
        
        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Extract main content
        content_selectors = [
            'article', '.article-content', '.post-content', '.content',
            '.story-body', 'main', '.entry-content', '.article-body'
        ]
        
        content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text() for elem in elements])
                break
        
        # If no specific content found, get all paragraphs
        if not content:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs])
        
        # Extract metadata
        author = ""
        author_selectors = ['.author', '.byline', '[rel="author"]', '.writer']
        for selector in author_selectors:
            author_elem = soup.select_one(selector)
            if author_elem:
                author = author_elem.get_text().strip()
                break
        
        # Extract publish date
        publish_date = ""
        date_selectors = ['time', '.date', '.publish-date', '[datetime]']
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                publish_date = date_elem.get('datetime') or date_elem.get_text().strip()
                break
        
        # Clean content
        content = re.sub(r'\s+', ' ', content).strip()
        combined_text = f"{title} {content}".strip()
        
        # Calculate political score
        political_score = calculate_political_score(combined_text)
        
        # Extract political entities
        entities = extract_political_entities(combined_text)
        
        # Check if domain is political
        is_political_source = is_political_domain(url)
        
        return {
            'url': url,
            'title': title,
            'content': content,
            'author': author,
            'publish_date': publish_date,
            'combined_text': combined_text,
            'political_score': political_score,
            'political_entities': entities,
            'is_political_source': is_political_source,
            'content_length': len(content.split()),
            'extracted_at': datetime.now().isoformat()
        }
        
    except requests.RequestException as e:
        return {'error': f"Network error: {str(e)}", 'url': url}
    except Exception as e:
        return {'error': f"Extraction error: {str(e)}", 'url': url}


def validate_political_classification(prediction, confidence, text):
    """Validate and provide reasoning for political classification"""
    validation = {
        'is_valid': True,
        'confidence_level': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low',
        'warnings': [],
        'suggestions': []
    }
    
    # Check text length
    word_count = len(text.split())
    if word_count < 50:
        validation['warnings'].append("Text is very short for reliable classification")
        validation['confidence_level'] = 'low'
    
    # Check for political keywords
    political_keywords = extract_political_keywords()
    text_lower = text.lower()
    keyword_count = sum(1 for keyword in political_keywords if keyword in text_lower)
    
    if prediction == 'Political' and keyword_count == 0:
        validation['warnings'].append("Classified as political but no political keywords found")
    
    if prediction == 'Non-Political' and keyword_count > 5:
        validation['warnings'].append("Classified as non-political but contains many political keywords")
    
    # Add suggestions
    if validation['confidence_level'] == 'low':
        validation['suggestions'].append("Consider providing more context or a longer text sample")
    
    if keyword_count < 2 and prediction == 'Political':
        validation['suggestions'].append("Classification may benefit from more political context")
    
    return validation


def get_political_news_categories():
    """Get standard political news categories"""
    return [
        'Elections', 'Government Policy', 'International Relations',
        'Legislation', 'Political Campaigns', 'Court Decisions',
        'Government Officials', 'Political Parties', 'Public Policy',
        'Political Scandals', 'Budget/Fiscal', 'Defense/Security'
    ]


def format_political_analysis_result(result):
    """Format political analysis result for consistent output"""
    formatted = {
        'classification': result.get('prediction', 'Unknown'),
        'confidence': result.get('confidence', 0.0),
        'confidence_percentage': f"{result.get('confidence', 0.0) * 100:.1f}%",
        'analysis': {
            'is_political': result.get('prediction') == 'Political',
            'reasoning': result.get('reasoning', 'No reasoning available'),
            'political_features': result.get('political_features', {}),
            'entities': result.get('political_entities', {}),
            'score': result.get('political_score', 0.0)
        },
        'probabilities': result.get('probabilities', {}),
        'timestamp': datetime.now().isoformat()
    }
    
    return formatted
