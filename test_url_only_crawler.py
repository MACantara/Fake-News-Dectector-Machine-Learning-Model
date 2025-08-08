#!/usr/bin/env python3
"""Test the URL-only NewsWebsiteCrawler"""

from web_app import NewsWebsiteCrawler

def test_url_only_crawler():
    print('Testing URL-only NewsWebsiteCrawler...')
    
    # Initialize the simplified crawler
    crawler = NewsWebsiteCrawler()
    
    # Test with a news website
    result = crawler.extract_article_links('https://www.pna.gov.ph', max_links=5)
    
    print('Success:', result['success'])
    print('Total URLs found:', result['total_found'])
    print('Classification method:', result['classification_method'])
    
    print('\nFirst 10 URLs:')
    for i, url in enumerate(result['articles'][:10]):
        print(f'{i+1}. {url}')

if __name__ == '__main__':
    test_url_only_crawler()
