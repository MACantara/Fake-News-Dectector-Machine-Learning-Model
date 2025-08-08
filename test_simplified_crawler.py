#!/usr/bin/env python3
"""Test the simplified NewsWebsiteCrawler"""

from web_app import NewsWebsiteCrawler

def test_simplified_crawler():
    print('Testing simplified NewsWebsiteCrawler...')
    
    # Initialize the simplified crawler
    crawler = NewsWebsiteCrawler()
    
    # Test with a news website
    result = crawler.extract_article_links('https://www.pna.gov.ph', max_links=5)
    
    print('Success:', result['success'])
    print('Total links found:', result['total_found'])
    print('Classification method:', result['classification_method'])
    
    print('\nFirst few links:')
    for i, link in enumerate(result['articles'][:5]):
        print(f'{i+1}. {link["html_format"]}')
        print(f'   Raw URL: {link["url"]}')
        print(f'   Title: {link["title"]}')
        print()

if __name__ == '__main__':
    test_simplified_crawler()
