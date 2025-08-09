"""
Routes package for the Fake News Detector application
"""

from .news_crawler_routes import news_crawler_bp
from .philippine_news_search_routes import philippine_news_bp

__all__ = ['news_crawler_bp', 'philippine_news_bp']
