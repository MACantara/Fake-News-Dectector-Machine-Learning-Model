# Philippine News Search Engine Integration

## Overview
This document outlines the specialized Philippine news search engine functionality that has been integrated into the Fake News Detector web application. The system automatically indexes Philippine news articles and provides advanced search capabilities.

## Features

### 1. Automatic Article Indexing
- **Auto-indexing**: Every URL submitted by users for analysis is automatically indexed into the Philippine news database in the background
- **Relevance Scoring**: Each article is scored (0-1) based on Philippine relevance using keywords, entities, and source domains
- **Content Extraction**: Advanced content extraction with Philippine news focus
- **Metadata Extraction**: Extracts title, author, publish date, category, and other metadata

### 2. Philippine-Specific Content Analysis
- **Location Detection**: Identifies Philippine locations (cities, provinces, regions)
- **Government Entity Recognition**: Detects government agencies, officials, and institutions
- **Cultural Keywords**: Recognizes Filipino cultural terms and events
- **News Source Recognition**: Identifies major Philippine news websites

### 3. Search Functionality
- **Full-text Search**: Powered by Whoosh search engine for fast results
- **Category Filtering**: Filter by news categories (politics, business, sports, etc.)
- **Source Filtering**: Filter by specific news sources
- **Result Limiting**: Configurable result limits (10-100 articles)
- **Relevance Ranking**: Results ranked by Philippine relevance score

### 4. Analytics Dashboard
- **Total Statistics**: Total articles, sources, categories indexed
- **Source Distribution**: Charts showing top news sources
- **Category Distribution**: Visual breakdown of article categories
- **Recent Activity**: Timeline of recent indexing activity
- **Search Analytics**: Top search queries and response times

## Technical Implementation

### Backend Components

#### PhilippineNewsSearchIndex Class
- **Database**: SQLite database with comprehensive schema
- **Search Index**: Whoosh full-text search index
- **Content Analysis**: Philippine-specific keyword and entity extraction
- **Relevance Scoring**: Multi-factor relevance calculation

#### Database Schema
```sql
-- Main articles table
philippine_articles (
    id, url, title, content, summary, author, publish_date,
    source_domain, category, tags, philippine_relevance_score,
    location_mentions, government_entities, language,
    sentiment_score, content_hash, indexed_date, last_updated,
    fake_news_prediction, political_prediction, view_count, search_count
)

-- Search analytics
search_queries (id, query, results_count, query_date, user_ip, response_time)

-- Indexing tasks
indexing_tasks (id, url, status, error_message, created_date, completed_date)
```

#### API Endpoints
- `POST /index-philippine-article` - Index a single article
- `POST /batch-index-philippine-articles` - Index multiple articles
- `GET/POST /search-philippine-news` - Search articles
- `GET /philippine-news-analytics` - Get analytics data
- `GET /philippine-article/<id>` - Get article details
- `GET /philippine-news-categories` - Get available categories
- `GET /philippine-news-sources` - Get available sources

### Frontend Components

#### New UI Sections
- **Philippine News Search Tab**: New input type for searching
- **Search Interface**: Query input with filters and options
- **Search Results Display**: Card-based results with relevance scores
- **Analytics Dashboard**: Comprehensive statistics and charts
- **Article Indexing Modal**: Manual article addition interface

#### JavaScript Integration
- **Search Methods**: Async search with error handling
- **Results Rendering**: Dynamic result card generation
- **Analytics Visualization**: Chart rendering and data display
- **Modal Management**: Article details and indexing modals

## Configuration

### Philippine Keywords Categories
- **Government**: DOH, DOF, DILG, DND, DepEd, etc.
- **Places**: Manila, Cebu, Davao, NCR, Luzon, Visayas, Mindanao, etc.
- **Officials**: President, Senator, Congressman, Mayor, Governor, etc.
- **Politics**: Malacañang, Senate, Congress, Supreme Court, COMELEC, etc.
- **Economy**: BSP, PSEI, Peso, GDP, OFW, BPO, etc.
- **Culture**: Filipino languages, festivals, cultural terms

### Philippine News Sources
- ABS-CBN (abs-cbn.com)
- GMA News (gma.tv)
- Philippine Daily Inquirer (inquirer.net)
- Rappler (rappler.com)
- Philippine Star (philstar.com)
- Manila Bulletin (manilabulletin.ph)
- CNN Philippines (cnnphilippines.com)
- Philippine News Agency (pna.gov.ph)

## Usage

### 1. Automatic Indexing
When users submit URLs for fake news analysis, articles are automatically:
1. Content extracted and analyzed
2. Philippine relevance calculated
3. Indexed in database (if relevance > 0.1)
4. Added to search index

### 2. Manual Search
Users can:
1. Click "Philippine News Search" tab
2. Enter search queries
3. Apply filters (category, source, limit)
4. View results with relevance scores
5. Click articles to analyze or view details

### 3. Analytics
Users can:
1. Click "View Analytics" button
2. See comprehensive statistics
3. Explore source and category distributions
4. Review recent indexing activity
5. Check top search queries

### 4. Manual Indexing
Users can:
1. Click "Add Article to Index"
2. Enter article URL
3. Submit for indexing
4. Receive confirmation or error messages

## Performance Considerations

### Indexing Performance
- Background threading for non-blocking indexing
- Content extraction timeout (15 seconds)
- Relevance threshold (0.1) to filter irrelevant content
- Batch processing support for multiple URLs

### Search Performance
- Whoosh full-text index for fast searches
- SQLite indexes on key fields
- Result limiting to prevent large responses
- Response time logging for monitoring

### Database Optimization
- Indexed fields: url, publish_date, source_domain, category, relevance_score
- Content hash for duplicate detection
- Periodic cleanup of old indexing tasks

## Dependencies

### Python Packages
```
whoosh==2.7.4          # Full-text search engine
python-dateutil==2.8.2 # Date parsing
textblob==0.17.1        # Sentiment analysis
fuzzywuzzy==0.18.0      # Fuzzy string matching
python-levenshtein==0.21.1  # String distance calculations
```

### Database
- SQLite (built-in Python support)
- No external database server required

## Future Enhancements

### Planned Features
1. **Real-time Updates**: WebSocket integration for live search results
2. **Advanced Filters**: Date range, sentiment, author filters
3. **Export Functions**: CSV/JSON export of search results
4. **Bulk Operations**: Bulk delete, update, re-index operations
5. **API Rate Limiting**: Prevent abuse of search endpoints
6. **Caching**: Redis cache for frequent queries
7. **Machine Learning**: Content classification and clustering

### Scalability Improvements
1. **PostgreSQL Migration**: For better performance with large datasets
2. **Elasticsearch**: Replace Whoosh for enterprise-scale search
3. **Microservices**: Split indexing and search into separate services
4. **Queue System**: Celery for background job processing
5. **CDN Integration**: Cache static search results

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Search Timeouts**: Check Whoosh index integrity
3. **Indexing Failures**: Verify URL accessibility and content extraction
4. **Low Relevance Scores**: Review Philippine keyword configuration

### Debug Commands
```bash
# Check dependencies
python -c "import whoosh, textblob, fuzzywuzzy; print('All imports successful')"

# Verify database
sqlite3 philippine_news_index.db ".tables"

# Check index directory
ls -la whoosh_index/
```

## Conclusion
The Philippine News Search Engine provides a comprehensive solution for indexing, searching, and analyzing Philippine news content. It integrates seamlessly with the existing fake news detection system while providing specialized functionality for Philippine news sources and content.
