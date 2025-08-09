# News Tracker - URL Classifier Feedback Integration

## Overview
The News Tracker system now sends user verification feedback to the URL Classifier to continuously improve the machine learning model. This creates a feedback loop where human verifications help the system become more accurate over time.

## Integration Details

### 🔄 How It Works
1. **Article Fetching**: News Tracker uses `/crawl-website` endpoint to get articles with ML predictions
2. **User Verification**: Users verify if articles are "news" or "not news" 
3. **Feedback Submission**: Verification data is automatically sent to `/url-classifier-feedback`
4. **Model Improvement**: URL Classifier uses feedback to retrain and improve accuracy

### 📊 Data Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  News Tracker   │───▶│   /crawl-website │───▶│  Article Queue  │
│                 │    │   (with ML data) │    │  (with metadata)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                                               │
         │                                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ URL Classifier  │◀───│ /url-classifier- │◀───│ User Verifies   │
│ Model Improves  │    │     feedback     │    │ Article         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 🗄️ Database Schema Updates

The `article_queue` table now includes crawler metadata:

```sql
ALTER TABLE article_queue ADD COLUMN confidence REAL DEFAULT 0.0;
ALTER TABLE article_queue ADD COLUMN is_news_prediction BOOLEAN DEFAULT TRUE;
ALTER TABLE article_queue ADD COLUMN probability_news REAL DEFAULT 0.0;
```

### 📝 Feedback Data Structure

When a user verifies an article, the following data is sent to the URL classifier:

```json
{
  "url": "https://example.com/article",
  "predicted_label": true,           // What the crawler predicted
  "actual_label": "news",           // What the user verified  
  "user_confidence": 1.0,           // High confidence (human verification)
  "comment": "News Tracker verification - Confidence: 0.85, Prob: 0.92"
}
```

### 🔧 Technical Implementation

#### Key Functions Added:

1. **`send_url_classifier_feedback()`**
   - Extracts crawler metadata from article data
   - Formats feedback for URL classifier endpoint
   - Handles errors gracefully (doesn't break verification)
   - Logs feedback success/failure

2. **Database Migration**
   - `migrate_database_schema()` adds new columns to existing databases
   - Preserves existing data while adding crawler metadata fields

3. **Enhanced Article Storage**
   - Articles now stored with confidence scores and predictions
   - Metadata used for intelligent feedback to URL classifier

#### Error Handling:
- Network errors don't break article verification
- Missing metadata is handled with sensible defaults
- All feedback attempts are logged for monitoring

### 🧪 Testing

Use the test script to verify integration:

```bash
python test_url_classifier_feedback_integration.py
```

The test:
- ✅ Adds a test website
- ✅ Fetches articles with crawler metadata  
- ✅ Verifies articles and checks feedback was sent
- ✅ Monitors URL classifier feedback count increase
- ✅ Cleans up test data

### 📈 Benefits

1. **Continuous Learning**: Model improves with every user verification
2. **Domain Adaptation**: Learns from real news sources being tracked
3. **Quality Improvement**: Human feedback refines ML predictions
4. **Automated Process**: No manual intervention required
5. **Graceful Degradation**: Works even if URL classifier is unavailable

### 🔍 Monitoring

Check URL classifier improvement:

```bash
# Check current model stats
curl http://localhost:5000/url-classifier-model-status

# View recent feedback
curl http://localhost:5000/url-classifier-stats
```

### 🎯 Expected Outcomes

- **Higher Accuracy**: URL classifier becomes more accurate over time
- **Better Filtering**: Crawler returns higher quality article candidates  
- **Reduced False Positives**: Fewer non-news URLs classified as news
- **Domain Expertise**: Model learns patterns specific to tracked news sources

### 📋 Configuration

The integration uses these default settings:
- Feedback confidence: `1.0` (high confidence for human verifications)
- Timeout: `10 seconds` for feedback requests
- Error handling: Non-blocking (verification succeeds even if feedback fails)
- Endpoint: `http://127.0.0.1:5000/url-classifier-feedback`

### 🔄 Retraining

The URL classifier automatically triggers retraining when:
- Every 10 new feedback entries are received
- Manual retraining is requested via `/retrain-url-classifier`

This ensures the model stays current with user feedback while avoiding excessive retraining.

---

## Summary

This integration creates a powerful feedback loop where:
1. **Smart Crawling** → Better article discovery
2. **User Verification** → Quality training data  
3. **Model Learning** → Improved predictions
4. **Repeat** → Continuously improving system

The News Tracker now not only consumes intelligent crawler services but also contributes back to improving the underlying ML models, making the entire system smarter over time! 🧠✨
