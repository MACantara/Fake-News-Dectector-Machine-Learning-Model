# Fake News Detector Web Application

A machine learning-powered web application that can detect fake news from text input or article URLs using the WELFake dataset.

## Features

- **Text Analysis**: Paste news article text directly for analysis
- **URL Analysis**: Extract and analyze content from article URLs automatically
- **Multiple ML Models**: Compares Logistic Regression, Random Forest, and Naive Bayes models
- **Real-time Results**: Get instant predictions with confidence scores
- **Modern UI**: Beautiful, responsive interface built with Tailwind CSS
- **Content Extraction**: Automatically extracts article content from web pages
- **🧠 Reinforcement Learning**: User feedback system that continuously improves model accuracy
- **📊 Learning Progress**: Real-time tracking of model improvements and feedback statistics
- **🔄 Auto-Retraining**: Automatic model retraining when sufficient feedback is collected

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Tailwind CSS, Bootstrap Icons
- **Machine Learning**: scikit-learn, NLTK
- **Data Processing**: pandas, numpy
- **Web Scraping**: BeautifulSoup, requests

## Installation

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd Fake-News-Dectector-Machine-Learning-Model
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the dataset is in place**:
   Make sure `WELFake_Dataset.csv` is in the project root directory.

4. **Train the model** (recommended):
   ```bash
   python train_model.py
   ```
   This will train the model once and save it for reuse. The web app will load much faster!

## Usage

### Quick Start (Recommended)

1. **Train the model first** (one-time setup):
   ```bash
   python train_model.py
   ```

2. **Start the web application**:
   ```bash
   python web_app.py
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

### Running the Web Application

**Option 1: With Pre-trained Model (Fast)**
- Run `train_model.py` first to create the model file
- Then run `web_app.py` - it will load instantly!

**Option 2: Auto-training (Slower)**
- Run `web_app.py` directly
- The app will train the model automatically on first run
- This takes 2-5 minutes but only happens once

### Running the Command Line Version

To run the original command-line version:
```bash
python app.py
```

## How to Use the Web Interface

### Text Analysis
1. Select "Text Input" tab
2. Paste your news article text in the textarea
3. Click "Analyze News"
4. View the prediction results with confidence scores

### URL Analysis
1. Select "Article URL" tab
2. Enter the URL of a news article
3. Click "Analyze News"
4. The app will extract the content and show both the prediction and extracted content

### 🧠 Providing Feedback for Model Improvement
1. After getting a prediction, you'll see a "Help Improve the Model" section
2. Click "Correct Prediction" if the model was right
3. Click "Incorrect - It's [opposite]" if the model was wrong
4. Optionally add a comment explaining why you think it was wrong
5. Your feedback is stored and used to retrain the model automatically

### 📊 Learning Progress
- The web interface shows real-time statistics about:
  - Total feedback collected
  - Feedback used for training
  - Pending feedback awaiting training
- The model automatically retrains when 10 new feedback entries are collected
- Each retraining cycle improves the model's accuracy

## Model Information

The application trains and compares three machine learning models:

- **Logistic Regression**: Linear model good for text classification
- **Random Forest**: Ensemble method that combines multiple decision trees
- **Naive Bayes**: Probabilistic classifier effective for text data

The best performing model is automatically selected and used for predictions.

### Text Preprocessing

The application performs the following text preprocessing steps:

1. Convert text to lowercase
2. Remove special characters and digits
3. Remove extra whitespace
4. Remove stop words
5. Apply stemming using Porter Stemmer
6. Vectorize using TF-IDF with 5000 max features

## Project Files

- **`app.py`** - Original command-line ML model with training and evaluation
- **`web_app.py`** - Flask web application backend with reinforcement learning
- **`train_model.py`** - Standalone script to train and save the model
- **`templates/index.html`** - Web application frontend with feedback interface
- **`requirements.txt`** - Python dependencies
- **`WELFake_Dataset.csv`** - Training dataset (required)
- **`fake_news_model.pkl`** - Saved trained model (created after training)
- **`user_feedback.json`** - User feedback data for model improvement (created automatically)

## 🧠 Reinforcement Learning System

The application includes an advanced reinforcement learning system that continuously improves the model based on user feedback:

### How It Works:
1. **User Interaction**: Users provide feedback on model predictions
2. **Data Collection**: Feedback is stored with timestamps and metadata
3. **Automatic Retraining**: Model retrains when 10+ feedback entries are collected
4. **Model Updates**: New model incorporates both original data and user corrections
5. **Continuous Improvement**: Each iteration makes the model more accurate

### Feedback Data Structure:
```json
{
  "timestamp": "2025-07-26T10:30:00",
  "text": "article content",
  "predicted_label": "Fake",
  "actual_label": "Real",
  "confidence": 0.75,
  "user_comment": "This is clearly real news",
  "processed_text": "processed article content",
  "used_for_training": true,
  "training_date": "2025-07-26T11:00:00"
}
```

### Benefits:
- **📈 Improved Accuracy**: Model learns from real-world corrections
- **🎯 Domain Adaptation**: Adapts to specific types of news articles
- **👥 Crowd Intelligence**: Leverages collective user knowledge
- **🔄 Continuous Learning**: Never stops improving

## Dataset

The application uses the WELFake dataset which contains:
- **index**: Unique identifier
- **title**: Article title
- **text**: Article content
- **label**: 0 = Fake, 1 = Real

## API Endpoints

### GET `/`
Serves the main web interface

### POST `/predict`
Analyzes text or URL content
- **Request Body**: 
  ```json
  {
    "type": "text|url",
    "text": "article text" // for text type
    "url": "article url"   // for url type
  }
  ```
- **Response**:
  ```json
  {
    "prediction": "Real|Fake",
    "confidence": 0.85,
    "probabilities": {
      "Fake": 0.15,
      "Real": 0.85
    },
    "extracted_content": { // only for URL requests
      "title": "Article Title",
      "content_preview": "First 500 characters..."
    }
  }
  ```

### POST `/submit-feedback`
Submit user feedback to improve the model
- **Request Body**:
  ```json
  {
    "text": "article text",
    "predicted_label": "Real|Fake",
    "actual_label": "Real|Fake",
    "confidence": 0.85,
    "comment": "optional user comment"
  }
  ```
- **Response**:
  ```json
  {
    "message": "Thank you for your feedback!",
    "feedback_stats": {
      "total_feedback": 15,
      "used_for_training": 10,
      "pending_training": 5,
      "needs_retraining": false
    }
  }
  ```

### GET `/feedback-stats`
Get statistics about collected feedback
- **Response**:
  ```json
  {
    "total_feedback": 15,
    "used_for_training": 10,
    "pending_training": 5,
    "retrain_threshold": 10,
    "needs_retraining": false
  }
  ```

### GET `/model-status`
Returns the current status of the ML model with feedback statistics
- **Response**:
  ```json
  {
    "is_trained": true,
    "status": "Model ready (Accuracy: 89.2%)",
    "accuracy": "0.8920",
    "feedback": {
      "total_feedback": 15,
      "used_for_training": 10,
      "pending_training": 5,
      "needs_retraining": false
    }
  }
  ```

## Browser Compatibility

The web application is compatible with modern browsers that support:
- ES6 JavaScript features
- CSS Grid and Flexbox
- Fetch API

## Troubleshooting

### Common Issues

1. **Model training takes too long**: This is normal for the first run. The model needs to process the entire dataset.

2. **URL extraction fails**: Some websites may block automated content extraction. Try using the text input method instead.

3. **NLTK data not found**: The application automatically downloads required NLTK data, but you may need to run:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. **Port already in use**: If port 5000 is busy, you can change it in `web_app.py`:
   ```python
   app.run(debug=True, port=5001)  # Change to different port
   ```

## Performance Notes

- **First-time Setup**: Run `train_model.py` once (2-5 minutes) to create the model file
- **Web App Startup**: Instant if model exists, 2-5 minutes if training needed
- **Predictions**: Typically very fast (< 1 second)
- **URL Content Extraction**: 3-10 seconds depending on the website
- **Model Reuse**: Saved model persists between sessions for faster startup

## Model Persistence

The application now uses model persistence for better performance:

- **First Run**: Train the model using `train_model.py` or let `web_app.py` train automatically
- **Subsequent Runs**: The saved model (`fake_news_model.pkl`) is loaded instantly
- **Retraining**: Delete `fake_news_model.pkl` or run `train_model.py` again to retrain

## Future Enhancements

- ✅ ~~Model persistence to avoid retraining on each restart~~ (Completed)
- ✅ ~~User feedback system to improve model accuracy~~ (Completed)
- Support for multiple languages
- Batch processing of multiple articles
- Advanced visualization of prediction confidence
- Export/import of feedback data
- A/B testing of different model configurations
- Real-time model performance monitoring

## License

This project is for educational purposes. Please ensure you have proper rights to analyze any content you submit to the application.
