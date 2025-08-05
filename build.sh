#!/bin/bash

# Build script for Render deployment
echo "Starting build process..."

# Install Python dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -c "
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print('Downloading NLTK punkt...')
nltk.download('punkt', quiet=True)
print('Downloading NLTK punkt_tab...')
nltk.download('punkt_tab', quiet=True)
print('Downloading NLTK stopwords...')
nltk.download('stopwords', quiet=True)
print('Downloading NLTK wordnet...')
nltk.download('wordnet', quiet=True)
print('Downloading NLTK omw-1.4...')
nltk.download('omw-1.4', quiet=True)
print('NLTK data download completed successfully!')
"

# Check if LFS files exist
echo "Checking for model files..."
if [ -f "fake_news_model.pkl" ]; then
    echo "✓ fake_news_model.pkl found"
else
    echo "⚠ fake_news_model.pkl not found - will be trained on first run"
fi

if [ -f "political_news_classifier.pkl" ]; then
    echo "✓ political_news_classifier.pkl found"
else
    echo "⚠ political_news_classifier.pkl not found - political classification will be unavailable"
fi

if [ -f "WELFake_Dataset.csv" ]; then
    echo "✓ WELFake_Dataset.csv found"
else
    echo "⚠ WELFake_Dataset.csv not found - this is required for training"
fi

echo "Build process completed!"
