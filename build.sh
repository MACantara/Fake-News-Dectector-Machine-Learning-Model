#!/bin/bash

# Build script for Render deployment with Git LFS support
echo "=== Starting build process ==="

# Install Python dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# Check Git LFS installation and status
echo "Checking Git LFS..."
if command -v git-lfs &> /dev/null; then
    echo "✓ Git LFS is available"
    git lfs version
    
    # Show LFS tracked files
    echo "Git LFS tracked files:"
    git lfs ls-files
    
    # Check if LFS files are downloaded
    echo "Checking LFS file status..."
    for file in "fake_news_model.pkl" "political_news_classifier.pkl" "WELFake_Dataset.csv" "News_Category_Dataset_v3.json"; do
        if [ -f "$file" ]; then
            size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
            if [ "$size" -gt 1000 ]; then
                echo "✓ $file: ${size} bytes (OK)"
            else
                echo "⚠ $file: ${size} bytes (might be LFS pointer)"
            fi
        else
            echo "✗ $file: not found"
        fi
    done
else
    echo "⚠ Git LFS not available - this might cause issues with model files"
fi

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
print('✓ NLTK data download completed successfully!')
"

# Verify Python dependencies
echo "Verifying critical Python packages..."
python -c "
import sys
try:
    import flask, pandas, numpy, sklearn, nltk, requests, joblib
    from bs4 import BeautifulSoup
    print('✓ All critical packages imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

echo "=== Build process completed! ==="
