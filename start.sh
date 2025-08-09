#!/bin/bash

# Start script for Render deployment
echo "Starting Fake News Detector application..."

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=production

# Start the application with Gunicorn
echo "Starting Gunicorn server..."
exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --preload web_app:app
