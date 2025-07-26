#!/usr/bin/env python3
"""
Train Model Script
This script trains the fake news detection model and saves it for use by the web application.
Run this script first before starting the web app for better performance.
"""

import os
import sys

def main():
    print("="*60)
    print("FAKE NEWS DETECTOR - MODEL TRAINING")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists('WELFake_Dataset.csv'):
        print("❌ Error: 'WELFake_Dataset.csv' not found in current directory.")
        print("Please ensure the dataset file is in the same directory as this script.")
        sys.exit(1)
    
    # Check if model already exists
    if os.path.exists('fake_news_model.pkl'):
        print("⚠️  A trained model already exists ('fake_news_model.pkl').")
        response = input("Do you want to retrain the model? (y/N): ").lower().strip()
        if response not in ['y', 'yes']:
            print("✅ Using existing model. You can start the web app now.")
            return
        else:
            print("🔄 Retraining model...")
    
    # Import and run the training
    try:
        from app import main as train_model
        print("🚀 Starting model training...")
        train_model()
        
        if os.path.exists('fake_news_model.pkl'):
            print("\n" + "="*60)
            print("✅ SUCCESS: Model training completed!")
            print("📁 Model saved as: fake_news_model.pkl")
            print("🌐 You can now start the web application with: python web_app.py")
            print("="*60)
        else:
            print("\n❌ Error: Model training completed but model file was not created.")
            
    except ImportError as e:
        print(f"❌ Error importing training module: {e}")
        print("Please ensure app.py is in the same directory.")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()
