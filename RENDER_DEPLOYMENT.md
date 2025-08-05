# Render Deployment Guide for Fake News Detector

This guide will help you deploy your Fake News Detector web application to Render.

## Prerequisites

1. Your GitHub repository should have Git LFS properly configured
2. All model files (*.pkl) and datasets (*.csv, *.json) should be tracked with Git LFS
3. Ensure all files are committed and pushed to GitHub

## Deployment Steps

### 1. Prepare Your Repository

Make sure your repository has these files:
- `web_app.py` - Your Flask application
- `requirements.txt` - Python dependencies
- `Procfile` - Process configuration
- `render.yaml` - Render service configuration
- `build.sh` - Build script
- `start.sh` - Start script
- `.renderignore` - Files to ignore during deployment
- `.gitattributes` - Git LFS configuration
- `.lfsconfig` - LFS configuration

### 2. Deploy to Render

#### Option A: Using Render Dashboard (Recommended)

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" and select "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: fake-news-detector (or your preferred name)
   - **Runtime**: Python 3
   - **Build Command**: `./build.sh`
   - **Start Command**: `gunicorn web_app:app`
   - **Plan**: Starter (or higher based on your needs)

#### Option B: Using render.yaml (Infrastructure as Code)

1. Ensure `render.yaml` is in your repository root
2. Go to Render Dashboard
3. Click "New +" and select "Blueprint"
4. Connect your repository and Render will automatically detect the `render.yaml`

### 3. Environment Configuration

Add these environment variables in Render:
- `PYTHON_VERSION`: 3.11.4
- `FLASK_ENV`: production
- `NLTK_DATA`: /opt/render/project/src/nltk_data

### 4. Git LFS Configuration for Render

Since Render supports Git LFS, your model files should be automatically pulled during deployment. However, if you encounter issues:

1. Ensure your `.gitattributes` file is configured correctly:
   ```
   *.pkl filter=lfs diff=lfs merge=lfs -text
   *.json filter=lfs diff=lfs merge=lfs -text
   *.csv filter=lfs diff=lfs merge=lfs -text
   ```

2. Verify files are in LFS:
   ```bash
   git lfs ls-files
   ```

3. If needed, manually add files to LFS:
   ```bash
   git lfs track "*.pkl"
   git lfs track "*.csv"
   git lfs track "*.json"
   git add .gitattributes
   git commit -m "Track large files with LFS"
   git push
   ```

### 5. Troubleshooting

#### Common Issues:

1. **LFS Files Not Downloaded**
   - Check if LFS is properly configured in your repository
   - Verify files are committed to LFS: `git lfs ls-files`
   - Render should automatically handle LFS, but ensure your repository is public or Render has proper access

2. **NLTK Data Download Fails**
   - The build script includes NLTK data download
   - If it fails, check the build logs in Render dashboard

3. **Model Training During Deployment**
   - If model files are missing, the app will train a new model on first run
   - This may cause the initial deployment to take longer
   - Ensure `WELFake_Dataset.csv` is available for training

4. **Memory Issues**
   - If you encounter memory issues, consider upgrading to a higher Render plan
   - The Starter plan may be insufficient for large models

#### Build Logs
Monitor your deployment in the Render dashboard under "Events" and "Logs" tabs.

### 6. Testing Your Deployment

Once deployed:
1. Access your application via the Render-provided URL
2. Test the fake news detection functionality
3. Verify both text input and URL analysis work correctly
4. Check the model status endpoint: `your-app-url.com/model-status`

### 7. Post-Deployment

- Monitor your application performance in Render dashboard
- Set up custom domain if needed
- Configure monitoring and alerting
- Consider setting up automated deployments from your main branch

## Important Notes

- **First Deployment**: May take 10-15 minutes due to model loading and NLTK data download
- **Subsequent Deployments**: Will be faster as dependencies are cached
- **Model Training**: If pre-trained models are not available, training will occur on first request
- **Scaling**: Consider upgrading your Render plan for production use

## Support

If you encounter issues:
1. Check Render deployment logs
2. Verify all LFS files are properly tracked and committed
3. Ensure your repository is accessible by Render
4. Contact Render support for platform-specific issues

## Security Considerations

- Environment variables are handled securely by Render
- No sensitive data should be hardcoded in your application
- Consider implementing rate limiting for production use
- Monitor usage to prevent abuse
