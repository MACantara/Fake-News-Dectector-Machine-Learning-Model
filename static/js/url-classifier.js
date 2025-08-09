// URL News Classifier Application
class URLClassifier {
    constructor() {
        this.currentUrl = '';
        this.currentPrediction = null;
        this.elements = {};
        
        this.init();
    }
    
    init() {
        this.cacheElements();
        this.bindEvents();
        this.checkModelStatus();
        this.loadStatistics();
        
        console.log('URL Classifier initialized successfully');
    }
    
    cacheElements() {
        const elementIds = [
            'urlInput', 'classifyBtn', 'loading', 'results', 'error',
            'modelStatus', 'predictionCard', 'predictionText', 'confidenceText',
            'modelType', 'newsPercentage', 'regularPercentage', 'newsBar', 'regularBar',
            'correctBtn', 'incorrectBtn', 'confidenceSlider', 'confidenceValue',
            'feedbackMessage', 'errorMessage', 'feedbackCount', 'modelAccuracy',
            'trainingIterations', 'retrainBtn', 'recentFeedback', 'feedbackList'
        ];
        
        elementIds.forEach(id => {
            this.elements[id] = document.getElementById(id);
        });
    }
    
    bindEvents() {
        // URL input monitoring
        if (this.elements.urlInput) {
            this.elements.urlInput.addEventListener('input', () => this.updateClassifyButton());
            this.elements.urlInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !this.elements.classifyBtn.disabled) {
                    this.classifyUrl();
                }
            });
        }
        
        // Classify button
        if (this.elements.classifyBtn) {
            this.elements.classifyBtn.addEventListener('click', () => this.classifyUrl());
        }
        
        // Feedback buttons
        if (this.elements.correctBtn) {
            this.elements.correctBtn.addEventListener('click', () => this.submitFeedback(true));
        }
        if (this.elements.incorrectBtn) {
            this.elements.incorrectBtn.addEventListener('click', () => this.submitFeedback(false));
        }
        
        // Confidence slider
        if (this.elements.confidenceSlider) {
            this.elements.confidenceSlider.addEventListener('input', (e) => {
                const value = Math.round(e.target.value * 100);
                this.elements.confidenceValue.textContent = `${value}%`;
            });
        }
        
        // Retrain button
        if (this.elements.retrainBtn) {
            this.elements.retrainBtn.addEventListener('click', () => this.retrainModel());
        }
    }
    
    async checkModelStatus() {
        try {
            const response = await fetch('/url-classifier-stats');
            const data = await response.json();
            
            if (data.success) {
                this.updateModelStatusDisplay(data.stats);
                this.updateRecentFeedback(data.recent_feedback);
            } else {
                this.updateModelStatusDisplay(null, data.error);
            }
        } catch (error) {
            console.error('Model status check failed:', error);
            this.updateModelStatusDisplay(null, error.message);
        }
    }
    
    updateModelStatusDisplay(stats, errorMessage) {
        if (!this.elements.modelStatus) return;
        
        if (errorMessage) {
            this.elements.modelStatus.innerHTML = `
                <div class="flex items-center justify-center">
                    <i class="bi bi-exclamation-triangle text-red-600 mr-2"></i>
                    <span class="text-red-700">${errorMessage}</span>
                </div>
            `;
            this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center bg-red-50 border border-red-200';
            return;
        }
        
        if (!stats) return;
        
        if (stats.is_trained) {
            this.elements.modelStatus.innerHTML = `
                <div class="flex items-center justify-center">
                    <i class="bi bi-check-circle text-green-600 mr-2"></i>
                    <span class="text-green-700">Reinforcement Learning Model Ready (${stats.feedback_count} feedback samples)</span>
                </div>
            `;
            this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center bg-green-50 border border-green-200';
        } else {
            this.elements.modelStatus.innerHTML = `
                <div class="flex items-center justify-center">
                    <i class="bi bi-info-circle text-blue-600 mr-2"></i>
                    <span class="text-blue-700">Using heuristic-based classification (needs training data)</span>
                </div>
            `;
            this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center bg-blue-50 border border-blue-200';
        }
    }
    
    updateClassifyButton() {
        if (!this.elements.classifyBtn || !this.elements.urlInput) return;
        
        const url = this.elements.urlInput.value.trim();
        const isValid = url.length > 0 && (url.startsWith('http') || url.includes('.'));
        
        this.elements.classifyBtn.disabled = !isValid;
    }
    
    async classifyUrl() {
        const url = this.elements.urlInput.value.trim();
        if (!url) {
            this.showError('Please enter a URL');
            return;
        }
        
        this.currentUrl = url;
        this.hideResults();
        this.hideError();
        this.showLoading(true);
        
        try {
            const response = await fetch('/classify-url', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: url })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentPrediction = data;
                this.displayResults(data);
            } else {
                this.showError(data.error || 'Classification failed');
            }
        } catch (error) {
            console.error('Classification error:', error);
            this.showError('Network error occurred. Please try again.');
        } finally {
            this.showLoading(false);
        }
    }
    
    displayResults(data) {
        // Update prediction card
        const predictionText = data.is_news_article ? 'News Article' : 'Regular Website';
        const confidence = Math.round(data.confidence * 100);
        
        this.elements.predictionText.textContent = predictionText;
        this.elements.confidenceText.textContent = `${confidence}%`;
        this.elements.modelType.textContent = data.heuristic_based ? 
            'Based on heuristic rules' : 'Based on trained ML model';
        
        // Update card styling
        const cardClass = data.is_news_article ? 
            'rounded-xl p-6 mb-6 border-l-4 shadow-lg bg-green-50 border-green-500' :
            'rounded-xl p-6 mb-6 border-l-4 shadow-lg bg-gray-50 border-gray-500';
        this.elements.predictionCard.className = cardClass;
        
        // Update text colors
        const textClass = data.is_news_article ? 'text-green-700' : 'text-gray-700';
        this.elements.predictionText.className = `text-3xl font-bold mb-2 ${textClass}`;
        this.elements.confidenceText.className = `text-2xl font-semibold ${textClass}`;
        
        // Update probability bars
        const newsProb = Math.round(data.probability_news * 100);
        const regularProb = Math.round(data.probability_not_news * 100);
        
        this.elements.newsPercentage.textContent = `${newsProb}%`;
        this.elements.regularPercentage.textContent = `${regularProb}%`;
        
        // Animate bars
        setTimeout(() => {
            this.elements.newsBar.style.width = `${newsProb}%`;
            this.elements.regularBar.style.width = `${regularProb}%`;
        }, 100);
        
        // Show results
        this.elements.results.classList.remove('hidden');
    }
    
    async submitFeedback(isCorrect) {
        if (!this.currentPrediction) return;
        
        const userConfidence = parseFloat(this.elements.confidenceSlider.value);
        const actualLabel = isCorrect ? this.currentPrediction.is_news_article : !this.currentPrediction.is_news_article;
        
        try {
            const response = await fetch('/url-classifier-feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    url: this.currentUrl,
                    predicted_label: this.currentPrediction.is_news_article,
                    actual_label: actualLabel,
                    user_confidence: userConfidence
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showFeedbackMessage(
                    `Thank you! Feedback recorded. Model accuracy: ${Math.round(data.model_accuracy * 100)}%`,
                    'success'
                );
                
                // Disable feedback buttons
                this.elements.correctBtn.disabled = true;
                this.elements.incorrectBtn.disabled = true;
                this.elements.correctBtn.textContent = 'Feedback Submitted';
                this.elements.incorrectBtn.style.display = 'none';
                
                // Refresh statistics
                this.loadStatistics();
            } else {
                this.showFeedbackMessage(data.error || 'Failed to submit feedback', 'error');
            }
        } catch (error) {
            console.error('Feedback submission error:', error);
            this.showFeedbackMessage('Network error occurred', 'error');
        }
    }
    
    showFeedbackMessage(message, type) {
        if (!this.elements.feedbackMessage) return;
        
        const bgClass = type === 'success' ? 'bg-green-100 border-green-400 text-green-700' : 'bg-red-100 border-red-400 text-red-700';
        const iconClass = type === 'success' ? 'bi-check-circle' : 'bi-x-circle';
        
        this.elements.feedbackMessage.className = `mt-4 p-3 rounded-lg border ${bgClass}`;
        this.elements.feedbackMessage.innerHTML = `
            <div class="flex items-center">
                <i class="bi ${iconClass} mr-2"></i>
                ${message}
            </div>
        `;
        this.elements.feedbackMessage.classList.remove('hidden');
        
        // Hide after 5 seconds
        setTimeout(() => {
            this.elements.feedbackMessage.classList.add('hidden');
        }, 5000);
    }
    
    async loadStatistics() {
        try {
            const response = await fetch('/url-classifier-stats');
            const data = await response.json();
            
            if (data.success) {
                this.updateStatistics(data.stats);
                this.updateRecentFeedback(data.recent_feedback);
            }
        } catch (error) {
            console.error('Failed to load statistics:', error);
        }
    }
    
    updateStatistics(stats) {
        if (this.elements.feedbackCount) {
            this.elements.feedbackCount.textContent = stats.feedback_count || 0;
        }
        if (this.elements.modelAccuracy) {
            const accuracy = stats.feedback_accuracy ? Math.round(stats.feedback_accuracy * 100) : 0;
            this.elements.modelAccuracy.textContent = `${accuracy}%`;
        }
        if (this.elements.trainingIterations) {
            this.elements.trainingIterations.textContent = stats.training_iterations || 0;
        }
    }
    
    updateRecentFeedback(feedbackList) {
        if (!this.elements.feedbackList || !feedbackList) return;
        
        if (feedbackList.length === 0) {
            this.elements.feedbackList.innerHTML = '<p class="text-gray-500 text-sm">No feedback submitted yet</p>';
            return;
        }
        
        const feedbackHtml = feedbackList.map(feedback => {
            const statusIcon = feedback.was_correct ? 
                '<i class="bi bi-check-circle text-green-600"></i>' : 
                '<i class="bi bi-x-circle text-red-600"></i>';
            const predictionText = feedback.predicted_label ? 'News' : 'Regular';
            const actualText = feedback.actual_label ? 'News' : 'Regular';
            const date = new Date(feedback.timestamp).toLocaleString();
            
            return `
                <div class="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <div class="flex items-center space-x-2">
                        ${statusIcon}
                        <span class="text-sm">
                            Predicted: <strong>${predictionText}</strong>, 
                            Actual: <strong>${actualText}</strong>
                        </span>
                    </div>
                    <span class="text-xs text-gray-500">${date}</span>
                </div>
            `;
        }).join('');
        
        this.elements.feedbackList.innerHTML = feedbackHtml;
    }
    
    async retrainModel() {
        this.elements.retrainBtn.disabled = true;
        this.elements.retrainBtn.innerHTML = '<i class="bi bi-hourglass-split animate-spin mr-1"></i>Retraining...';
        
        try {
            const response = await fetch('/retrain-url-classifier', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    force_retrain: true
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showFeedbackMessage(
                    `Model retrained successfully! Accuracy: ${Math.round(data.accuracy * 100)}%`,
                    'success'
                );
                this.loadStatistics();
                this.checkModelStatus();
            } else {
                this.showFeedbackMessage(data.error || 'Retraining failed', 'error');
            }
        } catch (error) {
            console.error('Retraining error:', error);
            this.showFeedbackMessage('Network error occurred', 'error');
        } finally {
            this.elements.retrainBtn.disabled = false;
            this.elements.retrainBtn.innerHTML = '<i class="bi bi-arrow-clockwise mr-1"></i>Retrain Model';
        }
    }
    
    showLoading(show) {
        if (show) {
            this.elements.loading.classList.remove('hidden');
            this.elements.classifyBtn.disabled = true;
            this.elements.classifyBtn.innerHTML = '<i class="bi bi-hourglass-split animate-spin mr-2"></i>Analyzing...';
        } else {
            this.elements.loading.classList.add('hidden');
            this.updateClassifyButton();
            this.elements.classifyBtn.innerHTML = '<i class="bi bi-search mr-2"></i>Classify URL';
        }
    }
    
    hideResults() {
        this.elements.results.classList.add('hidden');
        
        // Reset feedback buttons
        this.elements.correctBtn.disabled = false;
        this.elements.incorrectBtn.disabled = false;
        this.elements.correctBtn.innerHTML = '<i class="bi bi-check-circle mr-1"></i>Correct';
        this.elements.incorrectBtn.style.display = 'inline-block';
        this.elements.feedbackMessage.classList.add('hidden');
    }
    
    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.error.classList.remove('hidden');
        
        // Shake animation
        this.elements.error.style.animation = 'shake 0.5s';
        setTimeout(() => {
            this.elements.error.style.animation = '';
        }, 500);
    }
    
    hideError() {
        this.elements.error.classList.add('hidden');
    }
}

// Add shake animation CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
`;
document.head.appendChild(style);

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.urlClassifier = new URLClassifier();
});
