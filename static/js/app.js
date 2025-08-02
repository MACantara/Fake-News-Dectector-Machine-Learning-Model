// Main application logic for the Advanced News Analyzer
class NewsAnalyzer {
    constructor() {
        // Application state
        this.state = {
            currentInputType: Config.inputTypes.TEXT,
            currentAnalysisType: Config.analysisTypes.FAKE_NEWS,
            modelsReady: { 
                fake_news: false, 
                political: false 
            },
            isLoading: false
        };

        // DOM elements cache
        this.elements = {};
        
        // Initialize the application
        this.init();
    }

    // Initialize the application
    init() {
        this.cacheElements();
        this.bindEvents();
        this.checkModelStatus();
        this.updateTextCount();
        this.startPeriodicStatusCheck();
        
        // Initialize advanced features if available
        if (typeof AdvancedFeatures !== 'undefined') {
            this.advancedFeatures = new AdvancedFeatures(this);
        }
        
        console.log('News Analyzer initialized successfully');
    }

    // Cache DOM elements for performance
    cacheElements() {
        const elementIds = [
            'textBtn', 'urlBtn', 'fakeNewsBtn', 'politicalBtn', 'bothBtn',
            'textInput', 'urlInput', 'newsText', 'articleUrl', 'analyzeBtn',
            'loading', 'results', 'error', 'modelStatus', 'textCount',
            'fakeNewsResults', 'politicalResults', 'extractedContent',
            'fakeNewsPredictionCard', 'fakeNewsPredictionText', 'fakeNewsConfidenceText',
            'politicalPredictionCard', 'politicalPredictionText', 'politicalConfidenceText',
            'fakeBar', 'realBar', 'politicalBar', 'nonPoliticalBar',
            'fakePercentage', 'realPercentage', 'politicalPercentage', 'nonPoliticalPercentage',
            'reasoningText', 'extractedTitle', 'extractedPreview', 'errorMessage'
        ];

        elementIds.forEach(id => {
            this.elements[id] = Utils.dom.getElementById(id);
        });
    }

    // Bind event listeners
    bindEvents() {
        // Input type switching
        if (this.elements.textBtn) {
            this.elements.textBtn.addEventListener('click', () => this.switchInputType(Config.inputTypes.TEXT));
        }
        if (this.elements.urlBtn) {
            this.elements.urlBtn.addEventListener('click', () => this.switchInputType(Config.inputTypes.URL));
        }

        // Analysis type switching
        if (this.elements.fakeNewsBtn) {
            this.elements.fakeNewsBtn.addEventListener('click', () => this.switchAnalysisType(Config.analysisTypes.FAKE_NEWS));
        }
        if (this.elements.politicalBtn) {
            this.elements.politicalBtn.addEventListener('click', () => this.switchAnalysisType(Config.analysisTypes.POLITICAL));
        }
        if (this.elements.bothBtn) {
            this.elements.bothBtn.addEventListener('click', () => this.switchAnalysisType(Config.analysisTypes.BOTH));
        }

        // Input monitoring
        if (this.elements.newsText) {
            const debouncedUpdate = Utils.debounce(() => {
                this.updateTextCount();
                this.updateAnalyzeButton();
            }, 300);
            
            this.elements.newsText.addEventListener('input', debouncedUpdate);
        }
        
        if (this.elements.articleUrl) {
            this.elements.articleUrl.addEventListener('input', () => this.updateAnalyzeButton());
        }

        // Analyze button
        if (this.elements.analyzeBtn) {
            this.elements.analyzeBtn.addEventListener('click', () => this.analyzeContent());
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                if (!this.elements.analyzeBtn.disabled) {
                    this.analyzeContent();
                }
            }
        });
    }

    // Check model status
    async checkModelStatus() {
        try {
            const result = await Utils.http.get(Config.endpoints.modelStatus);
            
            if (result.success) {
                const data = result.data;
                this.state.modelsReady.fake_news = data.fake_news_model?.is_trained || false;
                this.state.modelsReady.political = data.political_model?.is_trained || false;
                
                this.updateModelStatusDisplay();
                this.updateAnalyzeButton();
            } else {
                this.showModelError();
            }
        } catch (error) {
            console.error('Failed to check model status:', error);
            this.showModelError();
        }
    }

    // Update model status display
    updateModelStatusDisplay() {
        if (!this.elements.modelStatus) return;

        let statusText = '';
        let statusClass = 'glass-effect rounded-xl p-4 mb-8 text-center';
        
        if (this.state.modelsReady.fake_news && this.state.modelsReady.political) {
            statusText = `
                <i class="bi bi-check-circle text-green-600 mr-2"></i>
                <span class="text-gray-700">${Config.messages.modelReady}</span>
            `;
            statusClass += ' border-green-200 status-ready';
        } else if (this.state.modelsReady.fake_news) {
            statusText = `
                <i class="bi bi-check-circle text-blue-600 mr-2"></i>
                <span class="text-gray-700">${Config.messages.modelPartial}</span>
            `;
            statusClass += ' border-blue-200 status-partial';
        } else {
            statusText = `
                <i class="bi bi-hourglass-split spin text-blue-600 mr-2"></i>
                <span class="text-gray-700">${Config.messages.modelLoading}</span>
            `;
            statusClass += ' status-loading';
        }
        
        Utils.dom.setHTML(this.elements.modelStatus, `<div class="flex items-center justify-center">${statusText}</div>`);
        this.elements.modelStatus.className = statusClass;
    }

    // Show model error
    showModelError() {
        if (!this.elements.modelStatus) return;

        const statusText = `
            <div class="flex items-center justify-center">
                <i class="bi bi-exclamation-triangle text-red-600 mr-2"></i>
                <span class="text-gray-700">${Config.messages.modelError}</span>
            </div>
        `;
        
        Utils.dom.setHTML(this.elements.modelStatus, statusText);
        this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center border border-red-200 status-error';
    }

    // Switch input type
    switchInputType(type) {
        this.state.currentInputType = type;
        
        if (type === Config.inputTypes.TEXT) {
            this.elements.textBtn.className = Config.cssClasses.button.active.textInput;
            this.elements.urlBtn.className = Config.cssClasses.button.inactive.input;
            Utils.dom.show(this.elements.textInput);
            Utils.dom.hide(this.elements.urlInput);
        } else {
            this.elements.urlBtn.className = Config.cssClasses.button.active.urlInput;
            this.elements.textBtn.className = Config.cssClasses.button.inactive.input;
            Utils.dom.show(this.elements.urlInput);
            Utils.dom.hide(this.elements.textInput);
        }
        
        this.updateAnalyzeButton();
        this.hideResults();
        
        // Save preference
        Utils.storage.set('preferredInputType', type);
    }

    // Switch analysis type
    switchAnalysisType(type) {
        this.state.currentAnalysisType = type;
        
        // Reset all buttons to inactive state
        this.elements.fakeNewsBtn.className = Config.cssClasses.button.inactive.default;
        this.elements.politicalBtn.className = Config.cssClasses.button.inactive.default;
        this.elements.bothBtn.className = Config.cssClasses.button.inactive.default;
        
        // Set active button
        switch (type) {
            case Config.analysisTypes.FAKE_NEWS:
                this.elements.fakeNewsBtn.className = Config.cssClasses.button.active.fakeNews;
                break;
            case Config.analysisTypes.POLITICAL:
                this.elements.politicalBtn.className = Config.cssClasses.button.active.political;
                break;
            case Config.analysisTypes.BOTH:
                this.elements.bothBtn.className = Config.cssClasses.button.active.both;
                break;
        }
        
        this.updateAnalyzeButton();
        this.hideResults();
        
        // Save preference
        Utils.storage.set('preferredAnalysisType', type);
    }

    // Update analyze button state
    updateAnalyzeButton() {
        if (!this.elements.analyzeBtn) return;

        const hasInput = this.state.currentInputType === Config.inputTypes.TEXT 
            ? this.elements.newsText?.value.trim().length > 0 
            : this.elements.articleUrl?.value.trim().length > 0;
        
        const modelAvailable = this.state.currentAnalysisType === Config.analysisTypes.FAKE_NEWS 
            ? this.state.modelsReady.fake_news
            : this.state.currentAnalysisType === Config.analysisTypes.POLITICAL 
            ? this.state.modelsReady.political
            : this.state.modelsReady.fake_news && this.state.modelsReady.political;
        
        this.elements.analyzeBtn.disabled = !modelAvailable || !hasInput || this.state.isLoading;
    }

    // Update text count
    updateTextCount() {
        if (!this.elements.newsText || !this.elements.textCount) return;

        const count = this.elements.newsText.value.length;
        Utils.dom.setText(this.elements.textCount, `${count} characters`);
        
        // Add visual feedback for text length
        if (count > Config.validation.maxTextLength) {
            this.elements.textCount.className = 'text-sm text-red-500';
        } else if (count < Config.validation.minTextLength) {
            this.elements.textCount.className = 'text-sm text-yellow-500';
        } else {
            this.elements.textCount.className = 'text-sm text-gray-500';
        }
    }

    // Hide results and errors
    hideResults() {
        Utils.dom.hide(this.elements.results);
        Utils.dom.hide(this.elements.error);
    }

    // Show error message
    showError(message) {
        Utils.dom.setText(this.elements.errorMessage, message);
        Utils.dom.show(this.elements.error);
        Utils.animations.shake(this.elements.error);
    }

    // Validate input before analysis
    validateInput() {
        if (this.state.currentInputType === Config.inputTypes.TEXT) {
            return Utils.validation.validateText(this.elements.newsText?.value);
        } else {
            return Utils.validation.validateURL(this.elements.articleUrl?.value);
        }
    }

    // Analyze content
    async analyzeContent() {
        // Validate input first
        const validation = this.validateInput();
        if (!validation.valid) {
            this.showError(validation.message);
            return;
        }

        this.hideResults();
        this.state.isLoading = true;
        Utils.dom.show(this.elements.loading);
        this.updateAnalyzeButton();
        
        try {
            const requestData = {
                type: this.state.currentInputType,
                analysis_type: this.state.currentAnalysisType
            };
            
            if (this.state.currentInputType === Config.inputTypes.TEXT) {
                requestData.text = this.elements.newsText.value.trim();
            } else {
                requestData.url = this.elements.articleUrl.value.trim();
            }
            
            const result = await Utils.http.post(Config.endpoints.predict, requestData);
            
            if (result.success) {
                this.displayResults(result.data);
                Utils.animations.pulse(this.elements.results);
            } else {
                this.showError(result.error || 'An error occurred during analysis');
            }
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError(Config.messages.networkError);
        } finally {
            this.state.isLoading = false;
            Utils.dom.hide(this.elements.loading);
            this.updateAnalyzeButton();
        }
    }

    // Display analysis results
    displayResults(data) {
        // Store results for export functionality
        Utils.storage.set('lastAnalysisResults', {
            ...data,
            timestamp: new Date().toISOString(),
            input_text: this.getCurrentInputText()
        });
        
        // Track analytics event if available
        if (this.advancedFeatures) {
            this.advancedFeatures.trackEvent('analysis_completed', {
                analysis_type: this.state.currentAnalysisType,
                input_type: this.state.currentInputType,
                has_fake_news: !!data.fake_news,
                has_political: !!data.political_classification
            });
        }
        
        // Hide individual result sections first
        Utils.dom.hide(this.elements.fakeNewsResults);
        Utils.dom.hide(this.elements.politicalResults);
        
        // Show fake news results if available
        if (data.fake_news && !data.fake_news.error) {
            this.displayFakeNewsResults(data.fake_news);
            Utils.dom.show(this.elements.fakeNewsResults);
        }
        
        // Show political results if available
        if (data.political_classification && !data.political_classification.error) {
            this.displayPoliticalResults(data.political_classification);
            Utils.dom.show(this.elements.politicalResults);
        }
        
        // Show extracted content if available
        if (data.extracted_content) {
            this.displayExtractedContent(data.extracted_content);
            Utils.dom.show(this.elements.extractedContent);
        } else {
            Utils.dom.hide(this.elements.extractedContent);
        }
        
        Utils.dom.show(this.elements.results);
    }

    // Display fake news results
    displayFakeNewsResults(data) {
        // Set prediction text and confidence
        Utils.dom.setText(this.elements.fakeNewsPredictionText, data.prediction);
        Utils.dom.setText(this.elements.fakeNewsConfidenceText, Utils.format.confidence(data.confidence));
        
        // Set card styling based on prediction
        if (data.prediction === 'Fake') {
            this.elements.fakeNewsPredictionCard.className = Config.cssClasses.prediction.fake;
            this.elements.fakeNewsPredictionText.className = Config.cssClasses.text.fake;
        } else {
            this.elements.fakeNewsPredictionCard.className = Config.cssClasses.prediction.real;
            this.elements.fakeNewsPredictionText.className = Config.cssClasses.text.real;
        }
        
        // Animate probability bars
        const fakeProb = data.probabilities.Fake * 100;
        const realProb = data.probabilities.Real * 100;
        
        Utils.animations.animateProgressBar(this.elements.fakeBar, fakeProb);
        Utils.animations.animateProgressBar(this.elements.realBar, realProb);
        Utils.dom.setText(this.elements.fakePercentage, Utils.format.percentage(data.probabilities.Fake));
        Utils.dom.setText(this.elements.realPercentage, Utils.format.percentage(data.probabilities.Real));
    }

    // Display political results
    displayPoliticalResults(data) {
        // Set prediction text and confidence
        Utils.dom.setText(this.elements.politicalPredictionText, data.prediction);
        Utils.dom.setText(this.elements.politicalConfidenceText, Utils.format.confidence(data.confidence));
        
        // Set card styling based on prediction
        if (data.prediction === 'Political') {
            this.elements.politicalPredictionCard.className = Config.cssClasses.prediction.political;
            this.elements.politicalPredictionText.className = Config.cssClasses.text.political;
        } else {
            this.elements.politicalPredictionCard.className = Config.cssClasses.prediction.nonPolitical;
            this.elements.politicalPredictionText.className = Config.cssClasses.text.nonPolitical;
        }
        
        // Animate probability bars
        const politicalProb = data.probabilities.Political * 100;
        const nonPoliticalProb = data.probabilities['Non-Political'] * 100;
        
        Utils.animations.animateProgressBar(this.elements.politicalBar, politicalProb);
        Utils.animations.animateProgressBar(this.elements.nonPoliticalBar, nonPoliticalProb);
        Utils.dom.setText(this.elements.politicalPercentage, Utils.format.percentage(data.probabilities.Political));
        Utils.dom.setText(this.elements.nonPoliticalPercentage, Utils.format.percentage(data.probabilities['Non-Political']));
        
        // Set reasoning
        Utils.dom.setText(this.elements.reasoningText, data.reasoning || 'No reasoning available');
    }

    // Display extracted content
    displayExtractedContent(content) {
        Utils.dom.setText(this.elements.extractedTitle, content.title || 'No title available');
        Utils.dom.setText(this.elements.extractedPreview, content.content_preview || 'No preview available');
    }

    // Start periodic model status checking
    startPeriodicStatusCheck() {
        setInterval(() => {
            if (!this.state.isLoading) {
                this.checkModelStatus();
            }
        }, Config.intervals.modelStatusCheck);
    }

    // Get current input text
    getCurrentInputText() {
        if (this.state.currentInputType === Config.inputTypes.TEXT) {
            return this.elements.newsText?.value || '';
        } else {
            return this.elements.articleUrl?.value || '';
        }
    }

    // Hide results
    hideResults() {
        Utils.dom.hide(this.elements.results);
        Utils.dom.hide(this.elements.fakeNewsResults);
        Utils.dom.hide(this.elements.politicalResults);
        Utils.dom.hide(this.elements.extractedContent);
    }

    // Load user preferences
    loadPreferences() {
        const savedInputType = Utils.storage.get('preferredInputType', Config.inputTypes.TEXT);
        const savedAnalysisType = Utils.storage.get('preferredAnalysisType', Config.analysisTypes.FAKE_NEWS);
        
        this.switchInputType(savedInputType);
        this.switchAnalysisType(savedAnalysisType);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.newsAnalyzer = new NewsAnalyzer();
});

// Export for testing purposes
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NewsAnalyzer;
}
