// Main application class for News Analyzer
class NewsAnalyzer {
    constructor() {
        // Initialize application state
        this.state = {
            currentInputType: Config.inputTypes.TEXT,
            currentAnalysisType: Config.analysisTypes.FAKE_NEWS,
            currentCrawlingMode: Config.crawlingModes.PREVIEW,
            modelsReady: { 
                fake_news: false, 
                political: false 
            },
            isLoading: false,
            bulkLabelingMode: false
        };

        // DOM elements cache
        this.elements = {};
        this.crawledArticles = [];
        this.bulkLabels = new Map(); // Store bulk labels for articles

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
            'textBtn', 'urlBtn', 'websiteBtn', 'fakeNewsBtn', 'politicalBtn', 'bothBtn',
            'textInput', 'urlInput', 'websiteInput', 'newsText', 'articleUrl', 'websiteUrl',
            'crawlOnlyBtn', 'crawlAnalyzeBtn', 'analyzeFoundArticlesBtn',
            'analyzeBtn', 'loading', 'results', 'websiteResults',
            'error', 'modelStatus', 'textCount',
            'fakeNewsResults', 'politicalResults', 'extractedContent',
            'fakeNewsPredictionCard', 'fakeNewsPredictionText', 'fakeNewsConfidenceText',
            'politicalPredictionCard', 'politicalPredictionText', 'politicalConfidenceText',
            'fakeBar', 'realBar', 'politicalBar', 'nonPoliticalBar',
            'fakePercentage', 'realPercentage', 'politicalPercentage', 'nonPoliticalPercentage',
            'reasoningText', 'extractedTitle', 'extractedPreview', 'errorMessage',
            'crawlingSummary', 'totalArticlesCount', 'successfulAnalysesCount', 'failedAnalysesCount',
            'analyzedWebsiteTitle', 'overallStats', 'fakeNewsStats', 'politicalStats',
            'fakeArticlesCount', 'realArticlesCount', 'fakePercentageOverall',
            'politicalArticlesCount', 'nonPoliticalArticlesCount', 'politicalPercentageOverall',
            'articleResultsList', 'crawledArticlesList', 'articleLinksContainer',
            // Bulk labeling elements
            'toggleBulkLabelingBtn', 'bulkActionControls', 'bulkLabelingInstructions',
            'selectAllBtn', 'deselectAllBtn', 'bulkLabelNewsBtn', 'bulkLabelNotNewsBtn',
            'submitBulkFeedbackBtn', 'labeledCount'
        ];

        elementIds.forEach(id => {
            this.elements[id] = document.getElementById(id);
        });

        // Add similar content elements
        this.elements.similarContentSection = document.getElementById('similarContentSection');
        this.elements.similarContentResults = document.getElementById('similarContentResults');
        this.elements.similarContentList = document.getElementById('similarContentList');
        this.elements.findSimilarBtn = document.getElementById('findSimilarBtn');
        this.elements.similarContentSummary = document.getElementById('similarContentSummary');
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
        if (this.elements.websiteBtn) {
            this.elements.websiteBtn.addEventListener('click', () => this.switchInputType(Config.inputTypes.WEBSITE));
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
            this.elements.newsText.addEventListener('input', () => {
                this.updateTextCount();
                this.updateAnalyzeButton();
            });
        }

        if (this.elements.articleUrl) {
            this.elements.articleUrl.addEventListener('input', () => this.updateAnalyzeButton());
        }

        // Analyze button
        if (this.elements.analyzeBtn) {
            this.elements.analyzeBtn.addEventListener('click', () => this.analyzeContent());
        }

        // Website crawling mode buttons
        if (this.elements.crawlOnlyBtn) {
            this.elements.crawlOnlyBtn.addEventListener('click', () => this.switchCrawlingMode(Config.crawlingModes.PREVIEW));
        }
        if (this.elements.crawlAnalyzeBtn) {
            this.elements.crawlAnalyzeBtn.addEventListener('click', () => this.switchCrawlingMode(Config.crawlingModes.ANALYZE));
        }

        // Analyze found articles button
        if (this.elements.analyzeFoundArticlesBtn) {
            this.elements.analyzeFoundArticlesBtn.addEventListener('click', () => this.analyzeFoundArticles());
        }

        // Bulk labeling event listeners
        if (this.elements.toggleBulkLabelingBtn) {
            this.elements.toggleBulkLabelingBtn.addEventListener('click', () => this.toggleBulkLabeling());
        }
        if (this.elements.selectAllBtn) {
            this.elements.selectAllBtn.addEventListener('click', () => this.selectAllArticles());
        }
        if (this.elements.deselectAllBtn) {
            this.elements.deselectAllBtn.addEventListener('click', () => this.deselectAllArticles());
        }
        if (this.elements.bulkLabelNewsBtn) {
            this.elements.bulkLabelNewsBtn.addEventListener('click', () => this.bulkLabelArticles(true));
        }
        if (this.elements.bulkLabelNotNewsBtn) {
            this.elements.bulkLabelNotNewsBtn.addEventListener('click', () => this.bulkLabelArticles(false));
        }
        if (this.elements.submitBulkFeedbackBtn) {
            this.elements.submitBulkFeedbackBtn.addEventListener('click', () => this.submitBulkFeedback());
        }

        // Website URL input monitoring
        if (this.elements.websiteUrl) {
            this.elements.websiteUrl.addEventListener('input', () => this.updateAnalyzeButton());
        }

        // Find similar content button
        if (this.elements.findSimilarBtn) {
            this.elements.findSimilarBtn.addEventListener('click', () => this.findSimilarContent());
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
            const response = await Utils.http.get(Config.endpoints.modelStatus);
            
            if (!response.success) {
                throw new Error(response.error || 'Failed to check model status');
            }

            // Handle both nested and direct response formats
            const status = response.data?.status || response.status || response.data || {};
            
            this.state.modelsReady = {
                fake_news: status.fake_news || false,
                political: status.political || false
            };

            this.updateModelStatusDisplay(status);
        } catch (error) {
            console.error('Model status check failed:', error);
            this.updateModelStatusDisplay(null, error.message);
        }
    }

    updateModelStatusDisplay(status, errorMessage) {
        if (!this.elements.modelStatus) return;

        if (errorMessage) {
            this.elements.modelStatus.innerHTML = `
                <div class="flex items-center justify-center">
                    <i class="bi bi-exclamation-triangle text-red-600 mr-2"></i>
                    <span class="text-red-700">${errorMessage}</span>
                </div>
            `;
            this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center border-l-4 border-red-500';
            return;
        }

        if (!status) return;

        const fakeNewsReady = status.fake_news;
        const politicalReady = status.political;
        const bothReady = fakeNewsReady && politicalReady;

        if (bothReady) {
            this.elements.modelStatus.innerHTML = `
                <div class="flex items-center justify-center">
                    <i class="bi bi-check-circle text-green-600 mr-2"></i>
                    <span class="text-green-700">All models ready</span>
                </div>
            `;
            this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center border-l-4 border-green-500';
        } else if (fakeNewsReady) {
            this.elements.modelStatus.innerHTML = `
                <div class="flex items-center justify-center">
                    <i class="bi bi-hourglass-split animate-spin text-yellow-600 mr-2"></i>
                    <span class="text-yellow-700">Fake news model ready, political classifier loading...</span>
                </div>
            `;
            this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center border-l-4 border-amber-500';
        } else {
            this.elements.modelStatus.innerHTML = `
                <div class="flex items-center justify-center">
                    <i class="bi bi-hourglass-split animate-spin text-blue-600 mr-2"></i>
                    <span class="text-gray-700">Loading models...</span>
                </div>
            `;
            this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center border-l-4 border-blue-500';
        }
    }

    // Start periodic model status checking
    startPeriodicStatusCheck() {
        setInterval(() => {
            this.checkModelStatus();
        }, Config.intervals.modelStatusCheck);
    }

    // Switch input type
    switchInputType(type) {
        this.state.currentInputType = type;
        
        // Reset button styles
        if (this.elements.textBtn) {
            this.elements.textBtn.className = Config.cssClasses.button.inactive.input;
        }
        if (this.elements.urlBtn) {
            this.elements.urlBtn.className = Config.cssClasses.button.inactive.input;
        }
        if (this.elements.websiteBtn) {
            this.elements.websiteBtn.className = Config.cssClasses.button.inactive.input;
        }
        
        // Hide all input sections
        Utils.dom.hide(this.elements.textInput);
        Utils.dom.hide(this.elements.urlInput);
        Utils.dom.hide(this.elements.websiteInput);
        
        // Get analysis sections for hiding/showing
        const analysisTypeSection = document.getElementById('analysisTypeSection');
        const analyzeButtonSection = document.getElementById('analyzeButtonSection');
        
        // Show appropriate section and activate button
        if (type === Config.inputTypes.TEXT) {
            if (this.elements.textBtn) {
                this.elements.textBtn.className = Config.cssClasses.button.active.textInput;
            }
            Utils.dom.show(this.elements.textInput);
            // Show analysis sections for text input
            if (analysisTypeSection) Utils.dom.show(analysisTypeSection);
            if (analyzeButtonSection) Utils.dom.show(analyzeButtonSection);
        } else if (type === Config.inputTypes.URL) {
            if (this.elements.urlBtn) {
                this.elements.urlBtn.className = Config.cssClasses.button.active.urlInput;
            }
            Utils.dom.show(this.elements.urlInput);
            // Show analysis sections for URL input
            if (analysisTypeSection) Utils.dom.show(analysisTypeSection);
            if (analyzeButtonSection) Utils.dom.show(analyzeButtonSection);
        } else if (type === Config.inputTypes.WEBSITE) {
            if (this.elements.websiteBtn) {
                this.elements.websiteBtn.className = Config.cssClasses.button.active.urlInput; // Reuse URL styling
            }
            Utils.dom.show(this.elements.websiteInput);
            // Show analysis sections for website input
            if (analysisTypeSection) Utils.dom.show(analysisTypeSection);
            if (analyzeButtonSection) Utils.dom.show(analyzeButtonSection);
        }
        
        this.updateAnalyzeButton();
        this.hideResults();
    }

    // Switch analysis type
    switchAnalysisType(type) {
        this.state.currentAnalysisType = type;
        
        // Update button styles
        if (this.elements.fakeNewsBtn) {
            this.elements.fakeNewsBtn.className = (type === Config.analysisTypes.FAKE_NEWS || type === Config.analysisTypes.BOTH) ? 
                Config.cssClasses.button.active.fakeNews : Config.cssClasses.button.inactive.default;
        }
        if (this.elements.politicalBtn) {
            this.elements.politicalBtn.className = (type === Config.analysisTypes.POLITICAL || type === Config.analysisTypes.BOTH) ?
                Config.cssClasses.button.active.political : Config.cssClasses.button.inactive.default;
        }
        if (this.elements.bothBtn) {
            this.elements.bothBtn.className = (type === Config.analysisTypes.BOTH) ?
                Config.cssClasses.button.active.both : Config.cssClasses.button.inactive.default;
        }
        
        this.hideResults();
    }

    // Switch crawling mode (preview vs analyze)
    switchCrawlingMode(mode) {
        this.state.currentCrawlingMode = mode;
        
        // Reset all button styles
        const buttons = [this.elements.crawlOnlyBtn, this.elements.crawlAnalyzeBtn];
        const inactiveClass = 'crawl-mode-btn bg-gray-200 text-gray-700 px-3 py-2 rounded-lg text-sm font-medium focus-ring';
        const activeClass = 'crawl-mode-btn bg-blue-600 text-white px-3 py-2 rounded-lg text-sm font-medium focus-ring';
        
        buttons.forEach(btn => {
            if (btn) btn.className = inactiveClass;
        });
        
        // Set active button
        if (mode === Config.crawlingModes.PREVIEW && this.elements.crawlOnlyBtn) {
            this.elements.crawlOnlyBtn.className = activeClass;
        } else if (mode === Config.crawlingModes.ANALYZE && this.elements.crawlAnalyzeBtn) {
            this.elements.crawlAnalyzeBtn.className = activeClass;
        }
    }

    // Hide results
    hideResults() {
        Utils.dom.hide(this.elements.results);
        Utils.dom.hide(this.elements.websiteResults);
        Utils.dom.hide(this.elements.similarContentSection);
        Utils.dom.hide(this.elements.error);
    }

    // Show error message
    showError(message) {
        Utils.dom.setText(this.elements.errorMessage, message);
        Utils.dom.show(this.elements.error);
        Utils.animations.shake(this.elements.error);
    }

    // Update text count
    updateTextCount() {
        if (this.elements.newsText && this.elements.textCount) {
            const length = this.elements.newsText.value.length;
            this.elements.textCount.textContent = `${length} characters`;
        }
    }

    // Update analyze button state
    updateAnalyzeButton() {
        if (!this.elements.analyzeBtn) return;
        
        const validation = this.validateInput();
        this.elements.analyzeBtn.disabled = this.state.isLoading || !validation.valid;
        
        if (this.state.isLoading) {
            this.elements.analyzeBtn.innerHTML = '<i class="bi bi-hourglass-split animate-spin mr-2"></i>Analyzing...';
        } else {
            this.elements.analyzeBtn.innerHTML = '<i class="bi bi-search mr-2"></i>Analyze News';
        }
    }

    // Validate user input
    validateInput() {
        if (this.state.currentInputType === Config.inputTypes.TEXT) {
            const text = this.elements.newsText?.value?.trim() || '';
            if (text.length < Config.validation.minTextLength) {
                return { valid: false, message: Config.messages.textTooShort };
            }
            if (text.length > Config.validation.maxTextLength) {
                return { valid: false, message: Config.messages.textTooLong };
            }
        } else if (this.state.currentInputType === Config.inputTypes.URL) {
            const url = this.elements.articleUrl?.value?.trim() || '';
            if (!Config.validation.urlPattern.test(url)) {
                return { valid: false, message: Config.messages.invalidUrl };
            }
        } else if (this.state.currentInputType === Config.inputTypes.WEBSITE) {
            const url = this.elements.websiteUrl?.value?.trim() || '';
            if (!Config.validation.websitePattern.test(url)) {
                return { valid: false, message: 'Please enter a valid website URL' };
            }
        }
        
        return { valid: true };
    }

    // Main analysis method
    async analyzeContent() {
        // Handle website crawling separately
        if (this.state.currentInputType === Config.inputTypes.WEBSITE) {
            return await this.handleWebsiteCrawling();
        }
        
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
            const analysisData = this.prepareAnalysisData();
            const response = await Utils.http.post(Config.endpoints.predict, analysisData);
            
            if (response.success) {
                this.displayResults(response.data);
            } else {
                this.showError(response.error || 'Analysis failed. Please try again.');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(Config.messages.networkError);
        } finally {
            this.state.isLoading = false;
            Utils.dom.hide(this.elements.loading);
            this.updateAnalyzeButton();
        }
    }

    // Prepare analysis data
    prepareAnalysisData() {
        const data = {
            type: this.state.currentInputType,
            analysis_type: this.state.currentAnalysisType
        };
        
        if (this.state.currentInputType === Config.inputTypes.TEXT) {
            data.text = this.elements.newsText.value.trim();
        } else if (this.state.currentInputType === Config.inputTypes.URL) {
            data.url = this.elements.articleUrl.value.trim();
        }
        
        return data;
    }

    // Handle website crawling
    async handleWebsiteCrawling() {
        const websiteUrl = this.elements.websiteUrl?.value.trim();
        if (!websiteUrl) {
            this.showError('Please enter a website URL');
            return;
        }

        const crawlingMode = this.state.currentCrawlingMode || Config.crawlingModes.PREVIEW;

        this.hideResults();
        this.state.isLoading = true;
        Utils.dom.show(this.elements.loading);
        this.updateAnalyzeButton();

        // Start timing the crawling operation
        const startTime = performance.now();

        try {
            if (crawlingMode === Config.crawlingModes.PREVIEW) {
                await this.crawlWebsitePreview(websiteUrl, startTime);
            } else {
                await this.crawlAndAnalyzeWebsite(websiteUrl, startTime);
            }
        } catch (error) {
            console.error('Website crawling error:', error);
            this.showError('Failed to analyze website. Please try again.');
        } finally {
            this.state.isLoading = false;
            Utils.dom.hide(this.elements.loading);
            this.updateAnalyzeButton();
        }
    }

    // Crawl website for preview (just show links)
    async crawlWebsitePreview(websiteUrl, startTime) {
        const response = await Utils.http.post(Config.endpoints.crawlWebsite, {
            website_url: websiteUrl
        });

        if (!response.success) {
            throw new Error(response.error || 'Failed to crawl website');
        }

        const result = response.data;
        if (!result.success) {
            throw new Error(result.error || 'Website crawling failed');
        }

        // Calculate response time
        const endTime = performance.now();
        const responseTime = endTime - startTime;
        result.responseTime = responseTime;

        this.displayCrawledArticles(result);
    }

    // Crawl and analyze website
    async crawlAndAnalyzeWebsite(websiteUrl, startTime) {
        const response = await Utils.http.post(Config.endpoints.analyzeWebsite, {
            website_url: websiteUrl,
            analysis_type: this.state.currentAnalysisType
        });

        if (!response.success) {
            throw new Error(response.error || 'Failed to analyze website');
        }

        const result = response.data;
        if (!result.success) {
            throw new Error(result.error || 'Website analysis failed');
        }

        // Calculate response time
        const endTime = performance.now();
        const responseTime = endTime - startTime;
        result.responseTime = responseTime;

        this.displayWebsiteAnalysisResults(result);
    }

    // Display crawled articles (preview mode)
    displayCrawledArticles(data) {
        Utils.dom.hide(this.elements.results);
        Utils.dom.hide(this.elements.crawledArticlesList);
        
        if (!data.articles || data.articles.length === 0) {
            this.showError('No articles found on this website');
            return;
        }

        // Store articles for later analysis (now just URLs)
        this.crawledArticles = data.articles;

        // Update summary 
        Utils.dom.setText(this.elements.totalArticlesCount, data.total_found);
        Utils.dom.setText(this.elements.analyzedWebsiteTitle, data.website_title || 'Unknown Website');

        // Add classification method info if available
        if (data.classification_method) {
            const responseTimeText = data.responseTime ? 
                `Response time: ${this.formatResponseTime(data.responseTime)}` : '';
            
            const summaryHtml = `
                <div class="mt-4 p-3 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200">
                    <div class="flex items-center space-x-2">
                        <i class="bi bi-link text-green-600"></i>
                        <span class="text-sm font-medium text-green-800">
                            ${data.classification_method}
                        </span>
                    </div>
                    <div class="text-xs text-green-700 mt-1">
                        Extracted ${data.total_found} URLs from website
                        ${data.responseTime ? ` • ${responseTimeText}` : ''}
                    </div>
                </div>
            `;
            this.elements.analyzedWebsiteTitle.insertAdjacentHTML('afterend', summaryHtml);
        } else if (data.responseTime) {
            // Show response time even if no classification method
            const responseTimeText = `Response time: ${this.formatResponseTime(data.responseTime)}`;
            const summaryHtml = `
                <div class="mt-4 p-3 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200">
                    <div class="flex items-center space-x-2">
                        <i class="bi bi-clock text-blue-600"></i>
                        <span class="text-sm font-medium text-blue-800">
                            URL Extraction Complete
                        </span>
                    </div>
                    <div class="text-xs text-blue-700 mt-1">
                        Extracted ${data.total_found} URLs • ${responseTimeText}
                    </div>
                </div>
            `;
            this.elements.analyzedWebsiteTitle.insertAdjacentHTML('afterend', summaryHtml);
        }

        // Clear and populate article links 
        this.elements.articleLinksContainer.innerHTML = '';
        
        data.articles.forEach((article, index) => {
            // Handle both URL strings and article objects
            let articleUrl, articleText, articleTitle, classificationInfo;
            
            if (typeof article === 'string') {
                // Simple URL string (backward compatibility)
                articleUrl = article;
                articleText = '';
                articleTitle = '';
                classificationInfo = null;
            } else if (typeof article === 'object' && article.url) {
                // Article object with metadata
                articleUrl = article.url;
                articleText = article.text || '';
                articleTitle = article.title || '';
                classificationInfo = article.classification || null;
            } else {
                console.warn('Invalid article format:', article);
                return; // Skip this article
            }
            
            // Ensure articleUrl is a string
            if (typeof articleUrl !== 'string') {
                console.warn('Article URL is not a string:', articleUrl);
                return; // Skip this article
            }
            
            const domainName = this.extractDomainFromUrl(articleUrl);
            
            // Create confidence display if classification info is available
            let classificationDisplay = '';
            if (classificationInfo) {
                const confidence = (classificationInfo.confidence * 100).toFixed(1);
                const isNews = classificationInfo.is_news_article;
                const confidenceClass = isNews ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800';
                const confidenceIcon = isNews ? 'check-circle' : 'x-circle';
                const confidenceText = isNews ? 'News Article' : 'Not News';
                
                classificationDisplay = `
                    <div class="mt-2">
                        <span class="inline-flex items-center px-2 py-1 rounded-full text-xs ${confidenceClass}">
                            <i class="bi bi-${confidenceIcon} mr-1"></i>
                            ${confidenceText} (${confidence}% confidence)
                        </span>
                    </div>
                `;
            }
            
            const articleHtml = `
                <div class="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-shadow" data-url="${Utils.format.escape(articleUrl)}">
                    <div class="flex items-start space-x-3">
                        ${this.state.bulkLabelingMode ? `
                            <input type="checkbox" class="article-checkbox mt-1" 
                                   onchange="newsAnalyzer.handleArticleSelection(this, '${Utils.format.escape(articleUrl)}')"
                                   data-url="${Utils.format.escape(articleUrl)}">
                        ` : ''}
                        <div class="flex-1 min-w-0">
                            <div class="flex items-center justify-between mb-2">
                                <span class="text-sm font-medium text-gray-900">#${index + 1}</span>
                                <div class="flex items-center gap-2">
                                    <!-- Domain badge -->
                                    <span class="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-800">
                                        <i class="bi bi-link mr-1"></i>
                                        ${domainName}
                                    </span>
                                    <!-- Feedback buttons -->
                                    <div class="flex gap-1">
                                        <button onclick="newsAnalyzer.submitUrlFeedback('${Utils.format.escape(articleUrl)}', true, 0.9)" 
                                                class="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs hover:bg-green-200 transition-colors"
                                                title="Mark as News Article">
                                            <i class="bi bi-check-circle"></i>
                                        </button>
                                        <button onclick="newsAnalyzer.submitUrlFeedback('${Utils.format.escape(articleUrl)}', false, 0.9)" 
                                                class="bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs hover:bg-red-200 transition-colors"
                                                title="Mark as Not News Article">
                                            <i class="bi bi-x-circle"></i>
                                        </button>
                                    </div>
                                </div>
                            </div>
                            
                            ${articleText ? `
                                <div class="mb-2">
                                    <span class="text-sm text-gray-700 font-medium">${Utils.format.escape(articleText)}</span>
                                </div>
                            ` : ''}
                            
                            <div class="mb-3">
                                <a href="${articleUrl}" target="_blank" 
                                   class="text-blue-600 hover:text-blue-800 text-sm break-all font-mono">
                                    ${articleUrl}
                                </a>
                            </div>
                            
                            ${classificationDisplay}
                        </div>
                    </div>
                </div>
            `;

            this.elements.articleLinksContainer.insertAdjacentHTML('beforeend', articleHtml);
        });

        // Show crawled articles section
        Utils.dom.show(this.elements.crawledArticlesList);
        Utils.dom.show(this.elements.websiteResults);
        
        // Initialize progress tracking
        this.updateLabelingProgress();
    }
    
    // Update labeling progress indicator
    updateLabelingProgress() {
        const container = document.getElementById('articleLinksContainer');
        const totalArticles = container.querySelectorAll('div[data-url]').length;
        const labeledArticles = container.querySelectorAll('div[data-url][data-labeled="true"], div[data-url][data-bulk-labeled="true"]').length;
        
        // Find or create progress indicator
        let progressIndicator = document.getElementById('labelingProgress');
        if (!progressIndicator && totalArticles > 0) {
            progressIndicator = document.createElement('div');
            progressIndicator.id = 'labelingProgress';
            progressIndicator.className = 'bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4';
            
            // Insert before article container
            const articlesHeader = document.querySelector('#crawledArticlesList h4');
            if (articlesHeader) {
                articlesHeader.parentNode.insertBefore(progressIndicator, articlesHeader.nextSibling);
            }
        }
        
        if (progressIndicator && totalArticles > 0) {
            const percentage = Math.round((labeledArticles / totalArticles) * 100);
            
            progressIndicator.innerHTML = `
                <div class="flex items-center justify-between mb-2">
                    <span class="text-sm font-medium text-blue-800">
                        <i class="bi bi-tags mr-1"></i>
                        Labeling Progress
                    </span>
                    <span class="text-sm font-semibold text-blue-600">
                        ${labeledArticles} / ${totalArticles} (${percentage}%)
                    </span>
                </div>
                <div class="w-full bg-blue-200 rounded-full h-2">
                    <div class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: ${percentage}%"></div>
                </div>
                ${percentage === 100 ? `
                    <div class="mt-2 text-xs text-green-600 flex items-center">
                        <i class="bi bi-check-circle mr-1"></i>
                        All articles labeled! Ready to submit feedback.
                    </div>
                ` : `
                    <div class="mt-2 text-xs text-blue-600">
                        ${totalArticles - labeledArticles} articles remaining
                    </div>
                `}
            `;
        }
    }

    // Helper method to extract domain from URL
    extractDomainFromUrl(url) {
        try {
            // Ensure url is a string
            if (typeof url !== 'string') {
                console.warn('extractDomainFromUrl: url is not a string:', typeof url, url);
                return 'Unknown Domain';
            }
            
            const urlObj = new URL(url);
            return urlObj.hostname;
        } catch (error) {
            // If URL parsing fails, try to extract domain manually
            if (typeof url === 'string') {
                const match = url.match(/https?:\/\/([^\/]+)/);
                return match ? match[1] : 'Unknown Domain';
            } else {
                console.warn('extractDomainFromUrl: Cannot process non-string URL:', url);
                return 'Invalid URL';
            }
        }
    }

    // Helper method to format response time
    formatResponseTime(responseTimeMs) {
        if (!responseTimeMs) return '';
        
        const seconds = (responseTimeMs / 1000).toFixed(2);
        if (responseTimeMs < 1000) {
            return `${Math.round(responseTimeMs)}ms`;
        } else if (responseTimeMs < 60000) {
            return `${seconds}s`;
        } else {
            const minutes = Math.floor(responseTimeMs / 60000);
            const remainingSeconds = ((responseTimeMs % 60000) / 1000).toFixed(0);
            return `${minutes}m ${remainingSeconds}s`;
        }
    }

    // Analyze found articles (when button is clicked in preview mode)
    async analyzeFoundArticles() {
        if (!this.crawledArticles || this.crawledArticles.length === 0) {
            this.showError('No articles to analyze');
            return;
        }

        this.state.isLoading = true;
        Utils.dom.show(this.elements.loading);
        this.updateAnalyzeButton();

        // Start timing the analysis operation
        const startTime = performance.now();

        try {
            // Convert URLs to article objects for analysis
            const articleObjects = this.crawledArticles.map(url => ({ url: url }));
            
            const response = await Utils.http.post(Config.endpoints.analyzeWebsite, {
                articles: articleObjects,
                analysis_type: this.state.currentAnalysisType
            });

            if (response.success) {
                // Calculate response time and add it to the response data
                const endTime = performance.now();
                const responseTime = endTime - startTime;
                response.data.responseTime = responseTime;
                
                this.displayWebsiteAnalysisResults(response.data);
            } else {
                this.showError(response.error || 'Failed to analyze articles');
            }
        } catch (error) {
            console.error('Article analysis error:', error);
            this.showError('Failed to analyze articles. Please try again.');
        } finally {
            this.state.isLoading = false;
            Utils.dom.hide(this.elements.loading);
            this.updateAnalyzeButton();
        }
    }

    // Display website analysis results
    displayWebsiteAnalysisResults(data) {
        Utils.dom.hide(this.elements.results);
        Utils.dom.hide(this.elements.crawledArticlesList);

        // Update summary statistics
        const summary = data.summary;
        Utils.dom.setText(this.elements.totalArticlesCount, summary.total_articles);
        Utils.dom.setText(this.elements.successfulAnalysesCount, summary.successful_analyses);
        Utils.dom.setText(this.elements.failedAnalysesCount, summary.failed_analyses);
        Utils.dom.setText(this.elements.analyzedWebsiteTitle, summary.website_title || 'Unknown Website');

        // Add response time info if available
        if (data.responseTime) {
            const responseTimeText = `Analysis completed in ${this.formatResponseTime(data.responseTime)}`;
            const responseTimeHtml = `
                <div class="mt-4 p-3 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg border border-purple-200">
                    <div class="flex items-center space-x-2">
                        <i class="bi bi-clock text-purple-600"></i>
                        <span class="text-sm font-medium text-purple-800">
                            Performance Metrics
                        </span>
                    </div>
                    <div class="text-xs text-purple-700 mt-1">
                        ${responseTimeText} • Analyzed ${summary.total_articles} articles
                    </div>
                </div>
            `;
            this.elements.analyzedWebsiteTitle.insertAdjacentHTML('afterend', responseTimeHtml);
        }

        // Show/hide statistics based on analysis type
        const showFakeNews = this.state.currentAnalysisType === Config.analysisTypes.FAKE_NEWS || 
                           this.state.currentAnalysisType === Config.analysisTypes.BOTH;
        const showPolitical = this.state.currentAnalysisType === Config.analysisTypes.POLITICAL || 
                            this.state.currentAnalysisType === Config.analysisTypes.BOTH;

        // Update fake news statistics
        if (showFakeNews && summary.fake_news_summary) {
            Utils.dom.setText(this.elements.fakeArticlesCount, summary.fake_news_summary.fake_count);
            Utils.dom.setText(this.elements.realArticlesCount, summary.fake_news_summary.real_count);
            Utils.dom.setText(this.elements.fakePercentageOverall, 
                `${summary.fake_news_summary.fake_percentage.toFixed(1)}%`);
            Utils.dom.show(this.elements.fakeNewsStats);
        } else {
            Utils.dom.hide(this.elements.fakeNewsStats);
        }

        // Update political statistics
        if (showPolitical && summary.political_summary) {
            Utils.dom.setText(this.elements.politicalArticlesCount, summary.political_summary.political_count);
            Utils.dom.setText(this.elements.nonPoliticalArticlesCount, summary.political_summary.non_political_count);
            Utils.dom.setText(this.elements.politicalPercentageOverall, 
                `${summary.political_summary.political_percentage.toFixed(1)}%`);
            Utils.dom.show(this.elements.politicalStats);
        } else {
            Utils.dom.hide(this.elements.politicalStats);
        }

        // Show overall stats if we have any data
        if (showFakeNews || showPolitical) {
            Utils.dom.show(this.elements.overallStats);
        }

        // Display individual article results
        this.displayIndividualArticleResults(data.results, showFakeNews, showPolitical);

        // Show website results section
        Utils.dom.show(this.elements.websiteResults);
    }

    // Display individual article results
    displayIndividualArticleResults(results, showFakeNews, showPolitical) {
        this.elements.articleResultsList.innerHTML = '';

        results.forEach((result, index) => {
            let resultHtml = `
                <div class="bg-white rounded-lg shadow-md p-6 border-l-4 border-blue-500">
                    <div class="flex justify-between items-start mb-4">
                        <div class="flex-1">
                            <h4 class="text-lg font-semibold text-gray-800 mb-2">${Utils.format.escape(result.title)}</h4>
                            <p class="text-sm text-gray-600 mb-2">${Utils.format.escape(result.url)}</p>
                            ${result.extracted_content?.title ? `<p class="text-sm text-gray-500">${Utils.format.escape(result.extracted_content.title)}</p>` : ''}
                        </div>
                        <span class="text-sm text-gray-500">#${index + 1}</span>
                    </div>
            `;

            // Add fake news results if available
            if (showFakeNews && result.fake_news && !result.fake_news.error) {
                const fakeNews = result.fake_news;
                const predictionClass = fakeNews.prediction === 'Fake' ? 'text-red-600' : 'text-green-600';
                resultHtml += `
                    <div class="mb-4 p-4 bg-gray-50 rounded-lg">
                        <h5 class="font-semibold text-gray-800 mb-2">Fake News Detection</h5>
                        <div class="flex justify-between items-center">
                            <span class="font-medium ${predictionClass}">${fakeNews.prediction}</span>
                            <span class="text-sm text-gray-600">${Utils.format.confidence(fakeNews.confidence)}</span>
                        </div>
                    </div>
                `;
            }

            // Add political results if available
            if (showPolitical && result.political_classification && !result.political_classification.error) {
                const political = result.political_classification;
                const predictionClass = political.prediction === 'Political' ? 'text-blue-600' : 'text-gray-600';
                resultHtml += `
                    <div class="mb-4 p-4 bg-gray-50 rounded-lg">
                        <h5 class="font-semibold text-gray-800 mb-2">Political Classification</h5>
                        <div class="flex justify-between items-center">
                            <span class="font-medium ${predictionClass}">${political.prediction}</span>
                            <span class="text-sm text-gray-600">${Utils.format.confidence(political.confidence)}</span>
                        </div>
                    </div>
                `;
            }

            resultHtml += `
                    <div class="flex justify-between items-center">
                        <button onclick="window.open('${result.url}', '_blank')" 
                                class="text-blue-600 hover:text-blue-800 text-sm font-medium">
                            <i class="bi bi-box-arrow-up-right mr-1"></i>Open Article
                        </button>
                        <span class="text-xs text-gray-400">
                            ${result.extracted_content?.word_count ? `${result.extracted_content.word_count} words` : ''}
                        </span>
                    </div>
                </div>
            `;

            this.elements.articleResultsList.insertAdjacentHTML('beforeend', resultHtml);
        });
    }

    // Display results for single article analysis
    displayResults(data) {
        // Handle nested data structure from backend response
        const actualData = data.data || data;
        
        console.log('Raw response data:', data);
        console.log('Actual data for processing:', actualData);
        
        // Store results for export functionality
        Utils.storage.set('lastAnalysisResults', {
            ...actualData,
            timestamp: new Date().toISOString(),
            input_text: this.getCurrentInputText()
        });
        
        // Track analytics event if available
        if (this.advancedFeatures) {
            this.advancedFeatures.trackAnalysis(actualData, this.state.currentInputType, this.state.currentAnalysisType);
        }
        
        // Hide individual result sections first
        Utils.dom.hide(this.elements.fakeNewsResults);
        Utils.dom.hide(this.elements.politicalResults);
        
        // Show fake news results if available
        if (actualData.fake_news && !actualData.fake_news.error) {
            console.log('Showing fake news results:', actualData.fake_news);
            this.displayFakeNewsResults(actualData.fake_news);
            Utils.dom.show(this.elements.fakeNewsResults);
        } else {
            console.log('Fake news results not shown:', actualData.fake_news);
        }
        
        // Show political results if available
        if (actualData.political_classification && !actualData.political_classification.error) {
            console.log('Showing political results:', actualData.political_classification);
            this.displayPoliticalResults(actualData.political_classification);
            Utils.dom.show(this.elements.politicalResults);
        } else {
            console.log('Political results not shown:', actualData.political_classification);
        }
        
        // Show extracted content if available
        if (actualData.extracted_content && this.state.currentInputType === Config.inputTypes.URL) {
            this.displayExtractedContent(actualData.extracted_content);
            Utils.dom.show(this.elements.extractedContent);
            
            // Show indexing status if available
            if (actualData.indexing_result) {
                this.displayIndexingStatus(actualData.indexing_result, this.elements.extractedContent);
            }
        }
        
        // Show results section
        Utils.dom.show(this.elements.results);
        Utils.animations.fadeIn(this.elements.results);
    }

    // Get current input text
    getCurrentInputText() {
        if (this.state.currentInputType === Config.inputTypes.TEXT) {
            return this.elements.newsText?.value || '';
        } else if (this.state.currentInputType === Config.inputTypes.URL) {
            return this.elements.articleUrl?.value || '';
        }
        return '';
    }

    // Display fake news detection results
    displayFakeNewsResults(fakeNewsData) {
        console.log('displayFakeNewsResults called with data:', fakeNewsData);
        
        if (!fakeNewsData || fakeNewsData.error) {
            console.error('Fake news data error:', fakeNewsData?.error || 'No data provided');
            return;
        }

        // Update prediction card
        const predictionClass = fakeNewsData.prediction === 'Fake' ? 
            Config.cssClasses.prediction.fake : Config.cssClasses.prediction.real;
        const textClass = fakeNewsData.prediction === 'Fake' ? 
            Config.cssClasses.text.fake : Config.cssClasses.text.real;
        
        if (this.elements.fakeNewsPredictionCard) {
            this.elements.fakeNewsPredictionCard.className = predictionClass;
        }

        // Update prediction text and confidence
        Utils.dom.setText(this.elements.fakeNewsPredictionText, fakeNewsData.prediction || 'Unknown');
        if (this.elements.fakeNewsPredictionText) {
            this.elements.fakeNewsPredictionText.className = textClass;
        }
        
        Utils.dom.setText(this.elements.fakeNewsConfidenceText, 
            Utils.format.confidence(fakeNewsData.confidence || 0));

        // Update probability bars
        const fakeProb = fakeNewsData.probabilities?.fake || fakeNewsData.probabilities?.Fake || 0;
        const realProb = fakeNewsData.probabilities?.real || fakeNewsData.probabilities?.Real || 0;

        if (this.elements.fakeBar && this.elements.realBar) {
            // Animate progress bars
            setTimeout(() => {
                this.elements.fakeBar.style.width = Utils.format.percentage(fakeProb);
                this.elements.realBar.style.width = Utils.format.percentage(realProb);
            }, 100);
        }

        Utils.dom.setText(this.elements.fakePercentage, Utils.format.percentage(fakeProb));
        Utils.dom.setText(this.elements.realPercentage, Utils.format.percentage(realProb));
    }

    // Display political classification results
    displayPoliticalResults(politicalData) {
        console.log('displayPoliticalResults called with data:', politicalData);
        
        if (!politicalData || politicalData.error) {
            console.error('Political data error:', politicalData?.error || 'No data provided');
            return;
        }

        // Update prediction card
        const predictionClass = politicalData.prediction === 'Political' ? 
            Config.cssClasses.prediction.political : Config.cssClasses.prediction.nonPolitical;
        const textClass = politicalData.prediction === 'Political' ? 
            Config.cssClasses.text.political : Config.cssClasses.text.nonPolitical;

        if (this.elements.politicalPredictionCard) {
            this.elements.politicalPredictionCard.className = predictionClass;
        }

        // Update prediction text and confidence
        Utils.dom.setText(this.elements.politicalPredictionText, politicalData.prediction || 'Unknown');
        if (this.elements.politicalPredictionText) {
            this.elements.politicalPredictionText.className = textClass;
        }

        Utils.dom.setText(this.elements.politicalConfidenceText, 
            Utils.format.confidence(politicalData.confidence || 0));

        // Update probability bars
        const politicalProb = politicalData.probabilities?.political || politicalData.probabilities?.Political || 0;
        const nonPoliticalProb = politicalData.probabilities?.['non-political'] || politicalData.probabilities?.['Non-Political'] || 0;

        if (this.elements.politicalBar && this.elements.nonPoliticalBar) {
            // Animate progress bars
            setTimeout(() => {
                this.elements.politicalBar.style.width = Utils.format.percentage(politicalProb);
                this.elements.nonPoliticalBar.style.width = Utils.format.percentage(nonPoliticalProb);
            }, 100);
        }

        Utils.dom.setText(this.elements.politicalPercentage, Utils.format.percentage(politicalProb));
        Utils.dom.setText(this.elements.nonPoliticalPercentage, Utils.format.percentage(nonPoliticalProb));

        // Update reasoning if available
        if (politicalData.reasoning && this.elements.reasoningText) {
            Utils.dom.setText(this.elements.reasoningText, politicalData.reasoning);
        }
    }

    // Display extracted content for URL analysis
    displayExtractedContent(contentData) {
        if (!contentData) {
            return;
        }

        Utils.dom.setText(this.elements.extractedTitle, contentData.title || 'No title extracted');
        Utils.dom.setText(this.elements.extractedPreview, 
            Utils.format.truncate(contentData.combined || contentData.content || 'No content extracted', 300));
    }

    // Display indexing status for URL analysis
    displayIndexingStatus(indexingResult, extractedContentElement) {
        if (!indexingResult || !extractedContentElement) {
            return;
        }

        // Create indexing status section
        const indexingStatusHtml = `
            <div class="mt-4 p-4 rounded-lg border" id="indexingStatus">
                <h5 class="text-sm font-semibold text-gray-800 mb-2 flex items-center">
                    <i class="bi bi-database mr-2"></i>
                    Search Engine Indexing
                </h5>
                <div class="indexing-status-content">
                    ${this.getIndexingStatusContent(indexingResult)}
                </div>
            </div>
        `;

        // Remove existing indexing status if present
        const existingStatus = document.getElementById('indexingStatus');
        if (existingStatus) {
            existingStatus.remove();
        }

        // Add indexing status after extracted content
        extractedContentElement.insertAdjacentHTML('beforeend', indexingStatusHtml);
    }

    // Get indexing status content based on result
    getIndexingStatusContent(indexingResult) {
        const status = indexingResult.status;
        
        switch (status) {
            case 'success':
                const relevanceScore = indexingResult.relevance_score || 0;
                const locations = indexingResult.locations || [];
                const govEntities = indexingResult.government_entities || [];
                
                return `
                    <div class="flex items-center mb-2">
                        <i class="bi bi-check-circle text-green-600 mr-2"></i>
                        <span class="text-green-700 font-medium">Successfully indexed in Philippine news database</span>
                    </div>
                    <div class="text-xs text-gray-600 space-y-1">
                        <div>Philippine Relevance Score: <span class="font-medium">${(relevanceScore * 100).toFixed(1)}%</span></div>
                        ${locations.length > 0 ? `<div>Locations found: <span class="font-medium">${locations.join(', ')}</span></div>` : ''}
                        ${govEntities.length > 0 ? `<div>Government entities: <span class="font-medium">${govEntities.join(', ')}</span></div>` : ''}
                        <div class="text-blue-600 mt-2">
                            <i class="bi bi-info-circle mr-1"></i>
                            This article is now searchable in our Philippine news search engine
                        </div>
                    </div>
                `;
                
            case 'skipped':
                return `
                    <div class="flex items-center mb-2">
                        <i class="bi bi-skip-forward text-yellow-600 mr-2"></i>
                        <span class="text-yellow-700 font-medium">Not indexed - Low Philippine relevance</span>
                    </div>
                    <div class="text-xs text-gray-600">
                        ${indexingResult.message || 'This article does not appear to be relevant to Philippine news'}
                    </div>
                `;
                
            case 'already_indexed':
                return `
                    <div class="flex items-center mb-2">
                        <i class="bi bi-bookmark-check text-blue-600 mr-2"></i>
                        <span class="text-blue-700 font-medium">Already in search database</span>
                    </div>
                    <div class="text-xs text-gray-600">
                        This article was previously indexed and is searchable in our Philippine news search engine
                    </div>
                `;
                
            case 'error':
                return `
                    <div class="flex items-center mb-2">
                        <i class="bi bi-exclamation-triangle text-red-600 mr-2"></i>
                        <span class="text-red-700 font-medium">Indexing failed</span>
                    </div>
                    <div class="text-xs text-gray-600">
                        ${indexingResult.message || 'An error occurred during indexing'}
                    </div>
                `;
                
            default:
                return `
                    <div class="flex items-center mb-2">
                        <i class="bi bi-question-circle text-gray-600 mr-2"></i>
                        <span class="text-gray-700 font-medium">Unknown indexing status</span>
                    </div>
                `;
        }
    }

    // Hide all sections
    hideAllSections() {
        const sections = [
            this.elements.results,
            this.elements.websiteResults,
            this.elements.similarContentSection,
            this.elements.error
        ];
        
        sections.forEach(section => {
            if (section) {
                Utils.dom.hide(section);
            }
        });
    }

    // Submit feedback for URL classification (RL integration)
    async submitUrlFeedback(url, isCorrect, confidence) {
        try {
            // Find the current prediction for this URL
            const article = this.crawledArticles?.find(a => {
                // Handle both URL strings and article objects
                if (typeof a === 'string') {
                    return a === url;
                } else if (typeof a === 'object' && a.url) {
                    return a.url === url;
                }
                return false;
            });
            
            if (!article) {
                console.error('Article not found for feedback:', url);
                return;
            }
            
            // Get classification info from the article
            let classificationInfo = null;
            if (typeof article === 'object') {
                // Try both old and new property names for compatibility
                classificationInfo = article.classification || article.ai_classification;
            }
            
            if (!classificationInfo) {
                console.warn('No classification info found for article. Using default values.');
                // Provide default classification when none exists
                classificationInfo = {
                    is_news_article: true, // Default assumption
                    confidence: 0.5
                };
            }

            const predictedLabel = classificationInfo.is_news_article;
            const actualLabel = isCorrect ? predictedLabel : !predictedLabel;

            const response = await Utils.http.post('/url-classifier-feedback', {
                url: url,
                predicted_label: predictedLabel,
                actual_label: actualLabel,
                user_confidence: confidence
            });

            if (response.success) {
                // Show success message
                this.showFeedbackMessage(
                    `✓ Thank you! Feedback submitted. Model accuracy: ${Math.round((response.model_accuracy || 0) * 100)}%`,
                    'success'
                );
                
                // Update the UI to show feedback was submitted
                this.updateArticleFeedbackUI(url, isCorrect);
            } else {
                this.showFeedbackMessage('✗ Failed to submit feedback: ' + (response.error || 'Unknown error'), 'error');
            }
        } catch (error) {
            console.error('Feedback submission error:', error);
            this.showFeedbackMessage('✗ Network error occurred while submitting feedback', 'error');
        }
    }

    // Update article UI after feedback submission
    updateArticleFeedbackUI(url, isCorrect) {
        // Find the article element and update it
        const container = document.getElementById('articleLinksContainer');
        const articles = container.querySelectorAll('div[data-url]');
        let targetArticle = null;
        
        articles.forEach(articleElement => {
            const articleUrl = articleElement.getAttribute('data-url');
            if (articleUrl === url) {
                targetArticle = articleElement;
                // Replace feedback buttons with confirmation in the top right area
                const buttonContainer = articleElement.querySelector('.flex.gap-1');
                if (buttonContainer) {
                    const icon = isCorrect ? 'check-circle' : 'x-circle';
                    const color = isCorrect ? 'green' : 'orange';
                    const message = isCorrect ? 'Correct' : 'Incorrect';
                    
                    buttonContainer.innerHTML = `
                        <div class="text-xs text-${color}-600 flex items-center bg-${color}-50 px-2 py-1 rounded-full">
                            <i class="bi bi-${icon} mr-1"></i>
                            ${message}
                        </div>
                    `;
                    
                    // Mark the article as labeled
                    articleElement.setAttribute('data-labeled', 'true');
                }
            }
        });
        
        // Move labeled article to bottom
        if (targetArticle) {
            // Remove from current position
            targetArticle.remove();
            
            // Add to bottom
            container.appendChild(targetArticle);
            
            // Add visual indication that it was moved
            targetArticle.style.opacity = '0.7';
            targetArticle.style.order = '999'; // Ensure it stays at bottom
            
            // Scroll to next unlabeled article if available
            this.scrollToNextUnlabeledArticle();
            
            // Update progress indicator
            this.updateLabelingProgress();
        }
    }
    
    // Scroll to the next unlabeled article for better user experience
    scrollToNextUnlabeledArticle() {
        const container = document.getElementById('articleLinksContainer');
        const articles = container.querySelectorAll('div[data-url]:not([data-labeled="true"]):not([data-bulk-labeled="true"])');
        
        if (articles.length > 0) {
            // Find the first unlabeled article
            const nextArticle = articles[0];
            
            // Smooth scroll to the article
            nextArticle.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'center' 
            });
            
            // Add a subtle highlight effect
            nextArticle.style.transition = 'box-shadow 0.3s ease';
            nextArticle.style.boxShadow = '0 0 15px rgba(59, 130, 246, 0.3)';
            
            // Remove highlight after 2 seconds
            setTimeout(() => {
                nextArticle.style.boxShadow = '';
            }, 2000);
        } else {
            // All articles are labeled, show completion message
            this.showFeedbackMessage('🎉 All articles have been labeled! Great job!', 'success');
        }
    }

    // Show feedback message
    showFeedbackMessage(message, type) {
        // Create or update feedback message element
        let feedbackElement = document.getElementById('urlFeedbackMessage');
        if (!feedbackElement) {
            feedbackElement = document.createElement('div');
            feedbackElement.id = 'urlFeedbackMessage';
            feedbackElement.className = 'fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg max-w-sm';
            document.body.appendChild(feedbackElement);
        }

        const bgClass = type === 'success' ? 'bg-green-100 border-green-400 text-green-700' : 'bg-red-100 border-red-400 text-red-700';
        feedbackElement.className = `fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg max-w-sm border ${bgClass}`;
        feedbackElement.innerHTML = `
            <div class="flex items-center">
                <div class="flex-1">${message}</div>
                <button onclick="this.parentElement.parentElement.remove()" class="ml-2 text-gray-500 hover:text-gray-700">
                    <i class="bi bi-x"></i>
                </button>
            </div>
        `;

        // Auto-hide after 5 seconds
        setTimeout(() => {
            if (feedbackElement.parentNode) {
                feedbackElement.remove();
            }
        }, 5000);
    }

    // Find similar content in Philippine news database
    async findSimilarContent() {
        let contentText = '';
        
        // Get content based on current input type
        if (this.state.currentInputType === Config.inputTypes.TEXT) {
            contentText = this.elements.newsText?.value?.trim() || '';
        } else if (this.state.currentInputType === Config.inputTypes.URL) {
            // For URL input, we need to check if we have extracted content from previous analysis
            const lastResults = Utils.storage.get('lastAnalysisResults');
            if (lastResults && lastResults.extracted_content) {
                contentText = lastResults.extracted_content.combined || 
                             lastResults.extracted_content.content || '';
            } else {
                this.showError('Please analyze the URL first to extract content before finding similar articles.');
                return;
            }
        } else {
            this.showError('Similar content search is only available for text input and article URL analysis.');
            return;
        }

        if (!contentText || contentText.length < 50) {
            this.showError('Content must be at least 50 characters long to find similar articles.');
            return;
        }

        // Hide previous similar content results
        Utils.dom.hide(this.elements.similarContentSection);
        
        // Show loading state
        if (this.elements.findSimilarBtn) {
            this.elements.findSimilarBtn.disabled = true;
            this.elements.findSimilarBtn.innerHTML = '<i class="bi bi-hourglass-split animate-spin mr-2"></i>Finding Similar Articles...';
        }

        try {
            const response = await Utils.http.post(Config.endpoints.findSimilarContent, {
                content: contentText,
                limit: 10,
                minimum_similarity: 0.15
            });

            if (response.success) {
                this.displaySimilarContent(response.data);
            } else {
                this.showError(response.error || 'Failed to find similar content');
            }
        } catch (error) {
            console.error('Similar content search error:', error);
            this.showError('An error occurred while searching for similar content');
        } finally {
            // Reset button state
            if (this.elements.findSimilarBtn) {
                this.elements.findSimilarBtn.disabled = false;
                this.elements.findSimilarBtn.innerHTML = '<i class="bi bi-search mr-2"></i>Find Similar Articles';
            }
        }
    }

    // Display similar content results
    displaySimilarContent(data) {
        if (!this.elements.similarContentResults || !this.elements.similarContentList) {
            console.error('Similar content elements not found');
            return;
        }

        // Update summary
        if (this.elements.similarContentSummary) {
            const summary = `Found ${data.total_count} similar articles (${(data.response_time * 1000).toFixed(0)}ms)`;
            const keywords = data.top_keywords ? ` • Keywords: ${data.top_keywords.slice(0, 5).join(', ')}` : '';
            this.elements.similarContentSummary.innerHTML = `
                <div class="text-sm text-gray-600">
                    <i class="bi bi-info-circle mr-1"></i>
                    ${summary}${keywords}
                </div>
            `;
        }

        // Clear previous results
        this.elements.similarContentList.innerHTML = '';

        if (data.results && data.results.length > 0) {
            data.results.forEach((article, index) => {
                this.createSimilarContentCard(article, index + 1);
            });
        } else {
            this.elements.similarContentList.innerHTML = `
                <div class="text-center py-8 text-gray-500">
                    <i class="bi bi-search text-4xl mb-4"></i>
                    <h4 class="text-lg font-semibold mb-2">No Similar Articles Found</h4>
                    <p>No articles in our Philippine news database match your content.</p>
                    <p class="text-sm mt-2">Try:</p>
                    <ul class="text-sm text-left mt-2 max-w-md mx-auto">
                        <li>• Using longer, more detailed content</li>
                        <li>• Including Philippine-specific terms or locations</li>
                        <li>• Analyzing recent Philippine news content</li>
                    </ul>
                </div>
            `;
        }

        // Show the similar content section
        Utils.dom.show(this.elements.similarContentSection);
        Utils.animations.fadeIn(this.elements.similarContentSection);
    }

    // Create similar content result card
    createSimilarContentCard(article, index) {
        const publishDate = article.publish_date ? 
            new Date(article.publish_date).toLocaleDateString() : 'Date unknown';
        
        const similarityScore = (article.similarity_score * 100).toFixed(1);
        const relevanceScore = (article.relevance_score * 100).toFixed(1);
        
        // Get similarity color class
        const scoreNum = parseFloat(similarityScore);
        let scoreClass = 'text-gray-600';
        if (scoreNum >= 70) scoreClass = 'text-green-600';
        else if (scoreNum >= 50) scoreClass = 'text-yellow-600';
        else if (scoreNum >= 30) scoreClass = 'text-orange-600';
        else scoreClass = 'text-red-600';

        const cardHtml = `
            <div class="bg-white rounded-lg shadow-md p-6 border-l-4 border-blue-500 hover:shadow-lg transition-shadow similar-content-card">
                <div class="flex justify-between items-start mb-3">
                    <div class="flex-1">
                        <h4 class="text-lg font-semibold text-gray-800 hover:text-blue-600 cursor-pointer mb-2" 
                            onclick="window.open('${article.url}', '_blank')">
                            ${Utils.format.escape(article.title)}
                        </h4>
                        <div class="flex items-center space-x-4 text-sm text-gray-600 mb-2">
                            <span class="flex items-center">
                                <i class="bi bi-calendar mr-1"></i>
                                ${publishDate}
                            </span>
                            <span class="flex items-center">
                                <i class="bi bi-globe mr-1"></i>
                                ${Utils.format.escape(article.source_domain)}
                            </span>
                            ${article.category ? `
                                <span class="flex items-center">
                                    <i class="bi bi-tag mr-1"></i>
                                    ${Utils.format.escape(article.category)}
                                </span>
                            ` : ''}
                        </div>
                    </div>
                    <div class="flex flex-col items-end space-y-1">
                        <span class="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
                            #${index}
                        </span>
                        <span class="text-xs px-2 py-1 rounded-full ${scoreClass.replace('text-', 'bg-').replace('-600', '-100')} ${scoreClass} similarity-score">
                            ${similarityScore}% similar
                        </span>
                        <span class="bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full">
                            ${relevanceScore}% PH
                        </span>
                    </div>
                </div>
                
                ${article.summary ? `
                    <p class="text-gray-700 text-sm mb-3 line-clamp-3">
                        ${Utils.format.escape(article.summary)}
                    </p>
                ` : ''}
                
                ${article.matching_keywords && article.matching_keywords.length > 0 ? `
                    <div class="mb-3">
                        <p class="text-xs text-gray-500 mb-1">Matching keywords:</p>
                        <div class="flex flex-wrap gap-1">
                            ${article.matching_keywords.slice(0, 6).map(keyword => 
                                `<span class="bg-gray-100 text-gray-700 text-xs px-2 py-1 rounded-full keyword-tag">${Utils.format.escape(keyword)}</span>`
                            ).join('')}
                            ${article.matching_keywords.length > 6 ? 
                                `<span class="text-xs text-gray-500">+${article.matching_keywords.length - 6} more</span>` : ''
                            }
                        </div>
                    </div>
                ` : ''}
                
                <div class="flex justify-between items-center">
                    <div class="flex space-x-2">
                        <button onclick="window.open('${article.url}', '_blank')" 
                                class="text-blue-600 hover:text-blue-800 text-sm font-medium flex items-center">
                            <i class="bi bi-box-arrow-up-right mr-1"></i>Read Original
                        </button>
                        <button onclick="newsAnalyzer.compareWithSimilar('${Utils.format.escape(article.url)}')" 
                                class="text-green-600 hover:text-green-800 text-sm font-medium flex items-center">
                            <i class="bi bi-arrows-angle-expand mr-1"></i>Compare
                        </button>
                    </div>
                    ${article.author ? `
                        <span class="text-xs text-gray-500">
                            by ${Utils.format.escape(article.author)}
                        </span>
                    ` : ''}
                </div>
            </div>
        `;

        this.elements.similarContentList.insertAdjacentHTML('beforeend', cardHtml);
    }

    // Compare current content with similar article (placeholder for future enhancement)
    compareWithSimilar(articleUrl) {
        // For now, just open the article and show a message
        window.open(articleUrl, '_blank');
        this.showFeedbackMessage('Article opened in new tab. Future versions will include side-by-side comparison.', 'success');
    }

    // Bulk Labeling Methods
    
    toggleBulkLabeling() {
        this.state.bulkLabelingMode = !this.state.bulkLabelingMode;
        
        if (this.state.bulkLabelingMode) {
            // Enable bulk labeling mode
            this.elements.toggleBulkLabelingBtn.innerHTML = '<i class="bi bi-tags mr-2"></i>Disable Bulk Labeling';
            this.elements.toggleBulkLabelingBtn.className = 'bg-red-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 transition-all duration-200';
            Utils.dom.show(this.elements.bulkActionControls);
            Utils.dom.show(this.elements.bulkLabelingInstructions);
        } else {
            // Disable bulk labeling mode
            this.elements.toggleBulkLabelingBtn.innerHTML = '<i class="bi bi-tags mr-2"></i>Enable Bulk Labeling';
            this.elements.toggleBulkLabelingBtn.className = 'bg-purple-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all duration-200';
            Utils.dom.hide(this.elements.bulkActionControls);
            Utils.dom.hide(this.elements.bulkLabelingInstructions);
            this.bulkLabels.clear();
        }
        
        // Re-render articles with or without checkboxes
        if (this.crawledArticles && this.crawledArticles.length > 0) {
            this.displayCrawledArticles({ articles: this.crawledArticles, total_found: this.crawledArticles.length });
        }
        
        this.updateBulkFeedbackButton();
    }

    selectAllArticles() {
        const checkboxes = document.querySelectorAll('.article-checkbox');
        checkboxes.forEach(checkbox => {
            checkbox.checked = true;
            const url = checkbox.closest('[data-url]').getAttribute('data-url');
            if (url) {
                this.handleArticleSelection(checkbox, url);
            }
        });
    }

    deselectAllArticles() {
        const checkboxes = document.querySelectorAll('.article-checkbox');
        checkboxes.forEach(checkbox => {
            checkbox.checked = false;
            const url = checkbox.closest('[data-url]').getAttribute('data-url');
            if (url) {
                this.handleArticleSelection(checkbox, url);
            }
        });
    }

    handleArticleSelection(checkbox, url) {
        if (checkbox.checked) {
            // Article is selected but not labeled yet
            if (!this.bulkLabels.has(url)) {
                this.bulkLabels.set(url, { selected: true, label: null });
            } else {
                this.bulkLabels.get(url).selected = true;
            }
        } else {
            // Article is deselected
            if (this.bulkLabels.has(url)) {
                this.bulkLabels.delete(url);
            }
        }
        this.updateBulkFeedbackButton();
    }

    bulkLabelArticles(isNews) {
        const selectedArticles = Array.from(this.bulkLabels.keys()).filter(url => 
            this.bulkLabels.get(url).selected
        );

        if (selectedArticles.length === 0) {
            this.showError('Please select articles to label first');
            return;
        }

        // Update labels for selected articles
        const container = document.getElementById('articleLinksContainer');
        const labeledElements = [];
        
        selectedArticles.forEach(url => {
            const labelData = this.bulkLabels.get(url);
            labelData.label = isNews;
            
            // Update UI to show label
            const articleElement = document.querySelector(`[data-url="${url}"]`);
            if (articleElement) {
                const statusElement = articleElement.querySelector('.bulk-label-status');
                if (statusElement) {
                    const labelClass = isNews ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800';
                    const labelIcon = isNews ? 'check-circle' : 'x-circle';
                    const labelText = isNews ? 'News' : 'Not News';
                    
                    statusElement.innerHTML = `
                        <span class="inline-flex items-center px-2 py-1 rounded-full text-xs ${labelClass}">
                            <i class="bi bi-${labelIcon} mr-1"></i>
                            Labeled as: ${labelText}
                        </span>
                    `;
                    statusElement.style.display = 'block';
                }
                
                // Mark as labeled and prepare for moving
                articleElement.setAttribute('data-bulk-labeled', 'true');
                articleElement.style.opacity = '0.7';
                labeledElements.push(articleElement);
            }
        });
        
        // Move all labeled articles to bottom
        labeledElements.forEach(element => {
            element.remove();
            container.appendChild(element);
        });
        
        // Scroll to next unlabeled article if any exist
        if (labeledElements.length > 0) {
            this.scrollToNextUnlabeledArticle();
            this.updateLabelingProgress();
        }

        this.updateBulkFeedbackButton();
        
        // Show success message
        const labelType = isNews ? 'News' : 'Not News';
        this.showFeedbackMessage(
            `Successfully labeled ${selectedArticles.length} articles as "${labelType}"`,
            'success'
        );
    }

    async submitBulkFeedback() {
        const labeledArticles = Array.from(this.bulkLabels.entries())
            .filter(([url, data]) => data.label !== null)
            .map(([url, data]) => ({
                url: url,
                actual_label: data.label,
                predicted_label: this.getArticlePrediction(url),
                user_confidence: 1.0
            }));

        if (labeledArticles.length === 0) {
            this.showError('No labeled articles to submit');
            return;
        }

        try {
            this.elements.submitBulkFeedbackBtn.disabled = true;
            this.elements.submitBulkFeedbackBtn.innerHTML = '<i class="bi bi-hourglass-split animate-spin mr-1"></i>Submitting...';

            const response = await Utils.http.post(Config.endpoints.urlFeedback, {
                feedback_batch: labeledArticles
            });

            if (response.success) {
                this.showFeedbackMessage(
                    `Successfully submitted feedback for ${labeledArticles.length} articles. Thank you for helping improve our AI!`,
                    'success'
                );
                
                // Clear labels after successful submission
                this.bulkLabels.clear();
                this.deselectAllArticles();
                this.updateBulkFeedbackButton();
                
                // Hide bulk labeling mode
                this.toggleBulkLabeling();
                
            } else {
                throw new Error(response.error || 'Failed to submit bulk feedback');
            }
        } catch (error) {
            console.error('Error submitting bulk feedback:', error);
            this.showError(`Failed to submit bulk feedback: ${error.message}`);
        } finally {
            this.elements.submitBulkFeedbackBtn.disabled = false;
            this.elements.submitBulkFeedbackBtn.innerHTML = '<i class="bi bi-cloud-upload mr-1"></i>Submit Feedback (<span id="labeledCount">0</span>)';
        }
    }

    getArticlePrediction(url) {
        const article = this.crawledArticles?.find(a => {
            // Handle both URL strings and article objects
            if (typeof a === 'string') {
                return a === url;
            } else if (typeof a === 'object' && a.url) {
                return a.url === url;
            }
            return false;
        });
        
        if (!article || typeof article === 'string') {
            return false; // Default for string URLs
        }
        
        // Try both old and new property names for compatibility
        const classificationInfo = article.classification || article.ai_classification;
        return classificationInfo?.prediction || classificationInfo?.is_news_article || false;
    }

    updateBulkFeedbackButton() {
        const labeledCount = Array.from(this.bulkLabels.values())
            .filter(data => data.label !== null).length;
        
        if (this.elements.labeledCount) {
            this.elements.labeledCount.textContent = labeledCount;
        }
        
        if (this.elements.submitBulkFeedbackBtn) {
            this.elements.submitBulkFeedbackBtn.disabled = labeledCount === 0;
        }
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
