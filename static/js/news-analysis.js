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
            isLoading: false
        };

        // DOM elements cache
        this.elements = {};
        this.crawledArticles = [];

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
            'maxArticles', 'crawlOnlyBtn', 'crawlAnalyzeBtn', 'analyzeFoundArticlesBtn',
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
            'articleResultsList', 'crawledArticlesList', 'articleLinksContainer'
        ];

        elementIds.forEach(id => {
            this.elements[id] = document.getElementById(id);
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

        // Website URL input monitoring
        if (this.elements.websiteUrl) {
            this.elements.websiteUrl.addEventListener('input', () => this.updateAnalyzeButton());
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
            this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center status-error';
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
            this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center status-ready';
        } else if (fakeNewsReady) {
            this.elements.modelStatus.innerHTML = `
                <div class="flex items-center justify-center">
                    <i class="bi bi-hourglass-split spin text-yellow-600 mr-2"></i>
                    <span class="text-yellow-700">Fake news model ready, political classifier loading...</span>
                </div>
            `;
            this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center status-partial';
        } else {
            this.elements.modelStatus.innerHTML = `
                <div class="flex items-center justify-center">
                    <i class="bi bi-hourglass-split spin text-blue-600 mr-2"></i>
                    <span class="text-gray-700">Loading models...</span>
                </div>
            `;
            this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center status-loading';
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
            this.elements.analyzeBtn.innerHTML = '<i class="bi bi-hourglass-split spin mr-2"></i>Analyzing...';
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

        const maxArticles = parseInt(this.elements.maxArticles?.value || '20');
        const crawlingMode = this.state.currentCrawlingMode || Config.crawlingModes.PREVIEW;

        this.hideResults();
        this.state.isLoading = true;
        Utils.dom.show(this.elements.loading);
        this.updateAnalyzeButton();

        try {
            if (crawlingMode === Config.crawlingModes.PREVIEW) {
                await this.crawlWebsitePreview(websiteUrl, maxArticles);
            } else {
                await this.crawlAndAnalyzeWebsite(websiteUrl, maxArticles);
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
    async crawlWebsitePreview(websiteUrl, maxArticles) {
        const response = await Utils.http.post(Config.endpoints.crawlWebsite, {
            website_url: websiteUrl,
            max_articles: maxArticles
        });

        if (!response.success) {
            throw new Error(response.error || 'Failed to crawl website');
        }

        const result = response.data;
        if (!result.success) {
            throw new Error(result.error || 'Website crawling failed');
        }

        this.displayCrawledArticles(result);
    }

    // Crawl and analyze website
    async crawlAndAnalyzeWebsite(websiteUrl, maxArticles) {
        const response = await Utils.http.post(Config.endpoints.analyzeWebsite, {
            website_url: websiteUrl,
            max_articles: maxArticles,
            analysis_type: this.state.currentAnalysisType
        });

        if (!response.success) {
            throw new Error(response.error || 'Failed to analyze website');
        }

        const result = response.data;
        if (!result.success) {
            throw new Error(result.error || 'Website analysis failed');
        }

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

        // Store articles for later analysis
        this.crawledArticles = data.articles;

        // Update summary
        Utils.dom.setText(this.elements.totalArticlesCount, data.total_found);
        Utils.dom.setText(this.elements.analyzedWebsiteTitle, data.website_title || 'Unknown Website');

        // Clear and populate article links
        this.elements.articleLinksContainer.innerHTML = '';
        
        data.articles.forEach((article, index) => {
            const articleHtml = `
                <div class="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-shadow">
                    <div class="flex items-start space-x-3">
                        <div class="flex-shrink-0">
                            <span class="inline-flex items-center justify-center w-8 h-8 bg-blue-100 text-blue-600 rounded-full text-sm font-medium">
                                ${index + 1}
                            </span>
                        </div>
                        <div class="flex-1 min-w-0">
                            <h4 class="text-sm font-medium text-gray-900 truncate">${Utils.format.escape(article.title)}</h4>
                            <p class="text-sm text-gray-500 truncate">${Utils.format.escape(article.url)}</p>
                        </div>
                        <div class="flex-shrink-0">
                            <button onclick="window.open('${article.url}', '_blank')" 
                                    class="text-blue-600 hover:text-blue-800 text-sm">
                                <i class="bi bi-box-arrow-up-right"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `;
            this.elements.articleLinksContainer.insertAdjacentHTML('beforeend', articleHtml);
        });

        // Show crawled articles section
        Utils.dom.show(this.elements.crawledArticlesList);
        Utils.dom.show(this.elements.websiteResults);
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

        try {
            const response = await Utils.http.post(Config.endpoints.analyzeWebsite, {
                articles: this.crawledArticles,
                analysis_type: this.state.currentAnalysisType
            });

            if (response.success) {
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

    // Hide all sections
    hideAllSections() {
        const sections = [
            this.elements.results,
            this.elements.websiteResults,
            this.elements.error
        ];
        
        sections.forEach(section => {
            if (section) {
                Utils.dom.hide(section);
            }
        });
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
