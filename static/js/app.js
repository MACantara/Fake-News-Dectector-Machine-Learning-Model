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
            'analyzeBtn', 'loading', 'results', 'websiteResults', 'error', 'modelStatus', 'textCount',
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
            const result = await Utils.http.get(Config.endpoints.modelStatus);
            
            if (result && result.fake_news_model && result.political_model) {
                this.state.modelsReady.fake_news = result.fake_news_model.is_trained;
                this.state.modelsReady.political = result.political_model.is_trained;
                
                const overallReady = result.fake_news_model.is_trained && result.political_model.is_trained;
                
                if (overallReady) {
                    Utils.dom.setText(this.elements.modelStatus, 
                        '<i class="bi bi-check-circle text-green-600 mr-2"></i>Models ready for analysis'
                    );
                    this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center status-ready';
                } else {
                    Utils.dom.setText(this.elements.modelStatus, 
                        '<i class="bi bi-hourglass-split text-yellow-600 mr-2"></i>Models are loading...'
                    );
                    this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center status-loading';
                }
            }
        } catch (error) {
            console.error('Failed to check model status:', error);
            if (this.elements.modelStatus) {
                Utils.dom.setText(this.elements.modelStatus, 
                    '<i class="bi bi-exclamation-triangle text-red-600 mr-2"></i>Error checking model status'
                );
                this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center status-error';
            }
        }
    }

    // Start periodic model status checking
    startPeriodicStatusCheck() {
        setInterval(() => {
            if (!this.state.isLoading) {
                this.checkModelStatus();
            }
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
        
        // Show appropriate section and activate button
        if (type === Config.inputTypes.TEXT) {
            if (this.elements.textBtn) {
                this.elements.textBtn.className = Config.cssClasses.button.active.textInput;
            }
            Utils.dom.show(this.elements.textInput);
        } else if (type === Config.inputTypes.URL) {
            if (this.elements.urlBtn) {
                this.elements.urlBtn.className = Config.cssClasses.button.active.urlInput;
            }
            Utils.dom.show(this.elements.urlInput);
        } else if (type === Config.inputTypes.WEBSITE) {
            if (this.elements.websiteBtn) {
                this.elements.websiteBtn.className = Config.cssClasses.button.active.urlInput; // Reuse URL styling
            }
            Utils.dom.show(this.elements.websiteInput);
        }
        
        this.updateAnalyzeButton();
        this.hideResults();
    }

    // Switch analysis type
    switchAnalysisType(type) {
        this.state.currentAnalysisType = type;
        
        // Update button styles
        if (this.elements.fakeNewsBtn) {
            this.elements.fakeNewsBtn.className = type === Config.analysisTypes.FAKE_NEWS ? 
                Config.cssClasses.button.active.analysis : Config.cssClasses.button.inactive.analysis;
        }
        if (this.elements.politicalBtn) {
            this.elements.politicalBtn.className = type === Config.analysisTypes.POLITICAL ? 
                Config.cssClasses.button.active.analysis : Config.cssClasses.button.inactive.analysis;
        }
        if (this.elements.bothBtn) {
            this.elements.bothBtn.className = type === Config.analysisTypes.BOTH ? 
                Config.cssClasses.button.active.analysis : Config.cssClasses.button.inactive.analysis;
        }
        
        this.hideResults();
    }

    // Switch crawling mode (preview vs analyze)
    switchCrawlingMode(mode) {
        this.state.currentCrawlingMode = mode;
        
        if (mode === Config.crawlingModes.PREVIEW) {
            if (this.elements.crawlOnlyBtn) {
                this.elements.crawlOnlyBtn.className = 'crawl-mode-btn bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium focus-ring';
            }
            if (this.elements.crawlAnalyzeBtn) {
                this.elements.crawlAnalyzeBtn.className = 'crawl-mode-btn bg-gray-200 text-gray-700 px-4 py-2 rounded-lg text-sm font-medium focus-ring';
            }
        } else {
            if (this.elements.crawlAnalyzeBtn) {
                this.elements.crawlAnalyzeBtn.className = 'crawl-mode-btn bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium focus-ring';
            }
            if (this.elements.crawlOnlyBtn) {
                this.elements.crawlOnlyBtn.className = 'crawl-mode-btn bg-gray-200 text-gray-700 px-4 py-2 rounded-lg text-sm font-medium focus-ring';
            }
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
            const count = this.elements.newsText.value.length;
            Utils.dom.setText(this.elements.textCount, `${count} characters`);
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
                return { valid: false, message: `Please enter at least ${Config.validation.minTextLength} characters` };
            }
            if (text.length > Config.validation.maxTextLength) {
                return { valid: false, message: `Text is too long. Maximum ${Config.validation.maxTextLength} characters allowed` };
            }
        } else if (this.state.currentInputType === Config.inputTypes.URL) {
            const url = this.elements.articleUrl?.value?.trim() || '';
            if (!Config.validation.urlPattern.test(url)) {
                return { valid: false, message: 'Please enter a valid URL starting with http:// or https://' };
            }
        } else if (this.state.currentInputType === Config.inputTypes.WEBSITE) {
            const url = this.elements.websiteUrl?.value?.trim() || '';
            if (!Config.validation.websitePattern.test(url)) {
                return { valid: false, message: 'Please enter a valid website URL starting with http:// or https://' };
            }
        }
        
        return { valid: true };
    }

    // Main analysis method
    async analyzeContent() {
        // Handle website crawling separately
        if (this.state.currentInputType === Config.inputTypes.WEBSITE) {
            await this.handleWebsiteCrawling();
            return;
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
            
            if (result.error) {
                this.showError(result.error);
            } else {
                this.displayResults(result);
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('An error occurred during analysis. Please try again.');
        } finally {
            this.state.isLoading = false;
            Utils.dom.hide(this.elements.loading);
            this.updateAnalyzeButton();
        }
    }

    // Handle website crawling
    async handleWebsiteCrawling() {
        const websiteUrl = this.elements.websiteUrl?.value.trim();
        if (!websiteUrl) {
            this.showError('Please enter a website URL');
            return;
        }

        const maxArticles = parseInt(this.elements.maxArticles?.value || '5');
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
            this.showError(`Website crawling failed: ${error.message}`);
        } finally {
            this.state.isLoading = false;
            Utils.dom.hide(this.elements.loading);
            this.updateAnalyzeButton();
        }
    }

    // Crawl website for preview (just show links)
    async crawlWebsitePreview(websiteUrl, maxArticles) {
        const result = await Utils.http.post(Config.endpoints.crawlWebsite, {
            website_url: websiteUrl,
            max_articles: maxArticles
        });

        if (!result.success) {
            throw new Error(result.error || 'Failed to crawl website');
        }

        this.displayCrawledArticles(result);
    }

    // Crawl and analyze website
    async crawlAndAnalyzeWebsite(websiteUrl, maxArticles) {
        const result = await Utils.http.post(Config.endpoints.analyzeWebsite, {
            website_url: websiteUrl,
            max_articles: maxArticles,
            analysis_type: this.state.currentAnalysisType
        });

        if (!result.success) {
            throw new Error(result.error || 'Failed to analyze website');
        }

        this.displayWebsiteAnalysisResults(result);
    }

    // Display crawled articles (preview mode)
    displayCrawledArticles(data) {
        Utils.dom.hide(this.elements.results);
        Utils.dom.hide(this.elements.crawledArticlesList);
        
        if (!data.articles || data.articles.length === 0) {
            this.showError('No articles found on the website');
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
            const articleElement = document.createElement('div');
            articleElement.className = 'bg-white rounded-lg p-4 border border-gray-200 hover:border-blue-300 transition-colors';
            articleElement.innerHTML = `
                <div class="flex items-start justify-between">
                    <div class="flex-1 mr-4">
                        <h5 class="font-medium text-gray-800 mb-2 line-clamp-2">${Utils.format.truncate(article.title, 80)}</h5>
                        <a href="${article.url}" target="_blank" class="text-blue-600 hover:text-blue-800 text-sm break-all">
                            ${Utils.format.truncate(article.url, 60)}
                        </a>
                    </div>
                    <div class="flex-shrink-0">
                        <span class="bg-gray-100 text-gray-600 px-2 py-1 rounded text-xs">
                            Article ${index + 1}
                        </span>
                    </div>
                </div>
            `;
            this.elements.articleLinksContainer.appendChild(articleElement);
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
            const websiteUrl = this.elements.websiteUrl?.value.trim();
            const result = await Utils.http.post(Config.endpoints.analyzeWebsite, {
                website_url: websiteUrl,
                max_articles: this.crawledArticles.length,
                analysis_type: this.state.currentAnalysisType
            });

            if (!result.success) {
                throw new Error(result.error || 'Failed to analyze articles');
            }

            this.displayWebsiteAnalysisResults(result);
        } catch (error) {
            console.error('Article analysis error:', error);
            this.showError(`Article analysis failed: ${error.message}`);
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
            const fakeStats = summary.fake_news_summary;
            Utils.dom.setText(this.elements.fakeArticlesCount, fakeStats.fake_articles);
            Utils.dom.setText(this.elements.realArticlesCount, fakeStats.real_articles);
            Utils.dom.setText(this.elements.fakePercentageOverall, `${fakeStats.fake_percentage.toFixed(1)}%`);
            Utils.dom.show(this.elements.fakeNewsStats);
        } else {
            Utils.dom.hide(this.elements.fakeNewsStats);
        }

        // Update political statistics
        if (showPolitical && summary.political_summary) {
            const politicalStats = summary.political_summary;
            Utils.dom.setText(this.elements.politicalArticlesCount, politicalStats.political_articles);
            Utils.dom.setText(this.elements.nonPoliticalArticlesCount, politicalStats.non_political_articles);
            Utils.dom.setText(this.elements.politicalPercentageOverall, `${politicalStats.political_percentage.toFixed(1)}%`);
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
            const articleDiv = document.createElement('div');
            articleDiv.className = 'bg-white rounded-xl shadow-lg border border-gray-200 p-6 space-y-4';

            let statusBadge = '';
            if (result.status === 'success') {
                statusBadge = '<span class="bg-green-100 text-green-800 text-xs font-medium px-2.5 py-0.5 rounded-full">Success</span>';
            } else if (result.status === 'failed') {
                statusBadge = '<span class="bg-red-100 text-red-800 text-xs font-medium px-2.5 py-0.5 rounded-full">Failed</span>';
            } else if (result.status === 'timeout') {
                statusBadge = '<span class="bg-yellow-100 text-yellow-800 text-xs font-medium px-2.5 py-0.5 rounded-full">Timeout</span>';
            }

            let content = `
                <div class="flex items-start justify-between mb-4">
                    <div class="flex-1">
                        <div class="flex items-center gap-2 mb-2">
                            <h4 class="text-lg font-semibold text-gray-800">Article ${index + 1}</h4>
                            ${statusBadge}
                        </div>
                        <h5 class="text-md font-medium text-gray-700 mb-2">${Utils.format.truncate(result.title, 100)}</h5>
                        <a href="${result.url}" target="_blank" class="text-blue-600 hover:text-blue-800 text-sm break-all">
                            ${Utils.format.truncate(result.url, 80)}
                        </a>
                    </div>
                </div>
            `;

            if (result.status === 'success') {
                if (result.content_preview) {
                    content += `
                        <div class="bg-gray-50 rounded-lg p-3 mb-4">
                            <p class="text-sm text-gray-600 font-medium mb-1">Content Preview:</p>
                            <p class="text-sm text-gray-700">${result.content_preview}</p>
                        </div>
                    `;
                }

                // Fake news results
                if (showFakeNews && result.fake_news && !result.fake_news.error) {
                    const fakeNews = result.fake_news;
                    const predictionClass = fakeNews.prediction === 'Fake' ? 'text-red-700' : 'text-green-700';
                    content += `
                        <div class="border-l-4 border-red-500 bg-red-50 p-4 rounded-r-lg">
                            <h6 class="font-medium text-gray-800 mb-2 flex items-center">
                                <i class="bi bi-shield-exclamation text-red-600 mr-2"></i>
                                Fake News Detection
                            </h6>
                            <div class="grid grid-cols-2 gap-4">
                                <div>
                                    <p class="text-sm text-gray-600">Prediction:</p>
                                    <p class="font-semibold ${predictionClass}">${fakeNews.prediction}</p>
                                </div>
                                <div>
                                    <p class="text-sm text-gray-600">Confidence:</p>
                                    <p class="font-semibold text-gray-800">${Utils.format.confidence(fakeNews.confidence)}</p>
                                </div>
                            </div>
                        </div>
                    `;
                }

                // Political classification results
                if (showPolitical && result.political_classification && !result.political_classification.error) {
                    const political = result.political_classification;
                    const predictionClass = political.prediction === 'Political' ? 'text-blue-700' : 'text-gray-700';
                    content += `
                        <div class="border-l-4 border-blue-500 bg-blue-50 p-4 rounded-r-lg">
                            <h6 class="font-medium text-gray-800 mb-2 flex items-center">
                                <i class="bi bi-bank text-blue-600 mr-2"></i>
                                Political Classification
                            </h6>
                            <div class="grid grid-cols-2 gap-4">
                                <div>
                                    <p class="text-sm text-gray-600">Classification:</p>
                                    <p class="font-semibold ${predictionClass}">${political.prediction}</p>
                                </div>
                                <div>
                                    <p class="text-sm text-gray-600">Confidence:</p>
                                    <p class="font-semibold text-gray-800">${Utils.format.confidence(political.confidence)}</p>
                                </div>
                            </div>
                        </div>
                    `;
                }
            } else {
                // Show error for failed articles
                content += `
                    <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                        <div class="flex items-center">
                            <i class="bi bi-exclamation-triangle text-red-500 mr-2"></i>
                            <span class="text-red-700 font-medium">Error: ${result.error || 'Analysis failed'}</span>
                        </div>
                    </div>
                `;
            }

            articleDiv.innerHTML = content;
            this.elements.articleResultsList.appendChild(articleDiv);
        });
    }

    // Display results for single article analysis
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
        if (data.extracted_content && this.state.currentInputType === Config.inputTypes.URL) {
            this.displayExtractedContent(data.extracted_content);
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
        } else {
            return this.elements.articleUrl?.value || '';
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
