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
            'textBtn', 'urlBtn', 'websiteBtn', 'searchBtn', 'fakeNewsBtn', 'politicalBtn', 'bothBtn',
            'textInput', 'urlInput', 'websiteInput', 'searchInput', 'newsText', 'articleUrl', 'websiteUrl',
            'searchQuery', 'searchCategory', 'searchSource', 'searchLimit',
            'performSearchBtn', 'viewAnalyticsBtn', 'addToIndexBtn', 'addArticleModal',
            'indexArticleUrl', 'submitIndexBtn', 'cancelIndexBtn',
            'maxArticles', 'crawlOnlyBtn', 'crawlAnalyzeBtn', 'analyzeFoundArticlesBtn',
            'analyzeBtn', 'loading', 'results', 'websiteResults', 'searchResults', 'analyticsResults',
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
            'searchSummary', 'searchSummaryText', 'searchResponseTime', 'searchResultsGrid',
            'loadMoreSection', 'loadMoreBtn', 'totalSourcesCount', 'avgRelevanceScore',
            'totalCategoriesCount', 'topSourcesChart', 'categoriesChart', 'recentActivityChart',
            'topQueriesTable', 'topQueriesBody'
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

        // Philippine News Search button
        if (this.elements.searchBtn) {
            this.elements.searchBtn.addEventListener('click', () => this.switchInputType(Config.inputTypes.SEARCH));
        }

        // Philippine search events
        if (this.elements.performSearchBtn) {
            this.elements.performSearchBtn.addEventListener('click', () => this.performPhilippineSearch());
        }
        if (this.elements.viewAnalyticsBtn) {
            this.elements.viewAnalyticsBtn.addEventListener('click', () => this.viewPhilippineAnalytics());
        }
        if (this.elements.addToIndexBtn) {
            this.elements.addToIndexBtn.addEventListener('click', () => this.showAddToIndexModal());
        }
        if (this.elements.submitIndexBtn) {
            this.elements.submitIndexBtn.addEventListener('click', () => this.submitIndexArticle());
        }
        if (this.elements.cancelIndexBtn) {
            this.elements.cancelIndexBtn.addEventListener('click', () => this.hideAddToIndexModal());
        }

        // Search input monitoring
        if (this.elements.searchQuery) {
            this.elements.searchQuery.addEventListener('input', () => this.updateSearchButton());
            this.elements.searchQuery.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.performPhilippineSearch();
                }
            });
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
            
            const result = response.data;
            if (result && result.fake_news_model && result.political_model) {
                this.state.modelsReady.fake_news = result.fake_news_model.is_trained;
                this.state.modelsReady.political = result.political_model.is_trained;
                
                const overallReady = result.fake_news_model.is_trained && result.political_model.is_trained;
                
                if (overallReady) {
                    Utils.dom.setHTML(this.elements.modelStatus, 
                        '<i class="bi bi-check-circle text-green-600 mr-2"></i>Models ready for analysis'
                    );
                    this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center status-ready';
                } else {
                    Utils.dom.setHTML(this.elements.modelStatus, 
                        '<i class="bi bi-hourglass-split text-yellow-600 mr-2"></i>Models are loading...'
                    );
                    this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center status-loading';
                }
            }
        } catch (error) {
            console.error('Failed to check model status:', error);
            if (this.elements.modelStatus) {
                Utils.dom.setHTML(this.elements.modelStatus, 
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
        if (this.elements.searchBtn) {
            this.elements.searchBtn.className = Config.cssClasses.button.inactive.input;
        }
        
        // Hide all input sections
        Utils.dom.hide(this.elements.textInput);
        Utils.dom.hide(this.elements.urlInput);
        Utils.dom.hide(this.elements.websiteInput);
        Utils.dom.hide(this.elements.searchInput);
        
        // Get analysis sections and Philippine search results for hiding/showing
        const analysisTypeSection = document.getElementById('analysisTypeSection');
        const analyzeButtonSection = document.getElementById('analyzeButtonSection');
        const searchResults = document.getElementById('searchResults');
        const analyticsResults = document.getElementById('analyticsResults');
        
        // Hide Philippine search results when switching away from search mode
        if (searchResults) Utils.dom.hide(searchResults);
        if (analyticsResults) Utils.dom.hide(analyticsResults);
        
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
        } else if (type === Config.inputTypes.SEARCH) {
            if (this.elements.searchBtn) {
                this.elements.searchBtn.className = Config.cssClasses.button.active.urlInput; // Reuse URL styling
            }
            Utils.dom.show(this.elements.searchInput);
            // Hide analysis sections for search mode
            if (analysisTypeSection) Utils.dom.hide(analysisTypeSection);
            if (analyzeButtonSection) Utils.dom.hide(analyzeButtonSection);
            // Philippine search results will be shown only when search is performed
            this.updateSearchButton();
            return; // Don't call updateAnalyzeButton for search mode
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
        
        // Reset all button styles
        const buttons = [this.elements.crawlOnlyBtn, this.elements.crawlAnalyzeBtn];
        const inactiveClass = 'crawl-mode-btn bg-gray-200 text-gray-700 px-3 py-2 rounded-lg text-sm font-medium focus-ring';
        const activeClass = 'crawl-mode-btn bg-blue-600 text-white px-3 py-2 rounded-lg text-sm font-medium focus-ring';
        
        buttons.forEach(btn => {
            if (btn) {
                btn.className = inactiveClass;
            }
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
            
            const response = await Utils.http.post(Config.endpoints.predict, requestData);
            
            if (!response.success) {
                this.showError(response.error || 'Analysis failed');
            } else {
                const result = response.data;
                if (result.error) {
                    this.showError(result.error);
                } else {
                    this.displayResults(result);
                }
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
        const response = await Utils.http.post(Config.endpoints.crawlWebsite, {
            website_url: websiteUrl,
            max_articles: maxArticles
        });

        if (!response.success) {
            throw new Error(response.error || 'Failed to crawl website');
        }

        const result = response.data;
        if (!result.success) {
            throw new Error(result.error || 'Failed to crawl website');
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
            const response = await Utils.http.post(Config.endpoints.analyzeWebsite, {
                website_url: websiteUrl,
                max_articles: this.crawledArticles.length,
                analysis_type: this.state.currentAnalysisType
            });

            if (!response.success) {
                throw new Error(response.error || 'Failed to analyze articles');
            }

            const result = response.data;
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

                // Philippine news indexing status
                if (result.indexing_status) {
                    const indexingClass = result.indexing_status === 'success' ? 'border-green-500 bg-green-50' : 
                                         result.indexing_status === 'skipped' ? 'border-yellow-500 bg-yellow-50' :
                                         result.indexing_status === 'already_indexed' ? 'border-blue-500 bg-blue-50' :
                                         'border-gray-500 bg-gray-50';
                    
                    const iconClass = result.indexing_status === 'success' ? 'bi-database-check text-green-600' :
                                     result.indexing_status === 'skipped' ? 'bi-database-dash text-yellow-600' :
                                     result.indexing_status === 'already_indexed' ? 'bi-database text-blue-600' :
                                     'bi-database-x text-gray-600';
                    
                    let indexingInfo = '';
                    if (result.indexing_status === 'success') {
                        indexingInfo = `
                            <div class="text-sm text-gray-600 mt-2">
                                <span>Philippine Relevance: ${(result.relevance_score * 100).toFixed(1)}%</span>
                                ${result.locations_found && result.locations_found.length ? `<span class="ml-3">Locations: ${result.locations_found.join(', ')}</span>` : ''}
                            </div>
                        `;
                    } else if (result.indexing_error) {
                        indexingInfo = `<div class="text-sm text-red-600 mt-2">Error: ${result.indexing_error}</div>`;
                    }
                    
                    content += `
                        <div class="border-l-4 ${indexingClass} p-4 rounded-r-lg">
                            <h6 class="font-medium text-gray-800 mb-2 flex items-center">
                                <i class="bi ${iconClass} mr-2"></i>
                                Philippine News Database
                            </h6>
                            <div>
                                <p class="text-sm text-gray-600">Indexing Status:</p>
                                <p class="font-semibold text-gray-800 capitalize">${result.indexing_status.replace('_', ' ')}</p>
                                ${indexingInfo}
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
        
        // Show indexing status if available
        if (data.indexing_status && this.state.currentInputType === Config.inputTypes.URL) {
            this.displayIndexingStatus(data.indexing_status);
        }
        
        // Show results section
        Utils.dom.show(this.elements.results);
        Utils.animations.fadeIn(this.elements.results);
    }

    // Display indexing status for single article
    displayIndexingStatus(indexingStatus) {
        // Create or update indexing status message
        const resultsContainer = this.elements.results;
        let statusElement = resultsContainer.querySelector('.indexing-status');
        
        if (!statusElement) {
            statusElement = document.createElement('div');
            statusElement.className = 'indexing-status bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4';
            resultsContainer.insertBefore(statusElement, resultsContainer.firstChild);
        }
        
        const statusIcon = indexingStatus === 'Article queued for indexing in Philippine news database' ? 
            'bi-database-add text-blue-600' : 'bi-info-circle text-blue-600';
        
        statusElement.innerHTML = `
            <div class="flex items-center">
                <i class="bi ${statusIcon} mr-2"></i>
                <span class="text-blue-800 font-medium">${indexingStatus}</span>
            </div>
        `;
    }

    // Get current input text
    getCurrentInputText() {
        if (this.state.currentInputType === Config.inputTypes.TEXT) {
            return this.elements.newsText?.value || '';
        } else {
            return this.elements.articleUrl?.value || '';
        }
    }

    // Display fake news detection results
    displayFakeNewsResults(fakeNewsData) {
        if (!fakeNewsData || fakeNewsData.error) {
            console.error('Fake news data error:', fakeNewsData?.error);
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
        const fakeProb = fakeNewsData.probabilities?.fake || 0;
        const realProb = fakeNewsData.probabilities?.real || 0;

        if (this.elements.fakeBar && this.elements.realBar) {
            Utils.animations.animateProgressBar(this.elements.fakeBar, fakeProb * 100);
            Utils.animations.animateProgressBar(this.elements.realBar, realProb * 100);
        }

        Utils.dom.setText(this.elements.fakePercentage, Utils.format.percentage(fakeProb));
        Utils.dom.setText(this.elements.realPercentage, Utils.format.percentage(realProb));
    }

    // Display political classification results
    displayPoliticalResults(politicalData) {
        if (!politicalData || politicalData.error) {
            console.error('Political data error:', politicalData?.error);
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
        const politicalProb = politicalData.probabilities?.political || 0;
        const nonPoliticalProb = politicalData.probabilities?.non_political || 0;

        if (this.elements.politicalBar && this.elements.nonPoliticalBar) {
            Utils.animations.animateProgressBar(this.elements.politicalBar, politicalProb * 100);
            Utils.animations.animateProgressBar(this.elements.nonPoliticalBar, nonPoliticalProb * 100);
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

    // Philippine News Search Methods
    
    // Update search button state
    updateSearchButton() {
        if (!this.elements.performSearchBtn) return;
        
        const query = this.elements.searchQuery?.value?.trim() || '';
        const isValid = query.length >= 3;
        
        this.elements.performSearchBtn.disabled = !isValid;
        
        if (isValid) {
            this.elements.performSearchBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        } else {
            this.elements.performSearchBtn.classList.add('opacity-50', 'cursor-not-allowed');
        }
    }

    // Perform Philippine news search
    async performPhilippineSearch() {
        const query = this.elements.searchQuery?.value?.trim();
        if (!query || query.length < 3) {
            this.showError('Please enter at least 3 characters to search.');
            return;
        }

        this.hideAllSections();
        this.state.isLoading = true;
        Utils.dom.show(this.elements.loading);

        try {
            const searchData = {
                query: query,
                category: this.elements.searchCategory?.value || '',
                source: this.elements.searchSource?.value || '',
                limit: parseInt(this.elements.searchLimit?.value || '20')
            };

            const response = await Utils.http.post(Config.endpoints.searchPhilippineNews, searchData);

            if (response.success) {
                this.displaySearchResults(response.data);
            } else {
                this.showError(response.error || 'Search failed');
            }
        } catch (error) {
            console.error('Search error:', error);
            this.showError('An error occurred while searching. Please try again.');
        } finally {
            this.state.isLoading = false;
            Utils.dom.hide(this.elements.loading);
        }
    }

    // Display search results
    displaySearchResults(data) {
        if (!this.elements.searchResults) return;

        // Update search summary
        if (this.elements.searchSummaryText) {
            this.elements.searchSummaryText.textContent = 
                `Found ${data.total_count} articles for "${data.query}"`;
        }
        if (this.elements.searchResponseTime) {
            this.elements.searchResponseTime.textContent = `${(data.response_time * 1000).toFixed(0)}ms`;
        }

        // Clear previous results
        if (this.elements.searchResultsGrid) {
            this.elements.searchResultsGrid.innerHTML = '';
        }

        if (data.results && data.results.length > 0) {
            data.results.forEach(article => {
                this.createSearchResultCard(article);
            });
        } else {
            this.elements.searchResultsGrid.innerHTML = `
                <div class="text-center py-8 text-gray-500">
                    <i class="bi bi-search text-4xl mb-4"></i>
                    <p class="text-lg">No articles found for your search.</p>
                    <p class="text-sm">Try using different keywords or filters.</p>
                </div>
            `;
        }

        Utils.dom.show(this.elements.searchResults);
    }

    // Create search result card
    createSearchResultCard(article) {
        if (!this.elements.searchResultsGrid) return;

        const publishDate = article.publish_date ? 
            new Date(article.publish_date).toLocaleDateString() : 'Date unknown';
        
        const relevanceScore = (article.relevance_score * 100).toFixed(1);
        
        const cardHtml = `
            <div class="bg-white rounded-lg shadow-md p-6 border-l-4 border-blue-500 hover:shadow-lg transition-shadow">
                <div class="flex justify-between items-start mb-3">
                    <h4 class="text-lg font-semibold text-gray-800 hover:text-blue-600 cursor-pointer" 
                        onclick="window.open('${article.url}', '_blank')">
                        ${Utils.format.escape(article.title)}
                    </h4>
                    <span class="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
                        ${relevanceScore}% PH
                    </span>
                </div>
                
                <p class="text-gray-600 mb-3 line-clamp-3">
                    ${Utils.format.escape(article.summary || 'No summary available')}
                </p>
                
                <div class="flex flex-wrap gap-2 mb-3">
                    ${article.category ? `<span class="bg-gray-100 text-gray-700 text-xs px-2 py-1 rounded">${article.category}</span>` : ''}
                    ${article.author ? `<span class="bg-green-100 text-green-700 text-xs px-2 py-1 rounded">By ${Utils.format.escape(article.author)}</span>` : ''}
                </div>
                
                <div class="flex justify-between items-center text-sm text-gray-500">
                    <div class="flex items-center">
                        <i class="bi bi-globe mr-1"></i>
                        <span>${Utils.format.escape(article.source_domain)}</span>
                    </div>
                    <div class="flex items-center">
                        <i class="bi bi-calendar mr-1"></i>
                        <span>${publishDate}</span>
                    </div>
                </div>
                
                <div class="mt-3 flex gap-2">
                    <button class="text-blue-600 hover:text-blue-800 text-sm font-medium"
                            onclick="newsAnalyzer.analyzeSearchResult('${article.url}')">
                        <i class="bi bi-search mr-1"></i>Analyze
                    </button>
                    <button class="text-gray-600 hover:text-gray-800 text-sm font-medium"
                            onclick="newsAnalyzer.viewSearchResultDetails(${article.id})">
                        <i class="bi bi-eye mr-1"></i>View Details
                    </button>
                </div>
            </div>
        `;

        this.elements.searchResultsGrid.insertAdjacentHTML('beforeend', cardHtml);
    }

    // Analyze search result
    async analyzeSearchResult(url) {
        // Switch to URL input mode and analyze
        this.switchInputType(Config.inputTypes.URL);
        if (this.elements.articleUrl) {
            this.elements.articleUrl.value = url;
            this.updateAnalyzeButton();
            await this.analyzeContent();
        }
    }

    // View search result details
    async viewSearchResultDetails(articleId) {
        try {
            const response = await Utils.http.get(`${Config.endpoints.getPhilippineArticle}${articleId}`);
            if (response.success) {
                this.displayArticleDetails(response.data);
            } else {
                this.showError('Failed to load article details');
            }
        } catch (error) {
            console.error('Error loading article details:', error);
            this.showError('Error loading article details');
        }
    }

    // Display article details modal
    displayArticleDetails(article) {
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4';
        modal.innerHTML = `
            <div class="bg-white rounded-lg max-w-4xl max-h-[90vh] overflow-y-auto">
                <div class="p-6">
                    <div class="flex justify-between items-start mb-4">
                        <h2 class="text-2xl font-bold text-gray-800">${Utils.format.escape(article.title)}</h2>
                        <button onclick="this.closest('.fixed').remove()" class="text-gray-500 hover:text-gray-700">
                            <i class="bi bi-x-lg text-xl"></i>
                        </button>
                    </div>
                    
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <div class="md:col-span-2">
                            <div class="prose max-w-none">
                                <p class="text-gray-600">${Utils.format.escape(article.content?.substring(0, 1000) || 'No content available')}${article.content?.length > 1000 ? '...' : ''}</p>
                            </div>
                        </div>
                        
                        <div class="space-y-4">
                            <div class="bg-gray-50 p-4 rounded-lg">
                                <h4 class="font-semibold text-gray-800 mb-2">Article Information</h4>
                                <div class="space-y-2 text-sm">
                                    <div><strong>Author:</strong> ${Utils.format.escape(article.author || 'Unknown')}</div>
                                    <div><strong>Source:</strong> ${Utils.format.escape(article.source_domain)}</div>
                                    <div><strong>Category:</strong> ${Utils.format.escape(article.category || 'General')}</div>
                                    <div><strong>Philippine Relevance:</strong> ${(article.philippine_relevance_score * 100).toFixed(1)}%</div>
                                    <div><strong>Publish Date:</strong> ${article.publish_date ? new Date(article.publish_date).toLocaleDateString() : 'Unknown'}</div>
                                </div>
                            </div>
                            
                            <div class="space-y-2">
                                <button onclick="window.open('${article.url}', '_blank')" 
                                        class="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700">
                                    <i class="bi bi-box-arrow-up-right mr-2"></i>Open Original
                                </button>
                                <button onclick="newsAnalyzer.analyzeSearchResult('${article.url}'); this.closest('.fixed').remove();" 
                                        class="w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700">
                                    <i class="bi bi-search mr-2"></i>Analyze Article
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }

    // View Philippine analytics
    async viewPhilippineAnalytics() {
        this.hideAllSections();
        this.state.isLoading = true;
        Utils.dom.show(this.elements.loading);

        try {
            const response = await Utils.http.get(Config.endpoints.philippineNewsAnalytics);
            
            if (response.success) {
                this.displayAnalytics(response.data);
            } else {
                this.showError(response.error || 'Failed to load analytics');
            }
        } catch (error) {
            console.error('Analytics error:', error);
            this.showError('An error occurred while loading analytics. Please try again.');
        } finally {
            this.state.isLoading = false;
            Utils.dom.hide(this.elements.loading);
        }
    }

    // Display analytics
    displayAnalytics(data) {
        if (!this.elements.analyticsResults) return;

        // Update key statistics
        if (this.elements.totalArticlesCount) {
            this.elements.totalArticlesCount.textContent = data.total_articles || '0';
        }
        if (this.elements.totalSourcesCount) {
            this.elements.totalSourcesCount.textContent = data.sources?.length || '0';
        }
        if (this.elements.avgRelevanceScore) {
            this.elements.avgRelevanceScore.textContent = data.average_relevance_score || '0.0';
        }
        if (this.elements.totalCategoriesCount) {
            this.elements.totalCategoriesCount.textContent = data.categories?.length || '0';
        }

        // Update charts
        this.updateSourcesChart(data.sources || []);
        this.updateCategoriesChart(data.categories || []);
        this.updateRecentActivityChart(data.recent_activity || []);
        this.updateTopQueriesTable(data.top_queries || []);

        Utils.dom.show(this.elements.analyticsResults);
    }

    // Update sources chart
    updateSourcesChart(sources) {
        if (!this.elements.topSourcesChart) return;
        
        this.elements.topSourcesChart.innerHTML = '';
        
        sources.slice(0, 10).forEach(source => {
            const percentage = sources[0]?.count ? (source.count / sources[0].count * 100) : 0;
            
            const chartItem = document.createElement('div');
            chartItem.className = 'flex items-center justify-between mb-2';
            chartItem.innerHTML = `
                <div class="flex-1 mr-4">
                    <div class="text-sm font-medium text-gray-700">${Utils.format.escape(source.domain)}</div>
                    <div class="bg-gray-200 rounded-full h-2 mt-1">
                        <div class="bg-blue-600 h-2 rounded-full transition-all duration-500" 
                             style="width: ${percentage}%"></div>
                    </div>
                </div>
                <div class="text-sm font-semibold text-gray-600">${source.count}</div>
            `;
            
            this.elements.topSourcesChart.appendChild(chartItem);
        });
    }

    // Update categories chart
    updateCategoriesChart(categories) {
        if (!this.elements.categoriesChart) return;
        
        this.elements.categoriesChart.innerHTML = '';
        
        categories.forEach(category => {
            const percentage = categories[0]?.count ? (category.count / categories[0].count * 100) : 0;
            
            const chartItem = document.createElement('div');
            chartItem.className = 'flex items-center justify-between mb-2';
            chartItem.innerHTML = `
                <div class="flex-1 mr-4">
                    <div class="text-sm font-medium text-gray-700">${Utils.format.escape(category.category || 'Unknown')}</div>
                    <div class="bg-gray-200 rounded-full h-2 mt-1">
                        <div class="bg-green-600 h-2 rounded-full transition-all duration-500" 
                             style="width: ${percentage}%"></div>
                    </div>
                </div>
                <div class="text-sm font-semibold text-gray-600">${category.count}</div>
            `;
            
            this.elements.categoriesChart.appendChild(chartItem);
        });
    }

    // Update recent activity chart
    updateRecentActivityChart(activity) {
        if (!this.elements.recentActivityChart) return;
        
        this.elements.recentActivityChart.innerHTML = '';
        
        if (activity.length === 0) {
            this.elements.recentActivityChart.innerHTML = '<p class="text-gray-500 text-sm">No recent activity</p>';
            return;
        }
        
        activity.forEach(day => {
            const chartItem = document.createElement('div');
            chartItem.className = 'flex items-center justify-between py-1';
            chartItem.innerHTML = `
                <div class="text-sm text-gray-700">${day.date}</div>
                <div class="text-sm font-semibold text-blue-600">${day.count} articles</div>
            `;
            
            this.elements.recentActivityChart.appendChild(chartItem);
        });
    }

    // Update top queries table
    updateTopQueriesTable(queries) {
        if (!this.elements.topQueriesBody) return;
        
        this.elements.topQueriesBody.innerHTML = '';
        
        if (queries.length === 0) {
            this.elements.topQueriesBody.innerHTML = `
                <tr>
                    <td colspan="3" class="text-center py-4 text-gray-500">No search queries yet</td>
                </tr>
            `;
            return;
        }
        
        queries.forEach(query => {
            const row = document.createElement('tr');
            row.className = 'border-b';
            row.innerHTML = `
                <td class="py-2 text-gray-700">${Utils.format.escape(query.query)}</td>
                <td class="py-2 text-center text-gray-600">${query.frequency}</td>
                <td class="py-2 text-center text-gray-600">${query.avg_results.toFixed(1)}</td>
            `;
            
            this.elements.topQueriesBody.appendChild(row);
        });
    }

    // Show add to index modal
    showAddToIndexModal() {
        if (this.elements.addArticleModal) {
            Utils.dom.show(this.elements.addArticleModal);
            if (this.elements.indexArticleUrl) {
                this.elements.indexArticleUrl.focus();
            }
        }
    }

    // Hide add to index modal
    hideAddToIndexModal() {
        if (this.elements.addArticleModal) {
            Utils.dom.hide(this.elements.addArticleModal);
            if (this.elements.indexArticleUrl) {
                this.elements.indexArticleUrl.value = '';
            }
        }
    }

    // Submit article to index
    async submitIndexArticle() {
        const url = this.elements.indexArticleUrl?.value?.trim();
        if (!url) {
            this.showError('Please enter a valid URL');
            return;
        }

        if (!Config.validation.urlPattern.test(url)) {
            this.showError('Please enter a valid URL starting with http:// or https://');
            return;
        }

        try {
            this.state.isLoading = true;
            Utils.dom.show(this.elements.loading);
            
            const response = await Utils.http.post(Config.endpoints.indexPhilippineArticle, {
                url: url,
                force_reindex: false
            });

            if (response.success) {
                this.hideAddToIndexModal();
                this.showError('Article indexed successfully!', 'success');
            } else {
                this.showError(response.error || 'Failed to index article');
            }
        } catch (error) {
            console.error('Indexing error:', error);
            this.showError('An error occurred while indexing the article');
        } finally {
            this.state.isLoading = false;
            Utils.dom.hide(this.elements.loading);
        }
    }

    // Crawl and index entire website
    // Crawl and index website (now just calls regular analysis which includes indexing)
    async crawlAndIndexWebsite() {
        // Set to analyze mode to ensure indexing happens
        this.switchCrawlingMode(Config.crawlingModes.ANALYZE);
        // Call the regular website analysis which now includes automatic indexing
        await this.handleWebsiteCrawling();
    }

    // Display crawl and index results
    displayCrawlAndIndexResults(response) {
        const { summary, results, website_title, website_url } = response;
        
        Utils.dom.show(this.elements.websiteResults);
        this.elements.websiteResults.innerHTML = `
            <div class="bg-white rounded-lg shadow-lg p-6">
                <div class="flex items-center justify-between mb-6">
                    <div>
                        <h3 class="text-2xl font-bold text-gray-800">Website Crawl & Index Complete</h3>
                        <p class="text-gray-600 mt-1">${Utils.format.escape(website_title || website_url)}</p>
                    </div>
                    <div class="text-right">
                        <div class="text-2xl font-bold text-green-600">${summary.successfully_indexed}</div>
                        <div class="text-sm text-gray-500">Articles Indexed</div>
                    </div>
                </div>

                <!-- Summary Statistics -->
                <div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
                    <div class="bg-blue-50 p-4 rounded-lg text-center">
                        <div class="text-2xl font-bold text-blue-600">${summary.total_articles_found}</div>
                        <div class="text-sm text-blue-800">Found</div>
                    </div>
                    <div class="bg-green-50 p-4 rounded-lg text-center">
                        <div class="text-2xl font-bold text-green-600">${summary.successfully_indexed}</div>
                        <div class="text-sm text-green-800">Indexed</div>
                    </div>
                    <div class="bg-yellow-50 p-4 rounded-lg text-center">
                        <div class="text-2xl font-bold text-yellow-600">${summary.skipped}</div>
                        <div class="text-sm text-yellow-800">Skipped</div>
                    </div>
                    <div class="bg-gray-50 p-4 rounded-lg text-center">
                        <div class="text-2xl font-bold text-gray-600">${summary.already_indexed}</div>
                        <div class="text-sm text-gray-800">Already Indexed</div>
                    </div>
                    <div class="bg-red-50 p-4 rounded-lg text-center">
                        <div class="text-2xl font-bold text-red-600">${summary.errors}</div>
                        <div class="text-sm text-red-800">Errors</div>
                    </div>
                </div>

                <!-- Detailed Results -->
                <div class="border-t pt-6">
                    <h4 class="text-lg font-semibold text-gray-800 mb-4">Article Processing Results</h4>
                    <div class="space-y-3 max-h-96 overflow-y-auto">
                        ${results.map(result => this.createCrawlResultCard(result)).join('')}
                    </div>
                </div>

                <!-- Actions -->
                <div class="border-t pt-6 mt-6">
                    <div class="flex flex-wrap gap-3">
                        <button onclick="newsAnalyzer.performPhilippineSearch('${Utils.format.escape(website_title || '')}')" 
                                class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors">
                            <i class="bi bi-search mr-2"></i>Search Indexed Articles
                        </button>
                        <button onclick="newsAnalyzer.viewCrawlHistory()" 
                                class="bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700 transition-colors">
                            <i class="bi bi-clock-history mr-2"></i>View Crawl History
                        </button>
                        <button onclick="newsAnalyzer.crawlAndIndexWebsite()" 
                                class="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition-colors">
                            <i class="bi bi-arrow-clockwise mr-2"></i>Crawl Another Website
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    // Create crawl result card
    createCrawlResultCard(result) {
        const statusConfig = {
            'success': { 
                bg: 'bg-green-50 border-green-200', 
                icon: 'bi-check-circle-fill text-green-600',
                text: 'text-green-800'
            },
            'skipped': { 
                bg: 'bg-yellow-50 border-yellow-200', 
                icon: 'bi-skip-forward-fill text-yellow-600',
                text: 'text-yellow-800'
            },
            'already_indexed': { 
                bg: 'bg-blue-50 border-blue-200', 
                icon: 'bi-info-circle-fill text-blue-600',
                text: 'text-blue-800'
            },
            'error': { 
                bg: 'bg-red-50 border-red-200', 
                icon: 'bi-exclamation-triangle-fill text-red-600',
                text: 'text-red-800'
            }
        };

        const config = statusConfig[result.status] || statusConfig['error'];

        return `
            <div class="border rounded-lg p-4 ${config.bg}">
                <div class="flex items-start space-x-3">
                    <i class="bi ${config.icon} text-lg mt-0.5"></i>
                    <div class="flex-1 min-w-0">
                        <h5 class="font-medium ${config.text} truncate">${Utils.format.escape(result.title)}</h5>
                        <p class="text-sm text-gray-600 truncate">${Utils.format.escape(result.url)}</p>
                        ${result.status === 'success' ? `
                            <div class="mt-2 text-sm ${config.text}">
                                <span class="inline-flex items-center">
                                    Relevance: ${(result.relevance_score * 100).toFixed(1)}%
                                    ${result.locations_found?.length ? ` • Locations: ${result.locations_found.join(', ')}` : ''}
                                </span>
                            </div>
                        ` : ''}
                        ${result.message ? `<p class="text-sm ${config.text} mt-1">${Utils.format.escape(result.message)}</p>` : ''}
                        ${result.error ? `<p class="text-sm ${config.text} mt-1">Error: ${Utils.format.escape(result.error)}</p>` : ''}
                    </div>
                    ${result.article_id ? `
                        <button onclick="newsAnalyzer.viewSearchResultDetails(${result.article_id})" 
                                class="text-blue-600 hover:text-blue-800 text-sm font-medium">
                            View
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    }

    // View crawl history
    async viewCrawlHistory() {
        try {
            this.state.isLoading = true;
            Utils.dom.show(this.elements.loading);

            const response = await Utils.http.get(Config.endpoints.getCrawlHistory + '?limit=20');

            if (response.success) {
                this.displayCrawlHistory(response.history);
            } else {
                this.showError('Failed to load crawl history');
            }
        } catch (error) {
            console.error('Error loading crawl history:', error);
            this.showError('Error loading crawl history');
        } finally {
            this.state.isLoading = false;
            Utils.dom.hide(this.elements.loading);
        }
    }

    // Display crawl history
    displayCrawlHistory(history) {
        this.hideAllSections();
        Utils.dom.show(this.elements.websiteResults);

        if (!history || history.length === 0) {
            this.elements.websiteResults.innerHTML = `
                <div class="bg-white rounded-lg shadow-lg p-6 text-center">
                    <div class="text-gray-500 mb-4">
                        <i class="bi bi-clock-history text-4xl"></i>
                    </div>
                    <h3 class="text-xl font-semibold text-gray-800 mb-2">No Crawl History</h3>
                    <p class="text-gray-600">No website crawling activities found.</p>
                </div>
            `;
            return;
        }

        this.elements.websiteResults.innerHTML = `
            <div class="bg-white rounded-lg shadow-lg p-6">
                <div class="flex items-center justify-between mb-6">
                    <h3 class="text-2xl font-bold text-gray-800">Website Crawl History</h3>
                    <span class="text-gray-500">${history.length} record(s)</span>
                </div>

                <div class="space-y-4">
                    ${history.map(record => `
                        <div class="border rounded-lg p-4 hover:bg-gray-50">
                            <div class="flex items-center justify-between">
                                <div class="flex-1">
                                    <h4 class="font-medium text-gray-800">${Utils.format.escape(record.url)}</h4>
                                    <p class="text-sm text-gray-600 mt-1">${Utils.format.escape(record.message)}</p>
                                    <div class="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                                        <span>Started: ${new Date(record.started_date).toLocaleString()}</span>
                                        ${record.completed_date ? `<span>Completed: ${new Date(record.completed_date).toLocaleString()}</span>` : ''}
                                    </div>
                                </div>
                                <div class="text-right">
                                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                        record.status === 'website_crawl_completed' ? 'bg-green-100 text-green-800' :
                                        record.status === 'failed' ? 'bg-red-100 text-red-800' :
                                        'bg-yellow-100 text-yellow-800'
                                    }">
                                        ${record.status.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                    </span>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>

                <div class="border-t pt-6 mt-6">
                    <button onclick="newsAnalyzer.crawlAndIndexWebsite()" 
                            class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors">
                        <i class="bi bi-plus-circle mr-2"></i>Crawl New Website
                    </button>
                </div>
            </div>
        `;
    }

    // Hide all sections
    hideAllSections() {
        const sections = [
            this.elements.results,
            this.elements.websiteResults,
            this.elements.searchResults,
            this.elements.analyticsResults,
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
