/**
 * RSS Feed Analysis JavaScript Module
 * Handles RSS feed fetching, article display, and analysis
 */

class RSSFeedAnalyzer {
    constructor() {
        this.state = {
            currentFeed: null,
            fetchedArticles: [],
            analyzedArticles: [],
            selectedFeeds: new Set(),
            isLoading: false,
            currentView: 'feeds'
        };

        this.elements = {};
        this.init();
    }

    init() {
        this.cacheElements();
        this.bindEvents();
        this.loadPredefinedFeeds();
        console.log('RSS Feed Analyzer initialized successfully');
    }

    cacheElements() {
        const elementIds = [
            'predefinedFeeds', 'customFeedUrl', 'addCustomFeedBtn',
            'articleLimit', 'timeFilter', 'analysisType',
            'fetchArticlesBtn', 'analyzeArticlesBtn', 'exportResultsBtn',
            'loadingIndicator', 'loadingTitle', 'loadingMessage', 'progressBar',
            'resultsSummary', 'totalArticlesCount', 'successfulCount',
            'fakeNewsSummary', 'fakeNewsCount', 'fakeNewsPercentage',
            'politicalSummary', 'politicalNewsCount', 'politicalNewsPercentage',
            'performanceMetrics', 'processingTime', 'feedSource', 'articlesPerSecond',
            'articlesContainer', 'articlesList', 'articleFilter', 'articleSort',
            'errorDisplay', 'errorMessage',
            'articleModal', 'modalTitle', 'modalContent', 'closeModalBtn'
        ];

        elementIds.forEach(id => {
            this.elements[id] = document.getElementById(id);
        });
    }

    bindEvents() {
        // Add custom feed
        this.elements.addCustomFeedBtn?.addEventListener('click', () => this.addCustomFeed());
        this.elements.customFeedUrl?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.addCustomFeed();
        });

        // Action buttons
        this.elements.fetchArticlesBtn?.addEventListener('click', () => this.fetchArticles());
        this.elements.analyzeArticlesBtn?.addEventListener('click', () => this.analyzeArticles());
        this.elements.exportResultsBtn?.addEventListener('click', () => this.exportResults());

        // Filter and sort
        this.elements.articleFilter?.addEventListener('change', () => this.filterAndSortArticles());
        this.elements.articleSort?.addEventListener('change', () => this.filterAndSortArticles());

        // Modal controls
        this.elements.closeModalBtn?.addEventListener('click', () => this.closeModal());
        this.elements.articleModal?.addEventListener('click', (e) => {
            if (e.target === this.elements.articleModal) this.closeModal();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.closeModal();
        });
    }

    async loadPredefinedFeeds() {
        try {
            const response = await fetch('/api/rss-feeds');
            const data = await response.json();

            if (data.success) {
                this.displayPredefinedFeeds(data.feeds);
            } else {
                this.showError('Failed to load predefined feeds');
            }
        } catch (error) {
            console.error('Error loading predefined feeds:', error);
            this.showError('Network error loading feeds');
        }
    }

    displayPredefinedFeeds(feeds) {
        if (!this.elements.predefinedFeeds) return;

        this.elements.predefinedFeeds.innerHTML = '';

        Object.entries(feeds).forEach(([key, feed]) => {
            const feedCard = document.createElement('div');
            feedCard.className = `feed-card p-4 border border-gray-200 rounded-lg cursor-pointer transition-all duration-200 hover:shadow-md ${feed.active ? 'bg-blue-50 border-blue-300' : 'bg-white'}`;
            feedCard.setAttribute('data-feed-key', key);

            feedCard.innerHTML = `
                <div class="flex items-center justify-between mb-2">
                    <h3 class="font-semibold text-gray-800">${feed.name}</h3>
                    <input type="checkbox" class="feed-checkbox w-5 h-5 text-blue-600 rounded focus:ring-blue-500" 
                           ${feed.active ? 'checked' : ''} data-feed-key="${key}">
                </div>
                <p class="text-sm text-gray-600 mb-2">${feed.category}</p>
                <p class="text-xs text-gray-500 break-all">${feed.url}</p>
                <div class="mt-3 flex items-center justify-between">
                    <button class="test-feed-btn text-blue-600 hover:text-blue-800 text-sm font-medium" 
                            data-feed-url="${feed.url}">
                        <i class="bi bi-play-circle mr-1"></i>Test Feed
                    </button>
                    <span class="text-xs px-2 py-1 rounded-full ${feed.active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'}">
                        ${feed.active ? 'Active' : 'Inactive'}
                    </span>
                </div>
            `;

            // Bind events for this card
            const checkbox = feedCard.querySelector('.feed-checkbox');
            checkbox.addEventListener('change', (e) => this.toggleFeedSelection(key, e.target.checked));

            const testBtn = feedCard.querySelector('.test-feed-btn');
            testBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.testFeed(feed.url, feed.name);
            });

            this.elements.predefinedFeeds.appendChild(feedCard);

            // Initialize selected feeds
            if (feed.active) {
                this.state.selectedFeeds.add(key);
            }
        });
    }

    toggleFeedSelection(feedKey, selected) {
        if (selected) {
            this.state.selectedFeeds.add(feedKey);
        } else {
            this.state.selectedFeeds.delete(feedKey);
        }

        // Update UI
        const feedCard = document.querySelector(`[data-feed-key="${feedKey}"]`);
        if (feedCard) {
            if (selected) {
                feedCard.classList.add('bg-blue-50', 'border-blue-300');
                feedCard.classList.remove('bg-white');
            } else {
                feedCard.classList.remove('bg-blue-50', 'border-blue-300');
                feedCard.classList.add('bg-white');
            }
        }

        // Update fetch button state
        this.updateFetchButtonState();
    }

    async testFeed(feedUrl, feedName) {
        this.showLoading('Testing RSS Feed', `Testing connection to ${feedName}...`);

        try {
            const response = await fetch('/api/rss-feed/parse', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    feed_url: feedUrl
                })
            });

            const data = await response.json();
            this.hideLoading();

            if (data.success) {
                this.showFeedTestResult(data, feedName);
            } else {
                this.showError(`Feed test failed: ${data.error}`);
            }
        } catch (error) {
            this.hideLoading();
            this.showError(`Network error testing feed: ${error.message}`);
        }
    }

    showFeedTestResult(data, feedName) {
        const feedInfo = data.feed_info;
        const sampleArticles = data.articles.slice(0, 3);

        const content = `
            <div class="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
                <h3 class="text-lg font-semibold text-green-800 mb-2">
                    <i class="bi bi-check-circle mr-2"></i>Feed Test Successful
                </h3>
                <p class="text-green-700">Successfully connected to ${feedName}</p>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                    <h4 class="font-semibold text-gray-800 mb-2">Feed Information</h4>
                    <ul class="text-sm text-gray-600 space-y-1">
                        <li><strong>Title:</strong> ${feedInfo.title}</li>
                        <li><strong>Language:</strong> ${feedInfo.language || 'Not specified'}</li>
                        <li><strong>Total Entries:</strong> ${feedInfo.total_entries}</li>
                        <li><strong>Last Updated:</strong> ${feedInfo.updated || 'Not specified'}</li>
                    </ul>
                </div>
                <div>
                    <h4 class="font-semibold text-gray-800 mb-2">Sample Articles</h4>
                    <ul class="text-sm text-gray-600 space-y-1">
                        ${sampleArticles.map(article => `
                            <li class="truncate">• ${article.title}</li>
                        `).join('')}
                    </ul>
                </div>
            </div>
            
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <p class="text-sm text-blue-700">
                    <i class="bi bi-info-circle mr-1"></i>
                    This feed contains ${data.total_found} articles ready for analysis
                </p>
            </div>
        `;

        this.showModal('Feed Test Results', content);
    }

    addCustomFeed() {
        const url = this.elements.customFeedUrl?.value.trim();
        if (!url) {
            this.showError('Please enter a valid RSS feed URL');
            return;
        }

        // Test the custom feed
        this.testFeed(url, 'Custom Feed');
        this.elements.customFeedUrl.value = '';
    }

    updateFetchButtonState() {
        if (this.elements.fetchArticlesBtn) {
            this.elements.fetchArticlesBtn.disabled = this.state.selectedFeeds.size === 0;
        }
    }

    async fetchArticles() {
        if (this.state.selectedFeeds.size === 0) {
            this.showError('Please select at least one RSS feed');
            return;
        }

        const limit = parseInt(this.elements.articleLimit?.value) || null;
        const hoursBack = parseInt(this.elements.timeFilter?.value) || null;

        this.showLoading('Fetching Articles', 'Downloading articles from RSS feeds...');
        this.updateProgress(10);

        try {
            // Get selected feed URLs
            const feedUrls = await this.getSelectedFeedUrls();
            const allArticles = [];
            let completedFeeds = 0;

            for (const feedData of feedUrls) {
                try {
                    const response = await fetch('/api/rss-feed/articles', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            feed_url: feedData.url,
                            limit: limit,
                            hours_back: hoursBack
                        })
                    });

                    const data = await response.json();

                    if (data.success) {
                        // Add feed source to each article
                        data.articles.forEach(article => {
                            article.feed_source = feedData.name;
                            article.feed_category = feedData.category;
                        });
                        allArticles.push(...data.articles);
                    } else {
                        console.error(`Failed to fetch from ${feedData.name}:`, data.error);
                    }

                    completedFeeds++;
                    this.updateProgress(10 + (completedFeeds / feedUrls.length) * 80);

                } catch (error) {
                    console.error(`Error fetching from ${feedData.name}:`, error);
                }
            }

            this.hideLoading();

            if (allArticles.length === 0) {
                this.showError('No articles were fetched from the selected feeds');
                return;
            }

            // Remove duplicates based on URL
            const uniqueArticles = this.removeDuplicateArticles(allArticles);

            // Sort by publication date (newest first)
            uniqueArticles.sort((a, b) => {
                const dateA = new Date(a.published_date || 0);
                const dateB = new Date(b.published_date || 0);
                return dateB - dateA;
            });

            this.state.fetchedArticles = uniqueArticles;
            this.displayFetchedArticles();
            this.elements.analyzeArticlesBtn.disabled = false;

        } catch (error) {
            this.hideLoading();
            this.showError(`Failed to fetch articles: ${error.message}`);
        }
    }

    async getSelectedFeedUrls() {
        const response = await fetch('/api/rss-feeds');
        const data = await response.json();

        if (!data.success) {
            throw new Error('Failed to get feed data');
        }

        return Array.from(this.state.selectedFeeds).map(key => {
            const feed = data.feeds[key];
            return {
                name: feed.name,
                url: feed.url,
                category: feed.category
            };
        });
    }

    removeDuplicateArticles(articles) {
        const seen = new Set();
        return articles.filter(article => {
            if (seen.has(article.url)) {
                return false;
            }
            seen.add(article.url);
            return true;
        });
    }

    displayFetchedArticles() {
        this.updateSummaryDisplay();
        this.displayArticlesList(this.state.fetchedArticles);
        this.elements.articlesContainer?.classList.remove('hidden');
    }

    async analyzeArticles() {
        if (this.state.fetchedArticles.length === 0) {
            this.showError('No articles to analyze. Please fetch articles first.');
            return;
        }

        const analysisType = this.elements.analysisType?.value || 'both';

        this.showLoading('Analyzing Articles', 'Running AI analysis on fetched articles...');
        this.updateProgress(0);

        try {
            const response = await fetch('/api/rss-feed/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    articles: this.state.fetchedArticles,
                    analysis_type: analysisType
                })
            });

            const data = await response.json();
            this.hideLoading();

            if (data.success) {
                this.state.analyzedArticles = data.results;
                this.displayAnalysisResults(data);
                this.elements.exportResultsBtn.disabled = false;
            } else {
                this.showError(`Analysis failed: ${data.error}`);
            }

        } catch (error) {
            this.hideLoading();
            this.showError(`Analysis error: ${error.message}`);
        }
    }

    displayAnalysisResults(data) {
        const summary = data.summary;

        // Update basic counts
        this.elements.totalArticlesCount.textContent = summary.total_articles;
        this.elements.successfulCount.textContent = summary.successful_analyses;

        // Show fake news summary if available
        if (summary.fake_news_summary) {
            this.elements.fakeNewsSummary?.classList.remove('hidden');
            this.elements.fakeNewsCount.textContent = summary.fake_news_summary.fake_count;
            this.elements.fakeNewsPercentage.textContent = `${summary.fake_news_summary.fake_percentage.toFixed(1)}%`;
        }

        // Show political summary if available
        if (summary.political_summary) {
            this.elements.politicalSummary?.classList.remove('hidden');
            this.elements.politicalNewsCount.textContent = summary.political_summary.political_count;
            this.elements.politicalNewsPercentage.textContent = `${summary.political_summary.political_percentage.toFixed(1)}%`;
        }

        this.elements.resultsSummary?.classList.remove('hidden');
        this.displayArticlesList(this.state.analyzedArticles);
    }

    displayArticlesList(articles) {
        if (!this.elements.articlesList) return;

        this.elements.articlesList.innerHTML = '';

        if (articles.length === 0) {
            this.elements.articlesList.innerHTML = `
                <div class="text-center py-8 text-gray-500">
                    <i class="bi bi-inbox text-4xl mb-4"></i>
                    <h3 class="text-lg font-semibold mb-2">No Articles Found</h3>
                    <p>Try adjusting your filters or fetch new articles.</p>
                </div>
            `;
            return;
        }

        articles.forEach((article, index) => {
            const articleCard = this.createArticleCard(article, index);
            this.elements.articlesList.appendChild(articleCard);
        });
    }

    createArticleCard(article, index) {
        const card = document.createElement('div');
        card.className = 'bg-white rounded-lg shadow-md p-6 border border-gray-200 hover:shadow-lg transition-shadow cursor-pointer';

        // Determine card border color based on analysis results
        let borderClass = 'border-gray-200';
        if (article.fake_news?.prediction === 'Fake') {
            borderClass = 'border-red-300 bg-red-50';
        } else if (article.political_classification?.prediction === 'Political') {
            borderClass = 'border-blue-300 bg-blue-50';
        }

        card.className += ` ${borderClass}`;

        // Format publication date
        let publishedDate = 'Unknown date';
        if (article.published_date) {
            try {
                publishedDate = new Date(article.published_date).toLocaleString();
            } catch (e) {
                publishedDate = article.published_date;
            }
        }

        card.innerHTML = `
            <div class="flex justify-between items-start mb-4">
                <div class="flex-1">
                    <h3 class="text-xl font-semibold text-gray-800 mb-2 line-clamp-2">${this.escapeHtml(article.title)}</h3>
                    <div class="flex items-center text-sm text-gray-600 mb-3 space-x-4">
                        <span class="flex items-center">
                            <i class="bi bi-calendar mr-1"></i>
                            ${publishedDate}
                        </span>
                        <span class="flex items-center">
                            <i class="bi bi-globe mr-1"></i>
                            ${article.source_domain}
                        </span>
                        ${article.feed_source ? `
                            <span class="flex items-center">
                                <i class="bi bi-rss mr-1"></i>
                                ${article.feed_source}
                            </span>
                        ` : ''}
                    </div>
                </div>
                <span class="bg-gray-100 text-gray-700 text-xs px-2 py-1 rounded-full ml-4">
                    #${index + 1}
                </span>
            </div>

            ${article.description ? `
                <p class="text-gray-700 mb-4 line-clamp-3">${this.escapeHtml(article.description)}</p>
            ` : ''}

            <!-- Analysis Results -->
            <div class="space-y-3 mb-4">
                ${this.createAnalysisResultDisplay(article)}
            </div>

            <!-- Action Buttons -->
            <div class="flex items-center justify-between pt-4 border-t border-gray-200">
                <button onclick="window.open('${article.url}', '_blank')" 
                        class="text-blue-600 hover:text-blue-800 text-sm font-medium flex items-center">
                    <i class="bi bi-box-arrow-up-right mr-1"></i>Read Original
                </button>
                <button onclick="rssAnalyzer.showArticleDetails(${index})" 
                        class="text-purple-600 hover:text-purple-800 text-sm font-medium flex items-center">
                    <i class="bi bi-info-circle mr-1"></i>View Details
                </button>
            </div>
        `;

        return card;
    }

    createAnalysisResultDisplay(article) {
        let html = '';

        // Fake news analysis
        if (article.fake_news && !article.fake_news.error) {
            const prediction = article.fake_news.prediction;
            const confidence = (article.fake_news.confidence * 100).toFixed(1);
            const isFake = prediction === 'Fake';
            
            html += `
                <div class="flex items-center justify-between p-3 rounded-lg ${isFake ? 'bg-red-100 border border-red-200' : 'bg-green-100 border border-green-200'}">
                    <div class="flex items-center">
                        <i class="bi bi-${isFake ? 'exclamation-triangle' : 'check-circle'} ${isFake ? 'text-red-600' : 'text-green-600'} mr-2"></i>
                        <span class="font-medium ${isFake ? 'text-red-800' : 'text-green-800'}">${prediction} News</span>
                    </div>
                    <span class="text-sm ${isFake ? 'text-red-600' : 'text-green-600'}">${confidence}% confidence</span>
                </div>
            `;
        }

        // Political classification
        if (article.political_classification && !article.political_classification.error) {
            const prediction = article.political_classification.prediction;
            const confidence = (article.political_classification.confidence * 100).toFixed(1);
            const isPolitical = prediction === 'Political';
            
            html += `
                <div class="flex items-center justify-between p-3 rounded-lg ${isPolitical ? 'bg-blue-100 border border-blue-200' : 'bg-gray-100 border border-gray-200'}">
                    <div class="flex items-center">
                        <i class="bi bi-${isPolitical ? 'building' : 'newspaper'} ${isPolitical ? 'text-blue-600' : 'text-gray-600'} mr-2"></i>
                        <span class="font-medium ${isPolitical ? 'text-blue-800' : 'text-gray-800'}">${prediction}</span>
                    </div>
                    <span class="text-sm ${isPolitical ? 'text-blue-600' : 'text-gray-600'}">${confidence}% confidence</span>
                </div>
            `;
        }

        // Analysis errors
        if (article.analysis_error) {
            html += `
                <div class="flex items-center p-3 rounded-lg bg-yellow-100 border border-yellow-200">
                    <i class="bi bi-exclamation-triangle text-yellow-600 mr-2"></i>
                    <span class="text-sm text-yellow-800">Analysis Error: ${article.analysis_error}</span>
                </div>
            `;
        }

        return html;
    }

    showArticleDetails(index) {
        const article = this.state.analyzedArticles[index] || this.state.fetchedArticles[index];
        if (!article) return;

        const content = `
            <div class="space-y-6">
                <!-- Article Info -->
                <div>
                    <h3 class="text-lg font-semibold text-gray-800 mb-3">Article Information</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <p class="text-sm text-gray-600 mb-1">Source</p>
                            <p class="font-medium">${article.source_domain}</p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-600 mb-1">Published</p>
                            <p class="font-medium">${article.published_date ? new Date(article.published_date).toLocaleString() : 'Unknown'}</p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-600 mb-1">Author</p>
                            <p class="font-medium">${article.author || 'Not specified'}</p>
                        </div>
                        <div>
                            <p class="text-sm text-gray-600 mb-1">Word Count</p>
                            <p class="font-medium">${article.word_count || 0} words</p>
                        </div>
                    </div>
                </div>

                <!-- Full Description -->
                ${article.description ? `
                    <div>
                        <h3 class="text-lg font-semibold text-gray-800 mb-3">Description</h3>
                        <p class="text-gray-700 leading-relaxed">${this.escapeHtml(article.description)}</p>
                    </div>
                ` : ''}

                <!-- Categories -->
                ${article.categories && article.categories.length > 0 ? `
                    <div>
                        <h3 class="text-lg font-semibold text-gray-800 mb-3">Categories</h3>
                        <div class="flex flex-wrap gap-2">
                            ${article.categories.map(cat => `
                                <span class="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">${cat}</span>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}

                <!-- Analysis Results -->
                <div>
                    <h3 class="text-lg font-semibold text-gray-800 mb-3">Analysis Results</h3>
                    <div class="space-y-3">
                        ${this.createAnalysisResultDisplay(article)}
                    </div>
                </div>

                <!-- Actions -->
                <div class="flex gap-4 pt-4 border-t border-gray-200">
                    <button onclick="window.open('${article.url}', '_blank')" 
                            class="bg-blue-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors">
                        <i class="bi bi-box-arrow-up-right mr-2"></i>Read Full Article
                    </button>
                    <button onclick="rssAnalyzer.copyArticleUrl('${article.url}')" 
                            class="bg-gray-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-gray-700 transition-colors">
                        <i class="bi bi-clipboard mr-2"></i>Copy URL
                    </button>
                </div>
            </div>
        `;

        this.showModal(article.title, content);
    }

    copyArticleUrl(url) {
        navigator.clipboard.writeText(url).then(() => {
            this.showToast('Article URL copied to clipboard', 'success');
        }).catch(() => {
            this.showToast('Failed to copy URL', 'error');
        });
    }

    filterAndSortArticles() {
        const filter = this.elements.articleFilter?.value || 'all';
        const sort = this.elements.articleSort?.value || 'newest';

        let articles = [...this.state.analyzedArticles];

        // Apply filters
        if (filter !== 'all') {
            articles = articles.filter(article => {
                switch (filter) {
                    case 'fake':
                        return article.fake_news?.prediction === 'Fake';
                    case 'real':
                        return article.fake_news?.prediction === 'Real';
                    case 'political':
                        return article.political_classification?.prediction === 'Political';
                    case 'non-political':
                        return article.political_classification?.prediction === 'Non-Political';
                    default:
                        return true;
                }
            });
        }

        // Apply sorting
        articles.sort((a, b) => {
            switch (sort) {
                case 'oldest':
                    return new Date(a.published_date || 0) - new Date(b.published_date || 0);
                case 'confidence':
                    const confA = Math.max(a.fake_news?.confidence || 0, a.political_classification?.confidence || 0);
                    const confB = Math.max(b.fake_news?.confidence || 0, b.political_classification?.confidence || 0);
                    return confB - confA;
                case 'newest':
                default:
                    return new Date(b.published_date || 0) - new Date(a.published_date || 0);
            }
        });

        this.displayArticlesList(articles);
    }

    updateSummaryDisplay() {
        if (this.elements.totalArticlesCount) {
            this.elements.totalArticlesCount.textContent = this.state.fetchedArticles.length;
        }
        if (this.elements.successfulCount) {
            this.elements.successfulCount.textContent = this.state.fetchedArticles.length;
        }
        this.elements.resultsSummary?.classList.remove('hidden');
    }

    exportResults() {
        if (this.state.analyzedArticles.length === 0) {
            this.showError('No analyzed articles to export');
            return;
        }

        try {
            const dataToExport = {
                export_info: {
                    exported_at: new Date().toISOString(),
                    total_articles: this.state.analyzedArticles.length,
                    analysis_type: this.elements.analysisType?.value || 'both'
                },
                articles: this.state.analyzedArticles
            };

            const blob = new Blob([JSON.stringify(dataToExport, null, 2)], {
                type: 'application/json'
            });

            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `rss-analysis-results-${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.showToast('Results exported successfully', 'success');

        } catch (error) {
            this.showError(`Export failed: ${error.message}`);
        }
    }

    showLoading(title, message) {
        this.state.isLoading = true;
        if (this.elements.loadingTitle) this.elements.loadingTitle.textContent = title;
        if (this.elements.loadingMessage) this.elements.loadingMessage.textContent = message;
        this.elements.loadingIndicator?.classList.remove('hidden');
        this.updateProgress(0);
    }

    hideLoading() {
        this.state.isLoading = false;
        this.elements.loadingIndicator?.classList.add('hidden');
    }

    updateProgress(percentage) {
        if (this.elements.progressBar) {
            this.elements.progressBar.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
        }
    }

    showError(message) {
        if (this.elements.errorMessage) this.elements.errorMessage.textContent = message;
        this.elements.errorDisplay?.classList.remove('hidden');
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            this.elements.errorDisplay?.classList.add('hidden');
        }, 10000);
    }

    showModal(title, content) {
        if (this.elements.modalTitle) this.elements.modalTitle.textContent = title;
        if (this.elements.modalContent) this.elements.modalContent.innerHTML = content;
        this.elements.articleModal?.classList.remove('hidden');
    }

    closeModal() {
        this.elements.articleModal?.classList.add('hidden');
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        const bgClass = type === 'success' ? 'bg-green-100 border-green-400 text-green-700' : 
                       type === 'error' ? 'bg-red-100 border-red-400 text-red-700' : 
                       'bg-blue-100 border-blue-400 text-blue-700';
        
        toast.className = `fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg border ${bgClass} transform translate-x-full opacity-0 transition-all duration-300`;
        toast.innerHTML = `
            <div class="flex items-center">
                <span class="text-sm font-medium">${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="ml-3 text-gray-500 hover:text-gray-700">
                    <i class="bi bi-x"></i>
                </button>
            </div>
        `;
        
        document.body.appendChild(toast);
        
        // Trigger animation
        setTimeout(() => {
            toast.classList.remove('translate-x-full', 'opacity-0');
            toast.classList.add('translate-x-0', 'opacity-100');
        }, 10);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.classList.add('translate-x-full', 'opacity-0');
                setTimeout(() => toast.remove(), 300);
            }
        }, 5000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the RSS Feed Analyzer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.rssAnalyzer = new RSSFeedAnalyzer();
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RSSFeedAnalyzer;
}
