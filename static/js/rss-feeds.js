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

        feeds.forEach((feed) => {
            const feedCard = document.createElement('div');
            feedCard.className = `feed-card p-4 border border-gray-200 rounded-lg cursor-pointer transition-all duration-200 hover:shadow-md ${feed.active ? 'bg-blue-50 border-blue-300' : 'bg-white'}`;
            feedCard.setAttribute('data-feed-id', feed.id);

            feedCard.innerHTML = `
                <div class="flex items-center justify-between mb-2">
                    <h3 class="font-semibold text-gray-800">${feed.name}</h3>
                    <div class="flex items-center space-x-2">
                        <input type="checkbox" class="feed-checkbox w-5 h-5 text-blue-600 rounded focus:ring-blue-500" 
                               ${feed.active ? 'checked' : ''} data-feed-id="${feed.id}">
                        <button class="edit-feed-btn text-gray-500 hover:text-gray-700" data-feed-id="${feed.id}">
                            <i class="bi bi-pencil-square"></i>
                        </button>
                    </div>
                </div>
                <p class="text-sm text-gray-600 mb-2">
                    <span class="inline-block bg-gray-100 text-gray-700 px-2 py-1 rounded-full text-xs">${feed.category}</span>
                </p>
                <p class="text-xs text-gray-500 mb-2">${feed.description || 'No description'}</p>
                <p class="text-xs text-gray-400 break-all mb-3">${feed.url}</p>
                <div class="flex items-center justify-between">
                    <button class="test-feed-btn text-blue-600 hover:text-blue-800 text-sm font-medium" 
                            data-feed-id="${feed.id}" data-feed-url="${feed.url}">
                        <i class="bi bi-play-circle mr-1"></i>Test Feed
                    </button>
                    <div class="flex items-center space-x-2">
                        <span class="text-xs px-2 py-1 rounded-full ${feed.active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'}">
                            ${feed.active ? 'Active' : 'Inactive'}
                        </span>
                        ${feed.last_fetched ? `<span class="text-xs text-gray-400">Last: ${new Date(feed.last_fetched).toLocaleDateString()}</span>` : ''}
                    </div>
                </div>
                ${feed.error_count > 0 ? `
                    <div class="mt-2 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
                        <i class="bi bi-exclamation-triangle mr-1"></i>
                        ${feed.error_count} error(s). Last: ${feed.last_error || 'Unknown error'}
                    </div>
                ` : ''}
            `;

            // Bind events for this card
            const checkbox = feedCard.querySelector('.feed-checkbox');
            checkbox.addEventListener('change', (e) => this.toggleFeedSelection(feed.id, e.target.checked));

            const testBtn = feedCard.querySelector('.test-feed-btn');
            testBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.testSingleFeed(feed.id, feed.url);
            });

            const editBtn = feedCard.querySelector('.edit-feed-btn');
            editBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.editFeed(feed);
            });

            this.elements.predefinedFeeds.appendChild(feedCard);
            
            // Initialize selected feeds for active feeds
            if (feed.active) {
                this.state.selectedFeeds.add(feed.id);
            }
        });

        // Update feed statistics
        this.updateFeedStats(feeds);
        
        // Update fetch button state after initializing selected feeds
        this.updateFetchButtonState();
    }

    updateFeedStats(feeds) {
        const totalFeeds = feeds.length;
        const activeFeeds = feeds.filter(feed => feed.active).length;
        const recentlyFetched = feeds.filter(feed => feed.last_fetched).length;
        
        // Add stats display if it doesn't exist
        let statsDiv = document.getElementById('feedStats');
        if (!statsDiv) {
            statsDiv = document.createElement('div');
            statsDiv.id = 'feedStats';
            statsDiv.className = 'mb-4 p-3 bg-gray-50 rounded-lg';
            this.elements.predefinedFeeds.parentNode.insertBefore(statsDiv, this.elements.predefinedFeeds);
        }
        
        statsDiv.innerHTML = `
            <div class="grid grid-cols-3 gap-4 text-center">
                <div>
                    <div class="text-2xl font-bold text-blue-600">${totalFeeds}</div>
                    <div class="text-sm text-gray-600">Total Feeds</div>
                </div>
                <div>
                    <div class="text-2xl font-bold text-green-600">${activeFeeds}</div>
                    <div class="text-sm text-gray-600">Active Feeds</div>
                </div>
                <div>
                    <div class="text-2xl font-bold text-purple-600">${recentlyFetched}</div>
                    <div class="text-sm text-gray-600">Recently Fetched</div>
                </div>
            </div>
        `;
    }

    toggleFeedSelection(feedId, selected) {
        if (selected) {
            this.state.selectedFeeds.add(feedId);
        } else {
            this.state.selectedFeeds.delete(feedId);
        }

        // Update UI
        const feedCard = document.querySelector(`[data-feed-id="${feedId}"]`);
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

    editFeed(feed) {
        // Create a simple modal for editing feed
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white p-6 rounded-lg max-w-md w-full mx-4">
                <h3 class="text-lg font-semibold mb-4">Edit RSS Feed</h3>
                <form id="editFeedForm">
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Name</label>
                        <input type="text" id="editFeedName" class="w-full p-2 border border-gray-300 rounded-md" value="${feed.name}">
                    </div>
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">URL</label>
                        <input type="url" id="editFeedUrl" class="w-full p-2 border border-gray-300 rounded-md" value="${feed.url}">
                    </div>
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Category</label>
                        <input type="text" id="editFeedCategory" class="w-full p-2 border border-gray-300 rounded-md" value="${feed.category}">
                    </div>
                    <div class="mb-4">
                        <label class="block text-sm font-medium text-gray-700 mb-2">Description</label>
                        <textarea id="editFeedDescription" class="w-full p-2 border border-gray-300 rounded-md" rows="3">${feed.description || ''}</textarea>
                    </div>
                    <div class="mb-4">
                        <label class="flex items-center">
                            <input type="checkbox" id="editFeedActive" class="mr-2" ${feed.active ? 'checked' : ''}>
                            <span class="text-sm font-medium text-gray-700">Active</span>
                        </label>
                    </div>
                    <div class="flex justify-end space-x-3">
                        <button type="button" id="cancelEdit" class="px-4 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50">Cancel</button>
                        <button type="button" id="deleteFeed" class="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700">Delete</button>
                        <button type="submit" class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">Save</button>
                    </div>
                </form>
            </div>
        `;

        document.body.appendChild(modal);

        // Bind events
        modal.querySelector('#cancelEdit').addEventListener('click', () => {
            document.body.removeChild(modal);
        });

        modal.querySelector('#deleteFeed').addEventListener('click', async () => {
            if (confirm('Are you sure you want to delete this RSS feed?')) {
                await this.deleteFeed(feed.id);
                document.body.removeChild(modal);
            }
        });

        modal.querySelector('#editFeedForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.updateFeed(feed.id, {
                name: modal.querySelector('#editFeedName').value,
                url: modal.querySelector('#editFeedUrl').value,
                category: modal.querySelector('#editFeedCategory').value,
                description: modal.querySelector('#editFeedDescription').value,
                active: modal.querySelector('#editFeedActive').checked
            });
            document.body.removeChild(modal);
        });
    }

    async updateFeed(feedId, feedData) {
        try {
            const response = await fetch(`/api/rss-feeds/${feedId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(feedData)
            });

            const data = await response.json();
            if (data.success) {
                this.showSuccess('RSS feed updated successfully');
                this.loadPredefinedFeeds(); // Refresh the feeds
            } else {
                this.showError(`Failed to update feed: ${data.error}`);
            }
        } catch (error) {
            console.error('Error updating feed:', error);
            this.showError('Network error updating feed');
        }
    }

    async deleteFeed(feedId) {
        try {
            const response = await fetch(`/api/rss-feeds/${feedId}`, {
                method: 'DELETE'
            });

            const data = await response.json();
            if (data.success) {
                this.showSuccess('RSS feed deleted successfully');
                this.loadPredefinedFeeds(); // Refresh the feeds
            } else {
                this.showError(`Failed to delete feed: ${data.error}`);
            }
        } catch (error) {
            console.error('Error deleting feed:', error);
            this.showError('Network error deleting feed');
        }
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

        const limit = parseInt(this.elements.articleLimit?.value) || 20;
        const hoursBack = parseInt(this.elements.timeFilter?.value) || 24;

        this.showLoading('Fetching Articles', 'Downloading articles from RSS feeds...');
        this.updateProgress(10);

        try {
            // Use batch fetch for better performance
            const response = await fetch('/api/rss-feed/batch-articles', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    feed_ids: Array.from(this.state.selectedFeeds),
                    limit_per_feed: Math.ceil(limit / this.state.selectedFeeds.size),
                    hours_back: hoursBack
                })
            });

            this.updateProgress(60);
            const data = await response.json();

            if (data.success) {
                // Store performance metrics
                this.state.performanceMetrics = {
                    total_processing_time: data.performance.total_processing_time,
                    feed_sources: data.performance.feed_sources,
                    articles_per_second: data.performance.articles_per_second,
                    successful_feeds: data.performance.successful_feeds,
                    failed_feeds: data.performance.failed_feeds
                };

                // Remove duplicates and limit total articles
                let articles = this.removeDuplicateArticles(data.articles);
                if (limit && articles.length > limit) {
                    articles = articles.slice(0, limit);
                }

                this.state.fetchedArticles = articles;
                this.updateProgress(100);
                
                setTimeout(() => {
                    this.hideLoading();
                    this.displayFetchedArticles();
                    this.showSuccess(`Successfully fetched ${articles.length} articles from ${this.state.selectedFeeds.size} RSS feeds`);
                }, 500);
            } else {
                this.hideLoading();
                this.showError(`Failed to fetch articles: ${data.error}`);
            }
        } catch (error) {
            console.error('Error fetching articles:', error);
            this.hideLoading();
            this.showError('Network error while fetching articles');
        }
    }

    async getSelectedFeedUrls() {
        const response = await fetch('/api/rss-feeds');
        const data = await response.json();

        if (!data.success) {
            throw new Error('Failed to get feed data');
        }

        return Array.from(this.state.selectedFeeds).map(feedId => {
            const feed = data.feeds.find(f => f.id === feedId);
            return feed ? {
                id: feed.id,
                name: feed.name,
                url: feed.url,
                category: feed.category
            } : null;
        }).filter(feed => feed !== null);
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
        
        // Update performance metrics if available
        if (this.state.performanceMetrics) {
            const metrics = this.state.performanceMetrics;
            
            if (this.elements.processingTime) {
                this.elements.processingTime.textContent = `${metrics.total_processing_time.toFixed(2)}s`;
            }
            if (this.elements.feedSource) {
                this.elements.feedSource.textContent = metrics.feed_sources;
            }
            if (this.elements.articlesPerSecond) {
                this.elements.articlesPerSecond.textContent = metrics.articles_per_second;
            }
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

    showSuccess(message) {
        this.showToast(message, 'success');
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
