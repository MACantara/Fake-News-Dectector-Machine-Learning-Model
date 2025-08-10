// Philippine News Search functionality
class PhilippineNewsSearch {
    constructor() {
        // Initialize application state
        this.state = {
            isLoading: false,
            currentQuery: '',
            results: [],
            totalResults: 0
        };

        // DOM elements cache
        this.elements = {};

        // Initialize the search functionality
        this.init();
    }

    // Initialize the search functionality
    init() {
        this.cacheElements();
        this.bindEvents();
        this.checkModelStatus();
        
        console.log('Philippine News Search initialized successfully');
    }

    // Cache DOM elements for performance
    cacheElements() {
        const elementIds = [
            'searchQuery', 'searchCategory', 'searchSource', 'searchLimit',
            'performSearchBtn', 'viewAnalyticsBtn', 'addToIndexBtn', 'addArticleModal',
            'indexArticleUrl', 'submitIndexBtn', 'cancelIndexBtn',
            'loading', 'searchResults', 'analyticsResults', 'error', 'modelStatus',
            'searchSummary', 'searchSummaryText', 'searchResponseTime', 'searchResultsGrid',
            'loadMoreSection', 'loadMoreBtn', 'totalArticlesCount', 'totalSourcesCount', 
            'avgRelevanceScore', 'totalCategoriesCount', 'topSourcesChart', 'categoriesChart', 
            'recentActivityChart', 'topQueriesTable', 'topQueriesBody', 'errorMessage'
        ];

        elementIds.forEach(id => {
            this.elements[id] = document.getElementById(id);
        });
    }

    // Bind event listeners
    bindEvents() {
        // Search button
        if (this.elements.performSearchBtn) {
            this.elements.performSearchBtn.addEventListener('click', () => this.performPhilippineSearch());
        }

        // Analytics button
        if (this.elements.viewAnalyticsBtn) {
            this.elements.viewAnalyticsBtn.addEventListener('click', () => this.viewPhilippineAnalytics());
        }

        // Add to index button
        if (this.elements.addToIndexBtn) {
            this.elements.addToIndexBtn.addEventListener('click', () => this.showAddToIndexModal());
        }

        // Index article buttons
        if (this.elements.submitIndexBtn) {
            this.elements.submitIndexBtn.addEventListener('click', () => this.submitIndexArticle());
        }
        if (this.elements.cancelIndexBtn) {
            this.elements.cancelIndexBtn.addEventListener('click', () => this.hideAddToIndexModal());
        }

        // Search input monitoring
        if (this.elements.searchQuery) {
            this.elements.searchQuery.addEventListener('input', () => this.updateSearchButton());
            this.elements.searchQuery.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !this.elements.performSearchBtn.disabled) {
                    this.performPhilippineSearch();
                }
            });
        }

        // Load more button
        if (this.elements.loadMoreBtn) {
            this.elements.loadMoreBtn.addEventListener('click', () => this.loadMoreResults());
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                if (!this.elements.performSearchBtn.disabled) {
                    this.performPhilippineSearch();
                }
            }
        });
    }

    // Check model status
    async checkModelStatus() {
        if (!this.elements.modelStatus) return;

        try {
            const response = await Utils.http.get(Config.endpoints.modelStatus);
            
            if (response.success && response.status) {
                const status = response.status;
                const isReady = status.fake_news && status.political;
                
                if (isReady) {
                    this.elements.modelStatus.innerHTML = `
                        <div class="flex items-center justify-center">
                            <i class="bi bi-check-circle text-green-600 mr-2"></i>
                            <span class="text-green-700">Models ready</span>
                        </div>
                    `;
                    this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center border-l-4 border-green-500';
                } else {
                    this.elements.modelStatus.innerHTML = `
                        <div class="flex items-center justify-center">
                            <i class="bi bi-exclamation-triangle text-yellow-600 mr-2"></i>
                            <span class="text-yellow-700">Some models not ready</span>
                        </div>
                    `;
                    this.elements.modelStatus.className = 'glass-effect rounded-xl p-4 mb-8 text-center status-warning';
                }
            }
        } catch (error) {
            console.error('Model status check failed:', error);
        }
    }

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
        this.state.currentQuery = query;
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
                this.state.results = response.data.results || [];
                this.state.totalResults = response.data.total_count || 0;
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
            this.elements.searchResponseTime.textContent = `${data.response_time.toFixed(0)}ms`;
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
                            onclick="window.open('/analyze-url?url=${encodeURIComponent(article.url)}', '_blank')">
                        <i class="bi bi-search mr-1"></i>Analyze
                    </button>
                    <button class="text-gray-600 hover:text-gray-800 text-sm font-medium"
                            onclick="philippineSearch.viewSearchResultDetails(${article.id})">
                        <i class="bi bi-eye mr-1"></i>View Details
                    </button>
                </div>
            </div>
        `;

        this.elements.searchResultsGrid.insertAdjacentHTML('beforeend', cardHtml);
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
        modal.className = 'fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4';
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
                                <button onclick="window.open('/analyze-url?url=${encodeURIComponent(article.url)}', '_blank'); this.closest('.fixed').remove();" 
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
                this.showSuccess('Article indexed successfully!');
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

    // Load more results
    async loadMoreResults() {
        // Implementation for pagination if needed
        console.log('Load more results not yet implemented');
    }

    // Hide all sections
    hideAllSections() {
        Utils.dom.hide(this.elements.searchResults);
        Utils.dom.hide(this.elements.analyticsResults);
        Utils.dom.hide(this.elements.error);
    }

    // Show error message
    showError(message) {
        if (this.elements.errorMessage) {
            Utils.dom.setText(this.elements.errorMessage, message);
            Utils.dom.show(this.elements.error);
            Utils.animations.shake(this.elements.error);
        }
    }

    // Show success message
    showSuccess(message) {
        // Create temporary success message
        const successDiv = document.createElement('div');
        successDiv.className = 'fixed top-4 right-4 bg-green-50 border border-green-200 rounded-lg p-4 z-50';
        successDiv.innerHTML = `
            <div class="flex items-center">
                <i class="bi bi-check-circle text-green-500 text-xl mr-3"></i>
                <div class="flex-1">
                    <h4 class="text-green-800 font-semibold">Success</h4>
                    <p class="text-green-700">${message}</p>
                </div>
            </div>
        `;
        
        document.body.appendChild(successDiv);
        
        // Remove after 3 seconds
        setTimeout(() => {
            successDiv.remove();
        }, 3000);
    }
}

// Initialize the Philippine News Search when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.philippineSearch = new PhilippineNewsSearch();
});

// Export for testing purposes
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PhilippineNewsSearch;
}
