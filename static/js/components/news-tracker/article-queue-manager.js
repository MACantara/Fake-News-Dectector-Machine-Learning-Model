/**
 * Article Queue Management Mixin
 * Handles article verification, batch operations, and queue rendering
 */

export const ArticleQueueManagerMixin = {
    /**
     * Bind article queue events
     */
    bindArticleQueueEvents() {
        // Article fetching
        document.getElementById('fetchNowBtn').addEventListener('click', () => this.fetchAllArticles());
        document.getElementById('clearQueueBtn').addEventListener('click', () => this.clearQueue());
        
        // Batch selection
        document.getElementById('selectAllBtn').addEventListener('click', () => this.selectAllArticles());
        document.getElementById('clearSelectionBtn').addEventListener('click', () => this.clearBatchSelection());
        document.getElementById('batchMarkNewsBtn').addEventListener('click', () => this.batchVerifyArticles(true));
        document.getElementById('batchMarkNotNewsBtn').addEventListener('click', () => this.batchVerifyArticles(false));
        
        // Batch size dropdown
        const batchSizeSelect = document.getElementById('batchSize');
        if (batchSizeSelect) {
            batchSizeSelect.addEventListener('change', (e) => this.updateBatchSize(parseInt(e.target.value)));
        }
        
        // Error/Success dismiss buttons
        document.getElementById('dismissErrorBtn')?.addEventListener('click', () => this.hideError());
        document.getElementById('dismissSuccessBtn')?.addEventListener('click', () => this.hideSuccess());
    },
    
    /**
     * Fetch articles from all tracked websites
     */
    async fetchAllArticles() {
        // Check if fetch is already in progress
        if (this.isFetchInProgress) {
            console.log('Fetch already in progress, skipping duplicate request');
            this.showWarning('Article fetch is already in progress, please wait...');
            return;
        }
        
        if (this.trackedWebsites.length === 0) {
            this.showError('No websites are being tracked');
            return;
        }
        
        // Set fetch lock
        this.isFetchInProgress = true;
        this.isPerformingOperation = true;
        this.updateFetchStatusIndicators(); // Update UI to show fetch in progress
        this.showLoading('Fetching articles from all websites...');
        this.showRealtimeActivity('Fetching articles from tracked websites...');
        
        try {
            const response = await fetch('/api/news-tracker/fetch-articles', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.updateRealtimeActivity('Processing fetched articles...');
                
                // Add new articles to queue
                const initialCount = this.articleQueue.length;
                this.articleQueue = [...this.articleQueue, ...data.articles];
                const newArticleCount = data.articles.length;
                
                // Update UI immediately
                this.renderArticleQueue();
                this.updateCounts();
                this.updateHighConfidenceQueue();
                this.updateStatistics();
                this.updateLastFetchInfo();
                
                if (newArticleCount > 0) {
                    this.showSuccess(`Found ${newArticleCount} new articles`);
                    
                    // Auto-index high confidence articles if enabled
                    if (this.autoIndexEnabled) {
                        this.updateRealtimeActivity('Auto-indexing high confidence articles...');
                        console.log('Manual fetch completed, triggering auto-indexing...');
                        
                        try {
                            const autoIndexResponse = await fetch('/api/news-tracker/auto-index-high-confidence', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    confidence_threshold: this.autoIndexThreshold / 100, // Convert percentage to decimal
                                    batch_size: this.autoIndexBatchSize
                                })
                            });
                            
                            const autoIndexResult = await autoIndexResponse.json();
                            
                            if (autoIndexResult.success && autoIndexResult.articles_indexed > 0) {
                                console.log(`Auto-indexing after manual fetch: ${autoIndexResult.articles_indexed} articles indexed`);
                                
                                // Update auto-indexing stats
                                this.autoIndexStats.totalIndexed += autoIndexResult.articles_indexed;
                                this.autoIndexStats.lastBatchSize = autoIndexResult.articles_indexed;
                                this.autoIndexStats.lastBatchTime = new Date().toISOString();
                                this.autoIndexStats.successRate = autoIndexResult.stats?.success_rate || 0;
                                this.updateAutoIndexStatsDisplay();
                                
                                // Refresh UI to show updated verification status
                                await this.loadTrackedWebsites();
                                this.updateHighConfidenceQueue();
                                this.updateStatistics();
                                
                                this.updateRealtimeActivity(`Completed: ${newArticleCount} articles fetched, ${autoIndexResult.articles_indexed} auto-indexed`);
                                setTimeout(() => this.hideRealtimeActivity(), 3000);
                                
                            } else {
                                console.log('Auto-indexing after manual fetch: No high confidence articles to index');
                                setTimeout(() => this.hideRealtimeActivity(), 2000);
                            }
                        } catch (autoIndexError) {
                            console.error('Auto-indexing error after manual fetch:', autoIndexError);
                            setTimeout(() => this.hideRealtimeActivity(), 1000);
                        }
                    } else {
                        // Hide activity notification if auto-indexing is disabled
                        setTimeout(() => this.hideRealtimeActivity(), 2000);
                    }
                    
                } else {
                    this.showSuccess('No new articles found');
                    this.hideRealtimeActivity();
                }
            } else {
                this.hideRealtimeActivity();
                this.showError(data.error || 'Failed to fetch articles');
            }
        } catch (error) {
            console.error('Error fetching articles:', error);
            this.hideRealtimeActivity();
            this.showError('Network error. Please try again.');
        } finally {
            this.isFetchInProgress = false; // Release fetch lock
            this.isPerformingOperation = false;
            this.updateFetchStatusIndicators(); // Update UI to show fetch completed
            this.hideLoading();
        }
    },
    
    /**
     * Clear article queue
     */
    async clearQueue() {
        if (!confirm('Are you sure you want to clear the entire article queue? This action cannot be undone.')) {
            return;
        }
        
        try {
            // Clear the local queue data (no backend endpoint needed)
            this.articleQueue = [];
            this.selectedArticles.clear();
            this.renderArticleQueue();
            this.updateCounts();
            this.updateSelectionCount();
            this.updateBatchActionButtons();
            this.updateDisplayedArticleCount();
            
            this.showSuccess('Article queue cleared successfully');
            
        } catch (error) {
            console.error('Error clearing queue:', error);
            this.showError('Failed to clear queue. Please try again.');
        }
    },
    
    /**
     * Verify single article
     */
    async verifyArticle(articleId, isNews) {
        try {
            const response = await fetch('/api/news-tracker/verify-article', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    articleId: articleId,
                    isNews: isNews,
                    url: this.articleQueue.find(a => a.id === articleId)?.url
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Update article in queue
                const article = this.articleQueue.find(a => a.id === articleId);
                if (article) {
                    article.verified = true;
                    article.isNews = isNews;
                    article.verifiedAt = new Date().toISOString();
                }
                
                // Real-time UI updates
                this.renderArticleQueue();
                this.updateCounts();
                this.updateStatistics();
                this.updateHighConfidenceQueue();
                
                this.showSuccess(`Article marked as ${isNews ? 'news' : 'not news'}`);
            } else {
                this.showError(data.error || 'Failed to verify article');
            }
        } catch (error) {
            console.error('Error verifying article:', error);
            this.showError('Network error. Please try again.');
        }
    },
    
    /**
     * Batch verify articles
     */
    async batchVerifyArticles(isNews) {
        const selectedElements = document.querySelectorAll('[data-batch-selected="true"]');
        
        if (selectedElements.length === 0) {
            this.showError('No articles selected for batch verification');
            return;
        }
        
        // Show progress indicator
        document.getElementById('batchProgress').classList.remove('hidden');
        
        // Disable batch action buttons during processing
        document.getElementById('batchMarkNewsBtn').disabled = true;
        document.getElementById('batchMarkNotNewsBtn').disabled = true;
        
        // Prepare batch data
        const articles = Array.from(selectedElements).map(element => {
            const articleId = element.dataset.articleId;
            const article = this.articleQueue.find(a => a.id == articleId);
            return {
                articleId: parseInt(articleId),
                isNews: isNews,
                url: article ? article.url : null
            };
        });
        
        try {
            const response = await fetch('/api/news-tracker/batch-verify-articles', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ articles })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Update article queue with verification results
                data.results.forEach(result => {
                    if (result.success) {
                        const article = this.articleQueue.find(a => a.id === result.articleId);
                        if (article) {
                            article.verified = true;
                            article.isNews = isNews;
                            article.verifiedAt = new Date().toISOString();
                        }
                    }
                });
                
                // Show success message with summary
                const summary = data.summary;
                const statusDiv = document.getElementById('batchStatus');
                statusDiv.className = 'mt-3 p-3 rounded-lg bg-green-50 border border-green-200';
                statusDiv.innerHTML = `
                    <div class="flex items-center">
                        <i class="bi bi-check-circle text-green-600 mr-2"></i>
                        <div>
                            <p class="font-semibold text-green-800">Batch verification completed!</p>
                            <p class="text-sm text-green-700">
                                ${summary.successful_verifications}/${summary.total_articles} articles verified as ${isNews ? 'news' : 'not news'}
                                ${summary.news_articles_found > 0 ? `, ${summary.successful_philippine_indexing}/${summary.news_articles_found} indexed in Philippine news system` : ''}
                            </p>
                        </div>
                    </div>
                `;
                statusDiv.classList.remove('hidden');
                
                // Clear selection and refresh display
                this.clearBatchSelection();
                this.renderArticleQueue();
                this.updateCounts();
                this.updateStatistics();
                this.updateHighConfidenceQueue();
                
                // Refresh prediction metrics after successful verification
                this.fetchPredictionMetrics();
                
            } else {
                this.showError(data.error || 'Failed to perform batch verification');
            }
            
        } catch (error) {
            console.error('Batch verification error:', error);
            this.showError('Network error during batch verification');
        } finally {
            // Hide progress indicator and re-enable buttons
            document.getElementById('batchProgress').classList.add('hidden');
            document.getElementById('batchMarkNewsBtn').disabled = false;
            document.getElementById('batchMarkNotNewsBtn').disabled = false;
            
            setTimeout(() => {
                document.getElementById('batchStatus').classList.add('hidden');
            }, 5000);
        }
    },
    
    /**
     * Select all articles
     */
    selectAllArticles() {
        // Get the current batch size from the dropdown
        const batchSizeSelect = document.getElementById('batchSize');
        const maxSelections = batchSizeSelect ? parseInt(batchSizeSelect.value) : 10;
        
        // Find unverified articles that can be selected
        const availableArticles = document.querySelectorAll('[data-batch-selectable="true"]:not([data-batch-selected="true"])');
        
        console.log(`Attempting to select ${maxSelections} articles from ${availableArticles.length} available`);
        
        if (availableArticles.length === 0) {
            this.showNotification('No unverified articles available for selection', 'info');
            return;
        }
        
        let selectedCount = 0;
        for (const element of availableArticles) {
            if (selectedCount >= maxSelections) break;
            
            const articleId = element.dataset.articleId;
            console.log(`Selecting article ${articleId}`);
            this.toggleArticleSelection(articleId, true);
            selectedCount++;
        }
        
        if (availableArticles.length > maxSelections) {
            this.showNotification(`Selected ${selectedCount} articles (maximum ${maxSelections} articles can be selected at once)`, 'info');
        } else {
            this.showSuccess(`Selected ${selectedCount} articles`);
        }
    },
    
    /**
     * Clear batch selection
     */
    clearBatchSelection() {
        this.selectedArticles.clear();
        document.querySelectorAll('[data-batch-selected="true"]').forEach(element => {
            element.dataset.batchSelected = 'false';
            element.classList.remove('bg-blue-50', 'border-blue-300');
            element.classList.add('bg-white', 'border-gray-200');
        });
        
        this.updateSelectionCount();
        this.updateBatchActionButtons();
    },
    
    /**
     * Toggle article selection
     */
    toggleArticleSelection(articleId, forceSelect = null) {
        const element = document.querySelector(`[data-article-id="${articleId}"]`);
        if (!element) return;
        
        const isCurrentlySelected = element.dataset.batchSelected === 'true';
        const shouldSelect = forceSelect !== null ? forceSelect : !isCurrentlySelected;
        
        if (shouldSelect && !isCurrentlySelected) {
            // Check selection limit using dynamic batch size
            const batchSizeSelect = document.getElementById('batchSize');
            const maxSelections = batchSizeSelect ? parseInt(batchSizeSelect.value) : 10;
            
            if (this.selectedArticles.size >= maxSelections) {
                this.showError(`Maximum ${maxSelections} articles can be selected at once`);
                return;
            }
            
            this.selectedArticles.add(parseInt(articleId));
            element.dataset.batchSelected = 'true';
            element.classList.remove('bg-white', 'border-gray-200');
            element.classList.add('bg-blue-50', 'border-blue-300');
        } else if (!shouldSelect && isCurrentlySelected) {
            this.selectedArticles.delete(parseInt(articleId));
            element.dataset.batchSelected = 'false';
            element.classList.remove('bg-blue-50', 'border-blue-300');
            element.classList.add('bg-white', 'border-gray-200');
        }
        
        this.updateSelectionCount();
        this.updateBatchActionButtons();
    },
    
    /**
     * Render article queue
     */
    renderArticleQueue() {
        const container = document.getElementById('articleQueue');
        if (!container) return;
        
        const filteredArticles = this.getFilteredArticles();
        const totalPages = Math.ceil(filteredArticles.length / this.itemsPerPage);
        const startIndex = (this.currentPage - 1) * this.itemsPerPage;
        const endIndex = Math.min(startIndex + this.itemsPerPage, filteredArticles.length);
        const pageArticles = filteredArticles.slice(startIndex, endIndex);
        
        if (filteredArticles.length === 0) {
            container.innerHTML = `
                <div id="emptyQueueMessage" class="text-center text-gray-500 py-8">
                    <i class="bi bi-inbox text-4xl mb-2 opacity-50"></i>
                    <p>No articles in queue yet.</p>
                    <p class="text-sm">Articles will appear here when fetched from tracked websites.</p>
                </div>
            `;
            this.hidePagination();
            return;
        }
        
        // Sort articles by confidence (lowest first) while keeping unverified articles first
        const sortedArticles = pageArticles.sort((a, b) => {
            // First, prioritize unverified articles
            if (!a.verified && b.verified) return -1;
            if (a.verified && !b.verified) return 1;
            
            // Within same verification status, sort by confidence (lowest first)
            const confidenceA = a.confidence || 0;
            const confidenceB = b.confidence || 0;
            return confidenceA - confidenceB;
        });
        
        container.innerHTML = sortedArticles.map(article => this.renderArticleItem(article)).join('');
        
        this.updatePagination(totalPages);
        this.updateDisplayedArticleCount();
    },
    
    /**
     * Render individual article item
     */
    renderArticleItem(article) {
        const isSelected = this.selectedArticles.has(article.id);
        const canSelect = !article.verified;
        const selectionClass = isSelected ? 'bg-blue-50 border-blue-300' : 'bg-white border-gray-200';
        
        return `
            <div class="article-item border rounded-lg p-4 mb-4 hover:shadow-md transition-shadow ${selectionClass} break-words"
                 data-article-id="${article.id}"
                 data-batch-selected="${isSelected}"
                 data-batch-selectable="${canSelect}"
                 ${canSelect ? `onclick="newsTracker.toggleArticleSelection(${article.id})"` : ''}
                 ${canSelect ? 'style="cursor: pointer;"' : ''}>
                
                <div class="flex items-start justify-between">
                    <div class="flex-1 min-w-0 pr-4">
                        ${canSelect ? `
                            <div class="flex items-center mb-2">
                                <input type="checkbox" 
                                       class="mr-3 flex-shrink-0" 
                                       ${isSelected ? 'checked' : ''}
                                       onclick="event.stopPropagation(); newsTracker.toggleArticleSelection(${article.id})">
                                <span class="text-xs text-blue-600 font-medium">
                                    ${isSelected ? 'Selected for batch verification' : 'Click to select'}
                                </span>
                            </div>
                        ` : ''}
                        
                        <h4 class="font-semibold text-gray-800 mb-2 break-words hyphens-auto">
                            ${article.title || 'No title available'}
                        </h4>
                        
                        <div class="flex items-center text-sm text-gray-600 mb-2 min-w-0">
                            <i class="bi bi-link-45deg mr-1 flex-shrink-0"></i>
                            <a href="${article.url}" target="_blank" class="text-blue-600 hover:text-blue-800 break-all">
                                ${article.url}
                            </a>
                        </div>
                        
                        <div class="flex flex-wrap items-center text-xs text-gray-500 gap-x-4 gap-y-1 mb-3">
                            <span class="flex items-center whitespace-nowrap">
                                <i class="bi bi-calendar mr-1"></i>
                                Found: ${this.formatDate(article.foundAt)}
                            </span>
                            ${article.verifiedAt ? `
                                <span class="flex items-center whitespace-nowrap">
                                    <i class="bi bi-check-circle mr-1"></i>
                                    Verified: ${this.formatDate(article.verifiedAt)}
                                </span>
                            ` : ''}
                        </div>
                        
                        ${this.renderPredictionMetrics(article)}
                    </div>
                    
                    <div class="flex flex-col items-end space-y-2 ml-4 flex-shrink-0">
                        ${this.renderVerificationBadge(article)}
                        ${!article.verified ? `
                            <div class="flex flex-col sm:flex-row gap-2">
                                <button 
                                    class="bg-green-600 text-white px-3 py-1 rounded text-sm hover:bg-green-700 transition-colors whitespace-nowrap"
                                    onclick="event.stopPropagation(); newsTracker.verifyArticle(${article.id}, true)"
                                    title="Mark as news"
                                >
                                    <i class="bi bi-check mr-1"></i>News
                                </button>
                                <button 
                                    class="bg-red-600 text-white px-3 py-1 rounded text-sm hover:bg-red-700 transition-colors whitespace-nowrap"
                                    onclick="event.stopPropagation(); newsTracker.verifyArticle(${article.id}, false)"
                                    title="Mark as not news"
                                >
                                    <i class="bi bi-x mr-1"></i>Not News
                                </button>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    },
    
    /**
     * Render verification badge
     */
    renderVerificationBadge(article) {
        if (!article.verified) {
            return '<span class="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-medium">Pending</span>';
        }
        
        if (article.isNews) {
            return '<span class="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium">Verified News</span>';
        } else {
            return '<span class="bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs font-medium">Not News</span>';
        }
    },
    
    /**
     * Render prediction metrics
     */
    renderPredictionMetrics(article) {
        if (!article.confidence && !article.probability_news) return '';
        
        const confidence = article.confidence || 0;
        const probability = article.probability_news || 0;
        const prediction = article.is_news_prediction;
        
        return `
            <div class="bg-gray-50 rounded-lg p-3 mb-3">
                <h5 class="text-sm font-medium text-gray-700 mb-2">
                    <i class="bi bi-graph-up mr-1"></i>
                    ML Prediction Metrics
                </h5>
                <div class="grid grid-cols-2 gap-3 text-xs">
                    <div>
                        <span class="text-gray-600">Confidence:</span>
                        <div class="mt-1">
                            <div class="bg-gray-200 rounded-full h-2">
                                <div class="bg-blue-600 h-2 rounded-full" style="width: ${confidence * 100}%"></div>
                            </div>
                            <span class="text-gray-700 font-medium">${(confidence * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                    <div>
                        <span class="text-gray-600">News Probability:</span>
                        <div class="mt-1">
                            <div class="bg-gray-200 rounded-full h-2">
                                <div class="bg-green-600 h-2 rounded-full" style="width: ${probability * 100}%"></div>
                            </div>
                            <span class="text-gray-700 font-medium">${(probability * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                </div>
                <div class="mt-2 flex items-center">
                    <span class="text-gray-600 text-xs mr-2">Prediction:</span>
                    <span class="text-xs px-2 py-1 rounded-full ${prediction ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                        <i class="bi bi-${prediction ? 'newspaper' : 'x-circle'} mr-1"></i>
                        ${prediction ? 'News' : 'Not News'}
                    </span>
                </div>
            </div>
        `;
    },
    
    /**
     * Update pagination
     */
    updatePagination(totalPages) {
        const paginationEl = document.getElementById('queuePagination');
        const prevBtn = document.getElementById('prevPageBtn');
        const nextBtn = document.getElementById('nextPageBtn');
        const pageInfo = document.getElementById('pageInfo');
        
        if (totalPages <= 1) {
            paginationEl?.classList.add('hidden');
            return;
        }
        
        paginationEl?.classList.remove('hidden');
        
        if (prevBtn) {
            prevBtn.disabled = this.currentPage <= 1;
        }
        
        if (nextBtn) {
            nextBtn.disabled = this.currentPage >= totalPages;
        }
        
        if (pageInfo) {
            pageInfo.textContent = `Page ${this.currentPage} of ${totalPages}`;
        }
    },
    
    /**
     * Hide pagination
     */
    hidePagination() {
        document.getElementById('queuePagination')?.classList.add('hidden');
    },
    
    /**
     * Update last fetch info
     */
    updateLastFetchInfo() {
        const lastFetchInfo = document.getElementById('lastFetchInfo');
        if (lastFetchInfo) {
            const now = new Date();
            const fetchTime = now.toLocaleString();
            lastFetchInfo.innerHTML = `
                <div class="flex items-center justify-between">
                    <div class="flex items-center">
                        <i class="bi bi-info-circle text-blue-600 mr-2"></i>
                        <span class="text-sm text-blue-800">Last fetch: ${fetchTime}</span>
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="text-xs text-blue-600">
                            <i class="bi bi-check-circle mr-1"></i>
                            Completed successfully
                        </span>
                    </div>
                </div>
            `;
        }
    },
    
    /**
     * Update batch size for article selection and display
     */
    updateBatchSize(batchSize) {
        // Update the displayed batch size in the button
        const selectedBatchSizeSpan = document.getElementById('selectedBatchSize');
        if (selectedBatchSizeSpan) {
            selectedBatchSizeSpan.textContent = batchSize.toString();
        }
        
        // Update the items per page for display
        this.itemsPerPage = batchSize;
        this.currentPage = 1; // Reset to first page when changing display count
        
        // Update the displayed article count info
        const displayedArticleCountSpan = document.getElementById('displayedArticleCount');
        if (displayedArticleCountSpan) {
            displayedArticleCountSpan.textContent = batchSize.toString();
        }
        
        // Clear current selection when batch size changes
        this.clearBatchSelection();
        
        // Re-render the queue with new pagination
        this.renderArticleQueue();
        
        // Store the batch size for future use
        this.currentBatchSize = batchSize;
        
        console.log(`Batch size updated to: ${batchSize}, items per page: ${this.itemsPerPage}`);
    }
};
