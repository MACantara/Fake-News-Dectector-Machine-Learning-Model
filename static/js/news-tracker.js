/**
 * News Tracker Application
 * Handles website tracking, article fetching, and verification
 */

class NewsTracker {
    constructor() {
        this.trackedWebsites = [];
        this.articleQueue = [];
        this.currentPage = 1;
        this.itemsPerPage = 10;
        this.autoFetchInterval = null;
        this.selectedArticles = new Set(); // Track selected articles
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadTrackedWebsites();
        this.loadArticleQueue();
        this.updateStatistics();
        this.setupAutoFetch();
        
        // Initialize selection interface
        this.updateSelectionCount();
        this.updateBatchActionButtons();
    }
    
    bindEvents() {
        // Website management
        document.getElementById('addWebsiteBtn').addEventListener('click', () => this.addWebsite());
        document.getElementById('websiteUrl').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.addWebsite();
        });
        
        // Auto-fetch controls
        document.getElementById('autoFetchToggle').addEventListener('change', (e) => {
            this.toggleAutoFetch(e.target.checked);
        });
        document.getElementById('fetchNowBtn').addEventListener('click', () => this.fetchAllArticles());
        document.getElementById('clearQueueBtn').addEventListener('click', () => this.clearQueue());
        
        // Queue management
        document.getElementById('queueFilter').addEventListener('change', (e) => {
            this.filterQueue(e.target.value);
        });
        
        // Batch verification controls
        document.getElementById('batchSize').addEventListener('change', (e) => {
            const size = e.target.value;
            document.getElementById('selectedBatchSize').textContent = size;
            this.clearBatchSelection();
        });
        document.getElementById('selectAllBtn').addEventListener('click', () => this.selectBatchArticles());
        document.getElementById('batchMarkNewsBtn').addEventListener('click', () => this.batchVerifyArticles(true));
        document.getElementById('batchMarkNotNewsBtn').addEventListener('click', () => this.batchVerifyArticles(false));
        document.getElementById('clearSelectionBtn').addEventListener('click', () => this.clearBatchSelection());
        
        // Pagination
        document.getElementById('prevPageBtn').addEventListener('click', () => this.previousPage());
        document.getElementById('nextPageBtn').addEventListener('click', () => this.nextPage());
        
        // Error/Success dismissal
        document.getElementById('dismissErrorBtn').addEventListener('click', () => this.hideError());
        document.getElementById('dismissSuccessBtn').addEventListener('click', () => this.hideSuccess());
    }
    
    async addWebsite() {
        const urlInput = document.getElementById('websiteUrl');
        const nameInput = document.getElementById('websiteName');
        const intervalSelect = document.getElementById('fetchInterval');
        
        const url = urlInput.value.trim();
        const name = nameInput.value.trim() || this.extractDomainName(url);
        const interval = parseInt(intervalSelect.value);
        
        if (!url) {
            this.showError('Please enter a website URL');
            return;
        }
        
        if (!this.isValidUrl(url)) {
            this.showError('Please enter a valid URL');
            return;
        }
        
        // Check if already tracking
        if (this.trackedWebsites.find(site => site.url === url)) {
            this.showError('This website is already being tracked');
            return;
        }
        
        this.showLoading('Adding website...');
        
        try {
            const response = await fetch('/api/news-tracker/add-website', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url, name, interval })
            });
            
            const data = await response.json();
            
            if (data.success) {
                const website = {
                    id: data.id,
                    url,
                    name,
                    interval,
                    addedAt: new Date().toISOString(),
                    lastFetch: null,
                    articleCount: 0,
                    status: 'active'
                };
                
                this.trackedWebsites.push(website);
                this.renderTrackedWebsites();
                this.updateCounts();
                
                // Clear inputs
                urlInput.value = '';
                nameInput.value = '';
                intervalSelect.value = '60';
                
                this.showSuccess(`Successfully added ${name} to tracking list`);
            } else {
                this.showError(data.error || 'Failed to add website');
            }
        } catch (error) {
            console.error('Error adding website:', error);
            this.showError('Network error. Please try again.');
        } finally {
            this.hideLoading();
        }
    }
    
    async removeWebsite(websiteId) {
        if (!confirm('Are you sure you want to stop tracking this website?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/news-tracker/remove-website/${websiteId}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.trackedWebsites = this.trackedWebsites.filter(site => site.id !== websiteId);
                this.renderTrackedWebsites();
                this.updateCounts();
                this.showSuccess('Website removed from tracking');
            } else {
                this.showError(data.error || 'Failed to remove website');
            }
        } catch (error) {
            console.error('Error removing website:', error);
            this.showError('Network error. Please try again.');
        }
    }
    
    async fetchAllArticles() {
        if (this.trackedWebsites.length === 0) {
            this.showError('No websites are being tracked');
            return;
        }
        
        this.showLoading('Fetching articles from all websites...');
        
        try {
            const response = await fetch('/api/news-tracker/fetch-articles', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.articleQueue = [...this.articleQueue, ...data.articles];
                this.renderArticleQueue();
                this.updateCounts();
                this.updateLastFetchInfo();
                
                if (data.articles.length > 0) {
                    this.showSuccess(`Found ${data.articles.length} new articles`);
                } else {
                    this.showSuccess('No new articles found');
                }
            } else {
                this.showError(data.error || 'Failed to fetch articles');
            }
        } catch (error) {
            console.error('Error fetching articles:', error);
            this.showError('Network error. Please try again.');
        } finally {
            this.hideLoading();
        }
    }
    
    async verifyArticle(isNews) {
        const currentArticle = this.articleQueue[this.currentArticleIndex];
        if (!currentArticle) return;
        
        try {
            const response = await fetch('/api/news-tracker/verify-article', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    articleId: currentArticle.id,
                    isNews,
                    url: currentArticle.url
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Update article in queue
                currentArticle.verified = true;
                currentArticle.isNews = isNews;
                currentArticle.verifiedAt = new Date().toISOString();
                
                this.showSuccess(`Article marked as ${isNews ? 'news' : 'not news'}`);
                this.nextArticle();
                this.updateCounts();
                this.updateStatistics();
            } else {
                this.showError(data.error || 'Failed to verify article');
            }
        } catch (error) {
            console.error('Error verifying article:', error);
            this.showError('Network error. Please try again.');
        }
    }
    
    skipArticle() {
        this.nextArticle();
    }
    
    nextArticle() {
        const unverifiedArticles = this.articleQueue.filter(article => !article.verified);
        
        if (unverifiedArticles.length === 0) {
            this.showSuccess('All articles have been verified!');
            return;
        }
        
        // Find next unverified article
        let nextIndex = this.currentArticleIndex + 1;
        while (nextIndex < this.articleQueue.length && this.articleQueue[nextIndex].verified) {
            nextIndex++;
        }
        
        if (nextIndex >= this.articleQueue.length) {
            // Loop back to start
            nextIndex = 0;
            while (nextIndex < this.articleQueue.length && this.articleQueue[nextIndex].verified) {
                nextIndex++;
            }
        }
        
        this.currentArticleIndex = nextIndex;
        this.displayCurrentArticle();
        this.updateVerificationProgress();
    }
    
    displayCurrentArticle() {
        const article = this.articleQueue[this.currentArticleIndex];
        if (!article) return;
        
        const currentArticleDiv = document.getElementById('currentArticle');
        currentArticleDiv.innerHTML = `
            <div class="border border-gray-200 rounded-lg p-4 bg-gray-50">
                <div class="flex items-start justify-between mb-3">
                    <div class="flex-1">
                        <h4 class="font-semibold text-gray-800 mb-1">${this.escapeHtml(article.title || 'No Title')}</h4>
                        <p class="text-sm text-blue-600 hover:text-blue-800">
                            <i class="bi bi-link-45deg mr-1"></i>
                            <a href="${article.url}" target="_blank" rel="noopener noreferrer">${article.url}</a>
                        </p>
                    </div>
                    <span class="text-xs text-gray-500 ml-4">
                        ${new Date(article.found_at || article.foundAt).toLocaleString()}
                    </span>
                </div>
                
                ${article.description ? `
                    <div class="mb-3">
                        <p class="text-sm text-gray-700">${this.escapeHtml(article.description)}</p>
                    </div>
                ` : ''}
                
                <div class="flex items-center justify-between text-xs text-gray-500">
                    <span>
                        <i class="bi bi-globe mr-1"></i>
                        ${this.escapeHtml(article.site_name || article.siteName || 'Unknown Source')}
                    </span>
                    <span>
                        Article ${this.currentArticleIndex + 1} of ${this.articleQueue.length}
                    </span>
                </div>
            </div>
        `;
    }
    
    updateVerificationProgress() {
        const verifiedCount = this.articleQueue.filter(article => article.verified).length;
        const totalCount = this.articleQueue.length;
        const percentage = totalCount > 0 ? (verifiedCount / totalCount) * 100 : 0;
        
        document.getElementById('verificationProgress').textContent = `${verifiedCount} / ${totalCount}`;
        document.getElementById('progressBar').style.width = `${percentage}%`;
    }
    
    renderTrackedWebsites() {
        const container = document.getElementById('trackedWebsitesList');
        
        if (this.trackedWebsites.length === 0) {
            container.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <i class="bi bi-inbox text-4xl mb-2 opacity-50"></i>
                    <p>No websites being tracked yet.</p>
                    <p class="text-sm">Add a website above to start tracking articles.</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = this.trackedWebsites.map(website => `
            <div class="flex items-center justify-between p-4 bg-gray-50 rounded-lg border">
                <div class="flex-1">
                    <h4 class="font-medium text-gray-800">${this.escapeHtml(website.name)}</h4>
                    <p class="text-sm text-gray-600">${website.url}</p>
                    <div class="flex items-center mt-1 text-xs text-gray-500">
                        <span class="mr-3">
                            <i class="bi bi-clock mr-1"></i>
                            Every ${website.fetch_interval || website.interval || 60} min
                        </span>
                        <span class="mr-3">
                            <i class="bi bi-newspaper mr-1"></i>
                            ${website.article_count || website.articleCount || 0} articles
                        </span>
                        <span>
                            <i class="bi bi-calendar mr-1"></i>
                            ${website.last_fetch || website.lastFetch ? new Date(website.last_fetch || website.lastFetch).toLocaleString() : 'Never fetched'}
                        </span>
                    </div>
                </div>
                <div class="flex items-center space-x-2">
                    <span class="px-2 py-1 text-xs rounded-full ${website.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                        ${website.status}
                    </span>
                    <button 
                        onclick="newsTracker.removeWebsite('${website.id}')"
                        class="text-red-600 hover:text-red-800 p-1"
                        title="Remove website"
                    >
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
        `).join('');
    }
    
    renderArticleQueue() {
        const container = document.getElementById('articleQueue');
        const  emptyMessage = document.getElementById('emptyQueueMessage');
        
        if (!container) {
            console.error('Article queue container not found');
            return;
        }
        
        if (this.articleQueue.length === 0) {
            if (emptyMessage) {
                emptyMessage.classList.remove('hidden');
            }
            container.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <i class="bi bi-inbox text-4xl mb-2 opacity-50"></i>
                    <p>No articles in queue yet.</p>
                    <p class="text-sm">Articles will appear here when fetched from tracked websites.</p>
                </div>
            `;
            return;
        }
        
        if (emptyMessage) {
            emptyMessage.classList.add('hidden');
        }
        
        // Apply current filter
        const filter = document.getElementById('queueFilter').value;
        const filteredArticles = this.filterArticlesByType(filter);
        
        // Pagination
        const startIndex = (this.currentPage - 1) * this.itemsPerPage;
        const endIndex = startIndex + this.itemsPerPage;
        const paginatedArticles = filteredArticles.slice(startIndex, endIndex);
        
        container.innerHTML = paginatedArticles.map(article => {
            const isSelectable = !article.verified;
            const baseClasses = isSelectable 
                ? 'cursor-pointer hover:shadow-lg hover:-translate-y-1 transition-all duration-200' 
                : 'opacity-75';
            
            return `
                <div data-article-id="${article.id}" 
                     class="${baseClasses} p-4 bg-gray-50 rounded-lg border border-gray-200 relative"
                     ${isSelectable ? `onclick="newsTracker.toggleArticleSelection('${article.id}')"` : ''}>
                    
                    ${isSelectable ? `
                        <!-- Selection indicator (hidden by default) -->
                        <div class="selection-indicator hidden absolute top-2 right-2 bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold z-10">
                            ✓
                        </div>
                        
                        <!-- Dashed border overlay for selection -->
                        <div class="selection-overlay absolute inset-0 rounded-lg border-2 border-dashed border-transparent pointer-events-none transition-all duration-200"></div>
                        
                        <!-- Click to select button -->
                        <div class="selection-button absolute top-3 left-3 bg-white border border-gray-300 rounded-full px-3 py-1 shadow-sm hover:bg-blue-50 hover:border-blue-400 transition-all duration-200 z-20">
                            <span class="text-xs font-medium text-gray-600 hover:text-blue-600 flex items-center">
                                <i class="bi bi-plus-circle mr-1"></i>
                                Click to select
                            </span>
                        </div>
                    ` : ''}
                    
                    <div class="content flex-1 ${isSelectable ? 'mt-8' : ''}">
                        <h4 class="font-medium text-gray-800 mb-1">${this.escapeHtml(article.title || 'No Title')}</h4>
                        <p class="text-sm text-blue-600 hover:text-blue-800 mb-2">
                            <a href="${article.url}" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation()">${article.url}</a>
                        </p>
                        ${article.description ? `<p class="text-sm text-gray-600 mb-2">${this.escapeHtml(article.description)}</p>` : ''}
                        <div class="flex items-center justify-between">
                            <div class="flex items-center text-xs text-gray-500">
                                <span class="mr-3">
                                    <i class="bi bi-globe mr-1"></i>
                                    ${this.escapeHtml(article.site_name || article.siteName || 'Unknown')}
                                </span>
                                <span class="mr-3">
                                    <i class="bi bi-calendar mr-1"></i>
                                    ${new Date(article.found_at || article.foundAt).toLocaleString()}
                                </span>
                                ${article.confidence ? `
                                    <span class="mr-3">
                                        <i class="bi bi-speedometer2 mr-1"></i>
                                        ${(article.confidence * 100).toFixed(0)}% conf.
                                    </span>
                                ` : ''}
                            </div>
                            <span class="px-2 py-1 text-xs rounded-full ${this.getStatusBadgeClass(article)}">
                                ${this.getStatusText(article)}
                            </span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        this.updatePagination(filteredArticles.length);
        
        // Initialize selection count
        this.updateSelectionCount();
        this.updateBatchActionButtons();
    }
    
    filterArticlesByType(filter) {
        switch (filter) {
            case 'unverified':
                return this.articleQueue.filter(article => !article.verified);
            case 'verified':
                return this.articleQueue.filter(article => article.verified);
            case 'news':
                return this.articleQueue.filter(article => {
                    const isNews = article.is_news !== undefined ? article.is_news : article.isNews;
                    return article.verified && isNews;
                });
            case 'not-news':
                return this.articleQueue.filter(article => {
                    const isNews = article.is_news !== undefined ? article.is_news : article.isNews;
                    return article.verified && !isNews;
                });
            default:
                return this.articleQueue;
        }
    }
    
    getStatusBadgeClass(article) {
        if (!article.verified) return 'bg-yellow-100 text-yellow-800';
        const isNews = article.is_news !== undefined ? article.is_news : article.isNews;
        if (isNews) return 'bg-green-100 text-green-800';
        return 'bg-red-100 text-red-800';
    }
    
    getStatusText(article) {
        if (!article.verified) return 'Pending';
        const isNews = article.is_news !== undefined ? article.is_news : article.isNews;
        if (isNews) return 'News';
        return 'Not News';
    }
    
    updateCounts() {
        const verifiedCount = this.articleQueue.filter(article => article.verified).length;
        const verifiedNewsCount = this.articleQueue.filter(article => {
            const isNews = article.is_news !== undefined ? article.is_news : article.isNews;
            return article.verified && isNews;
        }).length;
        const notNewsCount = this.articleQueue.filter(article => {
            const isNews = article.is_news !== undefined ? article.is_news : article.isNews;
            return article.verified && !isNews;
        }).length;
        const pendingCount = this.articleQueue.filter(article => !article.verified).length;
        
        document.getElementById('trackedWebsitesCount').textContent = this.trackedWebsites.length;
        document.getElementById('queueCount').textContent = this.articleQueue.length;
        document.getElementById('verifiedCount').textContent = verifiedCount;
        
        document.getElementById('pendingCount').textContent = pendingCount;
        document.getElementById('verifiedNewsCount').textContent = verifiedNewsCount;
        document.getElementById('notNewsCount').textContent = notNewsCount;
        document.getElementById('totalCount').textContent = this.articleQueue.length;
    }
    
    updateStatistics() {
        // This would typically fetch data from the server
        // For now, we'll calculate from current data
        
        const today = new Date().toDateString();
        const thisWeek = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
        
        const todayArticles = this.articleQueue.filter(article => 
            new Date(article.foundAt).toDateString() === today
        );
        const weekArticles = this.articleQueue.filter(article => 
            new Date(article.foundAt) >= thisWeek
        );
        
        const todayVerified = todayArticles.filter(article => article.verified).length;
        const weekVerified = weekArticles.filter(article => article.verified).length;
        const totalVerified = this.articleQueue.filter(article => article.verified).length;
        
        document.getElementById('todayFound').textContent = todayArticles.length;
        document.getElementById('todayVerified').textContent = todayVerified;
        document.getElementById('todaySuccessRate').textContent = 
            todayArticles.length > 0 ? Math.round((todayVerified / todayArticles.length) * 100) + '%' : '0%';
        
        document.getElementById('weekFound').textContent = weekArticles.length;
        document.getElementById('weekVerified').textContent = weekVerified;
        document.getElementById('weekAverage').textContent = Math.round(weekArticles.length / 7);
        
        document.getElementById('totalArticles').textContent = this.articleQueue.length;
        document.getElementById('totalVerified').textContent = totalVerified;
        document.getElementById('totalAccuracy').textContent = 
            this.articleQueue.length > 0 ? Math.round((totalVerified / this.articleQueue.length) * 100) + '%' : '0%';
    }
    
    // Utility methods
    isValidUrl(string) {
        try {
            new URL(string);
            return true;
        } catch (_) {
            return false;
        }
    }
    
    extractDomainName(url) {
        try {
            const domain = new URL(url).hostname;
            return domain.replace('www.', '');
        } catch (_) {
            return 'Unknown Website';
        }
    }
    
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, function(m) { return map[m]; });
    }
    
    showLoading(message = 'Loading...') {
        document.getElementById('loadingMessage').textContent = message;
        document.getElementById('loading').classList.remove('hidden');
    }
    
    hideLoading() {
        document.getElementById('loading').classList.add('hidden');
    }
    
    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('errorDisplay').classList.remove('hidden');
        setTimeout(() => this.hideError(), 5000);
    }
    
    hideError() {
        document.getElementById('errorDisplay').classList.add('hidden');
    }
    
    showSuccess(message) {
        document.getElementById('successMessage').textContent = message;
        document.getElementById('successDisplay').classList.remove('hidden');
        setTimeout(() => this.hideSuccess(), 3000);
    }
    
    hideSuccess() {
        document.getElementById('successDisplay').classList.add('hidden');
    }
    
    // Load data from server
    async loadTrackedWebsites() {
        try {
            const response = await fetch('/api/news-tracker/get-data');
            const data = await response.json();
            
            if (data.success) {
                this.trackedWebsites = data.websites || [];
                this.articleQueue = data.articles || [];
                this.renderTrackedWebsites();
                this.renderArticleQueue();
                this.updateCounts();
                this.updateStatistics();
                
            } else {
                console.error('Failed to load data:', data.error);
                this.showError('Failed to load tracked websites');
            }
        } catch (error) {
            console.error('Error loading data:', error);
            this.showError('Network error while loading data');
        }
    }
    
    async loadArticleQueue() {
        // This is now handled by loadTrackedWebsites()
        return;
    }
    
    setupAutoFetch() {
        // Setup auto-fetch intervals
    }
    
    toggleAutoFetch(enabled) {
        // Toggle auto-fetch functionality
    }
    
    updateLastFetchInfo() {
        document.getElementById('lastFetchInfo').innerHTML = `
            <div class="flex items-center">
                <i class="bi bi-info-circle text-blue-600 mr-2"></i>
                <span class="text-sm text-blue-800">Last fetch: ${new Date().toLocaleString()}</span>
            </div>
        `;
    }
    
    // Additional methods for missing functionality
    async clearQueue() {
        if (!confirm('Are you sure you want to clear the entire queue?')) return;
        
        this.articleQueue = [];
        this.renderArticleQueue();
        this.updateCounts();
        this.showSuccess('Queue cleared successfully');
    }
    
    filterQueue(filterType) {
        this.currentPage = 1;
        this.renderArticleQueue();
    }
    
    updatePagination(totalItems) {
        const totalPages = Math.ceil(totalItems / this.itemsPerPage);
        const pagination = document.getElementById('queuePagination');
        
        if (totalPages <= 1) {
            pagination.classList.add('hidden');
            return;
        }
        
        pagination.classList.remove('hidden');
        document.getElementById('pageInfo').textContent = `Page ${this.currentPage} of ${totalPages}`;
        document.getElementById('prevPageBtn').disabled = this.currentPage === 1;
        document.getElementById('nextPageBtn').disabled = this.currentPage === totalPages;
    }
    
    previousPage() {
        if (this.currentPage > 1) {
            this.currentPage--;
            this.renderArticleQueue();
        }
    }
    
    nextPage() {
        const filter = document.getElementById('queueFilter').value;
        const filteredArticles = this.filterArticlesByType(filter);
        const totalPages = Math.ceil(filteredArticles.length / this.itemsPerPage);
        
        if (this.currentPage < totalPages) {
            this.currentPage++;
            this.renderArticleQueue();
        }
    }
    
    // Batch verification methods (individual verification removed)
    
    updateSelectionDisplay() {
        // Auto-switch to unverified filter when starting selection
        const filter = document.getElementById('queueFilter').value;
        const unverifiedCount = this.articleQueue.filter(a => !a.verified).length;
        
        if (filter !== 'unverified' && unverifiedCount > 0) {
            // Show a hint about filtering to unverified articles
            const hint = document.createElement('div');
            hint.className = 'mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg';
            hint.innerHTML = `
                <div class="flex items-center justify-between">
                    <div class="flex items-center">
                        <i class="bi bi-lightbulb text-yellow-600 mr-2"></i>
                        <span class="text-sm text-yellow-800">
                            <strong>Tip:</strong> Switch to "Unverified" filter to see only articles that can be selected
                        </span>
                    </div>
                    <button onclick="document.getElementById('queueFilter').value='unverified'; newsTracker.filterQueue('unverified'); this.parentElement.parentElement.remove();" 
                            class="bg-yellow-600 text-white px-3 py-1 rounded text-xs hover:bg-yellow-700">
                        Switch Filter
                    </button>
                </div>
            `;
            
            const batchControls = document.getElementById('batchControls');
            if (!document.querySelector('.filter-hint')) {
                hint.classList.add('filter-hint');
                batchControls.parentNode.insertBefore(hint, batchControls.nextSibling);
            }
        }
    }
    toggleArticleSelection(articleId) {
        const articleElement = document.querySelector(`[data-article-id="${articleId}"]`);
        const article = this.articleQueue.find(a => a.id == articleId);
        
        if (!articleElement || !article || article.verified) {
            return; // Can't select verified articles
        }
        
        const isSelected = articleElement.dataset.batchSelected === 'true';
        
        if (isSelected) {
            this.unselectArticle(articleElement);
        } else {
            // Check if we've reached the max selection limit
            const currentSelections = document.querySelectorAll('[data-batch-selected="true"]').length;
            const maxSelections = parseInt(document.getElementById('batchSize').value);
            
            if (currentSelections >= maxSelections) {
                this.showError(`Maximum ${maxSelections} articles can be selected at once`);
                return;
            }
            
            this.selectArticle(articleElement);
        }
        
        this.updateSelectionCount();
        this.updateBatchActionButtons();
    }
    
    selectArticle(articleElement) {
        articleElement.dataset.batchSelected = 'true';
        
        // Update main container styling
        articleElement.classList.remove('bg-gray-50', 'border-gray-200');
        articleElement.classList.add('bg-blue-50', 'border-blue-400', 'border-2');
        
        // Show checkmark indicator
        const indicator = articleElement.querySelector('.selection-indicator');
        if (indicator) {
            indicator.classList.remove('hidden');
        }
        
        // Update the selection button
        const selectionButton = articleElement.querySelector('.selection-button');
        if (selectionButton) {
            selectionButton.innerHTML = `
                <span class="text-xs font-medium text-blue-600 flex items-center">
                    <i class="bi bi-check-circle-fill mr-1"></i>
                    Selected
                </span>
            `;
            selectionButton.classList.remove('bg-white', 'border-gray-300', 'hover:bg-blue-50', 'hover:border-blue-400');
            selectionButton.classList.add('bg-blue-100', 'border-blue-400');
        }
        
        // Update dashed border overlay
        const selectionOverlay = articleElement.querySelector('.selection-overlay');
        if (selectionOverlay) {
            selectionOverlay.classList.remove('border-transparent');
            selectionOverlay.classList.add('border-blue-400');
        }
    }
    
    unselectArticle(articleElement) {
        delete articleElement.dataset.batchSelected;
        
        // Reset main container styling
        articleElement.classList.remove('bg-blue-50', 'border-blue-400', 'border-2');
        articleElement.classList.add('bg-gray-50', 'border-gray-200');
        
        // Hide checkmark indicator
        const indicator = articleElement.querySelector('.selection-indicator');
        if (indicator) {
            indicator.classList.add('hidden');
        }
        
        // Reset the selection button
        const selectionButton = articleElement.querySelector('.selection-button');
        if (selectionButton) {
            selectionButton.innerHTML = `
                <span class="text-xs font-medium text-gray-600 hover:text-blue-600 flex items-center">
                    <i class="bi bi-plus-circle mr-1"></i>
                    Click to select
                </span>
            `;
            selectionButton.classList.remove('bg-blue-100', 'border-blue-400');
            selectionButton.classList.add('bg-white', 'border-gray-300', 'hover:bg-blue-50', 'hover:border-blue-400');
        }
        
        // Reset dashed border overlay
        const selectionOverlay = articleElement.querySelector('.selection-overlay');
        if (selectionOverlay) {
            selectionOverlay.classList.remove('border-blue-400');
            selectionOverlay.classList.add('border-transparent');
        }
    }
    
    updateSelectionCount() {
        const selectedCount = document.querySelectorAll('[data-batch-selected="true"]').length;
        const selectionCountElement = document.getElementById('selectionCount');
        if (selectionCountElement) {
            selectionCountElement.textContent = `${selectedCount} articles selected`;
        }
    }
    
    updateBatchActionButtons() {
        const selectedCount = document.querySelectorAll('[data-batch-selected="true"]').length;
        const newsBtn = document.getElementById('batchMarkNewsBtn');
        const notNewsBtn = document.getElementById('batchMarkNotNewsBtn');
        
        if (newsBtn && notNewsBtn) {
            if (selectedCount > 0) {
                newsBtn.disabled = false;
                notNewsBtn.disabled = false;
                newsBtn.classList.remove('opacity-50', 'cursor-not-allowed');
                notNewsBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            } else {
                newsBtn.disabled = true;
                notNewsBtn.disabled = true;
                newsBtn.classList.add('opacity-50', 'cursor-not-allowed');
                notNewsBtn.classList.add('opacity-50', 'cursor-not-allowed');
            }
        }
    }
    
    selectBatchArticles() {
        const batchSize = parseInt(document.getElementById('batchSize').value);
        
        // Get currently visible unverified articles from the DOM
        const unverifiedElements = Array.from(document.querySelectorAll('[data-article-id]')).filter(element => {
            const articleId = element.dataset.articleId;
            const article = this.articleQueue.find(a => a.id == articleId);
            return article && !article.verified;
        });
        
        if (unverifiedElements.length === 0) {
            this.showError('No unverified articles available for selection');
            return;
        }
        
        // Show hint about filtering if needed
        this.updateSelectionDisplay();
        
        // Clear any existing selections
        this.clearBatchSelection();
        
        // Select up to batchSize unverified articles
        const elementsToSelect = unverifiedElements.slice(0, batchSize);
        
        elementsToSelect.forEach(element => {
            this.selectArticle(element);
        });
        
        this.updateSelectionCount();
        this.updateBatchActionButtons();
        
        this.showInfo(`Selected ${elementsToSelect.length} articles for batch verification`);
    }
    
    clearBatchSelection() {
        // Unselect all selected articles
        document.querySelectorAll('[data-batch-selected="true"]').forEach(element => {
            this.unselectArticle(element);
        });
        
        this.updateSelectionCount();
        this.updateBatchActionButtons();
        
        // Hide any batch status messages
        const batchStatus = document.getElementById('batchStatus');
        if (batchStatus) {
            batchStatus.classList.add('hidden');
        }
        
        // Remove any filter hints
        const filterHints = document.querySelectorAll('.filter-hint');
        filterHints.forEach(hint => hint.remove());
    }
    
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
                
            } else {
                this.showError(data.error || 'Failed to perform batch verification');
            }
            
        } catch (error) {
            console.error('Batch verification error:', error);
            this.showError('Network error during batch verification');
        } finally {
            // Hide progress indicator and re-enable buttons
            document.getElementById('batchProgress').classList.add('hidden');
            this.updateBatchActionButtons();
        }
    }
    
    showInfo(message) {
        // Show info message (similar to showSuccess but with blue styling)
        const alertDiv = document.createElement('div');
        alertDiv.className = 'fixed top-4 right-4 bg-blue-100 border border-blue-400 text-blue-700 px-4 py-3 rounded shadow-lg z-50';
        alertDiv.innerHTML = `
            <div class="flex items-center">
                <i class="bi bi-info-circle mr-2"></i>
                <span>${message}</span>
                <button class="ml-4 text-blue-700 hover:text-blue-900" onclick="this.parentElement.parentElement.remove()">
                    <i class="bi bi-x"></i>
                </button>
            </div>
        `;
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentElement) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Initialize the application
let newsTracker;
document.addEventListener('DOMContentLoaded', () => {
    newsTracker = new NewsTracker();
});
