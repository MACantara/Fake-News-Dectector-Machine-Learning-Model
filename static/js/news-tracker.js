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
        this.currentArticleIndex = 0;
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadTrackedWebsites();
        this.loadArticleQueue();
        this.updateStatistics();
        this.setupAutoFetch();
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
        
        // Verification
        document.getElementById('verifyNewsBtn').addEventListener('click', () => this.verifyArticle(true));
        document.getElementById('verifyNotNewsBtn').addEventListener('click', () => this.verifyArticle(false));
        document.getElementById('skipArticleBtn').addEventListener('click', () => this.skipArticle());
        document.getElementById('viewFullArticleBtn').addEventListener('click', () => this.viewFullArticle());
        document.getElementById('reportIssueBtn').addEventListener('click', () => this.reportIssue());
        
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
                    this.showVerificationInterface();
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
            this.hideVerificationInterface();
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
    
    showVerificationInterface() {
        const verificationPanel = document.getElementById('verificationInterface');
        verificationPanel.classList.remove('hidden');
        
        // Find first unverified article
        const unverifiedIndex = this.articleQueue.findIndex(article => !article.verified);
        if (unverifiedIndex !== -1) {
            this.currentArticleIndex = unverifiedIndex;
            this.displayCurrentArticle();
            this.updateVerificationProgress();
        }
    }
    
    hideVerificationInterface() {
        document.getElementById('verificationInterface').classList.add('hidden');
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
        const emptyMessage = document.getElementById('emptyQueueMessage');
        
        if (this.articleQueue.length === 0) {
            emptyMessage.classList.remove('hidden');
            return;
        }
        
        emptyMessage.classList.add('hidden');
        
        // Apply current filter
        const filter = document.getElementById('queueFilter').value;
        const filteredArticles = this.filterArticlesByType(filter);
        
        // Pagination
        const startIndex = (this.currentPage - 1) * this.itemsPerPage;
        const endIndex = startIndex + this.itemsPerPage;
        const paginatedArticles = filteredArticles.slice(startIndex, endIndex);
        
        container.innerHTML = paginatedArticles.map(article => `
            <div class="flex items-start justify-between p-4 bg-gray-50 rounded-lg border hover:bg-gray-100 transition-colors">
                <div class="flex-1">
                    <h4 class="font-medium text-gray-800 mb-1">${this.escapeHtml(article.title || 'No Title')}</h4>
                    <p class="text-sm text-blue-600 hover:text-blue-800 mb-2">
                        <a href="${article.url}" target="_blank" rel="noopener noreferrer">${article.url}</a>
                    </p>
                    ${article.description ? `<p class="text-sm text-gray-600 mb-2">${this.escapeHtml(article.description)}</p>` : ''}
                    <div class="flex items-center text-xs text-gray-500">
                        <span class="mr-3">
                            <i class="bi bi-globe mr-1"></i>
                            ${this.escapeHtml(article.site_name || article.siteName || 'Unknown')}
                        </span>
                        <span>
                            <i class="bi bi-calendar mr-1"></i>
                            ${new Date(article.found_at || article.foundAt).toLocaleString()}
                        </span>
                    </div>
                </div>
                <div class="ml-4 flex flex-col items-end">
                    <span class="px-2 py-1 text-xs rounded-full mb-2 ${this.getStatusBadgeClass(article)}">
                        ${this.getStatusText(article)}
                    </span>
                    ${!article.verified ? `
                        <button 
                            onclick="newsTracker.verifyFromQueue('${article.id}')"
                            class="text-blue-600 hover:text-blue-800 text-xs"
                        >
                            <i class="bi bi-check-square mr-1"></i>
                            Verify
                        </button>
                    ` : ''}
                </div>
            </div>
        `).join('');
        
        this.updatePagination(filteredArticles.length);
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
                
                // Show verification interface if there are unverified articles
                const unverifiedArticles = this.articleQueue.filter(article => !article.verified);
                if (unverifiedArticles.length > 0) {
                    this.showVerificationInterface();
                }
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
        this.hideVerificationInterface();
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
    
    verifyFromQueue(articleId) {
        const articleIndex = this.articleQueue.findIndex(article => article.id === articleId);
        if (articleIndex !== -1) {
            this.currentArticleIndex = articleIndex;
            this.showVerificationInterface();
        }
    }
    
    viewFullArticle() {
        const article = this.articleQueue[this.currentArticleIndex];
        if (article) {
            window.open(article.url, '_blank');
        }
    }
    
    reportIssue() {
        const article = this.articleQueue[this.currentArticleIndex];
        if (article) {
            const subject = encodeURIComponent(`Issue with article: ${article.title || article.url}`);
            const body = encodeURIComponent(`Article URL: ${article.url}\n\nIssue description: `);
            window.open(`mailto:support@example.com?subject=${subject}&body=${body}`);
        }
    }
}

// Initialize the application
let newsTracker;
document.addEventListener('DOMContentLoaded', () => {
    newsTracker = new NewsTracker();
});
