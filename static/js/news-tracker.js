/**
 * News Tracker Application
 * Handles website tracking, article fetching, and verification
 */

class NewsTracker {
    constructor() {
        this.trackedWebsites = [];
        this.articleQueue = [];
        this.currentPage = 1;
        this.itemsPerPage = 20; // Default to 20 (double the default batch size of 10)
        this.autoFetchInterval = null;
        this.autoFetchEnabled = false;
        this.autoFetchIntervalMinutes = 30; // Default 30 minutes
        this.autoFetchStats = {
            totalRuns: 0,
            articlesFound: 0,
            lastRun: null,
            nextRun: null
        };
        this.selectedArticles = new Set(); // Track selected articles
        this.predictionMetrics = null; // Store comprehensive prediction metrics
        this.websiteViewMode = 'grouped'; // Default to grouped view
        
        // Auto-indexing settings
        this.autoIndexEnabled = false;
        this.autoIndexThreshold = 0.95; // 95% confidence threshold
        this.autoIndexBatchSize = 100;
        this.autoIndexStats = {
            totalIndexed: 0,
            lastBatchSize: 0,
            lastBatchTime: null,
            successRate: 0,
            highConfidenceQueue: 0
        };
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadViewPreferences();
        this.loadAutoIndexPreferences(); // Load auto-indexing preferences
        this.loadTrackedWebsites();
        this.loadArticleQueue();
        this.updateStatistics();
        this.setupAutoFetch();
        
        // Initialize selection interface
        this.updateSelectionCount();
        this.updateBatchActionButtons();
        
        // Initialize displayed article count indicator
        this.updateDisplayedArticleCount();
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            this.stopAutoFetch();
        });
    }
    
    loadViewPreferences() {
        // Load website view mode preference
        const savedViewMode = localStorage.getItem('newsTracker.websiteViewMode') || 'grouped';
        this.websiteViewMode = savedViewMode;
        
        // Update UI to reflect saved preference
        const viewModeSelect = document.getElementById('websiteViewMode');
        if (viewModeSelect) {
            viewModeSelect.value = savedViewMode;
        }
        
        // Update control visibility
        this.setWebsiteViewMode(savedViewMode);
    }
    
    bindEvents() {
        // Website management
        document.getElementById('addWebsiteBtn').addEventListener('click', () => this.addWebsite());
        document.getElementById('websiteUrl').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.addWebsite();
        });
        
        // Website view controls
        const websiteViewMode = document.getElementById('websiteViewMode');
        if (websiteViewMode) {
            websiteViewMode.addEventListener('change', (e) => {
                this.setWebsiteViewMode(e.target.value);
            });
        }
        
        const expandAllBtn = document.getElementById('expandAllDomainsBtn');
        if (expandAllBtn) {
            expandAllBtn.addEventListener('click', () => this.expandAllDomains());
        }
        
        const collapseAllBtn = document.getElementById('collapseAllDomainsBtn');
        if (collapseAllBtn) {
            collapseAllBtn.addEventListener('click', () => this.collapseAllDomains());
        }
        
        // Auto-fetch controls
        const autoFetchToggle = document.getElementById('autoFetchToggle');
        if (autoFetchToggle) {
            autoFetchToggle.addEventListener('change', (e) => {
                this.toggleAutoFetch(e.target.checked);
            });
            
            // Add click handlers to the visual toggle elements
            const toggleBg = autoFetchToggle.nextElementSibling;
            const toggleDot = toggleBg?.nextElementSibling;
            
            if (toggleBg) {
                toggleBg.addEventListener('click', () => {
                    autoFetchToggle.checked = !autoFetchToggle.checked;
                    autoFetchToggle.dispatchEvent(new Event('change'));
                });
            }
            
            if (toggleDot) {
                toggleDot.addEventListener('click', () => {
                    autoFetchToggle.checked = !autoFetchToggle.checked;
                    autoFetchToggle.dispatchEvent(new Event('change'));
                });
            }
        }
        
        const autoFetchInterval = document.getElementById('autoFetchInterval');
        if (autoFetchInterval) {
            autoFetchInterval.addEventListener('change', (e) => {
                this.updateAutoFetchInterval(parseInt(e.target.value));
            });
        }
        
        const fetchNowBtn = document.getElementById('fetchNowBtn');
        if (fetchNowBtn) {
            fetchNowBtn.addEventListener('click', () => this.fetchAllArticles());
        }
        
        const clearQueueBtn = document.getElementById('clearQueueBtn');
        if (clearQueueBtn) {
            clearQueueBtn.addEventListener('click', () => this.clearQueue());
        }
        
        const testAutoFetchBtn = document.getElementById('testAutoFetchBtn');
        if (testAutoFetchBtn) {
            testAutoFetchBtn.addEventListener('click', () => this.testAutoFetch());
        }
        
        // Auto-indexing controls
        const autoIndexToggle = document.getElementById('autoIndexToggle');
        if (autoIndexToggle) {
            autoIndexToggle.addEventListener('change', (e) => this.toggleAutoIndex(e.target.checked));
        }
        
        const autoIndexThreshold = document.getElementById('autoIndexThreshold');
        if (autoIndexThreshold) {
            autoIndexThreshold.addEventListener('change', (e) => this.updateAutoIndexThreshold(parseInt(e.target.value)));
        }
        
        const autoIndexBatchSize = document.getElementById('autoIndexBatchSize');
        if (autoIndexBatchSize) {
            autoIndexBatchSize.addEventListener('change', (e) => this.updateAutoIndexBatchSize(parseInt(e.target.value)));
        }
        
        const triggerAutoIndexBtn = document.getElementById('triggerAutoIndexBtn');
        if (triggerAutoIndexBtn) {
            triggerAutoIndexBtn.addEventListener('click', () => this.triggerAutoIndex());
        }
        
        const enableAllAutoFetchBtn = document.getElementById('enableAllAutoFetchBtn');
        if (enableAllAutoFetchBtn) {
            enableAllAutoFetchBtn.addEventListener('click', () => this.quickSetupAutoFetch());
        }
        
        const resetAutoFetchBtn = document.getElementById('resetAutoFetchBtn');
        if (resetAutoFetchBtn) {
            resetAutoFetchBtn.addEventListener('click', () => this.resetAutoFetchSettings());
        }
        
        // Queue management
        document.getElementById('queueFilter').addEventListener('change', (e) => {
            this.filterQueue(e.target.value);
        });
        
        // Batch verification controls
        document.getElementById('batchSize').addEventListener('change', (e) => {
            const size = parseInt(e.target.value);
            document.getElementById('selectedBatchSize').textContent = size;
            
            // Update items per page to show at least the batch size number of articles
            // This ensures users can see all selectable articles for their chosen batch size
            // For large batch sizes (100+), show the batch size + 20 extra for efficiency
            if (size >= 100) {
                this.itemsPerPage = size + 20; // For 100 articles, show 120 per page
            } else {
                this.itemsPerPage = Math.max(size * 2, 10); // Show at least double the batch size, minimum 10
            }
            
            // Update the displayed article count indicator
            const displayedCountElement = document.getElementById('displayedArticleCount');
            if (displayedCountElement) {
                displayedCountElement.textContent = this.itemsPerPage;
            }
            
            // Reset to first page and re-render to show the updated number of articles
            this.currentPage = 1;
            this.renderArticleQueue();
            
            // Clear any existing selections since the view has changed
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
        
        const url = urlInput.value.trim();
        
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
                body: JSON.stringify({ url })
            });
            
            const data = await response.json();
            
            if (data.success) {
                const website = {
                    id: data.id,
                    url,
                    name: this.extractDomainName(url), // Auto-generate display name
                    interval: 60, // Default interval
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
                
                this.showSuccess(data.message || 'Website added successfully');
                
                // Start auto-fetch if enabled and this is the first website
                if (this.autoFetchEnabled && this.trackedWebsites.length === 1) {
                    this.startAutoFetch();
                    this.updateAutoFetchStatus();
                }
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
                
                // Stop auto-fetch if no websites remain
                if (this.trackedWebsites.length === 0 && this.autoFetchEnabled) {
                    this.stopAutoFetch();
                    this.updateAutoFetchStatus();
                }
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
        // Check view mode and render accordingly
        if (this.websiteViewMode === 'list') {
            this.renderTrackedWebsitesSimple();
            return;
        }
        
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
        
        // Group websites by domain
        const domainGroups = this.groupWebsitesByDomain(this.trackedWebsites);
        
        container.innerHTML = domainGroups.map(group => `
            <div class="domain-group border border-gray-200 rounded-lg mb-4" data-domain-group="${group.domain}" data-expanded="false">
                <!-- Domain Group Header -->
                <div class="domain-header p-4 bg-gradient-to-r from-blue-50 to-blue-100 rounded-t-lg border-b border-blue-200 cursor-pointer hover:from-blue-100 hover:to-blue-150 transition-all duration-200" 
                     onclick="newsTracker.toggleDomainGroup('${group.domain}')">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center">
                            <i class="domain-toggle-icon bi bi-chevron-right text-blue-600 mr-2 transition-transform duration-200"></i>
                            <div>
                                <h3 class="font-semibold text-blue-800">${this.escapeHtml(group.displayName)}</h3>
                                <p class="text-sm text-blue-600">${group.domain}</p>
                            </div>
                        </div>
                        <div class="flex items-center space-x-4 text-sm text-blue-700">
                            <span class="flex items-center">
                                <i class="bi bi-globe mr-1"></i>
                                ${group.websites.length} source${group.websites.length !== 1 ? 's' : ''}
                            </span>
                            <span class="flex items-center">
                                <i class="bi bi-newspaper mr-1"></i>
                                ${group.totalArticles} articles
                            </span>
                            <span class="px-2 py-1 text-xs bg-blue-200 text-blue-800 rounded-full">
                                ${this.getGroupStatus(group.websites)}
                            </span>
                        </div>
                    </div>
                </div>
                
                <!-- Websites in Domain Group -->
                <div class="domain-websites" style="display: none;">
                    ${group.websites.map(website => `
                        <div class="flex items-center justify-between p-4 bg-gray-50 border-b border-gray-100 last:border-b-0 last:rounded-b-lg hover:bg-gray-100 transition-colors duration-150">
                            <div class="flex-1">
                                <h4 class="font-medium text-gray-800">${this.escapeHtml(website.name)}</h4>
                                <p class="text-sm text-gray-600 break-all">${website.url}</p>
                                <div class="flex items-center mt-2 text-xs text-gray-500">
                                    <span class="mr-4 flex items-center">
                                        <i class="bi bi-clock mr-1"></i>
                                        Every ${website.fetch_interval || website.interval || 60} min
                                    </span>
                                    <span class="mr-4 flex items-center">
                                        <i class="bi bi-newspaper mr-1"></i>
                                        ${website.article_count || website.articleCount || 0} articles
                                    </span>
                                    <span class="flex items-center">
                                        <i class="bi bi-calendar mr-1"></i>
                                        ${website.last_fetch || website.lastFetch ? new Date(website.last_fetch || website.lastFetch).toLocaleString() : 'Never fetched'}
                                    </span>
                                </div>
                            </div>
                            <div class="flex items-center space-x-2 ml-4">
                                <span class="px-2 py-1 text-xs rounded-full ${website.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                                    ${website.status}
                                </span>
                                <button 
                                    onclick="event.stopPropagation(); newsTracker.removeWebsite('${website.id}')"
                                    class="text-red-600 hover:text-red-800 p-1 rounded hover:bg-red-50 transition-colors duration-150"
                                    title="Remove website"
                                >
                                    <i class="bi bi-trash"></i>
                                </button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `).join('');
    }
    
    getGroupStatus(websites) {
        const activeCount = websites.filter(w => w.status === 'active').length;
        const totalCount = websites.length;
        
        if (activeCount === totalCount) {
            return 'All Active';
        } else if (activeCount === 0) {
            return 'None Active';
        } else {
            return `${activeCount}/${totalCount} Active`;
        }
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
        let filteredArticles = this.filterArticlesByType(filter);
        
        // Sort articles: unverified first, then by confidence score (lowest to highest), then by date
        filteredArticles = filteredArticles.sort((a, b) => {
            // Unverified articles should come first
            if (!a.verified && b.verified) return -1;
            if (a.verified && !b.verified) return 1;
            
            // Within same verification status, sort by confidence score
            const confidenceA = a.confidence || 0.5;
            const confidenceB = b.confidence || 0.5;
            
            // For unverified articles, sort by confidence (lowest first - most uncertain articles need attention first)
            if (!a.verified && !b.verified) {
                if (confidenceA !== confidenceB) {
                    return confidenceA - confidenceB; // Lowest confidence first
                }
            }
            
            // For verified articles, sort by confidence (highest first - best predictions first)
            if (a.verified && b.verified) {
                if (confidenceA !== confidenceB) {
                    return confidenceB - confidenceA; // Highest confidence first
                }
            }
            
            // If confidence scores are equal, sort by found date (newest first)
            const dateA = new Date(a.found_at || a.foundAt);
            const dateB = new Date(b.found_at || b.foundAt);
            return dateB - dateA;
        });
        
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
                            </div>
                            <span class="px-2 py-1 text-xs rounded-full ${this.getStatusBadgeClass(article)}">
                                ${this.getStatusText(article)}
                            </span>
                        </div>
                        
                        <!-- Confidence Score and Prediction Metrics -->
                        ${this.renderDetailedMetrics(article)}
                    </div>
                </div>
            `;
        }).join('');
        
        this.updatePagination(filteredArticles.length);
        
        // Initialize selection count
        this.updateSelectionCount();
        this.updateBatchActionButtons();
        
        // Update displayed article count
        this.updateDisplayedArticleCount();
        
        // Update high confidence queue count for auto-indexing
        this.updateHighConfidenceQueue();
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
    
    extractRootDomain(url) {
        try {
            const domain = new URL(url).hostname.replace('www.', '');
            
            // Handle special domain mappings first
            const specialDomains = this.getSpecialDomainMappings();
            if (specialDomains[domain]) {
                return specialDomains[domain];
            }
            
            // Check if any special domain is a subdomain of this domain
            for (const [key, value] of Object.entries(specialDomains)) {
                if (domain.endsWith('.' + key) || domain === key) {
                    return value;
                }
            }
            
            const parts = domain.split('.');
            
            // Handle country code TLDs (ccTLDs) and second-level domains
            if (this.isCountryCodeTLD(parts)) {
                return this.extractDomainWithCCTLD(parts);
            }
            
            // Handle standard domains
            if (parts.length >= 2) {
                // For domains like news.bbc.com, check if subdomain should be preserved
                if (parts.length >= 3 && this.shouldPreserveSubdomain(parts)) {
                    return parts.slice(-3).join('.');
                }
                return parts.slice(-2).join('.');
            }
            
            return domain;
        } catch (_) {
            return 'unknown';
        }
    }
    
    getSpecialDomainMappings() {
        // Map specific domains to their canonical grouping domain
        return {
            'mb.com.ph': 'manilabulletin.com.ph',
            'businessmirror.com.ph': 'businessmirror.com.ph',
            'news.abs-cbn.com': 'abs-cbn.com',
            'news.gma.network': 'gmanetwork.com',
            'cnnphilippines.com': 'cnn.com',
            'news.yahoo.com': 'yahoo.com',
            'abscbn.com': 'abs-cbn.com',
            'gmanews.tv': 'gmanetwork.com',
            'manilastandard.net': 'manilastandardtoday.com',
            'tribune.net.ph': 'tribuneonline.org',
            'bworldonline.com': 'businessworld.com.ph',
            'pna.gov.ph': 'pna.gov.ph',
            'philstar.com': 'philstar.com'
        };
    }
    
    isCountryCodeTLD(parts) {
        if (parts.length < 3) return false;
        
        const ccTLDs = [
            'com.ph', 'net.ph', 'org.ph', 'gov.ph', 'edu.ph',
            'co.uk', 'org.uk', 'ac.uk', 'gov.uk',
            'com.au', 'net.au', 'org.au', 'gov.au', 'edu.au',
            'co.jp', 'or.jp', 'ne.jp', 'go.jp', 'ac.jp',
            'com.sg', 'net.sg', 'org.sg', 'gov.sg', 'edu.sg',
            'com.my', 'net.my', 'org.my', 'gov.my', 'edu.my'
        ];
        
        const lastTwoParts = parts.slice(-2).join('.');
        return ccTLDs.includes(lastTwoParts);
    }
    
    extractDomainWithCCTLD(parts) {
        // For ccTLDs, take the domain name + ccTLD
        // e.g., businessmirror.com.ph -> businessmirror.com.ph
        // e.g., news.mb.com.ph -> mb.com.ph
        
        if (parts.length === 3) {
            // domain.com.ph
            return parts.join('.');
        } else if (parts.length >= 4) {
            // subdomain.domain.com.ph
            // Check if subdomain should be preserved
            const subdomain = parts[0];
            if (this.shouldPreserveSubdomainForCCTLD(subdomain)) {
                return parts.slice(-4).join('.');
            }
            return parts.slice(-3).join('.');
        }
        
        return parts.join('.');
    }
    
    shouldPreserveSubdomain(parts) {
        if (parts.length < 3) return false;
        
        const subdomain = parts[0];
        const preserveSubdomains = [
            'news', 'www', 'm', 'mobile', 'edition', 
            'international', 'cnn', 'bbc', 'sports'
        ];
        
        // Don't preserve common news subdomains unless it's a major organization
        if (preserveSubdomains.includes(subdomain)) {
            const domain = parts.slice(-2).join('.');
            const majorOrgs = [
                'abs-cbn.com', 'gmanetwork.com', 'rappler.com',
                'cnn.com', 'bbc.com', 'reuters.com', 'nytimes.com'
            ];
            return majorOrgs.includes(domain);
        }
        
        return false;
    }
    
    shouldPreserveSubdomainForCCTLD(subdomain) {
        // For ccTLD domains, be more conservative about preserving subdomains
        const preserveSubdomains = ['news', 'www'];
        return preserveSubdomains.includes(subdomain);
    }
    
    groupWebsitesByDomain(websites) {
        const groups = {};
        
        websites.forEach(website => {
            const rootDomain = this.extractRootDomain(website.url);
            
            if (!groups[rootDomain]) {
                groups[rootDomain] = {
                    domain: rootDomain,
                    displayName: this.getDomainDisplayName(rootDomain),
                    websites: [],
                    totalArticles: 0,
                    isExpanded: false // Start collapsed for cleaner view
                };
            }
            
            groups[rootDomain].websites.push(website);
            groups[rootDomain].totalArticles += (website.article_count || website.articleCount || 0);
        });
        
        // Sort websites within each group by article count
        Object.values(groups).forEach(group => {
            group.websites.sort((a, b) => 
                (b.article_count || b.articleCount || 0) - (a.article_count || a.articleCount || 0)
            );
        });
        
        // Sort groups by total articles (descending) and then by domain name
        const sortedGroups = Object.values(groups).sort((a, b) => {
            if (b.totalArticles !== a.totalArticles) {
                return b.totalArticles - a.totalArticles;
            }
            return a.domain.localeCompare(b.domain);
        });
        
        return sortedGroups;
    }
    
    getDomainDisplayName(domain) {
        // Convert domain to friendly display name
        const domainMap = {
            'cnn.com': 'CNN',
            'bbc.com': 'BBC',
            'reuters.com': 'Reuters',
            'nytimes.com': 'New York Times',
            'washingtonpost.com': 'Washington Post',
            'theguardian.com': 'The Guardian',
            'abc.net.au': 'ABC News',
            'yahoo.com': 'Yahoo News',
            'foxnews.com': 'Fox News',
            'nbcnews.com': 'NBC News',
            'cbsnews.com': 'CBS News',
            'npr.org': 'NPR',
            'apnews.com': 'Associated Press',
            'usatoday.com': 'USA Today',
            'wsj.com': 'Wall Street Journal',
            'bloomberg.com': 'Bloomberg',
            'time.com': 'Time',
            'newsweek.com': 'Newsweek',
            'huffpost.com': 'HuffPost',
            'politico.com': 'Politico',
            
            // Philippine news organizations
            'rappler.com': 'Rappler',
            'abs-cbn.com': 'ABS-CBN',
            'gmanetwork.com': 'GMA News',
            'inquirer.net': 'Philippine Daily Inquirer',
            'philstar.com': 'Philippine Star',
            'manilabulletin.com.ph': 'Manila Bulletin',
            'businessworld.com.ph': 'BusinessWorld',
            'businessmirror.com.ph': 'BusinessMirror',
            'manilastandard.net': 'Manila Standard',
            'manilastandardtoday.com': 'Manila Standard Today',
            'tribune.net.ph': 'Tribune',
            'tribuneonline.org': 'Tribune',
            'sunstar.com.ph': 'SunStar',
            'pna.gov.ph': 'Philippine News Agency',
            'malaya.com.ph': 'Malaya Business Insight',
            'remate.ph': 'Remate',
            'tempo.com.ph': 'Tempo',
            'journal.com.ph': 'The Journal',
            'manilatimes.net': 'Manila Times',
            'journal.ph': 'The Journal Online'
        };
        
        if (domainMap[domain]) {
            return domainMap[domain];
        }
        
        // Handle special patterns for Philippine domains
        if (domain.endsWith('.com.ph') || domain.endsWith('.net.ph') || domain.endsWith('.org.ph')) {
            const baseName = domain.split('.')[0];
            return this.capitalizeWords(baseName.replace(/[-_]/g, ' '));
        }
        
        // Generate display name from domain
        let name = domain
            .replace('.com', '')
            .replace('.org', '')
            .replace('.net', '')
            .replace('.ph', '')
            .replace('.co.uk', '')
            .replace('.com.au', '');
            
        return this.capitalizeWords(name.replace(/[-_.]/g, ' '));
    }
    
    capitalizeWords(str) {
        return str.split('.').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }
    
    toggleDomainGroup(domain) {
        const groupElement = document.querySelector(`[data-domain-group="${domain}"]`);
        const websitesContainer = groupElement?.querySelector('.domain-websites');
        const toggleIcon = groupElement?.querySelector('.domain-toggle-icon');
        
        if (!groupElement || !websitesContainer || !toggleIcon) return;
        
        const isExpanded = websitesContainer.style.display !== 'none';
        
        if (isExpanded) {
            websitesContainer.style.display = 'none';
            toggleIcon.classList.remove('bi-chevron-down');
            toggleIcon.classList.add('bi-chevron-right');
            groupElement.dataset.expanded = 'false';
        } else {
            websitesContainer.style.display = 'block';
            toggleIcon.classList.remove('bi-chevron-right');
            toggleIcon.classList.add('bi-chevron-down');
            groupElement.dataset.expanded = 'true';
        }
    }
    
    setWebsiteViewMode(mode) {
        this.websiteViewMode = mode;
        localStorage.setItem('newsTracker.websiteViewMode', mode);
        
        // Update control visibility
        const expandBtn = document.getElementById('expandAllDomainsBtn');
        const collapseBtn = document.getElementById('collapseAllDomainsBtn');
        
        if (mode === 'grouped') {
            expandBtn?.classList.remove('hidden');
            collapseBtn?.classList.remove('hidden');
        } else {
            expandBtn?.classList.add('hidden');
            collapseBtn?.classList.add('hidden');
        }
        
        this.renderTrackedWebsites();
    }
    
    expandAllDomains() {
        const domainGroups = document.querySelectorAll('[data-domain-group]');
        domainGroups.forEach(group => {
            const domain = group.dataset.domainGroup;
            const websitesContainer = group.querySelector('.domain-websites');
            const toggleIcon = group.querySelector('.domain-toggle-icon');
            
            if (websitesContainer && websitesContainer.style.display === 'none') {
                this.toggleDomainGroup(domain);
            }
        });
    }
    
    collapseAllDomains() {
        const domainGroups = document.querySelectorAll('[data-domain-group]');
        domainGroups.forEach(group => {
            const domain = group.dataset.domainGroup;
            const websitesContainer = group.querySelector('.domain-websites');
            const toggleIcon = group.querySelector('.domain-toggle-icon');
            
            if (websitesContainer && websitesContainer.style.display !== 'none') {
                this.toggleDomainGroup(domain);
            }
        });
    }
    
    renderTrackedWebsitesSimple() {
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
        
        // Sort websites by article count for simple view
        const sortedWebsites = [...this.trackedWebsites].sort((a, b) => 
            (b.article_count || b.articleCount || 0) - (a.article_count || a.articleCount || 0)
        );
        
        container.innerHTML = sortedWebsites.map(website => `
            <div class="flex items-center justify-between p-4 bg-gray-50 rounded-lg border hover:bg-gray-100 transition-colors duration-150">
                <div class="flex-1">
                    <h4 class="font-medium text-gray-800">${this.escapeHtml(website.name)}</h4>
                    <p class="text-sm text-gray-600 break-all">${website.url}</p>
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
                        class="text-red-600 hover:text-red-800 p-1 rounded hover:bg-red-50 transition-colors duration-150"
                        title="Remove website"
                    >
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
        `).join('');
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
    
    // Real-time activity notifications
    showRealtimeActivity(message) {
        const activityElement = document.getElementById('realtimeActivity');
        const messageElement = document.getElementById('activityMessage');
        
        if (activityElement && messageElement) {
            messageElement.textContent = message;
            activityElement.classList.remove('hidden');
        }
    }
    
    hideRealtimeActivity() {
        const activityElement = document.getElementById('realtimeActivity');
        if (activityElement) {
            activityElement.classList.add('hidden');
        }
    }
    
    updateRealtimeActivity(message) {
        const messageElement = document.getElementById('activityMessage');
        if (messageElement) {
            messageElement.textContent = message;
        }
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
                
                // Load prediction metrics if available
                if (data.prediction_metrics) {
                    this.predictionMetrics = data.prediction_metrics;
                    this.displayPredictionMetrics();
                } else {
                    // Fetch metrics separately if not included
                    this.fetchPredictionMetrics();
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
        // Load auto-fetch preferences from localStorage
        const autoFetchEnabled = localStorage.getItem('newsTracker.autoFetchEnabled') === 'true';
        const autoFetchInterval = parseInt(localStorage.getItem('newsTracker.autoFetchInterval')) || 30;
        const autoFetchStats = JSON.parse(localStorage.getItem('newsTracker.autoFetchStats') || '{"totalRuns":0,"articlesFound":0,"lastRun":null,"nextRun":null}');
        
        this.autoFetchEnabled = autoFetchEnabled;
        this.autoFetchIntervalMinutes = autoFetchInterval;
        this.autoFetchStats = autoFetchStats;
        
        // Update UI to reflect current state
        const toggle = document.getElementById('autoFetchToggle');
        const intervalSelect = document.getElementById('autoFetchInterval');
        
        if (toggle) {
            toggle.checked = autoFetchEnabled;
            this.updateToggleAppearance(toggle, autoFetchEnabled);
        }
        
        if (intervalSelect) {
            intervalSelect.value = autoFetchInterval.toString();
        }
        
        this.updateAutoFetchStatus();
        this.updateAutoFetchStats();
        
        // Start auto-fetch if enabled
        if (autoFetchEnabled && this.trackedWebsites.length > 0) {
            this.startAutoFetch();
        }
    }
    
    toggleAutoFetch(enabled) {
        console.log('toggleAutoFetch called with enabled:', enabled);
        this.autoFetchEnabled = enabled;
        
        // Save preference to localStorage
        localStorage.setItem('newsTracker.autoFetchEnabled', enabled.toString());
        
        // Update toggle appearance
        const toggle = document.getElementById('autoFetchToggle');
        if (toggle) {
            console.log('Updating toggle appearance for enabled:', enabled);
            this.updateToggleAppearance(toggle, enabled);
        } else {
            console.error('Toggle element not found');
        }
        
        this.updateAutoFetchStatus();
        
        if (enabled) {
            if (this.trackedWebsites.length === 0) {
                this.showError('Add websites to track before enabling auto-fetch');
                if (toggle) {
                    toggle.checked = false;
                }
                this.autoFetchEnabled = false;
                localStorage.setItem('newsTracker.autoFetchEnabled', 'false');
                this.updateToggleAppearance(toggle, false);
                this.updateAutoFetchStatus();
                return;
            }
            this.startAutoFetch();
            this.showSuccess(`Auto-fetch enabled. Articles will be fetched every ${this.autoFetchIntervalMinutes} minutes.`);
        } else {
            this.stopAutoFetch();
            this.showInfo('Auto-fetch disabled');
        }
    }
    
    startAutoFetch() {
        // Clear any existing interval
        this.stopAutoFetch();
        
        // Set up new interval
        const intervalMs = this.autoFetchIntervalMinutes * 60 * 1000;
        this.autoFetchInterval = setInterval(() => {
            this.autoFetchArticles();
        }, intervalMs);
        
        console.log(`Auto-fetch started with ${this.autoFetchIntervalMinutes} minute interval`);
    }
    
    stopAutoFetch() {
        if (this.autoFetchInterval) {
            clearInterval(this.autoFetchInterval);
            this.autoFetchInterval = null;
        }
        console.log('Auto-fetch stopped');
    }
    
    async autoFetchArticles() {
        try {
            console.log('Auto-fetch: Fetching articles...');
            const initialCount = this.articleQueue.length;
            await this.fetchAllArticles();
            const newCount = this.articleQueue.length;
            const foundArticles = newCount - initialCount;
            
            // Update stats
            this.autoFetchStats.totalRuns++;
            this.autoFetchStats.articlesFound += foundArticles;
            localStorage.setItem('newsTracker.autoFetchStats', JSON.stringify(this.autoFetchStats));
            this.updateAutoFetchStats();
            
            console.log(`Auto-fetch completed: Found ${foundArticles} new articles`);
            
            // Auto-indexing after fetch if enabled
            if (this.autoIndexEnabled && foundArticles > 0) {
                console.log('Auto-indexing: Checking for high confidence articles...');
                this.showRealtimeActivity('Auto-indexing high confidence articles...');
                
                try {
                    const response = await fetch('/api/news-tracker/auto-index-high-confidence', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            confidence_threshold: this.autoIndexThreshold,
                            batch_size: this.autoIndexBatchSize
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success && result.articles_indexed > 0) {
                        console.log(`Auto-indexing completed: ${result.articles_indexed} articles indexed`);
                        this.updateRealtimeActivity(`Auto-indexed ${result.articles_indexed} articles, updating UI...`);
                        
                        // Update auto-indexing stats
                        this.autoIndexStats.totalIndexed += result.articles_indexed;
                        this.autoIndexStats.lastBatchSize = result.articles_indexed;
                        this.autoIndexStats.lastBatchTime = new Date().toISOString();
                        this.autoIndexStats.successRate = result.stats?.success_rate || 0;
                        this.updateAutoIndexStatsDisplay();
                        
                        // Refresh the UI to show updated verification status
                        await this.loadTrackedWebsites();
                        this.updateHighConfidenceQueue();
                        this.updateStatistics();
                        
                        // Hide real-time activity after a brief delay
                        setTimeout(() => this.hideRealtimeActivity(), 3000);
                        
                    } else {
                        console.log('Auto-indexing: No high confidence articles to index');
                        this.hideRealtimeActivity();
                    }
                } catch (indexError) {
                    console.error('Auto-indexing error during auto-fetch:', indexError);
                    this.hideRealtimeActivity();
                    // Don't break auto-fetch if indexing fails
                }
            }
            
        } catch (error) {
            console.error('Auto-fetch error:', error);
            // Don't show error to user for auto-fetch to avoid spam
        }
    }
    
    updateToggleAppearance(toggle, enabled) {
        console.log('updateToggleAppearance called with enabled:', enabled);
        const toggleBg = toggle.nextElementSibling;
        const toggleDot = toggleBg?.nextElementSibling;
        
        console.log('Toggle elements found:', { toggleBg: !!toggleBg, toggleDot: !!toggleDot });
        
        if (enabled) {
            toggleBg?.classList.remove('bg-gray-200');
            toggleBg?.classList.add('bg-green-400');
            toggleDot?.classList.remove('-left-0.5');
            toggleDot?.classList.add('translate-x-6');
            console.log('Applied enabled styles');
        } else {
            toggleBg?.classList.remove('bg-green-400');
            toggleBg?.classList.add('bg-gray-200');
            toggleDot?.classList.remove('translate-x-6');
            toggleDot?.classList.add('-left-0.5');
            console.log('Applied disabled styles');
        }
    }
    
    updateAutoFetchStatus() {
        const statusElement = document.getElementById('autoFetchStatus');
        if (!statusElement) return;
        
        if (this.autoFetchEnabled) {
            const nextFetch = new Date(Date.now() + (this.autoFetchIntervalMinutes * 60 * 1000));
            statusElement.innerHTML = `
                <div class="flex items-center">
                    <i class="bi bi-play-circle text-green-500 mr-2"></i>
                    <span class="text-sm text-green-700">
                        Auto-fetch is running every ${this.autoFetchIntervalMinutes} minutes
                        <br>
                        <span class="text-xs">Next fetch: ${nextFetch.toLocaleTimeString()}</span>
                    </span>
                </div>
            `;
        } else {
            statusElement.innerHTML = `
                <div class="flex items-center">
                    <i class="bi bi-pause-circle text-gray-500 mr-2"></i>
                    <span class="text-sm text-gray-600">Auto-fetch is currently disabled</span>
                </div>
            `;
        }
    }
    
    updateLastFetchInfo() {
        const lastFetchInfo = document.getElementById('lastFetchInfo');
        if (lastFetchInfo) {
            const now = new Date();
            lastFetchInfo.innerHTML = `
                <div class="flex items-center justify-between">
                    <div class="flex items-center">
                        <i class="bi bi-info-circle text-blue-600 mr-2"></i>
                        <span class="text-sm text-blue-800">Last fetch: ${now.toLocaleString()}</span>
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="text-xs text-blue-600">
                            <i class="bi bi-clock mr-1"></i>
                            <span id="timeSinceLastFetch">just now</span>
                        </span>
                    </div>
                </div>
            `;
            
            // Update auto-fetch stats
            this.autoFetchStats.lastRun = now.toISOString();
            localStorage.setItem('newsTracker.autoFetchStats', JSON.stringify(this.autoFetchStats));
            this.updateAutoFetchStats();
        }
        
        // Update auto-fetch status to show next fetch time
        if (this.autoFetchEnabled) {
            this.updateAutoFetchStatus();
        }
    }
    
    updateAutoFetchInterval(minutes) {
        this.autoFetchIntervalMinutes = minutes;
        localStorage.setItem('newsTracker.autoFetchInterval', minutes.toString());
        
        // Restart auto-fetch with new interval if it's currently running
        if (this.autoFetchEnabled && this.autoFetchInterval) {
            this.startAutoFetch();
        }
        
        this.updateAutoFetchStatus();
        this.showInfo(`Auto-fetch interval updated to ${minutes} minutes`);
    }
    
    updateAutoFetchStats() {
        const totalRunsEl = document.getElementById('autoFetchTotalRuns');
        const articlesFoundEl = document.getElementById('autoFetchArticlesFound');
        const lastRunEl = document.getElementById('autoFetchLastRun');
        const nextRunEl = document.getElementById('autoFetchNextRun');
        
        if (totalRunsEl) totalRunsEl.textContent = this.autoFetchStats.totalRuns;
        if (articlesFoundEl) articlesFoundEl.textContent = this.autoFetchStats.articlesFound;
        if (lastRunEl) {
            lastRunEl.textContent = this.autoFetchStats.lastRun 
                ? new Date(this.autoFetchStats.lastRun).toLocaleString()
                : 'Never';
        }
        if (nextRunEl) {
            if (this.autoFetchEnabled && this.autoFetchInterval) {
                const nextRun = new Date(Date.now() + (this.autoFetchIntervalMinutes * 60 * 1000));
                nextRunEl.textContent = nextRun.toLocaleTimeString();
            } else {
                nextRunEl.textContent = '-';
            }
        }
    }
    
    async testAutoFetch() {
        if (this.trackedWebsites.length === 0) {
            this.showError('Add websites to track before testing auto-fetch');
            return;
        }
        
        this.showInfo('Running auto-fetch test...');
        
        try {
            const initialCount = this.articleQueue.length;
            await this.autoFetchArticles();
            const newCount = this.articleQueue.length;
            const foundArticles = newCount - initialCount;
            
            this.showSuccess(`Auto-fetch test completed! Found ${foundArticles} new articles.`);
            
            // Update stats
            this.autoFetchStats.totalRuns++;
            this.autoFetchStats.articlesFound += foundArticles;
            localStorage.setItem('newsTracker.autoFetchStats', JSON.stringify(this.autoFetchStats));
            this.updateAutoFetchStats();
            
        } catch (error) {
            console.error('Auto-fetch test error:', error);
            this.showError('Auto-fetch test failed. Check console for details.');
        }
    }
    
    quickSetupAutoFetch() {
        if (this.trackedWebsites.length === 0) {
            this.showError('Add websites to track before setting up auto-fetch');
            return;
        }
        
        // Set optimal settings
        this.autoFetchIntervalMinutes = 30; // 30 minutes
        this.autoFetchEnabled = true;
        
        // Update UI
        const toggle = document.getElementById('autoFetchToggle');
        const intervalSelect = document.getElementById('autoFetchInterval');
        
        if (toggle) {
            toggle.checked = true;
            this.updateToggleAppearance(toggle, true);
        }
        
        if (intervalSelect) {
            intervalSelect.value = '30';
        }
        
        // Save settings
        localStorage.setItem('newsTracker.autoFetchEnabled', 'true');
        localStorage.setItem('newsTracker.autoFetchInterval', '30');
        
        // Also enable auto-indexing with optimal settings
        this.autoIndexEnabled = true;
        this.autoIndexThreshold = 0.95; // 95% confidence
        this.autoIndexBatchSize = 100;
        
        const autoIndexToggle = document.getElementById('autoIndexToggle');
        if (autoIndexToggle) {
            autoIndexToggle.checked = true;
            this.updateToggleAppearance(autoIndexToggle, true);
        }
        
        const autoIndexSettings = document.getElementById('autoIndexSettings');
        if (autoIndexSettings) {
            autoIndexSettings.style.display = 'block';
        }
        
        const autoIndexStats = document.getElementById('autoIndexStats');
        if (autoIndexStats) {
            autoIndexStats.style.display = 'block';
        }
        
        // Save auto-indexing settings
        localStorage.setItem('newsTracker.autoIndexEnabled', 'true');
        localStorage.setItem('newsTracker.autoIndexThreshold', '0.95');
        localStorage.setItem('newsTracker.autoIndexBatchSize', '100');
        
        // Start auto-fetch
        this.startAutoFetch();
        this.updateAutoFetchStatus();
        this.updateAutoFetchStats();
        this.updateAutoIndexStatsDisplay();
        
        this.showSuccess('Auto-fetch and auto-indexing have been set up with optimal settings (30 min interval, 95% confidence threshold)');
    }
    
    resetAutoFetchSettings() {
        // Stop auto-fetch
        this.stopAutoFetch();
        
        // Reset to defaults
        this.autoFetchEnabled = false;
        this.autoFetchIntervalMinutes = 30;
        this.autoFetchStats = {
            totalRuns: 0,
            articlesFound: 0,
            lastRun: null,
            nextRun: null
        };
        
        // Update UI
        const toggle = document.getElementById('autoFetchToggle');
        const intervalSelect = document.getElementById('autoFetchInterval');
        
        if (toggle) {
            toggle.checked = false;
            this.updateToggleAppearance(toggle, false);
        }
        
        if (intervalSelect) {
            intervalSelect.value = '30';
        }
        
        // Clear localStorage
        localStorage.removeItem('newsTracker.autoFetchEnabled');
        localStorage.removeItem('newsTracker.autoFetchInterval');
        localStorage.removeItem('newsTracker.autoFetchStats');
        
        // Reset auto-indexing settings too
        this.autoIndexEnabled = false;
        this.autoIndexThreshold = 0.95;
        this.autoIndexBatchSize = 100;
        this.autoIndexStats = {
            totalIndexed: 0,
            lastBatchSize: 0,
            lastBatchTime: null,
            successRate: 0,
            highConfidenceQueue: 0
        };
        
        const autoIndexToggle = document.getElementById('autoIndexToggle');
        if (autoIndexToggle) {
            autoIndexToggle.checked = false;
            this.updateToggleAppearance(autoIndexToggle, false);
        }
        
        const autoIndexSettings = document.getElementById('autoIndexSettings');
        if (autoIndexSettings) {
            autoIndexSettings.style.display = 'none';
        }
        
        const autoIndexStats = document.getElementById('autoIndexStats');
        if (autoIndexStats) {
            autoIndexStats.style.display = 'none';
        }
        
        // Clear auto-indexing localStorage
        localStorage.removeItem('newsTracker.autoIndexEnabled');
        localStorage.removeItem('newsTracker.autoIndexThreshold');
        localStorage.removeItem('newsTracker.autoIndexBatchSize');
        
        // Update displays
        this.updateAutoFetchStatus();
        this.updateAutoFetchStats();
        this.updateAutoIndexStatsDisplay();
        
        this.showInfo('Auto-fetch and auto-indexing settings have been reset to defaults');
    }

    // Auto-indexing methods
    toggleAutoIndex(enabled) {
        this.autoIndexEnabled = enabled;
        
        // Update toggle appearance
        const toggle = document.getElementById('autoIndexToggle');
        if (toggle) {
            this.updateToggleAppearance(toggle, enabled);
        }
        
        // Show/hide auto-indexing settings
        const settings = document.getElementById('autoIndexSettings');
        if (settings) {
            settings.style.display = enabled ? 'block' : 'none';
        }
        
        // Show/hide auto-indexing stats
        const stats = document.getElementById('autoIndexStats');
        if (stats) {
            stats.style.display = enabled ? 'block' : 'none';
        }
        
        // Save preference
        localStorage.setItem('newsTracker.autoIndexEnabled', enabled.toString());
        
        this.showInfo(`Auto-indexing ${enabled ? 'enabled' : 'disabled'}`);
    }

    updateAutoIndexThreshold(threshold) {
        this.autoIndexThreshold = threshold / 100; // Convert percentage to decimal
        localStorage.setItem('newsTracker.autoIndexThreshold', this.autoIndexThreshold.toString());
        this.updateHighConfidenceQueue();
        this.showInfo(`Auto-indexing threshold updated to ${threshold}%`);
    }

    updateAutoIndexBatchSize(batchSize) {
        this.autoIndexBatchSize = batchSize;
        localStorage.setItem('newsTracker.autoIndexBatchSize', batchSize.toString());
        this.showInfo(`Auto-indexing batch size updated to ${batchSize} articles`);
    }

    async triggerAutoIndex() {
        try {
            this.showLoading('Finding high confidence articles to index...');
            this.showRealtimeActivity('Finding high confidence articles to index...');
            
            // Use the backend API endpoint instead of frontend processing
            const response = await fetch('/api/news-tracker/auto-index-high-confidence', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    confidence_threshold: this.autoIndexThreshold,
                    batch_size: this.autoIndexBatchSize
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.updateRealtimeActivity('Updating verification status and refreshing UI...');
                
                // Update stats
                this.autoIndexStats.totalIndexed += result.articles_indexed;
                this.autoIndexStats.lastBatchSize = result.articles_indexed;
                this.autoIndexStats.lastBatchTime = new Date().toISOString();
                this.autoIndexStats.successRate = result.stats?.success_rate || 0;
                this.updateAutoIndexStatsDisplay();
                
                // Refresh the article queue to show updated verification status
                await this.loadTrackedWebsites();
                
                // Update high confidence queue count
                this.updateHighConfidenceQueue();
                
                // Update statistics to reflect changes
                this.updateStatistics();
                
                this.showSuccess(result.message);
                
                // Hide real-time activity after a brief delay
                setTimeout(() => this.hideRealtimeActivity(), 2000);
                
            } else {
                this.hideRealtimeActivity();
                this.showError(result.error || 'Failed to trigger auto-indexing');
            }
            
        } catch (error) {
            console.error('Auto-indexing error:', error);
            this.hideRealtimeActivity();
            this.showError('Failed to trigger auto-indexing');
        } finally {
            this.hideLoading();
        }
    }

    getHighConfidenceArticles() {
        return this.articleQueue.filter(article => {
            const confidence = article.confidence || 0;
            return confidence >= this.autoIndexThreshold && 
                   !article.verified && 
                   article.is_news_prediction !== false;
        });
    }

    updateHighConfidenceQueue() {
        const highConfidenceArticles = this.getHighConfidenceArticles();
        this.autoIndexStats.highConfidenceQueue = highConfidenceArticles.length;
        
        const queueElement = document.getElementById('highConfidenceQueue');
        if (queueElement) {
            queueElement.textContent = highConfidenceArticles.length;
        }
    }

    updateAutoIndexStatsDisplay() {
        const autoIndexedCount = document.getElementById('autoIndexedCount');
        const lastAutoIndexBatch = document.getElementById('lastAutoIndexBatch');
        const highConfidenceQueue = document.getElementById('highConfidenceQueue');
        const autoIndexSuccessRate = document.getElementById('autoIndexSuccessRate');
        
        if (autoIndexedCount) autoIndexedCount.textContent = this.autoIndexStats.totalIndexed;
        if (lastAutoIndexBatch) lastAutoIndexBatch.textContent = this.autoIndexStats.lastBatchSize;
        if (highConfidenceQueue) highConfidenceQueue.textContent = this.autoIndexStats.highConfidenceQueue;
        
        // Use the success rate from backend if available
        if (autoIndexSuccessRate) {
            const successRate = this.autoIndexStats.successRate || 0;
            autoIndexSuccessRate.textContent = `${successRate}%`;
        }
        
        // Update last batch time if available
        const lastAutoIndexTime = document.getElementById('lastAutoIndexTime');
        if (lastAutoIndexTime && this.autoIndexStats.lastBatchTime) {
            const timeStr = new Date(this.autoIndexStats.lastBatchTime).toLocaleString();
            lastAutoIndexTime.textContent = timeStr;
        }
    }

    chunkArray(array, chunkSize) {
        const chunks = [];
        for (let i = 0; i < array.length; i += chunkSize) {
            chunks.push(array.slice(i, i + chunkSize));
        }
        return chunks;
    }

    loadAutoIndexPreferences() {
        // Load saved auto-indexing preferences
        const savedEnabled = localStorage.getItem('newsTracker.autoIndexEnabled');
        const savedThreshold = localStorage.getItem('newsTracker.autoIndexThreshold');
        const savedBatchSize = localStorage.getItem('newsTracker.autoIndexBatchSize');
        
        if (savedEnabled !== null) {
            this.autoIndexEnabled = savedEnabled === 'true';
            const toggle = document.getElementById('autoIndexToggle');
            if (toggle) {
                toggle.checked = this.autoIndexEnabled;
                this.updateToggleAppearance(toggle, this.autoIndexEnabled);
            }
        }
        
        if (savedThreshold !== null) {
            this.autoIndexThreshold = parseFloat(savedThreshold);
            const thresholdSelect = document.getElementById('autoIndexThreshold');
            if (thresholdSelect) {
                thresholdSelect.value = Math.round(this.autoIndexThreshold * 100).toString();
            }
        }
        
        if (savedBatchSize !== null) {
            this.autoIndexBatchSize = parseInt(savedBatchSize);
            const batchSizeSelect = document.getElementById('autoIndexBatchSize');
            if (batchSizeSelect) {
                batchSizeSelect.value = this.autoIndexBatchSize.toString();
            }
        }
        
        // Update UI visibility
        const settings = document.getElementById('autoIndexSettings');
        if (settings) {
            settings.style.display = this.autoIndexEnabled ? 'block' : 'none';
        }
        
        const stats = document.getElementById('autoIndexStats');
        if (stats) {
            stats.style.display = this.autoIndexEnabled ? 'block' : 'none';
        }
        
        this.updateHighConfidenceQueue();
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
        
        // Get currently visible unverified articles from the DOM (on current page)
        const unverifiedElements = Array.from(document.querySelectorAll('[data-article-id]')).filter(element => {
            const articleId = element.dataset.articleId;
            const article = this.articleQueue.find(a => a.id == articleId);
            return article && !article.verified;
        });
        
        if (unverifiedElements.length === 0) {
            this.showError('No unverified articles available for selection on this page');
            return;
        }
        
        // Clear any existing selections
        this.clearBatchSelection();
        
        // Select up to batchSize unverified articles
        const elementsToSelect = unverifiedElements.slice(0, batchSize);
        
        elementsToSelect.forEach(element => {
            this.selectArticle(element);
        });
        
        this.updateSelectionCount();
        this.updateBatchActionButtons();
        
        // Show informative message about selection
        const selectedCount = elementsToSelect.length;
        if (selectedCount < batchSize) {
            this.showInfo(`Selected ${selectedCount} of ${batchSize} requested articles (${selectedCount} available on current page)`);
        } else {
            this.showInfo(`Selected ${selectedCount} articles for batch verification`);
        }
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
    
    updateDisplayedArticleCount() {
        const displayedCountElement = document.getElementById('displayedArticleCount');
        if (displayedCountElement) {
            displayedCountElement.textContent = this.itemsPerPage;
        }
    }
    
    async fetchPredictionMetrics() {
        try {
            const response = await fetch('/api/news-tracker/prediction-metrics');
            const data = await response.json();
            
            if (data.success && data.metrics.status === 'success') {
                this.predictionMetrics = data.metrics;
                this.displayPredictionMetrics();
            } else if (data.metrics && data.metrics.status === 'insufficient_data') {
                this.displayInsufficientDataMessage(data.metrics.total_verified);
            }
        } catch (error) {
            console.error('Error fetching prediction metrics:', error);
        }
    }
    
    displayPredictionMetrics() {
        if (!this.predictionMetrics) return;
        
        const metrics = this.predictionMetrics;
        
        // Update existing accuracy display with comprehensive metrics
        this.updateMetricsDisplay(metrics);
        
        // Create or update advanced metrics dashboard
        this.createAdvancedMetricsDashboard(metrics);
    }
    
    updateMetricsDisplay(metrics) {
        // Update basic stats with enhanced information
        const basicMetrics = metrics.basic_metrics;
        const confusionMatrix = metrics.confusion_matrix;
        
        // Add metrics info to existing statistics
        const accuracyElements = document.querySelectorAll('[data-metric="accuracy"]');
        accuracyElements.forEach(el => {
            el.textContent = `${Math.round(basicMetrics.accuracy * 100)}%`;
            el.title = `Precision: ${Math.round(basicMetrics.precision * 100)}%, Recall: ${Math.round(basicMetrics.recall * 100)}%, F1: ${Math.round(basicMetrics.f1_score * 100)}%`;
        });
        
        // Update confusion matrix display if it exists
        this.updateConfusionMatrixDisplay(confusionMatrix);
    }
    
    createAdvancedMetricsDashboard(metrics) {
        let metricsContainer = document.getElementById('advancedMetricsContainer');
        
        if (!metricsContainer) {
            // Create metrics container if it doesn't exist
            const statisticsSection = document.querySelector('.bg-white.rounded-lg.shadow-md.p-6.mb-8');
            if (statisticsSection) {
                metricsContainer = document.createElement('div');
                metricsContainer.id = 'advancedMetricsContainer';
                metricsContainer.className = 'bg-white rounded-lg shadow-md p-6 mb-8';
                statisticsSection.parentNode.insertBefore(metricsContainer, statisticsSection.nextSibling);
            } else {
                return; // Can't find a place to add it
            }
        }
        
        const html = `
            <h3 class="text-xl font-semibold text-gray-800 mb-4">
                <i class="bi bi-graph-up-arrow"></i>
                Advanced Prediction Metrics
            </h3>
            
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
                <!-- Basic Metrics -->
                <div class="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4 border border-blue-200">
                    <h4 class="font-semibold text-blue-800 mb-3">
                        <i class="bi bi-bullseye mr-1"></i>
                        Classification Metrics
                    </h4>
                    <div class="space-y-2 text-sm">
                        <div class="flex justify-between">
                            <span class="text-blue-700">Accuracy:</span>
                            <span class="font-semibold text-blue-800">${Math.round(metrics.basic_metrics.accuracy * 100)}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-blue-700">Precision:</span>
                            <span class="font-semibold text-blue-800">${Math.round(metrics.basic_metrics.precision * 100)}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-blue-700">Recall:</span>
                            <span class="font-semibold text-blue-800">${Math.round(metrics.basic_metrics.recall * 100)}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-blue-700">F1-Score:</span>
                            <span class="font-semibold text-blue-800">${Math.round(metrics.basic_metrics.f1_score * 100)}%</span>
                        </div>
                    </div>
                </div>
                
                <!-- Advanced Metrics -->
                <div class="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-4 border border-green-200">
                    <h4 class="font-semibold text-green-800 mb-3">
                        <i class="bi bi-speedometer2 mr-1"></i>
                        Advanced Metrics
                    </h4>
                    <div class="space-y-2 text-sm">
                        <div class="flex justify-between">
                            <span class="text-green-700">MCC:</span>
                            <span class="font-semibold text-green-800">${metrics.advanced_metrics.matthews_correlation_coefficient.toFixed(3)}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-green-700">AUC-ROC:</span>
                            <span class="font-semibold text-green-800">${metrics.advanced_metrics.auc_roc.toFixed(3)}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-green-700">Sensitivity:</span>
                            <span class="font-semibold text-green-800">${Math.round(metrics.advanced_metrics.true_positive_rate * 100)}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-green-700">Specificity:</span>
                            <span class="font-semibold text-green-800">${Math.round(metrics.advanced_metrics.true_negative_rate * 100)}%</span>
                        </div>
                    </div>
                </div>
                
                <!-- Calibration Quality -->
                <div class="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-4 border border-purple-200">
                    <h4 class="font-semibold text-purple-800 mb-3">
                        <i class="bi bi-sliders mr-1"></i>
                        Prediction Quality
                    </h4>
                    <div class="space-y-2 text-sm">
                        <div class="flex justify-between">
                            <span class="text-purple-700">Calibration Error:</span>
                            <span class="font-semibold text-purple-800">${(metrics.calibration_metrics.expected_calibration_error * 100).toFixed(1)}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-purple-700">High Conf. Acc.:</span>
                            <span class="font-semibold text-purple-800">${this.calculateHighConfidenceAccuracy(metrics)}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-purple-700">Consistency:</span>
                            <span class="font-semibold text-purple-800">${metrics.temporal_consistency.is_consistent ? 'Good' : 'Variable'}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-purple-700">Total Verified:</span>
                            <span class="font-semibold text-purple-800">${metrics.total_verified_articles}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Confusion Matrix -->
            <div class="mb-6">
                <h4 class="font-semibold text-gray-800 mb-3">
                    <i class="bi bi-grid-3x3-gap mr-1"></i>
                    Confusion Matrix
                </h4>
                <div class="bg-gray-50 rounded-lg p-4">
                    ${this.renderConfusionMatrix(metrics.confusion_matrix)}
                </div>
            </div>
            
            <!-- Calibration Chart -->
            ${metrics.calibration_metrics.calibration_bins.length > 0 ? this.renderCalibrationChart(metrics.calibration_metrics) : ''}
            
            <!-- Temporal Consistency -->
            ${metrics.temporal_consistency.weekly_accuracies.length > 0 ? this.renderTemporalConsistency(metrics.temporal_consistency) : ''}
        `;
        
        metricsContainer.innerHTML = html;
    }
    
    calculateHighConfidenceAccuracy(metrics) {
        const quality = metrics.prediction_quality;
        if (quality.high_confidence_total === 0) return 'N/A';
        return Math.round((quality.high_confidence_correct / quality.high_confidence_total) * 100);
    }
    
    renderConfusionMatrix(cm) {
        return `
            <div class="grid grid-cols-2 gap-2 max-w-md mx-auto">
                <div class="text-center">
                    <div class="text-xs text-gray-600 mb-1">Predicted</div>
                    <div class="grid grid-cols-3 gap-1 text-xs">
                        <div></div>
                        <div class="font-medium">News</div>
                        <div class="font-medium">Not News</div>
                    </div>
                </div>
                <div class="grid grid-cols-3 gap-1">
                    <div class="flex items-center">
                        <div class="transform -rotate-90 text-xs text-gray-600">Actual</div>
                    </div>
                    <div class="grid grid-cols-2 gap-1">
                        <div class="bg-green-100 border border-green-300 p-2 text-center text-sm font-medium">
                            <div class="text-green-800">TP</div>
                            <div class="text-green-600">${cm.true_positives}</div>
                        </div>
                        <div class="bg-red-100 border border-red-300 p-2 text-center text-sm font-medium">
                            <div class="text-red-800">FN</div>
                            <div class="text-red-600">${cm.false_negatives}</div>
                        </div>
                        <div class="bg-red-100 border border-red-300 p-2 text-center text-sm font-medium">
                            <div class="text-red-800">FP</div>
                            <div class="text-red-600">${cm.false_positives}</div>
                        </div>
                        <div class="bg-green-100 border border-green-300 p-2 text-center text-sm font-medium">
                            <div class="text-green-800">TN</div>
                            <div class="text-green-600">${cm.true_negatives}</div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="mt-3 text-xs text-gray-600 text-center">
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <span class="font-medium">False Positive Rate:</span> 
                        ${Math.round((cm.false_positives / (cm.false_positives + cm.true_negatives)) * 100)}%
                    </div>
                    <div>
                        <span class="font-medium">False Negative Rate:</span> 
                        ${Math.round((cm.false_negatives / (cm.false_negatives + cm.true_positives)) * 100)}%
                    </div>
                </div>
            </div>
        `;
    }
    
    renderCalibrationChart(calibration) {
        const bins = calibration.calibration_bins.filter(bin => bin.count > 0);
        if (bins.length === 0) return '';
        
        return `
            <div class="mb-6">
                <h4 class="font-semibold text-gray-800 mb-3">
                    <i class="bi bi-bar-chart mr-1"></i>
                    Model Calibration
                </h4>
                <div class="bg-gray-50 rounded-lg p-4">
                    <div class="grid grid-cols-${Math.min(bins.length, 5)} gap-2">
                        ${bins.map(bin => `
                            <div class="text-center">
                                <div class="bg-blue-200 h-16 relative rounded">
                                    <div class="bg-blue-500 absolute bottom-0 w-full rounded" 
                                         style="height: ${bin.accuracy * 100}%"></div>
                                </div>
                                <div class="text-xs mt-1">
                                    <div class="font-medium">${bin.confidence_range}</div>
                                    <div class="text-gray-600">${Math.round(bin.accuracy * 100)}% (${bin.count})</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                    <div class="mt-3 text-xs text-gray-600 text-center">
                        <div>Expected Calibration Error: ${(calibration.expected_calibration_error * 100).toFixed(1)}%</div>
                        <div class="text-gray-500">Lower is better - indicates how well confidence matches actual accuracy</div>
                    </div>
                </div>
            </div>
        `;
    }
    
    renderTemporalConsistency(temporal) {
        if (temporal.weekly_accuracies.length === 0) return '';
        
        return `
            <div class="mb-6">
                <h4 class="font-semibold text-gray-800 mb-3">
                    <i class="bi bi-graph-up mr-1"></i>
                    Temporal Consistency
                </h4>
                <div class="bg-gray-50 rounded-lg p-4">
                    <div class="flex items-end space-x-2 h-24 mb-3">
                        ${temporal.weekly_accuracies.map((acc, i) => `
                            <div class="flex-1 flex flex-col items-center">
                                <div class="bg-purple-500 w-full rounded-t" style="height: ${acc * 100}%"></div>
                                <div class="text-xs mt-1">W${i + 1}</div>
                            </div>
                        `).join('')}
                    </div>
                    <div class="text-xs text-gray-600 text-center">
                        <div>Consistency Variance: ${temporal.consistency_variance.toFixed(4)}</div>
                        <div class="text-gray-500">
                            Model is ${temporal.is_consistent ? 'consistent' : 'variable'} over time
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    displayInsufficientDataMessage(totalVerified) {
        const metricsContainer = document.getElementById('advancedMetricsContainer');
        if (metricsContainer) {
            metricsContainer.innerHTML = `
                <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-center">
                    <i class="bi bi-info-circle text-yellow-600 text-2xl mb-2"></i>
                    <h4 class="font-semibold text-yellow-800 mb-2">Insufficient Data for Metrics</h4>
                    <p class="text-yellow-700 text-sm">
                        Need at least 2 verified articles for meaningful prediction metrics. 
                        Currently have ${totalVerified} verified article${totalVerified === 1 ? '' : 's'}.
                    </p>
                </div>
            `;
        }
    }
    
    updateConfusionMatrixDisplay(confusionMatrix) {
        // Update any existing confusion matrix displays
        const cmElements = document.querySelectorAll('[data-confusion-matrix]');
        cmElements.forEach(el => {
            el.innerHTML = this.renderConfusionMatrix(confusionMatrix);
        });
    }
    
    renderDetailedMetrics(article) {
        if (!article.confidence && !article.probability_news && !article.is_news_prediction) {
            return '';
        }

        const confidence = article.confidence || 0.5;
        const probabilityNews = article.probability_news || 0.5;
        const probabilityNotNews = 1 - probabilityNews;
        const prediction = article.is_news_prediction;

        // Determine confidence color and level
        let confidenceColor = '';
        let confidenceLevel = '';
        let confidenceIcon = '';
        
        if (confidence >= 0.8) {
            confidenceColor = 'text-green-600 bg-green-50 border-green-200';
            confidenceLevel = 'High Confidence';
            confidenceIcon = 'bi-check-circle-fill';
        } else if (confidence >= 0.6) {
            confidenceColor = 'text-yellow-600 bg-yellow-50 border-yellow-200';
            confidenceLevel = 'Medium Confidence';
            confidenceIcon = 'bi-exclamation-triangle-fill';
        } else {
            confidenceColor = 'text-red-600 bg-red-50 border-red-200';
            confidenceLevel = 'Low Confidence';
            confidenceIcon = 'bi-question-circle-fill';
        }

        // Determine prediction color
        const predictionColor = prediction ? 'text-blue-600 bg-blue-50 border-blue-200' : 'text-purple-600 bg-purple-50 border-purple-200';
        const predictionIcon = prediction ? 'bi-newspaper' : 'bi-x-circle';
        const predictionText = prediction ? 'Predicted as News' : 'Predicted as Not-News';

        return `
            <div class="mt-3 pt-3 border-t border-gray-200">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <!-- Confidence Score -->
                    <div class="flex items-center justify-between p-3 rounded-lg border ${confidenceColor}">
                        <div class="flex items-center">
                            <i class="${confidenceIcon} mr-2"></i>
                            <div>
                                <div class="font-semibold text-sm">${confidenceLevel}</div>
                                <div class="text-xs opacity-75">Model certainty</div>
                            </div>
                        </div>
                        <div class="text-right">
                            <div class="text-lg font-bold">${(confidence * 100).toFixed(1)}%</div>
                            <div class="text-xs opacity-75">Confidence</div>
                        </div>
                    </div>
                    
                    <!-- Prediction -->
                    <div class="flex items-center justify-between p-3 rounded-lg border ${predictionColor}">
                        <div class="flex items-center">
                            <i class="${predictionIcon} mr-2"></i>
                            <div>
                                <div class="font-semibold text-sm">${predictionText}</div>
                                <div class="text-xs opacity-75">Model prediction</div>
                            </div>
                        </div>
                        <div class="text-right">
                            <div class="text-lg font-bold">${prediction ? (probabilityNews * 100).toFixed(1) : (probabilityNotNews * 100).toFixed(1)}%</div>
                            <div class="text-xs opacity-75">Probability</div>
                        </div>
                    </div>
                </div>
                
                <!-- Detailed Probability Breakdown -->
                <div class="mt-2 p-2 bg-gray-50 rounded border">
                    <div class="text-xs font-medium text-gray-700 mb-1">Probability Breakdown:</div>
                    <div class="flex items-center space-x-4 text-xs">
                        <div class="flex items-center">
                            <div class="w-2 h-2 bg-blue-500 rounded-full mr-1"></div>
                            <span>News: ${(probabilityNews * 100).toFixed(1)}%</span>
                        </div>
                        <div class="flex items-center">
                            <div class="w-2 h-2 bg-purple-500 rounded-full mr-1"></div>
                            <span>Not-News: ${(probabilityNotNews * 100).toFixed(1)}%</span>
                        </div>
                        ${article.verified ? `
                            <div class="flex items-center text-gray-600">
                                <i class="bi bi-check-square mr-1"></i>
                                <span>Verified by user</span>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    renderPredictionQualityIndicator(article) {
        if (!article.confidence && !article.probability_news) return '';
        
        const confidence = article.confidence || 0.5;
        const probability = article.probability_news || 0.5;
        const prediction = article.is_news_prediction;
        
        // Determine quality level based on confidence and probability
        let qualityClass = '';
        let qualityIcon = '';
        let qualityText = '';
        
        if (confidence >= 0.8) {
            qualityClass = 'text-green-600';
            qualityIcon = 'bi-check-circle-fill';
            qualityText = 'High';
        } else if (confidence >= 0.6) {
            qualityClass = 'text-yellow-600';
            qualityIcon = 'bi-exclamation-triangle-fill';
            qualityText = 'Med';
        } else {
            qualityClass = 'text-red-600';
            qualityIcon = 'bi-question-circle-fill';
            qualityText = 'Low';
        }
        
        // Add prediction direction indicator
        const predictionIcon = prediction ? 'bi-newspaper' : 'bi-x-circle';
        const predictionText = prediction ? 'News' : 'Not News';
        
        return `
            <span class="mr-3 ${qualityClass}" title="Prediction Quality: ${qualityText}, Confidence: ${(confidence * 100).toFixed(1)}%, Probability: ${(probability * 100).toFixed(1)}%">
                <i class="${qualityIcon} mr-1"></i>
                <span class="text-xs">${qualityText}</span>
            </span>
            <span class="mr-3 text-gray-600" title="Model Prediction: ${predictionText}">
                <i class="${predictionIcon} mr-1"></i>
                <span class="text-xs">${predictionText}</span>
            </span>
        `;
    }
}

// Initialize the application
let newsTracker;
document.addEventListener('DOMContentLoaded', () => {
    newsTracker = new NewsTracker();
});
