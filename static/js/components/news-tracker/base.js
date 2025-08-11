/**
 * Base News Tracker Application
 * Core functionality and state management
 */

export class NewsTrackerBase {
    constructor() {
        // Core state
        this.trackedWebsites = [];
        this.articleQueue = [];
        this.currentPage = 1;
        this.itemsPerPage = 20;
        this.selectedArticles = new Set();
        this.predictionMetrics = null;
        this.websiteViewMode = 'grouped';
        this.isPerformingOperation = false;
        this.isFetchInProgress = false; // Lock to prevent concurrent fetch operations
        this.isRenderingWebsites = false; // Lock to prevent concurrent renders
        
        // Initialize properties that will be set by mixins
        this.autoFetchInterval = null;
        this.autoFetchEnabled = false;
        this.autoFetchIntervalMinutes = 30;
        this.autoFetchStats = {
            totalRuns: 0,
            articlesFound: 0,
            lastRun: null,
            nextRun: null
        };
        
        // Auto-indexing settings
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
        
        this.periodicRefreshInterval = null;
    }
    
    /**
     * Initialize the application
     */
    async init() {
        this.bindCoreEvents();
        this.loadViewPreferences();
        await this.loadTrackedWebsites();
        
        // Initialize UI components
        this.updateSelectionCount();
        this.updateBatchActionButtons();
        this.updateDisplayedArticleCount();
        
        // Setup cleanup
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }
    
    /**
     * Bind core event listeners
     */
    bindCoreEvents() {
        // Core navigation events
        const websiteViewMode = document.getElementById('websiteViewMode');
        if (websiteViewMode) {
            websiteViewMode.addEventListener('change', async (e) => {
                await this.setWebsiteViewMode(e.target.value);
            });
        }
        
        // Queue filter
        const queueFilter = document.getElementById('queueFilter');
        if (queueFilter) {
            queueFilter.addEventListener('change', () => this.renderArticleQueue());
        }
        
        // Pagination
        const prevPageBtn = document.getElementById('prevPageBtn');
        const nextPageBtn = document.getElementById('nextPageBtn');
        
        if (prevPageBtn) {
            prevPageBtn.addEventListener('click', () => this.previousPage());
        }
        
        if (nextPageBtn) {
            nextPageBtn.addEventListener('click', () => this.nextPage());
        }
    }
    
    /**
     * Load view preferences from localStorage
     */
    loadViewPreferences() {
        const savedViewMode = localStorage.getItem('newsTracker.websiteViewMode') || 'grouped';
        this.websiteViewMode = savedViewMode;
        
        const viewModeSelect = document.getElementById('websiteViewMode');
        if (viewModeSelect) {
            viewModeSelect.value = savedViewMode;
        }
        
        this.setWebsiteViewMode(savedViewMode);
    }
    
    /**
     * Set website view mode
     */
    async setWebsiteViewMode(mode) {
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
        
        // Only render if we have websites to display
        if (this.trackedWebsites && this.trackedWebsites.length > 0) {
            await this.renderTrackedWebsites();
        }
    }
    
    /**
     * Load tracked websites and article data
     */
    async loadTrackedWebsites() {
        try {
            const response = await fetch('/api/news-tracker/get-data');
            const data = await response.json();
            
            if (data.success) {
                this.trackedWebsites = data.websites || [];
                this.articleQueue = data.articles || [];
                await this.renderTrackedWebsites();
                this.renderArticleQueue();
                this.updateCounts();
                return true;
            } else {
                console.error('Failed to load data:', data.error);
                return false;
            }
        } catch (error) {
            console.error('Error loading tracked websites:', error);
            return false;
        }
    }
    
    /**
     * Update counts in header
     */
    updateCounts() {
        // Update header counts
        document.getElementById('trackedWebsitesCount').textContent = this.trackedWebsites.length;
        document.getElementById('queueCount').textContent = this.articleQueue.length;
        
        const verifiedCount = this.articleQueue.filter(article => article.verified).length;
        document.getElementById('verifiedCount').textContent = verifiedCount;
        
        // Update queue stats
        const pendingCount = this.articleQueue.filter(article => !article.verified).length;
        const verifiedNewsCount = this.articleQueue.filter(article => article.verified && article.isNews).length;
        const notNewsCount = this.articleQueue.filter(article => article.verified && !article.isNews).length;
        
        document.getElementById('pendingCount').textContent = pendingCount;
        document.getElementById('verifiedNewsCount').textContent = verifiedNewsCount;
        document.getElementById('notNewsCount').textContent = notNewsCount;
        document.getElementById('totalCount').textContent = this.articleQueue.length;
    }
    
    /**
     * Pagination controls
     */
    previousPage() {
        if (this.currentPage > 1) {
            this.currentPage--;
            this.renderArticleQueue();
        }
    }
    
    nextPage() {
        const totalPages = Math.ceil(this.getFilteredArticles().length / this.itemsPerPage);
        if (this.currentPage < totalPages) {
            this.currentPage++;
            this.renderArticleQueue();
        }
    }
    
    /**
     * Get filtered articles based on current filter
     */
    getFilteredArticles() {
        const filter = document.getElementById('queueFilter')?.value || 'all';
        
        switch (filter) {
            case 'unverified':
                return this.articleQueue.filter(article => !article.verified);
            case 'verified':
                return this.articleQueue.filter(article => article.verified);
            case 'news':
                return this.articleQueue.filter(article => article.verified && article.isNews);
            case 'not-news':
                return this.articleQueue.filter(article => article.verified && !article.isNews);
            default:
                return this.articleQueue;
        }
    }
    
    /**
     * Update selection count display
     */
    updateSelectionCount() {
        const count = this.selectedArticles.size;
        const selectionCountEl = document.getElementById('selectionCount');
        if (selectionCountEl) {
            selectionCountEl.textContent = `${count} selected`;
        }
    }
    
    /**
     * Update batch action buttons state
     */
    updateBatchActionButtons() {
        const hasSelection = this.selectedArticles.size > 0;
        const batchMarkNewsBtn = document.getElementById('batchMarkNewsBtn');
        const batchMarkNotNewsBtn = document.getElementById('batchMarkNotNewsBtn');
        
        if (batchMarkNewsBtn) batchMarkNewsBtn.disabled = !hasSelection;
        if (batchMarkNotNewsBtn) batchMarkNotNewsBtn.disabled = !hasSelection;
    }
    
    /**
     * Update displayed article count
     */
    updateDisplayedArticleCount() {
        const filteredArticles = this.getFilteredArticles();
        const availableEl = document.getElementById('availableForSelection');
        if (availableEl) {
            availableEl.textContent = `${filteredArticles.length} articles available`;
        }
    }
    
    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.autoFetchInterval) {
            clearInterval(this.autoFetchInterval);
        }
        if (this.periodicRefreshInterval) {
            clearInterval(this.periodicRefreshInterval);
        }
    }
    
    /**
     * Utility methods
     */
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
            const domain = new URL(url).hostname.toLowerCase();
            // Remove www. prefix and return clean domain
            return domain.replace(/^www\./, '');
        } catch (_) {
            return url;
        }
    }
    
    formatDate(dateString) {
        if (!dateString) return 'Never';
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }
    
    /**
     * Update fetch status indicators
     */
    updateFetchStatusIndicators() {
        // Update fetch button state
        const fetchBtn = document.getElementById('fetchNowBtn');
        const testAutoFetchBtn = document.getElementById('testAutoFetchBtn');
        
        if (fetchBtn) {
            if (this.isFetchInProgress) {
                fetchBtn.disabled = true;
                fetchBtn.innerHTML = '<i class="bi bi-hourglass-split mr-2"></i>Fetching...';
                fetchBtn.classList.add('opacity-50', 'cursor-not-allowed');
            } else {
                fetchBtn.disabled = false;
                fetchBtn.innerHTML = '<i class="bi bi-download mr-2"></i>Fetch All Articles Now';
                fetchBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
        }
        
        if (testAutoFetchBtn) {
            if (this.isFetchInProgress) {
                testAutoFetchBtn.disabled = true;
                testAutoFetchBtn.innerHTML = '<i class="bi bi-hourglass-split mr-2"></i>In Progress...';
                testAutoFetchBtn.classList.add('opacity-50', 'cursor-not-allowed');
            } else {
                testAutoFetchBtn.disabled = false;
                testAutoFetchBtn.innerHTML = '<i class="bi bi-play-circle mr-2"></i>Test Auto-fetch';
                testAutoFetchBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            }
        }
    }
    
    /**
     * Abstract methods to be implemented by mixins
     */
    async renderTrackedWebsites() {
        throw new Error('renderTrackedWebsites must be implemented by website manager');
    }
    
    renderArticleQueue() {
        throw new Error('renderArticleQueue must be implemented by article queue manager');
    }
}
