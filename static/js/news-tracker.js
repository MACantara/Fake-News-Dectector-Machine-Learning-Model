/**
 * News Tracker Application - Modular Version
 * Combines all modules into a unified application
 */

import { NewsTrackerBase } from './components/news-tracker/base.js';
import { WebsiteManagerMixin } from './components/news-tracker/website-manager.js';
import { ArticleQueueManagerMixin } from './components/news-tracker/article-queue-manager.js';
import { NotificationManagerMixin } from './components/news-tracker/notification-manager.js';
import { AutoFetchManagerMixin } from './components/news-tracker/auto-fetch-manager.js';
import { StatisticsManagerMixin } from './components/news-tracker/statistics-manager.js';
import { AutoIndexManagerMixin } from './components/news-tracker/auto-index-manager.js';

/**
 * Main News Tracker Application Class
 * Extends base class with all mixins for complete functionality
 */
class NewsTrackerApp extends NewsTrackerBase {
    constructor() {
        super();
        
        // Initialize flags for event handling
        this.globalEventsInitialized = false;
        this.isRealTimePaused = false;
        this.visibilityDebounceTimer = null;
        this.lastVisibilityChange = 0;
        
        // Configuration flag to disable visibility change handling if problematic
        this.disableVisibilityHandling = localStorage.getItem('newsTracker.disableVisibilityHandling') === 'true';
        
        this.initMixins();
    }
    
    /**
     * Initialize all mixins
     */
    initMixins() {
        // Apply all mixins to this instance
        Object.assign(this, WebsiteManagerMixin);
        Object.assign(this, ArticleQueueManagerMixin);
        Object.assign(this, NotificationManagerMixin);
        Object.assign(this, AutoFetchManagerMixin);
        Object.assign(this, StatisticsManagerMixin);
        Object.assign(this, AutoIndexManagerMixin);
    }
    
    /**
     * Initialize the complete application
     */
    async init() {
        try {
            console.log('Initializing News Tracker Application...');
            
            // Initialize base functionality first
            await super.init();
            
            // Setup all subsystems
            this.setupAutoFetch();
            this.setupAutoIndex();
            this.setupStatistics();
            
            // Bind all events
            this.bindAllEvents();
            
            // Load initial data
            await this.loadInitialData();
            
            // Start real-time features
            this.startRealTimeFeatures();
            
            console.log('News Tracker Application initialized successfully!');
            this.showSuccess('News Tracker loaded successfully!');
            
        } catch (error) {
            console.error('Failed to initialize News Tracker:', error);
            this.showError('Failed to load News Tracker. Please refresh the page.');
        }
    }
    
    /**
     * Bind all event handlers
     */
    bindAllEvents() {
        // Website management events
        this.bindWebsiteEvents();
        
        // Article queue events
        this.bindArticleQueueEvents();
        
        // Auto-fetch events
        this.bindAutoFetchEvents();
        
        // Auto-index events
        this.bindAutoIndexEvents();
        
        // Statistics events
        this.bindStatisticsEvents();
        
        // Global events
        this.bindGlobalEvents();
    }
    
    /**
     * Bind global application events
     */
    bindGlobalEvents() {
        // Prevent multiple event listeners
        if (this.globalEventsInitialized) {
            return;
        }
        
        this.globalEventsInitialized = true;
        this.visibilityDebounceTimer = null;
        this.isRealTimePaused = false;
        this.lastVisibilityChange = 0;
        
        // Page visibility change (pause/resume real-time features) with improved debouncing
        if (!this.disableVisibilityHandling) {
            document.addEventListener('visibilitychange', () => {
                const now = Date.now();
                const timeSinceLastChange = now - this.lastVisibilityChange;
                
                // Clear existing debounce timer
                if (this.visibilityDebounceTimer) {
                    clearTimeout(this.visibilityDebounceTimer);
                }
                
                // Log visibility change for debugging
                console.log(`Visibility change: document.hidden=${document.hidden}, timeSinceLastChange=${timeSinceLastChange}ms`);
                
                // Ignore rapid visibility changes (less than 1 second apart)
                if (timeSinceLastChange < 1000) {
                    console.log('Ignoring rapid visibility change');
                    return;
                }
                
                this.lastVisibilityChange = now;
                
                // Debounce visibility changes to prevent rapid toggling
                this.visibilityDebounceTimer = setTimeout(() => {
                    if (document.hidden && !this.isRealTimePaused) {
                        console.log('Page hidden - pausing real-time features');
                        this.pauseRealTimeFeatures();
                    } else if (!document.hidden && this.isRealTimePaused) {
                        console.log('Page visible - resuming real-time features');
                        this.resumeRealTimeFeatures();
                    }
                }, 1000); // Increased debounce to 1 second
            });
        } else {
            console.log('Visibility change handling disabled by user preference');
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'r':
                        e.preventDefault();
                        this.refreshAll();
                        break;
                    case 'f':
                        e.preventDefault();
                        this.focusAddWebsiteInput();
                        break;
                    case 's':
                        e.preventDefault();
                        this.updateStatistics();
                        break;
                }
            }
        });
        
        // Global error handling
        window.addEventListener('error', (e) => {
            console.error('Global error:', e.error);
            this.showError('An unexpected error occurred. Please refresh the page if issues persist.');
        });
        
        // Unhandled promise rejections
        window.addEventListener('unhandledrejection', (e) => {
            console.error('Unhandled promise rejection:', e.reason);
            this.showError('A network error occurred. Please check your connection.');
        });
    }
    
    /**
     * Load initial application data
     */
    async loadInitialData() {
        console.log('Loading initial data...');
        
        try {
            // Load tracked websites and articles
            await this.loadTrackedWebsites();
            
            // Update statistics
            this.updateStatistics();
            
            // Update high confidence queue
            this.updateHighConfidenceQueue();
            
            console.log('Initial data loaded successfully');
            
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showError('Failed to load some data. Please refresh the page.');
        }
    }
    
    /**
     * Start real-time features
     */
    startRealTimeFeatures() {
        // Start periodic refresh (every 30 seconds)
        this.startPeriodicRefresh();
        
        // Start auto-fetch if enabled
        if (this.autoFetchEnabled && this.trackedWebsites.length > 0) {
            this.startAutoFetch();
        }
        
        // Start statistics auto-refresh if enabled
        if (this.statisticsPrefs.autoRefresh) {
            this.startStatisticsAutoRefresh();
        }
        
        console.log('Real-time features started');
    }
    
    /**
     * Pause real-time features (when page is hidden)
     */
    pauseRealTimeFeatures() {
        if (this.isRealTimePaused) {
            return; // Already paused
        }
        
        console.log('Pausing real-time features');
        this.isRealTimePaused = true;
        this.stopPeriodicRefresh();
        this.stopAutoFetch();
        this.stopStatisticsAutoRefresh();
    }
    
    /**
     * Resume real-time features (when page becomes visible)
     */
    resumeRealTimeFeatures() {
        if (!this.isRealTimePaused) {
            return; // Already running
        }
        
        console.log('Resuming real-time features');
        this.isRealTimePaused = false;
        this.startRealTimeFeatures();
    }
    
    /**
     * Start periodic refresh system
     */
    startPeriodicRefresh() {
        this.stopPeriodicRefresh(); // Clear existing timer
        
        this.periodicRefreshInterval = setInterval(async () => {
            try {
                // Only refresh if not currently performing operations
                if (!this.isOperationInProgress) {
                    await this.refreshTrackedWebsites();
                    this.updateStatistics();
                }
            } catch (error) {
                console.error('Periodic refresh error:', error);
                // Don't show error to user for periodic refresh
            }
        }, 30000); // 30 seconds
    }
    
    /**
     * Stop periodic refresh system
     */
    stopPeriodicRefresh() {
        if (this.periodicRefreshInterval) {
            clearInterval(this.periodicRefreshInterval);
            this.periodicRefreshInterval = null;
        }
    }
    
    /**
     * Refresh all data
     */
    async refreshAll() {
        try {
            this.showLoading('Refreshing all data...');
            
            await Promise.all([
                this.loadTrackedWebsites(),
                this.updateStatistics()
            ]);
            
            this.updateHighConfidenceQueue();
            this.showSuccess('All data refreshed successfully!');
            
        } catch (error) {
            console.error('Error refreshing data:', error);
            this.showError('Failed to refresh some data. Please try again.');
        } finally {
            this.hideLoading();
        }
    }
    
    /**
     * Focus the add website input
     */
    focusAddWebsiteInput() {
        const input = document.getElementById('websiteUrl');
        if (input) {
            input.focus();
            input.select();
        }
    }
    
    /**
     * Refresh tracked websites data only
     */
    async refreshTrackedWebsites() {
        try {
            const response = await fetch('/api/news-tracker/get-data');
            if (!response.ok) throw new Error('Failed to fetch websites');
            
            const data = await response.json();
            const websites = data.websites || [];
            
            // Update tracked websites if data changed
            const currentIds = this.trackedWebsites.map(w => w.id).sort();
            const newIds = websites.map(w => w.id).sort();
            
            if (JSON.stringify(currentIds) !== JSON.stringify(newIds)) {
                this.trackedWebsites = websites;
                this.renderTrackedWebsites();
                console.log('Tracked websites updated via periodic refresh');
            }
            
        } catch (error) {
            console.error('Error refreshing tracked websites:', error);
        }
    }
    
    /**
     * Cleanup resources when page is unloaded
     */
    cleanup() {
        this.stopPeriodicRefresh();
        this.stopAutoFetch();
        this.stopStatisticsAutoRefresh();
        
        // Clear visibility debounce timer
        if (this.visibilityDebounceTimer) {
            clearTimeout(this.visibilityDebounceTimer);
            this.visibilityDebounceTimer = null;
        }
        
        console.log('News Tracker cleanup completed');
    }
    
    /**
     * Get application status
     */
    getStatus() {
        return {
            initialized: this.initialized,
            trackedWebsites: this.trackedWebsites.length,
            articleQueue: this.articleQueue.length,
            autoFetchEnabled: this.autoFetchEnabled,
            autoIndexEnabled: this.autoIndexEnabled,
            operationInProgress: this.isOperationInProgress,
            realTimeFeatures: {
                periodicRefresh: !!this.periodicRefreshInterval,
                autoFetch: !!this.autoFetchInterval,
                statisticsRefresh: !!this.statisticsRefreshInterval
            }
        };
    }
    
    /**
     * Enable debug mode
     */
    enableDebugMode() {
        this.debugMode = true;
        console.log('Debug mode enabled');
        console.log('Current status:', this.getStatus());
        
        // Add debug controls to page
        const debugInfo = document.getElementById('debugInfo');
        if (debugInfo) {
            debugInfo.style.display = 'block';
            debugInfo.innerHTML = `
                <div class="bg-gray-100 p-4 rounded-lg">
                    <h4 class="font-bold mb-2">Debug Information</h4>
                    <pre class="text-xs mb-3">${JSON.stringify(this.getStatus(), null, 2)}</pre>
                    <div class="space-x-2">
                        <button onclick="newsTracker.toggleVisibilityHandling()" class="px-2 py-1 text-xs bg-blue-500 text-white rounded">
                            ${this.disableVisibilityHandling ? 'Enable' : 'Disable'} Visibility Handling
                        </button>
                        <button onclick="newsTracker.forceResume()" class="px-2 py-1 text-xs bg-green-500 text-white rounded">
                            Force Resume
                        </button>
                        <button onclick="newsTracker.getStatus()" class="px-2 py-1 text-xs bg-purple-500 text-white rounded">
                            Log Status
                        </button>
                    </div>
                </div>
            `;
        }
    }
    
    /**
     * Toggle visibility change handling
     */
    toggleVisibilityHandling() {
        this.disableVisibilityHandling = !this.disableVisibilityHandling;
        localStorage.setItem('newsTracker.disableVisibilityHandling', this.disableVisibilityHandling.toString());
        
        const action = this.disableVisibilityHandling ? 'disabled' : 'enabled';
        console.log(`Visibility change handling ${action}`);
        this.showSuccess(`Visibility change handling ${action}. Refresh page to apply.`);
    }
    
    /**
     * Force resume real-time features
     */
    forceResume() {
        console.log('Force resuming real-time features');
        this.isRealTimePaused = false;
        this.startRealTimeFeatures();
        this.showSuccess('Real-time features force resumed');
    }
    
    /**
     * Disable debug mode
     */
    disableDebugMode() {
        this.debugMode = false;
        console.log('Debug mode disabled');
        
        const debugInfo = document.getElementById('debugInfo');
        if (debugInfo) {
            debugInfo.style.display = 'none';
        }
    }
}

// Initialize the application when DOM is ready
let newsTrackerApp;

document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOM loaded, initializing News Tracker...');
    
    try {
        newsTrackerApp = new NewsTrackerApp();
        await newsTrackerApp.init();
        
        // Make app globally available for debugging
        window.newsTracker = newsTrackerApp;
        
        // Debug mode toggle (Ctrl+Shift+D)
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey && e.key === 'D') {
                if (newsTrackerApp.debugMode) {
                    newsTrackerApp.disableDebugMode();
                } else {
                    newsTrackerApp.enableDebugMode();
                }
            }
        });
        
    } catch (error) {
        console.error('Failed to initialize News Tracker:', error);
        
        // Show fallback error message
        const errorContainer = document.getElementById('errorContainer') || document.body;
        errorContainer.innerHTML = `
            <div class="max-w-md mx-auto mt-8 p-6 bg-red-50 border border-red-200 rounded-lg">
                <div class="flex items-center mb-4">
                    <i class="bi bi-exclamation-triangle text-red-600 text-xl mr-3"></i>
                    <h3 class="text-lg font-semibold text-red-800">Application Error</h3>
                </div>
                <p class="text-red-700 mb-4">
                    Failed to initialize the News Tracker application. Please refresh the page to try again.
                </p>
                <button onclick="window.location.reload()" 
                        class="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors">
                    Refresh Page
                </button>
            </div>
        `;
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (newsTrackerApp) {
        newsTrackerApp.cleanup();
    }
});

// Export for module usage
export { NewsTrackerApp };
