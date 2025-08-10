/**
 * Robust Auto-Fetch Scheduler
 * Precise timing-based auto-fetch system with recovery mechanisms
 */

export class AutoFetchScheduler {
    constructor(newsTrackerApp) {
        this.app = newsTrackerApp;
        this.isRunning = false;
        this.intervalMinutes = 30;
        this.nextRunTime = null;
        this.schedulerInterval = null;
        this.checkInterval = 10000; // Check every 10 seconds for precision
        
        // Statistics
        this.stats = {
            totalRuns: 0,
            articlesFound: 0,
            lastRun: null,
            lastRunDuration: 0,
            successfulRuns: 0,
            failedRuns: 0,
            averageArticlesPerRun: 0
        };
        
        this.loadSettings();
        this.setupScheduler();
    }
    
    /**
     * Load scheduler settings from localStorage
     */
    loadSettings() {
        const savedSettings = localStorage.getItem('newsTracker.autoFetchScheduler');
        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            this.intervalMinutes = settings.intervalMinutes || 30;
            this.stats = { ...this.stats, ...settings.stats };
        }
        
        // Load enabled state
        this.isRunning = localStorage.getItem('newsTracker.autoFetchEnabled') === 'true';
        
        console.log(`Auto-fetch scheduler initialized: ${this.isRunning ? 'enabled' : 'disabled'}, interval: ${this.intervalMinutes} minutes`);
    }
    
    /**
     * Save scheduler settings to localStorage
     */
    saveSettings() {
        const settings = {
            intervalMinutes: this.intervalMinutes,
            stats: this.stats
        };
        localStorage.setItem('newsTracker.autoFetchScheduler', JSON.stringify(settings));
    }
    
    /**
     * Setup the precision scheduler
     */
    setupScheduler() {
        // Clear any existing scheduler
        this.stopScheduler();
        
        // Start the precision checker that runs every 10 seconds
        this.schedulerInterval = setInterval(() => {
            this.checkSchedule();
        }, this.checkInterval);
        
        console.log('Auto-fetch scheduler setup complete');
    }
    
    /**
     * Check if it's time to run auto-fetch
     */
    checkSchedule() {
        if (!this.isRunning || !this.nextRunTime) {
            return;
        }
        
        const now = new Date();
        const timeUntilRun = this.nextRunTime.getTime() - now.getTime();
        
        // If it's time to run (within 10 seconds of scheduled time)
        if (timeUntilRun <= this.checkInterval) {
            console.log(`Auto-fetch scheduled run triggered at ${now.toLocaleTimeString()}`);
            this.executeAutoFetch();
        }
    }
    
    /**
     * Start the auto-fetch scheduler
     */
    start() {
        if (this.isRunning) {
            console.log('Auto-fetch scheduler already running');
            return;
        }
        
        if (!this.app.trackedWebsites || this.app.trackedWebsites.length === 0) {
            console.log('Cannot start auto-fetch: no tracked websites');
            return;
        }
        
        this.isRunning = true;
        localStorage.setItem('newsTracker.autoFetchEnabled', 'true');
        
        // Schedule first run
        this.scheduleNextRun();
        
        console.log(`Auto-fetch scheduler started with ${this.intervalMinutes} minute intervals`);
        this.app.showSuccess(`Auto-fetch started (${this.intervalMinutes} minute intervals)`);
        
        // Update UI
        this.updateUI();
    }
    
    /**
     * Stop the auto-fetch scheduler
     */
    stop() {
        if (!this.isRunning) {
            console.log('Auto-fetch scheduler already stopped');
            return;
        }
        
        this.isRunning = false;
        this.nextRunTime = null;
        localStorage.setItem('newsTracker.autoFetchEnabled', 'false');
        
        console.log('Auto-fetch scheduler stopped');
        this.app.showSuccess('Auto-fetch stopped');
        
        // Update UI
        this.updateUI();
    }
    
    /**
     * Schedule the next auto-fetch run
     */
    scheduleNextRun() {
        if (!this.isRunning) {
            return;
        }
        
        const now = new Date();
        this.nextRunTime = new Date(now.getTime() + (this.intervalMinutes * 60 * 1000));
        
        console.log(`Next auto-fetch scheduled for: ${this.nextRunTime.toLocaleString()}`);
        
        // Save to localStorage for persistence
        localStorage.setItem('newsTracker.nextAutoFetchTime', this.nextRunTime.toISOString());
        
        // Update UI
        this.updateUI();
    }
    
    /**
     * Execute auto-fetch operation
     */
    async executeAutoFetch() {
        // Check if fetch is already in progress
        if (this.app.isFetchInProgress) {
            console.log('Auto-fetch skipped: Fetch operation already in progress');
            // Schedule next run without executing
            this.scheduleNextRun();
            return;
        }
        
        const startTime = Date.now();
        
        try {
            console.log('Executing scheduled auto-fetch...');
            
            // Update stats
            this.stats.totalRuns++;
            this.stats.lastRun = new Date().toISOString();
            
            // Get initial article count
            const initialCount = this.app.articleQueue ? this.app.articleQueue.length : 0;
            
            // Execute the fetch (fetchAllArticles will handle its own locking)
            await this.app.fetchAllArticles();
            
            // Calculate new articles found
            const finalCount = this.app.articleQueue ? this.app.articleQueue.length : 0;
            const newArticles = Math.max(0, finalCount - initialCount);
            
            // Update statistics
            this.stats.articlesFound += newArticles;
            this.stats.successfulRuns++;
            this.stats.lastRunDuration = Date.now() - startTime;
            this.stats.averageArticlesPerRun = this.stats.totalRuns > 0 ? 
                Math.round(this.stats.articlesFound / this.stats.totalRuns * 10) / 10 : 0;
            
            console.log(`Auto-fetch completed: ${newArticles} new articles found in ${this.stats.lastRunDuration}ms`);
            
            // Note: Auto-indexing is handled within fetchAllArticles
            
        } catch (error) {
            console.error('Auto-fetch execution failed:', error);
            this.stats.failedRuns++;
        } finally {
            // Save statistics
            this.saveSettings();
            
            // Schedule next run
            this.scheduleNextRun();
            
            // Update UI
            this.updateUI();
        }
    }
    
    /**
     * Execute auto-indexing for high confidence articles
     */
    async executeAutoIndexing() {
        try {
            console.log('Running auto-indexing after auto-fetch...');
            
            const response = await fetch('/api/news-tracker/auto-index-high-confidence', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    confidence_threshold: (this.app.autoIndexThreshold || 95) / 100, // Convert percentage to decimal
                    batch_size: this.app.autoIndexBatchSize || 10
                })
            });
            
            const result = await response.json();
            
            if (result.success && result.articles_indexed > 0) {
                console.log(`Auto-indexing completed: ${result.articles_indexed} articles indexed`);
                
                // Update auto-indexing stats if available
                if (this.app.autoIndexStats) {
                    this.app.autoIndexStats.totalIndexed += result.articles_indexed;
                    this.app.autoIndexStats.lastBatchSize = result.articles_indexed;
                    this.app.autoIndexStats.lastBatchTime = new Date().toISOString();
                    
                    if (this.app.updateAutoIndexStatsDisplay) {
                        this.app.updateAutoIndexStatsDisplay();
                    }
                }
                
                // Refresh UI
                await this.app.loadTrackedWebsites();
                if (this.app.updateHighConfidenceQueue) {
                    this.app.updateHighConfidenceQueue();
                }
            }
            
        } catch (error) {
            console.error('Auto-indexing after auto-fetch failed:', error);
        }
    }
    
    /**
     * Set auto-fetch interval
     */
    setInterval(minutes) {
        const oldInterval = this.intervalMinutes;
        this.intervalMinutes = Math.max(1, parseInt(minutes)) || 30;
        
        console.log(`Auto-fetch interval changed from ${oldInterval} to ${this.intervalMinutes} minutes`);
        
        // Reschedule if running
        if (this.isRunning) {
            this.scheduleNextRun();
        }
        
        // Save settings
        this.saveSettings();
        
        // Update UI
        this.updateUI();
    }
    
    /**
     * Get time until next run in minutes
     */
    getTimeUntilNextRun() {
        if (!this.nextRunTime || !this.isRunning) {
            return null;
        }
        
        const now = new Date();
        const diff = this.nextRunTime.getTime() - now.getTime();
        return Math.max(0, Math.round(diff / 60000)); // Convert to minutes
    }
    
    /**
     * Update UI elements
     */
    updateUI() {
        // Update status display
        this.updateStatusDisplay();
        
        // Update statistics display
        this.updateStatsDisplay();
        
        // Update next run display
        this.updateNextRunDisplay();
    }
    
    /**
     * Update status display
     */
    updateStatusDisplay() {
        const statusEl = document.getElementById('autoFetchStatus');
        if (!statusEl) return;
        
        if (this.isRunning) {
            const nextRun = this.nextRunTime ? 
                this.nextRunTime.toLocaleTimeString() : 'calculating...';
            
            statusEl.innerHTML = `
                <div class="flex items-center">
                    <i class="bi bi-play-circle text-green-600 mr-2"></i>
                    <span class="text-sm text-green-700">
                        Auto-fetch is running (every ${this.intervalMinutes} minutes)
                    </span>
                </div>
                <div class="text-xs text-green-600 mt-1">
                    Next run: ${nextRun}
                </div>
            `;
            statusEl.className = 'p-3 bg-green-50 rounded-lg border border-green-200';
        } else {
            statusEl.innerHTML = `
                <div class="flex items-center">
                    <i class="bi bi-pause-circle text-gray-500 mr-2"></i>
                    <span class="text-sm text-gray-600">Auto-fetch is currently disabled</span>
                </div>
            `;
            statusEl.className = 'p-3 bg-gray-50 rounded-lg border border-gray-200';
        }
    }
    
    /**
     * Update statistics display
     */
    updateStatsDisplay() {
        const elements = {
            totalRuns: document.getElementById('autoFetchTotalRuns'),
            articlesFound: document.getElementById('autoFetchArticlesFound'),
            lastRun: document.getElementById('autoFetchLastRun'),
            successRate: document.getElementById('autoFetchSuccessRate')
        };
        
        if (elements.totalRuns) elements.totalRuns.textContent = this.stats.totalRuns;
        if (elements.articlesFound) elements.articlesFound.textContent = this.stats.articlesFound;
        if (elements.lastRun) {
            elements.lastRun.textContent = this.stats.lastRun ? 
                new Date(this.stats.lastRun).toLocaleString() : 'Never';
        }
        if (elements.successRate) {
            const rate = this.stats.totalRuns > 0 ? 
                Math.round((this.stats.successfulRuns / this.stats.totalRuns) * 100) : 0;
            elements.successRate.textContent = `${rate}%`;
        }
    }
    
    /**
     * Update next run countdown display
     */
    updateNextRunDisplay() {
        const nextRunEl = document.getElementById('autoFetchNextRun');
        if (!nextRunEl) return;
        
        const minutesUntilRun = this.getTimeUntilNextRun();
        if (minutesUntilRun !== null) {
            if (minutesUntilRun === 0) {
                nextRunEl.textContent = 'Soon';
            } else {
                nextRunEl.textContent = `${minutesUntilRun} min`;
            }
        } else {
            nextRunEl.textContent = '-';
        }
    }
    
    /**
     * Stop the scheduler completely
     */
    stopScheduler() {
        if (this.schedulerInterval) {
            clearInterval(this.schedulerInterval);
            this.schedulerInterval = null;
        }
    }
    
    /**
     * Reset all statistics and settings to defaults
     */
    reset() {
        // Stop the scheduler
        this.stop();
        
        // Reset statistics
        this.stats = {
            totalRuns: 0,
            articlesFound: 0,
            lastRun: null,
            lastRunDuration: 0,
            successfulRuns: 0,
            failedRuns: 0,
            averageArticlesPerRun: 0
        };
        
        // Reset settings
        this.intervalMinutes = 30;
        this.nextRunTime = null;
        
        // Save to localStorage
        this.saveSettings();
        
        // Update displays
        this.updateStatsDisplay();
        this.updateNextRunDisplay();
        
        console.log('AutoFetchScheduler reset to defaults');
    }
    
    /**
     * Update statistics with external data
     */
    updateStats(newStats) {
        if (newStats.articlesFound !== undefined) {
            this.stats.articlesFound += newStats.articlesFound;
        }
        if (newStats.lastRun !== undefined) {
            this.stats.lastRun = newStats.lastRun;
        }
        if (newStats.totalRuns !== undefined) {
            this.stats.totalRuns = newStats.totalRuns;
        }
        if (newStats.successfulRuns !== undefined) {
            this.stats.successfulRuns = newStats.successfulRuns;
        }
        if (newStats.failedRuns !== undefined) {
            this.stats.failedRuns = newStats.failedRuns;
        }
        
        // Recalculate average
        if (this.stats.totalRuns > 0) {
            this.stats.averageArticlesPerRun = this.stats.articlesFound / this.stats.totalRuns;
        }
        
        // Save to localStorage
        this.saveSettings();
        
        // Update display
        this.updateStatsDisplay();
    }
    
    /**
     * Get current statistics
     */
    getStats() {
        return {
            totalRuns: this.stats.totalRuns,
            articlesFound: this.stats.articlesFound,
            lastRun: this.stats.lastRun,
            nextRun: this.nextRunTime ? this.nextRunTime.toISOString() : null,
            successfulRuns: this.stats.successfulRuns,
            failedRuns: this.stats.failedRuns,
            lastRunDuration: this.stats.lastRunDuration,
            averageArticlesPerRun: this.stats.averageArticlesPerRun
        };
    }
    
    /**
     * Get scheduler status for debugging
     */
    getStatus() {
        return {
            isRunning: this.isRunning,
            intervalMinutes: this.intervalMinutes,
            nextRunTime: this.nextRunTime ? this.nextRunTime.toISOString() : null,
            timeUntilNextRun: this.getTimeUntilNextRun(),
            stats: { ...this.stats }
        };
    }
}
