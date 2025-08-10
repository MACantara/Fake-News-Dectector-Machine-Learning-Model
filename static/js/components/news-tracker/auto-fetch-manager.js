/**
 * Auto-fetch Management Mixin
 * Handles automatic article fetching with configurable intervals
 */

export const AutoFetchManagerMixin = {
    /**
     * Bind auto-fetch events
     */
    bindAutoFetchEvents() {
        // Auto-fetch controls
        const autoFetchToggle = document.getElementById('autoFetchToggle');
        if (autoFetchToggle) {
            autoFetchToggle.addEventListener('change', (e) => this.toggleAutoFetch(e.target.checked));
        }
        
        const autoFetchInterval = document.getElementById('autoFetchInterval');
        if (autoFetchInterval) {
            autoFetchInterval.addEventListener('change', (e) => this.setAutoFetchInterval(parseInt(e.target.value)));
        }
        
        // Manual controls
        document.getElementById('testAutoFetchBtn')?.addEventListener('click', () => this.testAutoFetch());
        document.getElementById('enableAllAutoFetchBtn')?.addEventListener('click', () => this.enableAllAutoFetch());
        document.getElementById('resetAutoFetchBtn')?.addEventListener('click', () => this.resetAutoFetch());
    },
    
    /**
     * Setup auto-fetch system
     */
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
    },
    
    /**
     * Toggle auto-fetch on/off
     */
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
        
        if (enabled) {
            if (this.trackedWebsites.length === 0) {
                this.showError('Add websites to track before enabling auto-fetch');
                this.autoFetchEnabled = false;
                localStorage.setItem('newsTracker.autoFetchEnabled', 'false');
                if (toggle) {
                    toggle.checked = false;
                    this.updateToggleAppearance(toggle, false);
                }
                return;
            }
            
            this.startAutoFetch();
            this.showSuccess(`Auto-fetch enabled. Articles will be fetched every ${this.autoFetchIntervalMinutes} minutes.`);
        } else {
            this.stopAutoFetch();
            this.showSuccess('Auto-fetch disabled.');
        }
        
        this.updateAutoFetchStatus();
    },
    
    /**
     * Set auto-fetch interval
     */
    setAutoFetchInterval(minutes) {
        this.autoFetchIntervalMinutes = minutes;
        localStorage.setItem('newsTracker.autoFetchInterval', minutes.toString());
        
        // Restart auto-fetch if enabled to apply new interval
        if (this.autoFetchEnabled) {
            this.stopAutoFetch();
            this.startAutoFetch();
        }
        
        this.updateAutoFetchStatus();
    },
    
    /**
     * Start auto-fetch timer
     */
    startAutoFetch() {
        this.stopAutoFetch(); // Clear existing timer
        
        console.log(`Starting auto-fetch with ${this.autoFetchIntervalMinutes} minute interval`);
        
        const intervalMs = this.autoFetchIntervalMinutes * 60 * 1000;
        
        this.autoFetchInterval = setInterval(() => {
            this.autoFetchArticles();
        }, intervalMs);
        
        // Update next run time
        this.autoFetchStats.nextRun = new Date(Date.now() + intervalMs).toISOString();
        localStorage.setItem('newsTracker.autoFetchStats', JSON.stringify(this.autoFetchStats));
        this.updateAutoFetchStats();
    },
    
    /**
     * Stop auto-fetch timer
     */
    stopAutoFetch() {
        if (this.autoFetchInterval) {
            clearInterval(this.autoFetchInterval);
            this.autoFetchInterval = null;
            console.log('Auto-fetch stopped');
        }
        
        // Clear next run time
        this.autoFetchStats.nextRun = null;
        localStorage.setItem('newsTracker.autoFetchStats', JSON.stringify(this.autoFetchStats));
        this.updateAutoFetchStats();
    },
    
    /**
     * Perform auto-fetch operation
     */
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
            this.autoFetchStats.lastRun = new Date().toISOString();
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
    },
    
    /**
     * Test auto-fetch manually
     */
    async testAutoFetch() {
        if (this.trackedWebsites.length === 0) {
            this.showError('Add websites to track before testing auto-fetch');
            return;
        }
        
        this.showLoading('Testing auto-fetch...');
        
        try {
            const initialCount = this.articleQueue.length;
            await this.fetchAllArticles();
            const foundArticles = this.articleQueue.length - initialCount;
            
            this.showSuccess(`Auto-fetch test completed! Found ${foundArticles} new articles.`);
        } catch (error) {
            console.error('Test auto-fetch error:', error);
            this.showError('Auto-fetch test failed. Please try again.');
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Enable auto-fetch with optimal settings
     */
    async enableAllAutoFetch() {
        if (this.trackedWebsites.length === 0) {
            this.showError('Add websites to track before setting up auto-fetch');
            return;
        }
        
        // Set optimal settings
        this.autoFetchIntervalMinutes = 30;
        this.autoFetchEnabled = true;
        
        // Save to localStorage
        localStorage.setItem('newsTracker.autoFetchEnabled', 'true');
        localStorage.setItem('newsTracker.autoFetchInterval', '30');
        
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
        
        // Start auto-fetch
        this.startAutoFetch();
        this.updateAutoFetchStatus();
        
        this.showSuccess('Auto-fetch enabled with optimal settings (30 minutes interval)');
    },
    
    /**
     * Reset auto-fetch settings
     */
    resetAutoFetch() {
        // Reset to defaults
        this.autoFetchEnabled = false;
        this.autoFetchIntervalMinutes = 30;
        this.autoFetchStats = {
            totalRuns: 0,
            articlesFound: 0,
            lastRun: null,
            nextRun: null
        };
        
        // Clear localStorage
        localStorage.removeItem('newsTracker.autoFetchEnabled');
        localStorage.removeItem('newsTracker.autoFetchInterval');
        localStorage.removeItem('newsTracker.autoFetchStats');
        
        // Stop auto-fetch
        this.stopAutoFetch();
        
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
        
        this.updateAutoFetchStatus();
        this.updateAutoFetchStats();
        
        this.showSuccess('Auto-fetch settings reset to defaults');
    },
    
    /**
     * Update auto-fetch status display
     */
    updateAutoFetchStatus() {
        const statusEl = document.getElementById('autoFetchStatus');
        if (!statusEl) return;
        
        if (this.autoFetchEnabled) {
            const nextRun = this.autoFetchStats.nextRun ? 
                new Date(this.autoFetchStats.nextRun).toLocaleTimeString() : 
                'calculating...';
            
            statusEl.innerHTML = `
                <div class="flex items-center">
                    <i class="bi bi-play-circle text-green-600 mr-2"></i>
                    <span class="text-sm text-green-700">
                        Auto-fetch is running (every ${this.autoFetchIntervalMinutes} minutes)
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
    },
    
    /**
     * Update auto-fetch statistics display
     */
    updateAutoFetchStats() {
        const totalRunsEl = document.getElementById('autoFetchTotalRuns');
        const articlesFoundEl = document.getElementById('autoFetchArticlesFound');
        const lastRunEl = document.getElementById('autoFetchLastRun');
        const nextRunEl = document.getElementById('autoFetchNextRun');
        
        if (totalRunsEl) totalRunsEl.textContent = this.autoFetchStats.totalRuns;
        if (articlesFoundEl) articlesFoundEl.textContent = this.autoFetchStats.articlesFound;
        
        if (lastRunEl) {
            lastRunEl.textContent = this.autoFetchStats.lastRun ? 
                this.formatDate(this.autoFetchStats.lastRun) : 'Never';
        }
        
        if (nextRunEl) {
            if (this.autoFetchEnabled && this.autoFetchStats.nextRun) {
                const nextRun = new Date(this.autoFetchStats.nextRun);
                const now = new Date();
                const diffMs = nextRun - now;
                
                if (diffMs > 0) {
                    const diffMinutes = Math.ceil(diffMs / (1000 * 60));
                    nextRunEl.textContent = `${diffMinutes} min`;
                } else {
                    nextRunEl.textContent = 'Soon';
                }
            } else {
                nextRunEl.textContent = '-';
            }
        }
    },
    
    /**
     * Update toggle appearance
     */
    updateToggleAppearance(toggle, enabled) {
        console.log('updateToggleAppearance called with enabled:', enabled);
        const toggleBg = toggle.nextElementSibling;
        const toggleDot = toggleBg?.querySelector('.toggle-dot') || toggleBg?.nextElementSibling;
        
        console.log('Toggle elements found:', { toggleBg: !!toggleBg, toggleDot: !!toggleDot });
        
        if (toggleBg) {
            if (enabled) {
                toggleBg.classList.remove('bg-gray-200');
                toggleBg.classList.add('bg-blue-600');
            } else {
                toggleBg.classList.remove('bg-blue-600');
                toggleBg.classList.add('bg-gray-200');
            }
        }
        
        if (toggleDot) {
            if (enabled) {
                toggleDot.classList.add('translate-x-6');
            } else {
                toggleDot.classList.remove('translate-x-6');
            }
        }
    }
};
