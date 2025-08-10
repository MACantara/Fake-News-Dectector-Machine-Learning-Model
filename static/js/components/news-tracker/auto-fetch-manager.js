/**
 * Auto-fetch Management Mixin
 * Handles automatic article fetching with robust scheduling system
 */

import { AutoFetchScheduler } from './auto-fetch-scheduler.js';

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
     * Setup auto-fetch system with robust scheduler
     */
    setupAutoFetch() {
        // Initialize the robust scheduler
        this.autoFetchScheduler = new AutoFetchScheduler(this);
        
        // Load auto-fetch preferences from localStorage
        const autoFetchEnabled = localStorage.getItem('newsTracker.autoFetchEnabled') === 'true';
        const autoFetchInterval = parseInt(localStorage.getItem('newsTracker.autoFetchInterval')) || 30;
        
        this.autoFetchEnabled = autoFetchEnabled;
        this.autoFetchIntervalMinutes = autoFetchInterval;
        
        // Set scheduler interval
        this.autoFetchScheduler.setInterval(autoFetchInterval);
        
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
        
        // Start scheduler if enabled
        if (autoFetchEnabled && this.trackedWebsites.length > 0) {
            this.autoFetchScheduler.start();
        }
        
        // Update UI displays
        this.updateAutoFetchStatus();
        this.updateAutoFetchStats();
        
        console.log('Auto-fetch system setup complete with robust scheduler');
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
            
            this.autoFetchScheduler.start();
            this.showSuccess(`Auto-fetch enabled with robust scheduling (${this.autoFetchIntervalMinutes} minute intervals)`);
        } else {
            this.autoFetchScheduler.stop();
            this.showSuccess('Auto-fetch disabled');
        }
        
        this.updateAutoFetchStatus();
    },
    
    /**
     * Set auto-fetch interval
     */
    setAutoFetchInterval(minutes) {
        this.autoFetchIntervalMinutes = minutes;
        localStorage.setItem('newsTracker.autoFetchInterval', minutes.toString());
        
        // Update scheduler interval
        if (this.autoFetchScheduler) {
            this.autoFetchScheduler.setInterval(minutes);
        }
        
        this.updateAutoFetchStatus();
        console.log(`Auto-fetch interval set to ${minutes} minutes`);
    },
    
    /**
     * Start auto-fetch (legacy compatibility)
     */
    startAutoFetch() {
        if (this.autoFetchScheduler) {
            this.autoFetchScheduler.start();
        }
    },
    
    /**
     * Stop auto-fetch (legacy compatibility)
     */
    stopAutoFetch() {
        if (this.autoFetchScheduler) {
            this.autoFetchScheduler.stop();
        }
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
            
            // Update scheduler stats
            this.autoFetchScheduler.updateStats({
                articlesFound: foundArticles,
                lastRun: new Date().toISOString()
            });
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
                        
                        this.showSuccess(`Auto-indexed ${result.articles_indexed} high confidence articles`);
                    } else if (result.success && result.articles_indexed === 0) {
                        console.log('Auto-indexing: No high confidence articles found for indexing');
                        this.updateRealtimeActivity('No high confidence articles found');
                        setTimeout(() => this.hideRealtimeActivity(), 2000);
                    } else {
                        console.error('Auto-indexing failed:', result.error);
                        this.updateRealtimeActivity('Auto-indexing failed');
                        setTimeout(() => this.hideRealtimeActivity(), 3000);
                    }
                } catch (error) {
                    console.error('Auto-indexing error:', error);
                    this.updateRealtimeActivity('Auto-indexing error');
                    setTimeout(() => this.hideRealtimeActivity(), 3000);
                }
            } else if (this.autoIndexEnabled && foundArticles === 0) {
                console.log('Auto-indexing: No new articles to check for high confidence');
            }
            
            // Show notification about the auto-fetch result
            if (foundArticles > 0) {
                this.showRealtimeActivity(`Auto-fetch found ${foundArticles} new articles`);
                setTimeout(() => this.hideRealtimeActivity(), 2000);
            }
            
        } catch (error) {
            console.error('Auto-fetch error:', error);
            this.showError('Auto-fetch failed: ' + error.message);
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
        this.autoFetchScheduler.start();
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
        
        // Clear localStorage
        localStorage.removeItem('newsTracker.autoFetchEnabled');
        localStorage.removeItem('newsTracker.autoFetchInterval');
        
        // Reset scheduler
        this.autoFetchScheduler.reset();
        
        // Stop auto-fetch
        this.autoFetchScheduler.stop();
        
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
            const schedulerStats = this.autoFetchScheduler.getStats();
            const nextRun = schedulerStats.nextRun ? 
                new Date(schedulerStats.nextRun).toLocaleTimeString() : 
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
                    <span class="text-sm text-gray-600">Auto-fetch is stopped</span>
                </div>
            `;
            statusEl.className = 'p-3 bg-gray-50 rounded-lg border border-gray-200';
        }
    },
    
    /**
     * Update auto-fetch statistics display
     */
    updateAutoFetchStats() {
        const schedulerStats = this.autoFetchScheduler.getStats();
        
        const totalRunsEl = document.getElementById('autoFetchTotalRuns');
        const articlesFoundEl = document.getElementById('autoFetchArticlesFound');
        const lastRunEl = document.getElementById('autoFetchLastRun');
        const nextRunEl = document.getElementById('autoFetchNextRun');
        
        if (totalRunsEl) totalRunsEl.textContent = schedulerStats.totalRuns;
        if (articlesFoundEl) articlesFoundEl.textContent = schedulerStats.articlesFound;
        
        if (lastRunEl) {
            lastRunEl.textContent = schedulerStats.lastRun ? 
                this.formatDate(schedulerStats.lastRun) : 'Never';
        }
        
        if (nextRunEl) {
            if (this.autoFetchEnabled && schedulerStats.nextRun) {
                const nextRun = new Date(schedulerStats.nextRun);
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
