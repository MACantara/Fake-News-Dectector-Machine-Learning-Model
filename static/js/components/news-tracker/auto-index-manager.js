/**
 * Auto-Index Management Mixin
 * Handles automatic indexing of high confidence articles
 */

export const AutoIndexManagerMixin = {
    /**
     * Bind auto-index events
     */
    bindAutoIndexEvents() {
        // Auto-index controls
        const autoIndexToggle = document.getElementById('autoIndexToggle');
        if (autoIndexToggle) {
            autoIndexToggle.addEventListener('change', (e) => this.toggleAutoIndex(e.target.checked));
        }
        
        const autoIndexThreshold = document.getElementById('autoIndexThreshold');
        if (autoIndexThreshold) {
            autoIndexThreshold.addEventListener('change', (e) => this.setAutoIndexThreshold(parseFloat(e.target.value)));
        }
        
        const autoIndexBatchSize = document.getElementById('autoIndexBatchSize');
        if (autoIndexBatchSize) {
            autoIndexBatchSize.addEventListener('change', (e) => this.setAutoIndexBatchSize(parseInt(e.target.value)));
        }
        
        // Manual controls
        const triggerAutoIndexBtn = document.getElementById('triggerAutoIndexBtn');
        if (triggerAutoIndexBtn) {
            triggerAutoIndexBtn.addEventListener('click', () => this.triggerAutoIndex());
            console.log('triggerAutoIndexBtn event listener bound successfully');
        } else {
            console.warn('triggerAutoIndexBtn not found in DOM');
        }
        
        document.getElementById('testAutoIndexBtn')?.addEventListener('click', () => this.testAutoIndex());
        document.getElementById('runAutoIndexBtn')?.addEventListener('click', () => this.runAutoIndex());
        document.getElementById('resetAutoIndexBtn')?.addEventListener('click', () => this.resetAutoIndex());
    },
    
    /**
     * Setup auto-index system
     */
    setupAutoIndex() {
        // Load auto-index preferences from localStorage
        const autoIndexEnabled = localStorage.getItem('newsTracker.autoIndexEnabled') === 'true';
        const autoIndexThreshold = parseFloat(localStorage.getItem('newsTracker.autoIndexThreshold')) || 95.0;
        const autoIndexBatchSize = parseInt(localStorage.getItem('newsTracker.autoIndexBatchSize')) || 10;
        const autoIndexStats = JSON.parse(localStorage.getItem('newsTracker.autoIndexStats') || '{"totalIndexed":0,"lastBatchSize":0,"lastBatchTime":null,"successRate":0}');
        
        this.autoIndexEnabled = autoIndexEnabled;
        this.autoIndexThreshold = autoIndexThreshold;
        this.autoIndexBatchSize = autoIndexBatchSize;
        this.autoIndexStats = autoIndexStats;
        
        // Update UI to reflect current state
        const toggle = document.getElementById('autoIndexToggle');
        const thresholdInput = document.getElementById('autoIndexThreshold');
        const batchSizeInput = document.getElementById('autoIndexBatchSize');
        
        if (toggle) {
            toggle.checked = autoIndexEnabled;
            this.updateToggleAppearance(toggle, autoIndexEnabled);
        }
        
        if (thresholdInput) {
            thresholdInput.value = autoIndexThreshold.toString();
        }
        
        if (batchSizeInput) {
            batchSizeInput.value = autoIndexBatchSize.toString();
        }
        
        // Set initial visibility of auto-index settings
        const autoIndexSettings = document.getElementById('autoIndexSettings');
        if (autoIndexSettings) {
            autoIndexSettings.style.display = autoIndexEnabled ? 'block' : 'none';
        }
        
        // Set initial visibility of auto-index stats
        const autoIndexStatsEl = document.getElementById('autoIndexStats');
        if (autoIndexStatsEl) {
            autoIndexStatsEl.style.display = autoIndexEnabled ? 'block' : 'none';
        }
        
        this.updateAutoIndexStatus();
        this.updateAutoIndexStatsDisplay();
        this.updateHighConfidenceQueue();
    },
    
    /**
     * Toggle auto-index on/off
     */
    toggleAutoIndex(enabled) {
        console.log('toggleAutoIndex called with enabled:', enabled);
        this.autoIndexEnabled = enabled;
        
        // Save preference to localStorage
        localStorage.setItem('newsTracker.autoIndexEnabled', enabled.toString());
        
        // Update toggle appearance
        const toggle = document.getElementById('autoIndexToggle');
        if (toggle) {
            this.updateToggleAppearance(toggle, enabled);
        }
        
        // Show/hide auto-index settings
        const autoIndexSettings = document.getElementById('autoIndexSettings');
        if (autoIndexSettings) {
            autoIndexSettings.style.display = enabled ? 'block' : 'none';
        }
        
        // Show/hide auto-index stats
        const autoIndexStatsEl = document.getElementById('autoIndexStats');
        if (autoIndexStatsEl) {
            autoIndexStatsEl.style.display = enabled ? 'block' : 'none';
        }
        
        if (enabled) {
            this.showSuccess(`Auto-indexing enabled for articles with ${this.autoIndexThreshold}%+ confidence.`);
        } else {
            this.showSuccess('Auto-indexing disabled.');
        }
        
        this.updateAutoIndexStatus();
    },
    
    /**
     * Set auto-index confidence threshold
     */
    setAutoIndexThreshold(threshold) {
        this.autoIndexThreshold = threshold;
        localStorage.setItem('newsTracker.autoIndexThreshold', threshold.toString());
        
        this.updateAutoIndexStatus();
        this.updateHighConfidenceQueue();
        
        console.log(`Auto-index threshold set to ${threshold}%`);
    },
    
    /**
     * Set auto-index batch size
     */
    setAutoIndexBatchSize(batchSize) {
        this.autoIndexBatchSize = batchSize;
        localStorage.setItem('newsTracker.autoIndexBatchSize', batchSize.toString());
        
        this.updateAutoIndexStatus();
        
        console.log(`Auto-index batch size set to ${batchSize}`);
    },
    
    /**
     * Trigger auto-index operation (alias for runAutoIndex)
     */
    async triggerAutoIndex() {
        console.log('triggerAutoIndex called');
        return await this.runAutoIndex();
    },
    
    /**
     * Run auto-index operation manually
     */
    async runAutoIndex() {
        try {
            this.showLoading('Running auto-index...');
            
            const response = await fetch('/api/news-tracker/auto-index-high-confidence', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    confidence_threshold: this.autoIndexThreshold / 100, // Convert percentage to decimal
                    batch_size: this.autoIndexBatchSize
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                if (result.articles_indexed > 0) {
                    this.showSuccess(`Successfully indexed ${result.articles_indexed} high confidence articles!`);
                    
                    // Update auto-indexing stats
                    this.autoIndexStats.totalIndexed += result.articles_indexed;
                    this.autoIndexStats.lastBatchSize = result.articles_indexed;
                    this.autoIndexStats.lastBatchTime = new Date().toISOString();
                    this.autoIndexStats.successRate = result.stats?.success_rate || 0;
                    localStorage.setItem('newsTracker.autoIndexStats', JSON.stringify(this.autoIndexStats));
                    this.updateAutoIndexStatsDisplay();
                    
                    // Refresh the UI to show updated verification status
                    await this.loadTrackedWebsites();
                    this.updateHighConfidenceQueue();
                    this.updateStatistics();
                    
                } else {
                    this.showNotification('No articles found that meet the high confidence criteria.', 'info');
                }
            } else {
                this.showError(`Auto-index failed: ${result.message || 'Unknown error'}`);
            }
            
        } catch (error) {
            console.error('Auto-index error:', error);
            this.showError('Auto-index operation failed. Please try again.');
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Test auto-index with current settings
     */
    async testAutoIndex() {
        try {
            this.showLoading('Testing auto-index settings...');
            
            // Count eligible articles
            const eligibleArticles = this.articleQueue.filter(article => 
                (article.verified === 1 || article.verified === true) &&
                (article.is_news === 0 || article.is_news === false || !article.is_news) &&
                parseFloat(article.fake_news_confidence) >= this.autoIndexThreshold
            );
            
            if (eligibleArticles.length === 0) {
                this.showNotification(`No articles found with ${this.autoIndexThreshold}%+ confidence that haven't been indexed yet.`, 'info');
                return;
            }
            
            // Show test results
            const message = `Found ${eligibleArticles.length} articles eligible for auto-indexing:\n` +
                `- Confidence threshold: ${this.autoIndexThreshold}%\n` +
                `- Batch size: ${this.autoIndexBatchSize}\n` +
                `- Next batch would process: ${Math.min(eligibleArticles.length, this.autoIndexBatchSize)} articles`;
            
            this.showNotification(message, 'info');
            
        } catch (error) {
            console.error('Test auto-index error:', error);
            this.showError('Auto-index test failed. Please try again.');
        } finally {
            this.hideLoading();
        }
    },
    
    /**
     * Reset auto-index settings
     */
    resetAutoIndex() {
        // Reset to defaults
        this.autoIndexEnabled = false;
        this.autoIndexThreshold = 95.0;
        this.autoIndexBatchSize = 10;
        this.autoIndexStats = {
            totalIndexed: 0,
            lastBatchSize: 0,
            lastBatchTime: null,
            successRate: 0
        };
        
        // Clear localStorage
        localStorage.removeItem('newsTracker.autoIndexEnabled');
        localStorage.removeItem('newsTracker.autoIndexThreshold');
        localStorage.removeItem('newsTracker.autoIndexBatchSize');
        localStorage.removeItem('newsTracker.autoIndexStats');
        
        // Update UI
        const toggle = document.getElementById('autoIndexToggle');
        const thresholdInput = document.getElementById('autoIndexThreshold');
        const batchSizeInput = document.getElementById('autoIndexBatchSize');
        
        if (toggle) {
            toggle.checked = false;
            this.updateToggleAppearance(toggle, false);
        }
        
        if (thresholdInput) {
            thresholdInput.value = '95.0';
        }
        
        if (batchSizeInput) {
            batchSizeInput.value = '10';
        }
        
        this.updateAutoIndexStatus();
        this.updateAutoIndexStatsDisplay();
        this.updateHighConfidenceQueue();
        
        this.showSuccess('Auto-index settings reset to defaults');
    },
    
    /**
     * Update high confidence queue display
     */
    updateHighConfidenceQueue() {
        const queueEl = document.getElementById('highConfidenceQueue');
        if (!queueEl) return;
        
        // Find articles that meet the high confidence criteria
        const eligibleArticles = this.articleQueue.filter(article => 
            (article.verified === 1 || article.verified === true) &&
            (article.is_news === 0 || article.is_news === false || !article.is_news) &&
            parseFloat(article.fake_news_confidence) >= this.autoIndexThreshold
        );
        
        if (eligibleArticles.length === 0) {
            queueEl.innerHTML = `
                <div class="text-center py-8 text-gray-500">
                    <i class="bi bi-inbox text-4xl mb-2"></i>
                    <p>No articles in high confidence queue</p>
                    <p class="text-sm">Articles with ${this.autoIndexThreshold}%+ confidence will appear here</p>
                </div>
            `;
            return;
        }
        
        // Sort by confidence (highest first)
        eligibleArticles.sort((a, b) => 
            parseFloat(b.fake_news_confidence) - parseFloat(a.fake_news_confidence)
        );
        
        // Show up to 10 articles in the preview
        const previewArticles = eligibleArticles.slice(0, 10);
        
        queueEl.innerHTML = `
            <div class="space-y-3">
                <div class="flex justify-between items-center">
                    <h4 class="font-medium text-gray-900">High Confidence Queue (${eligibleArticles.length})</h4>
                    <span class="text-xs text-gray-500">Showing ${previewArticles.length} of ${eligibleArticles.length}</span>
                </div>
                
                ${previewArticles.map(article => `
                    <div class="border border-gray-200 rounded-lg p-3 bg-white">
                        <div class="flex justify-between items-start mb-2">
                            <div class="flex-1 pr-3">
                                <h5 class="font-medium text-sm text-gray-900 line-clamp-2">
                                    ${this.escapeHtml(article.title || 'Untitled')}
                                </h5>
                                <p class="text-xs text-gray-600 mt-1">
                                    ${this.escapeHtml(article.website_name || 'Unknown Source')}
                                </p>
                            </div>
                            <div class="flex flex-col items-end space-y-1">
                                <span class="text-xs font-medium px-2 py-1 rounded-full ${
                                    parseFloat(article.fake_news_confidence) >= 98 ? 'bg-green-100 text-green-800' :
                                    parseFloat(article.fake_news_confidence) >= 95 ? 'bg-blue-100 text-blue-800' :
                                    'bg-yellow-100 text-yellow-800'
                                }">
                                    ${parseFloat(article.fake_news_confidence).toFixed(1)}%
                                </span>
                                <span class="text-xs px-2 py-1 rounded-full ${
                                    article.fake_news_prediction === 'FAKE' || article.fake_news_prediction === 1 ?
                                    'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'
                                }">
                                    ${article.fake_news_prediction === 'FAKE' || article.fake_news_prediction === 1 ? 'FAKE' : 'REAL'}
                                </span>
                            </div>
                        </div>
                        
                        <div class="flex justify-between items-center text-xs text-gray-500">
                            <span>${this.formatDate(article.published_at || article.created_at)}</span>
                            <span class="flex items-center">
                                <i class="bi bi-clock mr-1"></i>
                                Queued for indexing
                            </span>
                        </div>
                    </div>
                `).join('')}
                
                ${eligibleArticles.length > 10 ? `
                    <div class="text-center py-2">
                        <span class="text-xs text-gray-500">
                            ${eligibleArticles.length - 10} more articles in queue
                        </span>
                    </div>
                ` : ''}
            </div>
        `;
    },
    
    /**
     * Update auto-index status display
     */
    updateAutoIndexStatus() {
        const statusEl = document.getElementById('autoIndexStatus');
        if (!statusEl) return;
        
        if (this.autoIndexEnabled) {
            statusEl.innerHTML = `
                <div class="flex items-center">
                    <i class="bi bi-gear-fill text-green-600 mr-2"></i>
                    <span class="text-sm text-green-700">
                        Auto-indexing is enabled (${this.autoIndexThreshold}%+ confidence)
                    </span>
                </div>
                <div class="text-xs text-green-600 mt-1">
                    Batch size: ${this.autoIndexBatchSize} articles
                </div>
            `;
            statusEl.className = 'p-3 bg-green-50 rounded-lg border border-green-200';
        } else {
            statusEl.innerHTML = `
                <div class="flex items-center">
                    <i class="bi bi-gear text-gray-500 mr-2"></i>
                    <span class="text-sm text-gray-600">Auto-indexing is currently disabled</span>
                </div>
            `;
            statusEl.className = 'p-3 bg-gray-50 rounded-lg border border-gray-200';
        }
    },
    
    /**
     * Update auto-index statistics display
     */
    updateAutoIndexStatsDisplay() {
        const totalIndexedEl = document.getElementById('autoIndexTotalIndexed');
        const lastBatchSizeEl = document.getElementById('autoIndexLastBatchSize');
        const lastBatchTimeEl = document.getElementById('autoIndexLastBatchTime');
        const successRateEl = document.getElementById('autoIndexSuccessRate');
        
        if (totalIndexedEl) totalIndexedEl.textContent = this.autoIndexStats.totalIndexed;
        if (lastBatchSizeEl) lastBatchSizeEl.textContent = this.autoIndexStats.lastBatchSize;
        if (successRateEl) successRateEl.textContent = `${this.autoIndexStats.successRate.toFixed(1)}%`;
        
        if (lastBatchTimeEl) {
            lastBatchTimeEl.textContent = this.autoIndexStats.lastBatchTime ? 
                this.formatDate(this.autoIndexStats.lastBatchTime) : 'Never';
        }
    },
    
    /**
     * Update toggle appearance
     */
    updateToggleAppearance(toggle, enabled) {
        const toggleBg = toggle.nextElementSibling;
        const toggleDot = toggleBg?.querySelector('.toggle-dot') || toggleBg?.nextElementSibling;
        
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
