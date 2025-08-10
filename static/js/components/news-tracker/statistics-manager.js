/**
 * Statistics Management Mixin
 * Handles metrics display, analytics, and performance tracking
 */

export const StatisticsManagerMixin = {
    /**
     * Bind statistics events
     */
    bindStatisticsEvents() {
        // Statistics refresh controls
        document.getElementById('refreshStatsBtn')?.addEventListener('click', () => this.updateStatistics());
        document.getElementById('clearStatsBtn')?.addEventListener('click', () => this.clearStatistics());
        document.getElementById('exportStatsBtn')?.addEventListener('click', () => this.exportStatistics());
        
        // Chart controls
        document.getElementById('chartTypeSelect')?.addEventListener('change', (e) => this.changeChartType(e.target.value));
        document.getElementById('timeRangeSelect')?.addEventListener('change', (e) => this.changeTimeRange(e.target.value));
    },
    
    /**
     * Setup statistics system
     */
    setupStatistics() {
        // Load statistics preferences
        this.statisticsPrefs = JSON.parse(localStorage.getItem('newsTracker.statisticsPrefs') || '{"chartType":"pie","timeRange":"week","autoRefresh":true}');
        
        // Initialize statistics counters
        this.statisticsCounters = {
            totalWebsites: 0,
            totalArticles: 0,
            verifiedArticles: 0,
            indexedArticles: 0,
            fakeNews: 0,
            realNews: 0,
            politicalNews: 0,
            averageConfidence: 0
        };
        
        this.updateStatistics();
        
        // Auto-refresh statistics every 2 minutes if enabled
        if (this.statisticsPrefs.autoRefresh) {
            this.startStatisticsAutoRefresh();
        }
    },
    
    /**
     * Update all statistics
     */
    async updateStatistics() {
        try {
            await Promise.all([
                this.updateBasicStats(),
                this.updateVerificationStats(),
                this.updateConfidenceStats(),
                this.updateCategoryStats(),
                this.updateTimelineStats()
            ]);
            
            this.renderStatisticsCharts();
            
        } catch (error) {
            console.error('Error updating statistics:', error);
        }
    },
    
    /**
     * Update basic statistics
     */
    updateBasicStats() {
        // Count tracked websites
        this.statisticsCounters.totalWebsites = this.trackedWebsites.length;
        
        // Count articles
        this.statisticsCounters.totalArticles = this.articleQueue.length;
        
        // Count verified articles
        this.statisticsCounters.verifiedArticles = this.articleQueue.filter(article => 
            article.verified === 1 || article.verified === true
        ).length;
        
        // Count indexed articles
        this.statisticsCounters.indexedArticles = this.articleQueue.filter(article => 
            article.is_news === 1 || article.is_news === true
        ).length;
        
        // Update DOM
        this.updateStatElement('totalWebsites', this.statisticsCounters.totalWebsites);
        this.updateStatElement('totalArticles', this.statisticsCounters.totalArticles);
        this.updateStatElement('verifiedArticles', this.statisticsCounters.verifiedArticles);
        this.updateStatElement('indexedArticles', this.statisticsCounters.indexedArticles);
        
        // Calculate percentages
        const verificationRate = this.statisticsCounters.totalArticles > 0 ? 
            (this.statisticsCounters.verifiedArticles / this.statisticsCounters.totalArticles * 100).toFixed(1) : 0;
        const indexingRate = this.statisticsCounters.totalArticles > 0 ? 
            (this.statisticsCounters.indexedArticles / this.statisticsCounters.totalArticles * 100).toFixed(1) : 0;
        
        this.updateStatElement('verificationRate', `${verificationRate}%`);
        this.updateStatElement('indexingRate', `${indexingRate}%`);
    },
    
    /**
     * Update verification statistics
     */
    updateVerificationStats() {
        const verifiedArticles = this.articleQueue.filter(article => 
            article.verified === 1 || article.verified === true
        );
        
        // Count by fake news prediction
        this.statisticsCounters.fakeNews = verifiedArticles.filter(article => 
            article.fake_news_prediction === 1 || article.fake_news_prediction === 'FAKE'
        ).length;
        
        this.statisticsCounters.realNews = verifiedArticles.filter(article => 
            article.fake_news_prediction === 0 || article.fake_news_prediction === 'REAL'
        ).length;
        
        // Count political news
        this.statisticsCounters.politicalNews = verifiedArticles.filter(article => 
            article.political_news_prediction === 1 || article.political_news_prediction === 'POLITICAL'
        ).length;
        
        // Update DOM
        this.updateStatElement('fakeNewsCount', this.statisticsCounters.fakeNews);
        this.updateStatElement('realNewsCount', this.statisticsCounters.realNews);
        this.updateStatElement('politicalNewsCount', this.statisticsCounters.politicalNews);
        
        // Calculate fake news percentage
        const fakeNewsRate = verifiedArticles.length > 0 ? 
            (this.statisticsCounters.fakeNews / verifiedArticles.length * 100).toFixed(1) : 0;
        this.updateStatElement('fakeNewsRate', `${fakeNewsRate}%`);
    },
    
    /**
     * Update confidence statistics
     */
    updateConfidenceStats() {
        const verifiedArticles = this.articleQueue.filter(article => 
            article.verified === 1 || article.verified === true
        );
        
        if (verifiedArticles.length === 0) {
            this.statisticsCounters.averageConfidence = 0;
            this.updateStatElement('averageConfidence', '0%');
            return;
        }
        
        // Calculate average confidence
        const totalConfidence = verifiedArticles.reduce((sum, article) => {
            const confidence = parseFloat(article.fake_news_confidence) || 0;
            return sum + confidence;
        }, 0);
        
        this.statisticsCounters.averageConfidence = totalConfidence / verifiedArticles.length;
        this.updateStatElement('averageConfidence', `${this.statisticsCounters.averageConfidence.toFixed(1)}%`);
        
        // Confidence distribution
        const highConfidence = verifiedArticles.filter(article => 
            parseFloat(article.fake_news_confidence) >= 95
        ).length;
        const mediumConfidence = verifiedArticles.filter(article => {
            const conf = parseFloat(article.fake_news_confidence);
            return conf >= 75 && conf < 95;
        }).length;
        const lowConfidence = verifiedArticles.filter(article => 
            parseFloat(article.fake_news_confidence) < 75
        ).length;
        
        this.updateStatElement('highConfidenceCount', highConfidence);
        this.updateStatElement('mediumConfidenceCount', mediumConfidence);
        this.updateStatElement('lowConfidenceCount', lowConfidence);
    },
    
    /**
     * Update category statistics
     */
    updateCategoryStats() {
        const indexedArticles = this.articleQueue.filter(article => 
            article.is_news === 1 || article.is_news === true
        );
        
        // Count by news categories
        const categoryStats = {};
        indexedArticles.forEach(article => {
            const category = article.category || 'Uncategorized';
            categoryStats[category] = (categoryStats[category] || 0) + 1;
        });
        
        // Sort categories by count
        const sortedCategories = Object.entries(categoryStats)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5); // Top 5 categories
        
        // Update top categories display
        const topCategoriesEl = document.getElementById('topCategories');
        if (topCategoriesEl && sortedCategories.length > 0) {
            topCategoriesEl.innerHTML = sortedCategories
                .map(([category, count]) => `
                    <div class="flex justify-between items-center py-1">
                        <span class="text-sm text-gray-600">${category}</span>
                        <span class="text-sm font-medium text-gray-900">${count}</span>
                    </div>
                `).join('');
        } else if (topCategoriesEl) {
            topCategoriesEl.innerHTML = '<div class="text-sm text-gray-500">No categories available</div>';
        }
    },
    
    /**
     * Update timeline statistics
     */
    updateTimelineStats() {
        const now = new Date();
        const ranges = {
            today: new Date(now.getFullYear(), now.getMonth(), now.getDate()),
            week: new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000),
            month: new Date(now.getFullYear(), now.getMonth(), 1)
        };
        
        Object.entries(ranges).forEach(([period, startDate]) => {
            const articlesInPeriod = this.articleQueue.filter(article => {
                const articleDate = new Date(article.created_at || article.published_at || '');
                return articleDate >= startDate;
            });
            
            this.updateStatElement(`articles${period.charAt(0).toUpperCase() + period.slice(1)}`, articlesInPeriod.length);
        });
    },
    
    /**
     * Render statistics charts
     */
    renderStatisticsCharts() {
        this.renderVerificationChart();
        this.renderConfidenceChart();
        this.renderTimelineChart();
        this.renderCategoryChart();
    },
    
    /**
     * Render verification pie chart
     */
    renderVerificationChart() {
        const chartEl = document.getElementById('verificationChart');
        if (!chartEl) return;
        
        const fakeCount = this.statisticsCounters.fakeNews;
        const realCount = this.statisticsCounters.realNews;
        const total = fakeCount + realCount;
        
        if (total === 0) {
            chartEl.innerHTML = '<div class="text-center text-gray-500 py-8">No verified articles yet</div>';
            return;
        }
        
        const fakePercentage = (fakeCount / total * 100).toFixed(1);
        const realPercentage = (realCount / total * 100).toFixed(1);
        
        chartEl.innerHTML = `
            <div class="space-y-4">
                <div class="flex items-center justify-between">
                    <div class="flex items-center">
                        <div class="w-4 h-4 bg-red-500 rounded mr-2"></div>
                        <span class="text-sm">Fake News</span>
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="text-sm font-medium">${fakeCount}</span>
                        <span class="text-xs text-gray-500">(${fakePercentage}%)</span>
                    </div>
                </div>
                <div class="w-full bg-gray-200 rounded-full h-2">
                    <div class="bg-red-500 h-2 rounded-full" style="width: ${fakePercentage}%"></div>
                </div>
                
                <div class="flex items-center justify-between">
                    <div class="flex items-center">
                        <div class="w-4 h-4 bg-green-500 rounded mr-2"></div>
                        <span class="text-sm">Real News</span>
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="text-sm font-medium">${realCount}</span>
                        <span class="text-xs text-gray-500">(${realPercentage}%)</span>
                    </div>
                </div>
                <div class="w-full bg-gray-200 rounded-full h-2">
                    <div class="bg-green-500 h-2 rounded-full" style="width: ${realPercentage}%"></div>
                </div>
            </div>
        `;
    },
    
    /**
     * Render confidence distribution chart
     */
    renderConfidenceChart() {
        const chartEl = document.getElementById('confidenceChart');
        if (!chartEl) return;
        
        const verifiedArticles = this.articleQueue.filter(article => 
            article.verified === 1 || article.verified === true
        );
        
        if (verifiedArticles.length === 0) {
            chartEl.innerHTML = '<div class="text-center text-gray-500 py-8">No confidence data available</div>';
            return;
        }
        
        // Create confidence buckets
        const buckets = {
            'Very High (95-100%)': 0,
            'High (85-94%)': 0,
            'Medium (75-84%)': 0,
            'Low (60-74%)': 0,
            'Very Low (<60%)': 0
        };
        
        verifiedArticles.forEach(article => {
            const confidence = parseFloat(article.fake_news_confidence) || 0;
            if (confidence >= 95) buckets['Very High (95-100%)']++;
            else if (confidence >= 85) buckets['High (85-94%)']++;
            else if (confidence >= 75) buckets['Medium (75-84%)']++;
            else if (confidence >= 60) buckets['Low (60-74%)']++;
            else buckets['Very Low (<60%)']++;
        });
        
        const maxCount = Math.max(...Object.values(buckets));
        
        chartEl.innerHTML = Object.entries(buckets)
            .map(([range, count]) => {
                const percentage = maxCount > 0 ? (count / maxCount * 100) : 0;
                return `
                    <div class="flex items-center justify-between mb-3">
                        <span class="text-xs text-gray-600 w-24">${range}</span>
                        <div class="flex-1 mx-3">
                            <div class="w-full bg-gray-200 rounded-full h-2">
                                <div class="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                                     style="width: ${percentage}%"></div>
                            </div>
                        </div>
                        <span class="text-xs font-medium w-8 text-right">${count}</span>
                    </div>
                `;
            }).join('');
    },
    
    /**
     * Render timeline chart
     */
    renderTimelineChart() {
        const chartEl = document.getElementById('timelineChart');
        if (!chartEl) return;
        
        // Create 7-day timeline
        const days = [];
        for (let i = 6; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            days.push(date);
        }
        
        const dailyCounts = days.map(date => {
            const startOfDay = new Date(date.getFullYear(), date.getMonth(), date.getDate());
            const endOfDay = new Date(startOfDay.getTime() + 24 * 60 * 60 * 1000);
            
            return this.articleQueue.filter(article => {
                const articleDate = new Date(article.created_at || article.published_at || '');
                return articleDate >= startOfDay && articleDate < endOfDay;
            }).length;
        });
        
        const maxCount = Math.max(...dailyCounts, 1);
        
        chartEl.innerHTML = days.map((date, index) => {
            const count = dailyCounts[index];
            const height = (count / maxCount * 100);
            const dayName = date.toLocaleDateString('en-US', { weekday: 'short' });
            
            return `
                <div class="flex flex-col items-center">
                    <div class="w-8 bg-gray-200 rounded-sm relative" style="height: 60px;">
                        <div class="bg-blue-500 rounded-sm absolute bottom-0 w-full transition-all duration-300" 
                             style="height: ${height}%"></div>
                    </div>
                    <span class="text-xs text-gray-500 mt-1">${dayName}</span>
                    <span class="text-xs font-medium">${count}</span>
                </div>
            `;
        }).join('');
    },
    
    /**
     * Render category distribution chart
     */
    renderCategoryChart() {
        const chartEl = document.getElementById('categoryChart');
        if (!chartEl) return;
        
        const indexedArticles = this.articleQueue.filter(article => 
            article.is_news === 1 || article.is_news === true
        );
        
        if (indexedArticles.length === 0) {
            chartEl.innerHTML = '<div class="text-center text-gray-500 py-8">No categorized articles yet</div>';
            return;
        }
        
        // Count categories
        const categoryStats = {};
        indexedArticles.forEach(article => {
            const category = article.category || 'Uncategorized';
            categoryStats[category] = (categoryStats[category] || 0) + 1;
        });
        
        const sortedCategories = Object.entries(categoryStats)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 8); // Top 8 categories
        
        const maxCount = Math.max(...sortedCategories.map(([, count]) => count));
        
        chartEl.innerHTML = sortedCategories
            .map(([category, count]) => {
                const percentage = (count / maxCount * 100);
                return `
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-xs text-gray-600 truncate w-20">${category}</span>
                        <div class="flex-1 mx-2">
                            <div class="w-full bg-gray-200 rounded-full h-2">
                                <div class="bg-purple-500 h-2 rounded-full transition-all duration-300" 
                                     style="width: ${percentage}%"></div>
                            </div>
                        </div>
                        <span class="text-xs font-medium w-6 text-right">${count}</span>
                    </div>
                `;
            }).join('');
    },
    
    /**
     * Clear all statistics
     */
    clearStatistics() {
        if (!confirm('Are you sure you want to clear all statistics? This action cannot be undone.')) {
            return;
        }
        
        // Reset counters
        this.statisticsCounters = {
            totalWebsites: 0,
            totalArticles: 0,
            verifiedArticles: 0,
            indexedArticles: 0,
            fakeNews: 0,
            realNews: 0,
            politicalNews: 0,
            averageConfidence: 0
        };
        
        // Clear auto-fetch stats
        this.autoFetchStats = {
            totalRuns: 0,
            articlesFound: 0,
            lastRun: null,
            nextRun: null
        };
        
        // Clear auto-index stats  
        this.autoIndexStats = {
            totalIndexed: 0,
            lastBatchSize: 0,
            lastBatchTime: null,
            successRate: 0
        };
        
        // Save to localStorage
        localStorage.setItem('newsTracker.autoFetchStats', JSON.stringify(this.autoFetchStats));
        localStorage.setItem('newsTracker.autoIndexStats', JSON.stringify(this.autoIndexStats));
        
        // Update displays
        this.updateStatistics();
        this.updateAutoFetchStats();
        this.updateAutoIndexStatsDisplay();
        
        this.showSuccess('Statistics cleared successfully');
    },
    
    /**
     * Export statistics to JSON
     */
    exportStatistics() {
        const exportData = {
            timestamp: new Date().toISOString(),
            basicStats: this.statisticsCounters,
            autoFetchStats: this.autoFetchStats,
            autoIndexStats: this.autoIndexStats,
            trackedWebsites: this.trackedWebsites.length,
            articleQueue: this.articleQueue.length,
            preferences: this.statisticsPrefs
        };
        
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `news-tracker-stats-${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        
        this.showSuccess('Statistics exported successfully');
    },
    
    /**
     * Start auto-refresh for statistics
     */
    startStatisticsAutoRefresh() {
        // Refresh every 2 minutes
        this.statisticsRefreshInterval = setInterval(() => {
            this.updateStatistics();
        }, 2 * 60 * 1000);
    },
    
    /**
     * Stop auto-refresh for statistics
     */
    stopStatisticsAutoRefresh() {
        if (this.statisticsRefreshInterval) {
            clearInterval(this.statisticsRefreshInterval);
            this.statisticsRefreshInterval = null;
        }
    },
    
    /**
     * Update statistic element
     */
    updateStatElement(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
        }
    },
    
    /**
     * Change chart type
     */
    changeChartType(type) {
        this.statisticsPrefs.chartType = type;
        localStorage.setItem('newsTracker.statisticsPrefs', JSON.stringify(this.statisticsPrefs));
        this.renderStatisticsCharts();
    },
    
    /**
     * Change time range
     */
    changeTimeRange(range) {
        this.statisticsPrefs.timeRange = range;
        localStorage.setItem('newsTracker.statisticsPrefs', JSON.stringify(this.statisticsPrefs));
        this.updateTimelineStats();
        this.renderTimelineChart();
    }
};
