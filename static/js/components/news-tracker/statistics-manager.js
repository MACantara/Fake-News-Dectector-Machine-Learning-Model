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
        
        // Initialize data containers
        this.trackerData = null;
        this.trackedWebsites = [];
        this.articleQueue = [];
        this.predictionMetrics = null;
        
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
            // Fetch data from the backend API
            const response = await fetch('/api/news-tracker/get-data');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            if (!data.success) {
                throw new Error('Failed to fetch tracker data');
            }
            
            // Store the fetched data
            this.trackerData = data;
            this.trackedWebsites = data.websites || [];
            this.articleQueue = data.articles || [];
            this.predictionMetrics = data.prediction_metrics;
            
            // Update all statistics using the fetched data
            await Promise.all([
                this.updateBasicStats(),
                this.updateVerificationStats(),
                this.updateConfidenceStats(),
                this.updateCategoryStats(),
                this.updateTimelineStats(),
                this.updateTopWebsites()
            ]);
            
            this.renderStatisticsCharts();
            
        } catch (error) {
            console.error('Error updating statistics:', error);
            this.showError('Failed to update statistics');
        }
    },
    
    /**
     * Update basic statistics
     */
    updateBasicStats() {
        // Use backend data if available, otherwise fall back to local data
        const stats = this.trackerData?.stats || {};
        
        // Count tracked websites
        this.statisticsCounters.totalWebsites = stats.total_websites || this.trackedWebsites.length;
        
        // Count articles
        this.statisticsCounters.totalArticles = stats.total_articles || this.articleQueue.length;
        
        // Count verified articles
        this.statisticsCounters.verifiedArticles = stats.verified_articles || 
            this.articleQueue.filter(article => article.verified === 1 || article.verified === true).length;
        
        // Count indexed articles (is_news = true)
        this.statisticsCounters.indexedArticles = this.articleQueue.filter(article => 
            article.is_news === 1 || article.is_news === true || article.isNews === true
        ).length;
        
        // Update DOM elements
        this.updateStatElement('totalArticles', this.statisticsCounters.totalArticles);
        this.updateStatElement('totalVerified', this.statisticsCounters.verifiedArticles);
        
        // Calculate and update accuracy from prediction metrics
        if (this.predictionMetrics?.basic_metrics) {
            const accuracy = (this.predictionMetrics.basic_metrics.accuracy * 100).toFixed(1);
            this.updateStatElement('totalAccuracy', `${accuracy}%`);
        } else {
            const accuracy = this.statisticsCounters.totalArticles > 0 ? 
                (this.statisticsCounters.verifiedArticles / this.statisticsCounters.totalArticles * 100).toFixed(1) : 0;
            this.updateStatElement('totalAccuracy', `${accuracy}%`);
        }
        
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
        
        // Count by fake news prediction - handle multiple field names
        this.statisticsCounters.fakeNews = verifiedArticles.filter(article => 
            article.fake_news_prediction === 1 || article.fake_news_prediction === 'FAKE' ||
            article.prediction === 'FAKE' || article.prediction === 1
        ).length;
        
        this.statisticsCounters.realNews = verifiedArticles.filter(article => 
            article.fake_news_prediction === 0 || article.fake_news_prediction === 'REAL' ||
            article.prediction === 'REAL' || article.prediction === 0
        ).length;
        
        // Count political news - handle multiple field names
        this.statisticsCounters.politicalNews = verifiedArticles.filter(article => 
            article.political_news_prediction === 1 || article.political_news_prediction === 'POLITICAL' ||
            article.political_prediction === 'POLITICAL' || article.political_prediction === 1
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
        
        // Calculate average confidence - handle multiple field names
        const totalConfidence = verifiedArticles.reduce((sum, article) => {
            const confidence = parseFloat(
                article.fake_news_confidence || article.confidence || 
                article.prediction_confidence || 0
            );
            return sum + confidence;
        }, 0);
        
        this.statisticsCounters.averageConfidence = totalConfidence / verifiedArticles.length;
        this.updateStatElement('averageConfidence', `${this.statisticsCounters.averageConfidence.toFixed(1)}%`);
        
        // Confidence distribution
        const highConfidence = verifiedArticles.filter(article => {
            const confidence = parseFloat(
                article.fake_news_confidence || article.confidence || 
                article.prediction_confidence || 0
            );
            return confidence >= 95;
        }).length;
        
        const mediumConfidence = verifiedArticles.filter(article => {
            const confidence = parseFloat(
                article.fake_news_confidence || article.confidence || 
                article.prediction_confidence || 0
            );
            return confidence >= 75 && confidence < 95;
        }).length;
        
        const lowConfidence = verifiedArticles.filter(article => {
            const confidence = parseFloat(
                article.fake_news_confidence || article.confidence || 
                article.prediction_confidence || 0
            );
            return confidence < 75;
        }).length;
        
        this.updateStatElement('highConfidenceCount', highConfidence);
        this.updateStatElement('mediumConfidenceCount', mediumConfidence);
        this.updateStatElement('lowConfidenceCount', lowConfidence);
    },
    
    /**
     * Update category statistics
     */
    updateCategoryStats() {
        const indexedArticles = this.articleQueue.filter(article => 
            article.is_news === 1 || article.is_news === true || article.isNews === true
        );
        
        // Count by news categories
        const categoryStats = {};
        indexedArticles.forEach(article => {
            // Try multiple possible category field names
            const category = article.category || article.news_category || 
                           article.predicted_category || 'Uncategorized';
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
        
        // Count articles for each time period
        Object.entries(ranges).forEach(([period, startDate]) => {
            const articlesInPeriod = this.articleQueue.filter(article => {
                // Use multiple timestamp fields to find article date
                const articleDate = new Date(
                    article.found_at || article.foundAt || 
                    article.created_at || article.published_at || 
                    article.added_at || ''
                );
                return articleDate >= startDate && !isNaN(articleDate.getTime());
            });
            
            const verifiedInPeriod = articlesInPeriod.filter(article => 
                article.verified === 1 || article.verified === true
            );
            
            // Update based on period
            if (period === 'today') {
                this.updateStatElement('todayFound', articlesInPeriod.length);
                this.updateStatElement('todayVerified', verifiedInPeriod.length);
                
                const successRate = articlesInPeriod.length > 0 ? 
                    (verifiedInPeriod.length / articlesInPeriod.length * 100).toFixed(1) : 0;
                this.updateStatElement('todaySuccessRate', `${successRate}%`);
            } else if (period === 'week') {
                this.updateStatElement('weekFound', articlesInPeriod.length);
                this.updateStatElement('weekVerified', verifiedInPeriod.length);
                
                const avgPerDay = (articlesInPeriod.length / 7).toFixed(1);
                this.updateStatElement('weekAverage', avgPerDay);
            }
        });
    },
    
    /**
     * Update top performing websites
     */
    updateTopWebsites() {
        const topWebsitesEl = document.getElementById('topWebsites');
        if (!topWebsitesEl) return;
        
        // Use domain groups from backend if available
        const domainGroups = this.trackerData?.domain_groups || [];
        
        if (domainGroups.length === 0) {
            topWebsitesEl.innerHTML = `
                <div class="text-center text-gray-500">
                    <i class="bi bi-bar-chart text-2xl mb-2 opacity-50"></i>
                    <p class="text-sm">No data available yet.</p>
                </div>
            `;
            return;
        }
        
        // Take top 5 performing websites
        const topSites = domainGroups.slice(0, 5);
        
        topWebsitesEl.innerHTML = `
            <div class="space-y-3">
                ${topSites.map((group, index) => {
                    const totalArticles = group.total_articles || 0;
                    const verifiedArticles = group.verified_articles || 0;
                    const websiteCount = group.website_count || 0;
                    const successRate = totalArticles > 0 ? 
                        (verifiedArticles / totalArticles * 100).toFixed(1) : 0;
                    
                    // Determine badge color based on ranking
                    const badgeColors = [
                        'bg-yellow-500', // 1st place - gold
                        'bg-gray-400',   // 2nd place - silver  
                        'bg-yellow-600', // 3rd place - bronze
                        'bg-blue-500',   // 4th place - blue
                        'bg-green-500'   // 5th place - green
                    ];
                    
                    return `
                        <div class="flex items-center justify-between p-3 bg-white rounded-lg border border-gray-200 hover:shadow-sm transition-shadow">
                            <div class="flex items-center space-x-3">
                                <div class="flex items-center justify-center w-6 h-6 rounded-full ${badgeColors[index]} text-white text-xs font-bold">
                                    ${index + 1}
                                </div>
                                <div>
                                    <div class="font-medium text-gray-900">${group.display_name || group.domain}</div>
                                    <div class="text-xs text-gray-500">${websiteCount} website${websiteCount !== 1 ? 's' : ''}</div>
                                </div>
                            </div>
                            <div class="text-right">
                                <div class="text-sm font-semibold text-gray-900">${totalArticles}</div>
                                <div class="text-xs text-gray-500">${successRate}% verified</div>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
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
            const confidence = parseFloat(
                article.fake_news_confidence || article.confidence || 
                article.prediction_confidence || 0
            );
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
                // Use multiple timestamp fields to find article date
                const articleDate = new Date(
                    article.found_at || article.foundAt || 
                    article.created_at || article.published_at || 
                    article.added_at || ''
                );
                return articleDate >= startOfDay && articleDate < endOfDay && !isNaN(articleDate.getTime());
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
            article.is_news === 1 || article.is_news === true || article.isNews === true
        );
        
        if (indexedArticles.length === 0) {
            chartEl.innerHTML = '<div class="text-center text-gray-500 py-8">No categorized articles yet</div>';
            return;
        }
        
        // Count categories
        const categoryStats = {};
        indexedArticles.forEach(article => {
            // Try multiple possible category field names
            const category = article.category || article.news_category || 
                           article.predicted_category || 'Uncategorized';
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
    },
    
    /**
     * Show error message
     */
    showError(message) {
        console.error(message);
        // Create a temporary error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'fixed top-4 right-4 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg z-50';
        errorDiv.textContent = message;
        document.body.appendChild(errorDiv);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }
};
