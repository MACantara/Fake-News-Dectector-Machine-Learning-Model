// Advanced features and user preferences module
class AdvancedFeatures {
    constructor(newsAnalyzer) {
        this.analyzer = newsAnalyzer;
        this.preferences = this.loadPreferences();
        this.initializeFeatures();
    }

    // Initialize advanced features
    initializeFeatures() {
        this.setupKeyboardShortcuts();
        this.setupAutoSave();
        this.setupThemeToggle();
        this.setupExportFeatures();
        this.setupAnalyticsTracking();
    }

    // Load user preferences
    loadPreferences() {
        return {
            theme: Utils.storage.get('theme', 'light'),
            autoSave: Utils.storage.get('autoSave', true),
            keyboardShortcuts: Utils.storage.get('keyboardShortcuts', true),
            analytics: Utils.storage.get('analytics', false),
            exportFormat: Utils.storage.get('exportFormat', 'json'),
            ...Utils.storage.get('userPreferences', {})
        };
    }

    // Save user preferences
    savePreferences() {
        Utils.storage.set('userPreferences', this.preferences);
        Utils.storage.set('theme', this.preferences.theme);
        Utils.storage.set('autoSave', this.preferences.autoSave);
        Utils.storage.set('keyboardShortcuts', this.preferences.keyboardShortcuts);
        Utils.storage.set('analytics', this.preferences.analytics);
        Utils.storage.set('exportFormat', this.preferences.exportFormat);
    }

    // Setup keyboard shortcuts
    setupKeyboardShortcuts() {
        if (!this.preferences.keyboardShortcuts) return;

        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter: Analyze
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                if (!this.analyzer.elements.analyzeBtn.disabled) {
                    this.analyzer.analyzeContent();
                }
            }

            // Ctrl/Cmd + 1: Switch to text input
            if ((e.ctrlKey || e.metaKey) && e.key === '1') {
                e.preventDefault();
                this.analyzer.switchInputType(Config.inputTypes.TEXT);
            }

            // Ctrl/Cmd + 2: Switch to URL input
            if ((e.ctrlKey || e.metaKey) && e.key === '2') {
                e.preventDefault();
                this.analyzer.switchInputType(Config.inputTypes.URL);
            }

            // Ctrl/Cmd + F: Focus on text input
            if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
                e.preventDefault();
                if (this.analyzer.state.currentInputType === Config.inputTypes.TEXT) {
                    this.analyzer.elements.newsText?.focus();
                } else {
                    this.analyzer.elements.articleUrl?.focus();
                }
            }

            // Escape: Clear results
            if (e.key === 'Escape') {
                this.analyzer.hideResults();
            }
        });
    }

    // Setup auto-save functionality
    setupAutoSave() {
        if (!this.preferences.autoSave) return;

        const autoSaveInterval = Config.intervals.autoSave || 60000;
        
        setInterval(() => {
            const textContent = this.analyzer.elements.newsText?.value;
            const urlContent = this.analyzer.elements.articleUrl?.value;
            
            if (textContent && textContent.length > 10) {
                Utils.storage.set('autoSavedText', textContent);
            }
            
            if (urlContent) {
                Utils.storage.set('autoSavedUrl', urlContent);
            }
        }, autoSaveInterval);

        // Restore auto-saved content on load
        this.restoreAutoSavedContent();
    }

    // Restore auto-saved content
    restoreAutoSavedContent() {
        const savedText = Utils.storage.get('autoSavedText');
        const savedUrl = Utils.storage.get('autoSavedUrl');

        if (savedText && this.analyzer.elements.newsText) {
            this.analyzer.elements.newsText.value = savedText;
            this.analyzer.updateTextCount();
        }

        if (savedUrl && this.analyzer.elements.articleUrl) {
            this.analyzer.elements.articleUrl.value = savedUrl;
        }
    }

    // Setup theme toggle
    setupThemeToggle() {
        // Apply saved theme
        this.applyTheme(this.preferences.theme);

        // Create theme toggle button (if element exists)
        const themeToggle = Utils.dom.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                this.toggleTheme();
            });
        }
    }

    // Toggle between light and dark themes
    toggleTheme() {
        this.preferences.theme = this.preferences.theme === 'light' ? 'dark' : 'light';
        this.applyTheme(this.preferences.theme);
        this.savePreferences();
    }

    // Apply theme to document
    applyTheme(theme) {
        document.documentElement.classList.toggle('dark', theme === 'dark');
        
        // Update theme toggle button if it exists
        const themeToggle = Utils.dom.getElementById('themeToggle');
        if (themeToggle) {
            const icon = themeToggle.querySelector('i');
            if (icon) {
                icon.className = theme === 'dark' ? 'bi bi-sun' : 'bi bi-moon';
            }
        }
    }

    // Setup export features
    setupExportFeatures() {
        // Add export button event listeners if they exist
        const exportJsonBtn = Utils.dom.getElementById('exportJson');
        const exportCsvBtn = Utils.dom.getElementById('exportCsv');
        const exportPdfBtn = Utils.dom.getElementById('exportPdf');

        if (exportJsonBtn) {
            exportJsonBtn.addEventListener('click', () => this.exportResults('json'));
        }
        if (exportCsvBtn) {
            exportCsvBtn.addEventListener('click', () => this.exportResults('csv'));
        }
        if (exportPdfBtn) {
            exportPdfBtn.addEventListener('click', () => this.exportResults('pdf'));
        }
    }

    // Export analysis results
    exportResults(format) {
        const results = this.getLastResults();
        if (!results) {
            Utils.dom.show(this.createNotification('No results to export', 'warning'));
            return;
        }

        switch (format) {
            case 'json':
                this.exportAsJson(results);
                break;
            case 'csv':
                this.exportAsCsv(results);
                break;
            case 'pdf':
                this.exportAsPdf(results);
                break;
            default:
                console.warn('Unknown export format:', format);
        }
    }

    // Get last analysis results
    getLastResults() {
        return Utils.storage.get('lastAnalysisResults');
    }

    // Export as JSON
    exportAsJson(results) {
        const dataStr = JSON.stringify(results, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `news-analysis-${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        
        URL.revokeObjectURL(url);
    }

    // Export as CSV
    exportAsCsv(results) {
        const csvRows = [
            ['Timestamp', 'Analysis Type', 'Prediction', 'Confidence', 'Text Preview']
        ];

        if (results.fake_news) {
            csvRows.push([
                new Date().toISOString(),
                'Fake News Detection',
                results.fake_news.prediction,
                results.fake_news.confidence,
                Utils.format.truncate(results.input_text || '', 100)
            ]);
        }

        if (results.political_classification) {
            csvRows.push([
                new Date().toISOString(),
                'Political Classification',
                results.political_classification.prediction,
                results.political_classification.confidence,
                Utils.format.truncate(results.input_text || '', 100)
            ]);
        }

        const csvContent = csvRows.map(row => 
            row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(',')
        ).join('\n');

        const dataBlob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `news-analysis-${new Date().toISOString().split('T')[0]}.csv`;
        link.click();
        
        URL.revokeObjectURL(url);
    }

    // Export as PDF (simplified version)
    exportAsPdf(results) {
        // Create a new window with printable content
        const printWindow = window.open('', '_blank');
        const printContent = this.generatePrintableContent(results);
        
        printWindow.document.write(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>News Analysis Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .result-section { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; }
                    .confidence { font-weight: bold; }
                    .timestamp { color: #666; font-size: 12px; }
                </style>
            </head>
            <body>
                ${printContent}
            </body>
            </html>
        `);
        
        printWindow.document.close();
        printWindow.print();
    }

    // Generate printable content
    generatePrintableContent(results) {
        const timestamp = new Date().toLocaleString();
        let content = `
            <div class="header">
                <h1>News Analysis Report</h1>
                <p class="timestamp">Generated on ${timestamp}</p>
            </div>
        `;

        if (results.fake_news) {
            content += `
                <div class="result-section">
                    <h2>Fake News Detection</h2>
                    <p><strong>Prediction:</strong> ${results.fake_news.prediction}</p>
                    <p><strong>Confidence:</strong> <span class="confidence">${Utils.format.confidence(results.fake_news.confidence)}</span></p>
                </div>
            `;
        }

        if (results.political_classification) {
            content += `
                <div class="result-section">
                    <h2>Political Classification</h2>
                    <p><strong>Classification:</strong> ${results.political_classification.prediction}</p>
                    <p><strong>Confidence:</strong> <span class="confidence">${Utils.format.confidence(results.political_classification.confidence)}</span></p>
                    ${results.political_classification.reasoning ? `<p><strong>Reasoning:</strong> ${results.political_classification.reasoning}</p>` : ''}
                </div>
            `;
        }

        if (results.input_text) {
            content += `
                <div class="result-section">
                    <h2>Analyzed Text</h2>
                    <p>${Utils.format.truncate(results.input_text, 500)}</p>
                </div>
            `;
        }

        return content;
    }

    // Setup analytics tracking
    setupAnalyticsTracking() {
        if (!this.preferences.analytics) return;

        // Track page views, analysis requests, etc.
        this.trackEvent('page_view', { page: 'news_analyzer' });
    }

    // Track analytics events
    trackEvent(eventName, eventData = {}) {
        if (!this.preferences.analytics) return;

        // Store analytics data locally (can be extended to send to analytics service)
        const analyticsData = Utils.storage.get('analyticsData', []);
        analyticsData.push({
            event: eventName,
            data: eventData,
            timestamp: new Date().toISOString()
        });

        // Keep only last 1000 events
        if (analyticsData.length > 1000) {
            analyticsData.splice(0, analyticsData.length - 1000);
        }

        Utils.storage.set('analyticsData', analyticsData);
    }

    // Create notification element
    createNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type} fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50`;
        notification.innerHTML = `
            <div class="flex items-center">
                <i class="bi bi-info-circle mr-2"></i>
                <span>${message}</span>
                <button class="ml-4 text-lg" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;

        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);

        return notification;
    }

    // Get analytics summary
    getAnalyticsSummary() {
        const analyticsData = Utils.storage.get('analyticsData', []);
        
        const summary = {
            totalEvents: analyticsData.length,
            eventTypes: {},
            lastWeek: analyticsData.filter(event => {
                const eventDate = new Date(event.timestamp);
                const weekAgo = new Date();
                weekAgo.setDate(weekAgo.getDate() - 7);
                return eventDate > weekAgo;
            }).length
        };

        // Count event types
        analyticsData.forEach(event => {
            summary.eventTypes[event.event] = (summary.eventTypes[event.event] || 0) + 1;
        });

        return summary;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdvancedFeatures;
} else {
    window.AdvancedFeatures = AdvancedFeatures;
}
