// Configuration module for the Advanced News Analyzer
const Config = {
    // API endpoints
    endpoints: {
        predict: '/predict',
        modelStatus: '/model-status',
        submitFeedback: '/submit-feedback',
        retrainModel: '/retrain-model',
        crawlWebsite: '/crawl-website',
        analyzeWebsite: '/analyze-website',
        indexPhilippineArticle: '/index-philippine-article',
        searchPhilippineNews: '/search-philippine-news',
        findSimilarContent: '/find-similar-content',
        philippineNewsAnalytics: '/philippine-news-analytics',
        getPhilippineArticle: '/philippine-article/',
        batchIndexPhilippineArticles: '/batch-index-philippine-articles',
        philippineNewsCategories: '/philippine-news-categories',
        philippineNewsSources: '/philippine-news-sources',
        getCrawlHistory: '/get-crawl-history',
        urlFeedback: '/url-classifier-feedback',
        // RSS Feed endpoints
        rssFeeds: '/api/rss-feeds',
        rssFeedArticles: '/api/rss-feed/articles',
        rssFeedParse: '/api/rss-feed/parse',
        rssFeedAnalyze: '/api/rss-feed/analyze'
    },
    
    // UI update intervals
    intervals: {
        modelStatusCheck: 30000, // 30 seconds
        autoSave: 60000 // 1 minute (if implemented)
    },
    
    // Input validation
    validation: {
        minTextLength: 10,
        maxTextLength: 50000,
        urlPattern: /^https?:\/\/.+/i,
        websitePattern: /^https?:\/\/[^\/]+/i
    },
    
    // Animation durations (in milliseconds)
    animations: {
        fadeIn: 500,
        slideIn: 300,
        progressBar: 800
    },
    
    // Analysis types
    analysisTypes: {
        FAKE_NEWS: 'fake_news',
        POLITICAL: 'political',
        BOTH: 'both'
    },
    
    // Input types
    inputTypes: {
        TEXT: 'text',
        URL: 'url',
        WEBSITE: 'website',
        SEARCH: 'search'
    },
    
    // Crawling modes
    crawlingModes: {
        PREVIEW: 'preview',
        ANALYZE: 'analyze'
    },
    
    // CSS classes for different states
    cssClasses: {
        button: {
            active: {
                fakeNews: 'analysis-type-btn bg-red-600 text-white px-6 py-4 rounded-lg font-medium transition-all duration-200 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500',
                political: 'analysis-type-btn bg-blue-600 text-white px-6 py-4 rounded-lg font-medium transition-all duration-200 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500',
                both: 'analysis-type-btn bg-purple-600 text-white px-6 py-4 rounded-lg font-medium transition-all duration-200 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500',
                textInput: 'input-type-btn bg-blue-600 text-white px-6 py-3 rounded-lg font-medium transition-all duration-200 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500',
                urlInput: 'input-type-btn bg-blue-600 text-white px-6 py-3 rounded-lg font-medium transition-all duration-200 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500'
            },
            inactive: {
                default: 'analysis-type-btn bg-gray-200 text-gray-700 px-6 py-4 rounded-lg font-medium transition-all duration-200 hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-500',
                input: 'input-type-btn bg-gray-200 text-gray-700 px-6 py-3 rounded-lg font-medium transition-all duration-200 hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-500'
            }
        },
        prediction: {
            fake: 'rounded-xl p-6 mb-6 border-l-4 bg-red-50 border-red-500',
            real: 'rounded-xl p-6 mb-6 border-l-4 bg-green-50 border-green-500',
            political: 'rounded-xl p-6 mb-6 border-l-4 bg-blue-50 border-blue-500',
            nonPolitical: 'rounded-xl p-6 mb-6 border-l-4 bg-gray-50 border-gray-500'
        },
        text: {
            fake: 'text-2xl font-bold text-red-700',
            real: 'text-2xl font-bold text-green-700',
            political: 'text-2xl font-bold text-blue-700',
            nonPolitical: 'text-2xl font-bold text-gray-700'
        }
    },
    
    // Default messages
    messages: {
        loading: 'Analyzing content...',
        modelLoading: 'Models loading...',
        modelReady: 'Both models ready! Choose your analysis type.',
        modelPartial: 'Fake news model ready. Political classifier loading...',
        modelError: 'Error loading models',
        networkError: 'Network error. Please try again.',
        noInput: 'Please enter some text or a URL to analyze.',
        invalidUrl: 'Please enter a valid URL.',
        textTooShort: 'Text must be at least 10 characters long.',
        textTooLong: 'Text is too long. Please keep it under 50,000 characters.'
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Config;
} else {
    window.Config = Config;
}
