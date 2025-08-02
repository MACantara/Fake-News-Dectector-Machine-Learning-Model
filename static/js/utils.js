// Utility functions for the Advanced News Analyzer
const Utils = {
    // DOM manipulation utilities
    dom: {
        // Get element by ID with error handling
        getElementById(id) {
            const element = document.getElementById(id);
            if (!element) {
                console.warn(`Element with ID '${id}' not found`);
            }
            return element;
        },

        // Add class with error handling
        addClass(element, className) {
            if (element && element.classList) {
                element.classList.add(className);
            }
        },

        // Remove class with error handling
        removeClass(element, className) {
            if (element && element.classList) {
                element.classList.remove(className);
            }
        },

        // Toggle class with error handling
        toggleClass(element, className) {
            if (element && element.classList) {
                element.classList.toggle(className);
            }
        },

        // Show element
        show(element) {
            if (element) {
                element.classList.remove('hidden');
            }
        },

        // Hide element
        hide(element) {
            if (element) {
                element.classList.add('hidden');
            }
        },

        // Set text content safely
        setText(element, text) {
            if (element) {
                element.textContent = text || '';
            }
        },

        // Set HTML content safely
        setHTML(element, html) {
            if (element) {
                element.innerHTML = html || '';
            }
        }
    },

    // Animation utilities
    animations: {
        // Fade in element
        fadeIn(element, duration = Config.animations.fadeIn) {
            if (!element) return;
            
            element.style.opacity = '0';
            element.style.transition = `opacity ${duration}ms ease-in-out`;
            Utils.dom.show(element);
            
            // Trigger animation on next frame
            requestAnimationFrame(() => {
                element.style.opacity = '1';
            });
        },

        // Fade out element
        fadeOut(element, duration = Config.animations.fadeIn) {
            if (!element) return;
            
            element.style.transition = `opacity ${duration}ms ease-in-out`;
            element.style.opacity = '0';
            
            setTimeout(() => {
                Utils.dom.hide(element);
            }, duration);
        },

        // Animate progress bar
        animateProgressBar(bar, percentage, duration = Config.animations.progressBar) {
            if (!bar) return;
            
            bar.style.transition = `width ${duration}ms ease-in-out`;
            bar.style.width = `${percentage}%`;
        },

        // Add shake animation for errors
        shake(element) {
            if (!element) return;
            
            element.classList.add('error-shake');
            setTimeout(() => {
                element.classList.remove('error-shake');
            }, 500);
        },

        // Add pulse animation for success
        pulse(element) {
            if (!element) return;
            
            element.classList.add('success-pulse');
            setTimeout(() => {
                element.classList.remove('success-pulse');
            }, 1000);
        }
    },

    // Validation utilities
    validation: {
        // Validate text input
        validateText(text) {
            if (!text || text.trim().length === 0) {
                return { valid: false, message: Config.messages.noInput };
            }
            
            if (text.length < Config.validation.minTextLength) {
                return { valid: false, message: Config.messages.textTooShort };
            }
            
            if (text.length > Config.validation.maxTextLength) {
                return { valid: false, message: Config.messages.textTooLong };
            }
            
            return { valid: true };
        },

        // Validate URL input
        validateURL(url) {
            if (!url || url.trim().length === 0) {
                return { valid: false, message: Config.messages.noInput };
            }
            
            if (!Config.validation.urlPattern.test(url)) {
                return { valid: false, message: Config.messages.invalidUrl };
            }
            
            return { valid: true };
        }
    },

    // Formatting utilities
    format: {
        // Format percentage
        percentage(value, decimals = 1) {
            return `${(value * 100).toFixed(decimals)}%`;
        },

        // Format confidence score
        confidence(value, decimals = 1) {
            return `${(value * 100).toFixed(decimals)}%`;
        },

        // Truncate text with ellipsis
        truncate(text, maxLength = 100) {
            if (!text) return '';
            return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
        },

        // Format text for display (preserve line breaks)
        displayText(text) {
            if (!text) return '';
            return text.replace(/\n/g, '<br>');
        }
    },

    // HTTP utilities
    http: {
        // Generic fetch wrapper with error handling
        async request(url, options = {}) {
            try {
                const response = await fetch(url, {
                    headers: {
                        'Content-Type': 'application/json',
                        ...options.headers
                    },
                    ...options
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || `HTTP error! status: ${response.status}`);
                }

                return { success: true, data };
            } catch (error) {
                console.error('HTTP request failed:', error);
                return { success: false, error: error.message };
            }
        },

        // POST request helper
        async post(url, data) {
            return this.request(url, {
                method: 'POST',
                body: JSON.stringify(data)
            });
        },

        // GET request helper
        async get(url) {
            return this.request(url, {
                method: 'GET'
            });
        }
    },

    // Local storage utilities
    storage: {
        // Set item in localStorage with error handling
        set(key, value) {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (error) {
                console.warn('Failed to save to localStorage:', error);
                return false;
            }
        },

        // Get item from localStorage with error handling
        get(key, defaultValue = null) {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (error) {
                console.warn('Failed to read from localStorage:', error);
                return defaultValue;
            }
        },

        // Remove item from localStorage
        remove(key) {
            try {
                localStorage.removeItem(key);
                return true;
            } catch (error) {
                console.warn('Failed to remove from localStorage:', error);
                return false;
            }
        }
    },

    // Debounce utility for performance
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Throttle utility for performance
    throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Utils;
} else {
    window.Utils = Utils;
}
