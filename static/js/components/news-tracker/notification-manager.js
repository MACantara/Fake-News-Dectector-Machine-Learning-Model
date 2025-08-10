/**
 * Notification Management Mixin
 * Handles success, error, loading, and real-time activity notifications
 */

export const NotificationManagerMixin = {
    /**
     * Show loading indicator with message
     */
    showLoading(message = 'Loading...') {
        document.getElementById('loadingMessage').textContent = message;
        document.getElementById('loading').classList.remove('hidden');
    },
    
    /**
     * Hide loading indicator
     */
    hideLoading() {
        document.getElementById('loading').classList.add('hidden');
    },
    
    /**
     * Show error message
     */
    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('errorDisplay').classList.remove('hidden');
        setTimeout(() => this.hideError(), 5000);
    },
    
    /**
     * Hide error message
     */
    hideError() {
        document.getElementById('errorDisplay').classList.add('hidden');
    },
    
    /**
     * Show success message
     */
    showSuccess(message) {
        document.getElementById('successMessage').textContent = message;
        document.getElementById('successDisplay').classList.remove('hidden');
        setTimeout(() => this.hideSuccess(), 3000);
    },
    
    /**
     * Hide success message
     */
    hideSuccess() {
        document.getElementById('successDisplay').classList.add('hidden');
    },
    
    /**
     * Show real-time activity notification
     */
    showRealtimeActivity(message) {
        const activityElement = document.getElementById('realtimeActivity');
        const messageElement = document.getElementById('activityMessage');
        
        if (activityElement && messageElement) {
            messageElement.textContent = message;
            activityElement.classList.remove('hidden');
        }
    },
    
    /**
     * Hide real-time activity notification
     */
    hideRealtimeActivity() {
        const activityElement = document.getElementById('realtimeActivity');
        if (activityElement) {
            activityElement.classList.add('hidden');
        }
    },
    
    /**
     * Update real-time activity message
     */
    updateRealtimeActivity(message) {
        const messageElement = document.getElementById('activityMessage');
        if (messageElement) {
            messageElement.textContent = message;
        }
    },
    
    /**
     * Show notification with custom type and duration
     */
    showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg transition-all duration-300 transform translate-x-full`;
        
        // Set notification style based on type
        const styles = {
            success: 'bg-green-50 border border-green-200 text-green-800',
            error: 'bg-red-50 border border-red-200 text-red-800',
            warning: 'bg-yellow-50 border border-yellow-200 text-yellow-800',
            info: 'bg-blue-50 border border-blue-200 text-blue-800'
        };
        
        const icons = {
            success: 'bi-check-circle',
            error: 'bi-exclamation-triangle',
            warning: 'bi-exclamation-triangle',
            info: 'bi-info-circle'
        };
        
        notification.className += ` ${styles[type] || styles.info}`;
        notification.innerHTML = `
            <div class="flex items-center">
                <i class="${icons[type] || icons.info} text-lg mr-2"></i>
                <span class="text-sm font-medium">${message}</span>
                <button class="ml-4 hover:opacity-70" onclick="this.parentElement.parentElement.remove()">
                    <i class="bi bi-x text-lg"></i>
                </button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.classList.remove('translate-x-full');
        }, 100);
        
        // Auto-remove after duration
        setTimeout(() => {
            notification.classList.add('translate-x-full');
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 300);
        }, duration);
        
        return notification;
    },
    
    /**
     * Show progress notification with progress bar
     */
    showProgressNotification(message, progress = 0) {
        let notification = document.getElementById('progress-notification');
        
        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'progress-notification';
            notification.className = 'fixed top-4 right-4 z-50 bg-white border border-gray-200 rounded-lg shadow-lg p-4 min-w-80';
            document.body.appendChild(notification);
        }
        
        notification.innerHTML = `
            <div class="flex items-center mb-2">
                <div class="animate-spin rounded-full h-4 w-4 border-2 border-blue-600 border-t-transparent mr-2"></div>
                <span class="text-sm font-medium text-gray-800">${message}</span>
            </div>
            <div class="w-full bg-gray-200 rounded-full h-2">
                <div class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: ${progress}%"></div>
            </div>
            <div class="text-xs text-gray-600 mt-1">${progress.toFixed(1)}% complete</div>
        `;
        
        return notification;
    },
    
    /**
     * Update progress notification
     */
    updateProgressNotification(message, progress) {
        const notification = document.getElementById('progress-notification');
        if (notification) {
            const messageEl = notification.querySelector('.text-sm.font-medium');
            const progressBar = notification.querySelector('.bg-blue-600');
            const progressText = notification.querySelector('.text-xs.text-gray-600');
            
            if (messageEl) messageEl.textContent = message;
            if (progressBar) progressBar.style.width = `${progress}%`;
            if (progressText) progressText.textContent = `${progress.toFixed(1)}% complete`;
        }
    },
    
    /**
     * Hide progress notification
     */
    hideProgressNotification() {
        const notification = document.getElementById('progress-notification');
        if (notification) {
            notification.classList.add('translate-x-full');
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 300);
        }
    },
    
    /**
     * Show confirmation dialog
     */
    showConfirmDialog(message, title = 'Confirm Action') {
        return new Promise((resolve) => {
            const modal = document.createElement('div');
            modal.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50';
            modal.innerHTML = `
                <div class="bg-white rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
                    <h3 class="text-lg font-semibold text-gray-800 mb-4">${title}</h3>
                    <p class="text-gray-600 mb-6">${message}</p>
                    <div class="flex justify-end space-x-3">
                        <button id="modal-cancel" class="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors">
                            Cancel
                        </button>
                        <button id="modal-confirm" class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                            Confirm
                        </button>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            
            const cancelBtn = modal.querySelector('#modal-cancel');
            const confirmBtn = modal.querySelector('#modal-confirm');
            
            cancelBtn.addEventListener('click', () => {
                modal.remove();
                resolve(false);
            });
            
            confirmBtn.addEventListener('click', () => {
                modal.remove();
                resolve(true);
            });
            
            // Close on backdrop click
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.remove();
                    resolve(false);
                }
            });
            
            // Focus confirm button
            confirmBtn.focus();
        });
    },
    
    /**
     * Show input dialog
     */
    showInputDialog(message, title = 'Input Required', defaultValue = '') {
        return new Promise((resolve) => {
            const modal = document.createElement('div');
            modal.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50';
            modal.innerHTML = `
                <div class="bg-white rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
                    <h3 class="text-lg font-semibold text-gray-800 mb-4">${title}</h3>
                    <p class="text-gray-600 mb-4">${message}</p>
                    <input type="text" id="modal-input" class="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 mb-6" value="${defaultValue}">
                    <div class="flex justify-end space-x-3">
                        <button id="modal-cancel" class="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors">
                            Cancel
                        </button>
                        <button id="modal-confirm" class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                            OK
                        </button>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            
            const input = modal.querySelector('#modal-input');
            const cancelBtn = modal.querySelector('#modal-cancel');
            const confirmBtn = modal.querySelector('#modal-confirm');
            
            cancelBtn.addEventListener('click', () => {
                modal.remove();
                resolve(null);
            });
            
            confirmBtn.addEventListener('click', () => {
                const value = input.value.trim();
                modal.remove();
                resolve(value);
            });
            
            // Submit on Enter
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    const value = input.value.trim();
                    modal.remove();
                    resolve(value);
                }
            });
            
            // Close on backdrop click
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.remove();
                    resolve(null);
                }
            });
            
            // Focus input
            input.focus();
            input.select();
        });
    },
    
    /**
     * Show toast notification (bottom-right corner)
     */
    showToast(message, type = 'info', duration = 3000) {
        const toast = document.createElement('div');
        toast.className = 'fixed bottom-4 right-4 z-50 p-3 rounded-lg shadow-lg transform translate-y-full transition-transform duration-300';
        
        const styles = {
            success: 'bg-green-600 text-white',
            error: 'bg-red-600 text-white',
            warning: 'bg-yellow-600 text-white',
            info: 'bg-blue-600 text-white'
        };
        
        const icons = {
            success: 'bi-check-circle',
            error: 'bi-exclamation-triangle',
            warning: 'bi-exclamation-triangle',
            info: 'bi-info-circle'
        };
        
        toast.className += ` ${styles[type] || styles.info}`;
        toast.innerHTML = `
            <div class="flex items-center">
                <i class="${icons[type] || icons.info} mr-2"></i>
                <span class="text-sm">${message}</span>
            </div>
        `;
        
        document.body.appendChild(toast);
        
        // Animate in
        setTimeout(() => {
            toast.classList.remove('translate-y-full');
        }, 100);
        
        // Auto-remove
        setTimeout(() => {
            toast.classList.add('translate-y-full');
            setTimeout(() => {
                if (toast.parentElement) {
                    toast.remove();
                }
            }, 300);
        }, duration);
        
        return toast;
    }
};
