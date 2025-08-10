/**
 * Website Management Mixin
 * Handles website tracking, adding, removing, and rendering
 */

export const WebsiteManagerMixin = {
    /**
     * Bind website management events
     */
    bindWebsiteEvents() {
        // Website management
        document.getElementById('addWebsiteBtn').addEventListener('click', () => this.addWebsite());
        document.getElementById('websiteUrl').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.addWebsite();
        });
        
        // Domain expansion/collapse controls
        const expandAllBtn = document.getElementById('expandAllDomainsBtn');
        if (expandAllBtn) {
            expandAllBtn.addEventListener('click', () => this.expandAllDomains());
        }
        
        const collapseAllBtn = document.getElementById('collapseAllDomainsBtn');
        if (collapseAllBtn) {
            collapseAllBtn.addEventListener('click', () => this.collapseAllDomains());
        }
    },
    
    /**
     * Add a new website to track
     */
    async addWebsite() {
        const urlInput = document.getElementById('websiteUrl');
        const url = urlInput.value.trim();
        
        if (!url) {
            this.showError('Please enter a website URL');
            return;
        }
        
        if (!this.isValidUrl(url)) {
            this.showError('Please enter a valid URL');
            return;
        }
        
        // Check if already tracking
        if (this.trackedWebsites.find(site => site.url === url)) {
            this.showError('This website is already being tracked');
            return;
        }
        
        this.isPerformingOperation = true;
        this.showLoading('Adding website...');
        
        try {
            const response = await fetch('/api/news-tracker/add-website', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });
            
            const data = await response.json();
            
            if (data.success) {
                const website = {
                    id: data.id,
                    url,
                    name: this.extractDomainName(url),
                    interval: 60,
                    addedAt: new Date().toISOString(),
                    lastFetch: null,
                    articleCount: 0,
                    status: 'active'
                };
                
                this.trackedWebsites.push(website);
                
                // Real-time UI updates
                this.renderTrackedWebsites();
                this.updateCounts();
                this.updateStatistics();
                
                // Clear input
                urlInput.value = '';
                
                this.showSuccess(data.message || 'Website added successfully');
                
                // Start auto-fetch if enabled and this is the first website
                if (this.autoFetchEnabled && this.trackedWebsites.length === 1) {
                    this.startAutoFetch();
                    this.updateAutoFetchStatus();
                }
            } else {
                this.showError(data.error || 'Failed to add website');
            }
        } catch (error) {
            console.error('Error adding website:', error);
            this.showError('Network error. Please try again.');
        } finally {
            this.isPerformingOperation = false;
            this.hideLoading();
        }
    },
    
    /**
     * Remove a website from tracking
     */
    async removeWebsite(websiteId) {
        if (!confirm('Are you sure you want to stop tracking this website?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/news-tracker/remove-website/${websiteId}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.trackedWebsites = this.trackedWebsites.filter(site => site.id !== websiteId);
                
                // Real-time UI updates
                this.renderTrackedWebsites();
                this.updateCounts();
                this.updateStatistics();
                
                this.showSuccess('Website removed from tracking');
                
                // Stop auto-fetch if no websites remain
                if (this.trackedWebsites.length === 0 && this.autoFetchEnabled) {
                    this.stopAutoFetch();
                    this.updateAutoFetchStatus();
                }
            } else {
                this.showError(data.error || 'Failed to remove website');
            }
        } catch (error) {
            console.error('Error removing website:', error);
            this.showError('Network error. Please try again.');
        }
    },
    
    /**
     * Render tracked websites list
     */
    renderTrackedWebsites() {
        const container = document.getElementById('trackedWebsitesList');
        if (!container) return;
        
        if (this.trackedWebsites.length === 0) {
            container.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <i class="bi bi-inbox text-4xl mb-2 opacity-50"></i>
                    <p>No websites being tracked yet.</p>
                    <p class="text-sm">Add a website above to start tracking articles.</p>
                </div>
            `;
            return;
        }
        
        if (this.websiteViewMode === 'grouped') {
            this.renderGroupedWebsites(container);
        } else {
            this.renderSimpleWebsiteList(container);
        }
    },
    
    /**
     * Render websites grouped by domain
     */
    renderGroupedWebsites(container) {
        const grouped = this.groupWebsitesByDomain();
        
        container.innerHTML = Object.entries(grouped).map(([domain, websites]) => {
            const isExpanded = this.isGroupExpanded(domain);
            const websiteList = websites.map(website => this.renderWebsiteItem(website, true)).join('');
            
            return `
                <div class="domain-group border border-gray-200 rounded-lg mb-4">
                    <div class="domain-header bg-gray-50 p-4 border-b border-gray-200 cursor-pointer"
                         onclick="newsTracker.toggleDomainGroup('${domain}')">
                        <div class="flex items-center justify-between">
                            <div class="flex items-center">
                                <i class="bi bi-${isExpanded ? 'chevron-down' : 'chevron-right'} mr-2 text-gray-600"></i>
                                <span class="font-semibold text-gray-800">${this.getDomainDisplayName(domain)}</span>
                                <span class="ml-2 bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
                                    ${websites.length} ${websites.length === 1 ? 'site' : 'sites'}
                                </span>
                            </div>
                            <div class="text-sm text-gray-600">
                                ${websites.reduce((sum, w) => sum + (w.articleCount || 0), 0)} articles
                            </div>
                        </div>
                    </div>
                    <div class="website-list ${isExpanded ? '' : 'hidden'}" id="group-${domain}">
                        ${websiteList}
                    </div>
                </div>
            `;
        }).join('');
    },
    
    /**
     * Render simple website list
     */
    renderSimpleWebsiteList(container) {
        container.innerHTML = this.trackedWebsites.map(website => 
            this.renderWebsiteItem(website, false)
        ).join('');
    },
    
    /**
     * Render individual website item
     */
    renderWebsiteItem(website, isGrouped) {
        const containerClass = isGrouped ? 'p-4 border-b border-gray-100 last:border-b-0' : 
                                         'p-4 border border-gray-200 rounded-lg mb-3';
        
        return `
            <div class="${containerClass} hover:bg-gray-50 transition-colors">
                <div class="flex items-center justify-between">
                    <div class="flex-1">
                        <div class="flex items-center mb-2">
                            <i class="bi bi-globe text-blue-600 mr-2"></i>
                            <h4 class="font-medium text-gray-800 truncate">${website.name || website.url}</h4>
                            <span class="ml-2 text-xs px-2 py-1 bg-green-100 text-green-800 rounded-full">
                                ${website.status || 'active'}
                            </span>
                        </div>
                        <p class="text-sm text-gray-600 truncate mb-1">${website.url}</p>
                        <div class="flex items-center text-xs text-gray-500 space-x-4">
                            <span>
                                <i class="bi bi-calendar-plus mr-1"></i>
                                Added: ${this.formatDate(website.addedAt)}
                            </span>
                            <span>
                                <i class="bi bi-clock mr-1"></i>
                                Last fetch: ${this.formatDate(website.lastFetch)}
                            </span>
                            <span>
                                <i class="bi bi-file-text mr-1"></i>
                                ${website.articleCount || 0} articles
                            </span>
                        </div>
                    </div>
                    <div class="flex items-center space-x-2 ml-4">
                        <button 
                            class="bg-red-500 text-white p-2 rounded-lg hover:bg-red-600 transition-colors"
                            onclick="event.stopPropagation(); newsTracker.removeWebsite('${website.id}')"
                            title="Remove website"
                        >
                            <i class="bi bi-trash text-sm"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
    },
    
    /**
     * Group websites by domain
     */
    groupWebsitesByDomain() {
        const grouped = {};
        
        this.trackedWebsites.forEach(website => {
            const domain = this.extractRootDomain(website.url);
            if (!grouped[domain]) {
                grouped[domain] = [];
            }
            grouped[domain].push(website);
        });
        
        return grouped;
    },
    
    /**
     * Extract root domain for grouping
     */
    extractRootDomain(url) {
        try {
            const parsed = new URL(url);
            let hostname = parsed.hostname.replace('www.', '');
            
            // Handle special domain mappings
            const specialMappings = {
                'mb.com.ph': 'manilabulletin.com.ph',
                'news.abs-cbn.com': 'abs-cbn.com',
                'news.gma.network': 'gmanetwork.com',
                'cnnphilippines.com': 'cnn.com'
            };
            
            if (specialMappings[hostname]) {
                return specialMappings[hostname];
            }
            
            return hostname;
        } catch (error) {
            return url;
        }
    },
    
    /**
     * Get display name for domain
     */
    getDomainDisplayName(domain) {
        const displayNames = {
            'manilabulletin.com.ph': 'Manila Bulletin',
            'abs-cbn.com': 'ABS-CBN News',
            'gmanetwork.com': 'GMA News',
            'cnn.com': 'CNN Philippines',
            'philstar.com': 'The Philippine Star'
        };
        
        return displayNames[domain] || domain.replace(/\.(com|org|net|gov|edu)(\.[a-z]{2})?$/i, '');
    },
    
    /**
     * Toggle domain group expansion
     */
    toggleDomainGroup(domain) {
        const groupElement = document.getElementById(`group-${domain}`);
        if (groupElement) {
            const isHidden = groupElement.classList.contains('hidden');
            groupElement.classList.toggle('hidden');
            
            // Update chevron icon
            const chevron = groupElement.previousElementSibling.querySelector('.bi-chevron-down, .bi-chevron-right');
            if (chevron) {
                chevron.className = `bi bi-${isHidden ? 'chevron-down' : 'chevron-right'} mr-2 text-gray-600`;
            }
            
            // Save expansion state
            const expandedGroups = JSON.parse(localStorage.getItem('newsTracker.expandedGroups') || '{}');
            expandedGroups[domain] = isHidden;
            localStorage.setItem('newsTracker.expandedGroups', JSON.stringify(expandedGroups));
        }
    },
    
    /**
     * Check if group is expanded
     */
    isGroupExpanded(domain) {
        const expandedGroups = JSON.parse(localStorage.getItem('newsTracker.expandedGroups') || '{}');
        return expandedGroups[domain] !== false; // Default to expanded
    },
    
    /**
     * Expand all domain groups
     */
    expandAllDomains() {
        const allGroups = document.querySelectorAll('.website-list');
        const expandedGroups = {};
        
        allGroups.forEach(group => {
            group.classList.remove('hidden');
            const domain = group.id.replace('group-', '');
            expandedGroups[domain] = true;
            
            // Update chevron
            const chevron = group.previousElementSibling.querySelector('.bi-chevron-right');
            if (chevron) {
                chevron.className = 'bi bi-chevron-down mr-2 text-gray-600';
            }
        });
        
        localStorage.setItem('newsTracker.expandedGroups', JSON.stringify(expandedGroups));
    },
    
    /**
     * Collapse all domain groups
     */
    collapseAllDomains() {
        const allGroups = document.querySelectorAll('.website-list');
        const expandedGroups = {};
        
        allGroups.forEach(group => {
            group.classList.add('hidden');
            const domain = group.id.replace('group-', '');
            expandedGroups[domain] = false;
            
            // Update chevron
            const chevron = group.previousElementSibling.querySelector('.bi-chevron-down');
            if (chevron) {
                chevron.className = 'bi bi-chevron-right mr-2 text-gray-600';
            }
        });
        
        localStorage.setItem('newsTracker.expandedGroups', JSON.stringify(expandedGroups));
    }
};
