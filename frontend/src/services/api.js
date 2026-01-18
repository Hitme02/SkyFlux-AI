/**
 * SkyFlux AI - API Service
 * 
 * Handles all API calls with caching integration.
 * All responses are cached and served from cache when valid.
 */

import {
    getCached,
    setCached,
    isValid,
    setDataVersion,
    makeCacheKey,
} from './cache.js';

// API base URL - use relative path for proxy in dev, full URL in production
const API_BASE = '/api';

/**
 * Fetch with caching
 */
async function fetchWithCache(endpoint, params = {}) {
    const cacheKey = makeCacheKey(endpoint, params);

    // Try cache first
    try {
        const cached = await getCached(cacheKey);
        if (isValid(cached, endpoint)) {
            console.log(`[API] Cache hit: ${endpoint}`);
            return cached.data;
        }
    } catch (err) {
        console.warn('[API] Cache read error:', err);
    }

    // Build URL with query params
    const url = new URL(endpoint, window.location.origin + API_BASE + '/');
    Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
            url.searchParams.set(key, value);
        }
    });

    console.log(`[API] Fetching: ${url.pathname}${url.search}`);

    // Fetch from API
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    // Update data version if present
    if (data.data_version) {
        setDataVersion(data.data_version);
    }

    // Cache the response
    try {
        await setCached(cacheKey, data, data.data_version);
        console.log(`[API] Cached: ${endpoint}`);
    } catch (err) {
        console.warn('[API] Cache write error:', err);
    }

    return data;
}

/**
 * API Methods
 */

export async function getHealth() {
    return fetchWithCache('health');
}

export async function getMetadata() {
    return fetchWithCache('metadata');
}

export async function getDensity(params = {}) {
    return fetchWithCache('density', params);
}

export async function getPredictions(params = {}) {
    return fetchWithCache('predictions', params);
}

export async function getAnomalies(params = {}) {
    return fetchWithCache('anomalies', params);
}

export async function getStress(params = {}) {
    return fetchWithCache('stress', params);
}

export async function getTrajectories(params = {}) {
    return fetchWithCache('trajectories', params);
}

export async function triggerRetrain(options = {}) {
    const response = await fetch(`${API_BASE}/admin/retrain`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${options.token || ''}`,
        },
        body: JSON.stringify({
            training_window_days: options.trainingDays || 15,
            models: options.models || ['trajectory', 'anomaly'],
        }),
    });

    if (!response.ok) {
        throw new Error(`Retrain error: ${response.status}`);
    }

    return response.json();
}

export async function getRetrainStatus(jobId) {
    const response = await fetch(`${API_BASE}/admin/retrain/${jobId}`);
    return response.json();
}

export { API_BASE };
