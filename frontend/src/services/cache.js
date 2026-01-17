/**
 * SkyFlux AI - Cache Service
 * 
 * Implements aggressive caching with:
 * - IndexedDB persistence
 * - TTL-based expiration
 * - Version-aware invalidation
 */

const DB_NAME = 'skyflux-cache';
const DB_VERSION = 1;
const STORE_NAME = 'cache';

// Cache configuration per endpoint
const CACHE_CONFIG = {
    metadata: { ttl: 3600000, versionAware: false }, // 1 hour
    density: { ttl: 86400000, versionAware: true },  // 24 hours
    predictions: { ttl: 86400000, versionAware: true },
    anomalies: { ttl: 86400000, versionAware: true },
    stress: { ttl: 86400000, versionAware: true },
};

let db = null;
let currentDataVersion = null;

/**
 * Initialize IndexedDB
 */
async function initDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
            db = request.result;
            resolve(db);
        };

        request.onupgradeneeded = (event) => {
            const database = event.target.result;
            if (!database.objectStoreNames.contains(STORE_NAME)) {
                database.createObjectStore(STORE_NAME, { keyPath: 'key' });
            }
        };
    });
}

/**
 * Get cached data
 */
async function getCached(key) {
    if (!db) await initDB();

    return new Promise((resolve, reject) => {
        const transaction = db.transaction([STORE_NAME], 'readonly');
        const store = transaction.objectStore(STORE_NAME);
        const request = store.get(key);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
    });
}

/**
 * Set cached data
 */
async function setCached(key, data, dataVersion) {
    if (!db) await initDB();

    const record = {
        key,
        data,
        dataVersion,
        cachedAt: Date.now(),
    };

    return new Promise((resolve, reject) => {
        const transaction = db.transaction([STORE_NAME], 'readwrite');
        const store = transaction.objectStore(STORE_NAME);
        const request = store.put(record);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve();
    });
}

/**
 * Clear all cached data
 */
async function clearCache() {
    if (!db) await initDB();

    return new Promise((resolve, reject) => {
        const transaction = db.transaction([STORE_NAME], 'readwrite');
        const store = transaction.objectStore(STORE_NAME);
        const request = store.clear();

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve();
    });
}

/**
 * Check if cached entry is valid
 */
function isValid(cached, endpoint) {
    if (!cached) return false;

    const config = CACHE_CONFIG[endpoint] || { ttl: 3600000, versionAware: true };

    // Check TTL
    const age = Date.now() - cached.cachedAt;
    if (age > config.ttl) {
        console.log(`[Cache] ${endpoint}: Expired (age: ${Math.round(age / 1000)}s)`);
        return false;
    }

    // Check version
    if (config.versionAware && currentDataVersion && cached.dataVersion !== currentDataVersion) {
        console.log(`[Cache] ${endpoint}: Version mismatch (${cached.dataVersion} vs ${currentDataVersion})`);
        return false;
    }

    return true;
}

/**
 * Update current data version
 */
function setDataVersion(version) {
    if (currentDataVersion && currentDataVersion !== version) {
        console.log(`[Cache] Data version changed: ${currentDataVersion} -> ${version}`);
        // Clear version-aware caches on version change
        clearCache().catch(console.error);
    }
    currentDataVersion = version;
}

/**
 * Get current data version
 */
function getDataVersion() {
    return currentDataVersion;
}

/**
 * Generate cache key from endpoint and params
 */
function makeCacheKey(endpoint, params = {}) {
    const sortedParams = Object.keys(params)
        .sort()
        .map(k => `${k}=${params[k]}`)
        .join('&');
    return sortedParams ? `${endpoint}?${sortedParams}` : endpoint;
}

export {
    initDB,
    getCached,
    setCached,
    clearCache,
    isValid,
    setDataVersion,
    getDataVersion,
    makeCacheKey,
    CACHE_CONFIG,
};
