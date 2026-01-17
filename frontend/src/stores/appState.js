/**
 * SkyFlux AI - App State Store
 * 
 * Simple reactive state management.
 */

const state = {
    // Current view
    currentView: 'map',

    // Date/time selection
    selectedDate: new Date().toISOString().split('T')[0],
    selectedHour: 12,

    // Data
    metadata: null,
    density: [],
    predictions: [],
    anomalies: [],
    stress: [],

    // UI State
    isLoading: false,
    error: null,

    // Time travel
    timeTravelPosition: 0,
    isPlaying: false,

    // What-if
    densityFactor: 100,
    altitudeSpread: 100,
};

// Subscribers
const subscribers = new Set();

/**
 * Subscribe to state changes
 */
function subscribe(callback) {
    subscribers.add(callback);
    return () => subscribers.delete(callback);
}

/**
 * Update state and notify subscribers
 */
function setState(updates) {
    Object.assign(state, updates);
    subscribers.forEach(cb => cb(state));
}

/**
 * Get current state
 */
function getState() {
    return { ...state };
}

export { state, subscribe, setState, getState };
