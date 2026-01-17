/**
 * SkyFlux AI - Main Entry Point
 * 
 * Initializes the application and coordinates all components.
 */

import { initDB, clearCache } from './services/cache.js';
import * as api from './services/api.js';
import { state, subscribe, setState } from './stores/appState.js';
import { initMap, fitToBounds } from './components/MapView.js';
import { initAnomalyCards } from './components/AnomalyCards.js';

// DOM Elements
let elements = {};

/**
 * Initialize DOM element references
 */
function initElements() {
    elements = {
        // Navigation
        navBtns: document.querySelectorAll('.nav-btn'),

        // Status
        statusDot: document.querySelector('.status-dot'),
        statusText: document.querySelector('.status-text'),

        // Date/time
        datePicker: document.getElementById('date-picker'),
        hourSlider: document.getElementById('hour-slider'),
        hourValue: document.getElementById('hour-value'),

        // Time travel
        playBtn: document.getElementById('play-btn'),
        timeSlider: document.getElementById('time-slider'),
        predTime: document.getElementById('pred-time'),
        predError: document.getElementById('pred-error'),

        // What-if
        densityFactor: document.getElementById('density-factor'),
        altitudeSpread: document.getElementById('altitude-spread'),
        applyWhatIf: document.getElementById('apply-what-if'),

        // Stats
        statFlights: document.getElementById('stat-flights'),
        statAnomalies: document.getElementById('stat-anomalies'),
        statStress: document.getElementById('stat-stress'),
        statVersion: document.getElementById('stat-version'),

        // Footer
        dataVersion: document.getElementById('data-version'),
        lastTrained: document.getElementById('last-trained'),

        // Admin
        retrainBtn: document.getElementById('retrain-btn'),
    };
}

/**
 * Bind event listeners
 */
function bindEvents() {
    // Navigation
    elements.navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.dataset.view;
            setState({ currentView: view });

            // Update active state
            elements.navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Reload data for the view
            loadDataForView(view);
        });
    });

    // Date picker
    elements.datePicker.addEventListener('change', (e) => {
        setState({ selectedDate: e.target.value });
        loadDataForView(state.currentView);
    });

    // Hour slider
    elements.hourSlider.addEventListener('input', (e) => {
        const hour = parseInt(e.target.value);
        elements.hourValue.textContent = hour;
        setState({ selectedHour: hour });
    });

    elements.hourSlider.addEventListener('change', () => {
        loadDataForView(state.currentView);
    });

    // What-if sliders
    elements.densityFactor.addEventListener('input', (e) => {
        e.target.nextElementSibling.textContent = `${e.target.value}%`;
    });

    elements.altitudeSpread.addEventListener('input', (e) => {
        e.target.nextElementSibling.textContent = `${e.target.value}%`;
    });

    elements.applyWhatIf.addEventListener('click', () => {
        setState({
            densityFactor: parseInt(elements.densityFactor.value),
            altitudeSpread: parseInt(elements.altitudeSpread.value),
        });
        // Trigger re-render with new factors
        loadDataForView(state.currentView);
    });

    // Retrain button
    elements.retrainBtn.addEventListener('click', async () => {
        if (!confirm('Trigger model retraining? This may take some time.')) return;

        const token = prompt('Enter admin token:');
        if (!token) return;

        try {
            setStatus('loading', 'Triggering retrain...');
            const result = await api.triggerRetrain({ token });
            alert(`Retraining started!\nJob ID: ${result.job_id}\nEstimated: ${result.estimated_completion}`);

            // Clear cache after retrain
            await clearCache();
            setStatus('ok', 'Connected');
        } catch (err) {
            console.error('Retrain error:', err);
            alert(`Retrain failed: ${err.message}`);
            setStatus('error', 'Retrain failed');
        }
    });

    // Time travel play button
    elements.playBtn.addEventListener('click', () => {
        setState({ isPlaying: !state.isPlaying });
        elements.playBtn.textContent = state.isPlaying ? '⏸ Pause' : '▶ Play';

        if (state.isPlaying) {
            startTimeTravel();
        }
    });
}

/**
 * Set status indicator
 */
function setStatus(type, text) {
    elements.statusDot.className = `status-dot ${type}`;
    elements.statusText.textContent = text;
}

/**
 * Load data for current view
 */
async function loadDataForView(view) {
    const params = {
        date: state.selectedDate,
        hour: state.selectedHour,
    };

    setState({ isLoading: true });
    setStatus('loading', 'Loading...');

    try {
        switch (view) {
            case 'map':
                const predictions = await api.getPredictions({ ...params, limit: 100 });
                setState({ predictions: predictions.predictions || [] });
                break;

            case 'density':
                const density = await api.getDensity(params);
                setState({ density: density.grid || [] });
                if (density.grid?.length) fitToBounds(density.grid);
                break;

            case 'anomalies':
                const anomalies = await api.getAnomalies({ ...params, min_score: 0.5, limit: 50 });
                setState({ anomalies: anomalies.anomalies || [] });
                updateStats({ anomalyCount: anomalies.count });
                break;

            case 'stress':
                const stress = await api.getStress({ ...params, min_stress: 25 });
                setState({ stress: stress.stress_grid || [] });
                if (stress.stress_grid?.length) fitToBounds(stress.stress_grid);
                updateStats({ avgStress: calculateAvgStress(stress.stress_grid) });
                break;
        }

        setStatus('ok', 'Connected');
    } catch (err) {
        console.error('Data load error:', err);
        setStatus('error', 'Error loading data');
    } finally {
        setState({ isLoading: false });
    }
}

/**
 * Calculate average stress
 */
function calculateAvgStress(stressData) {
    if (!stressData || stressData.length === 0) return 0;
    const sum = stressData.reduce((acc, s) => acc + (s.stress_index || 0), 0);
    return Math.round(sum / stressData.length);
}

/**
 * Update statistics display
 */
function updateStats(stats = {}) {
    if (stats.flightCount !== undefined) {
        elements.statFlights.textContent = stats.flightCount.toLocaleString();
    }
    if (stats.anomalyCount !== undefined) {
        elements.statAnomalies.textContent = stats.anomalyCount.toLocaleString();
    }
    if (stats.avgStress !== undefined) {
        elements.statStress.textContent = stats.avgStress;
    }
}

/**
 * Time travel animation
 */
let timeTravelInterval = null;

function startTimeTravel() {
    if (timeTravelInterval) clearInterval(timeTravelInterval);

    timeTravelInterval = setInterval(() => {
        let pos = parseInt(elements.timeSlider.value) + 1;

        if (pos > 100) {
            pos = 0;
            setState({ isPlaying: false });
            elements.playBtn.textContent = '▶ Play';
            clearInterval(timeTravelInterval);
        }

        elements.timeSlider.value = pos;
        setState({ timeTravelPosition: pos });

        // Update prediction display
        updateTimeTravelDisplay(pos);
    }, 100);
}

function updateTimeTravelDisplay(position) {
    // This would interpolate between prediction and actual positions
    const minutes = Math.round(position * 0.05 * 60);
    elements.predTime.textContent = `+${minutes}s`;
}

/**
 * Load initial metadata
 */
async function loadMetadata() {
    try {
        const metadata = await api.getMetadata();
        setState({ metadata });

        // Update footer
        elements.dataVersion.textContent = metadata.data_version || '--';
        elements.lastTrained.textContent = metadata.last_trained
            ? new Date(metadata.last_trained).toLocaleDateString()
            : '--';

        // Update stats
        elements.statVersion.textContent = metadata.data_version?.split('-')[0]?.replace('v', '') || '--';

        // Set date range if available
        if (metadata.date_range?.end) {
            elements.datePicker.value = metadata.date_range.end;
            setState({ selectedDate: metadata.date_range.end });
        }

    } catch (err) {
        console.error('Metadata load error:', err);
    }
}

/**
 * Application initialization
 */
async function init() {
    console.log('SkyFlux AI initializing...');

    // Initialize cache
    await initDB();

    // Initialize DOM references
    initElements();

    // Set initial date
    elements.datePicker.value = state.selectedDate;

    // Bind events
    bindEvents();

    // Initialize map
    initMap();

    // Initialize anomaly cards
    initAnomalyCards();

    // Load metadata
    await loadMetadata();

    // Load initial data
    await loadDataForView('map');

    console.log('SkyFlux AI ready!');
}

// Start application
init().catch(console.error);
