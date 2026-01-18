/**
 * SkyFlux AI - Main Entry Point
 * 
 * Initializes the application and coordinates all components.
 */

import { initDB, clearCache } from './services/cache.js';
import * as api from './services/api.js';
import { state, subscribe, setState } from './stores/appState.js';
import { initMap, fitToBounds, setTrajectories, renderPlanesAtTime, getTimeAtPosition, timeRange, stopTimeTravel } from './components/MapView.js';
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
        } else {
            // Stopped - clear interval
            if (timeTravelInterval) {
                clearInterval(timeTravelInterval);
                timeTravelInterval = null;
            }
        }
    });

    // Time travel slider for manual scrubbing
    elements.timeSlider.addEventListener('input', (e) => {
        const pos = parseInt(e.target.value);

        // Build set of anomaly flight IDs for red highlighting
        const anomalyFlightIds = new Set();
        if (state.anomalies) {
            state.anomalies.forEach(a => anomalyFlightIds.add(a.flight_id));
        }

        renderPlanesAtTime(pos, anomalyFlightIds);
        updateTimeTravelDisplay(pos);
    });

    // Region filter handlers
    const regionGlobal = document.getElementById('region-global');
    const regionCustom = document.getElementById('region-custom');
    const regionBounds = document.getElementById('region-bounds');
    const applyRegionFilter = document.getElementById('apply-region-filter');

    if (regionGlobal && regionCustom && regionBounds) {
        // Toggle bounds visibility based on mode
        regionGlobal.addEventListener('change', () => {
            regionBounds.style.display = 'none';
            setState({ regionFilter: null });
        });

        regionCustom.addEventListener('change', () => {
            regionBounds.style.display = 'block';
        });

        // Apply custom bounds filter
        applyRegionFilter?.addEventListener('click', () => {
            const minLat = parseFloat(document.getElementById('bound-min-lat').value) || -90;
            const maxLat = parseFloat(document.getElementById('bound-max-lat').value) || 90;
            const minLon = parseFloat(document.getElementById('bound-min-lon').value) || -180;
            const maxLon = parseFloat(document.getElementById('bound-max-lon').value) || 180;

            setState({
                regionFilter: { minLat, maxLat, minLon, maxLon }
            });

            console.log(`[Region] Filter applied: ${minLat},${minLon} to ${maxLat},${maxLon}`);

            // Re-render current view with filter (client-side only, no refetch)
            if (state.map) {
                state.map.invalidateSize();
            }
        });
    }
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
    // Toggle views
    const isModelView = view === 'models';
    const mapContainer = document.getElementById('map-view-container');
    const modelsContainer = document.getElementById('models-view-container');

    if (mapContainer && modelsContainer) {
        mapContainer.style.display = isModelView ? 'none' : 'block';
        modelsContainer.style.display = isModelView ? 'block' : 'none';

        if (!isModelView && state.map) {
            setTimeout(() => state.map.invalidateSize(), 100);
        }
    }

    const params = {
        date: state.selectedDate,
        hour: state.selectedHour,
    };

    setState({ isLoading: true });
    setStatus('loading', 'Loading...');

    try {
        switch (view) {
            case 'models':
                try {
                    const metadata = await api.getMetadata();
                    renderModelsList(metadata.models || []);
                } catch (e) {
                    console.error("Failed to load models", e);
                    renderModelsList([]);
                }
                break;

            case 'map':
                // Fetch predictions, anomalies, AND trajectories for time travel
                const [predictionsRes, anomaliesRes, trajectoriesRes] = await Promise.all([
                    api.getPredictions({ ...params, limit: 500 }),
                    api.getAnomalies({ ...params, type: 'route', limit: 100 }),
                    api.getTrajectories({ ...params, limit: 100 }),
                ]);
                setState({
                    predictions: predictionsRes.predictions || [],
                    anomalies: anomaliesRes.anomalies || []
                });

                // Set trajectories for time travel animation
                if (trajectoriesRes.trajectories) {
                    setTrajectories(trajectoriesRes.trajectories);
                    // Reset time slider to start
                    elements.timeSlider.value = 0;
                    updateTimeTravelDisplay(0);
                }
                break;

            case 'density':
                // Fetch density AND predictions AND trajectories concurrently
                const [densityRes, densPreds, densTraj] = await Promise.all([
                    api.getDensity(params),
                    api.getPredictions({ ...params, limit: 500 }),
                    api.getTrajectories({ ...params, limit: 1000 })
                ]);
                setState({
                    density: densityRes.grid || [],
                    predictions: densPreds.predictions || []
                });

                if (densTraj.trajectories) {
                    setTrajectories(densTraj.trajectories);
                }

                if (densityRes.grid?.length) fitToBounds(densityRes.grid);
                break;

            case 'anomalies':
                const anomalies = await api.getAnomalies({ ...params, min_score: 0.5, limit: 50 });
                setState({ anomalies: anomalies.anomalies || [] });
                updateStats({ anomalyCount: anomalies.count });
                break;

            case 'stress':
                // Fetch stress AND predictions AND trajectories concurrently
                const [stressRes, stressPreds, stressTraj] = await Promise.all([
                    api.getStress({ ...params, min_stress: 25 }),
                    api.getPredictions({ ...params, limit: 500 }),
                    api.getTrajectories({ ...params, limit: 1000 })
                ]);
                setState({
                    stress: stressRes.stress_grid || [],
                    predictions: stressPreds.predictions || []
                });

                if (stressTraj.trajectories) {
                    setTrajectories(stressTraj.trajectories);
                }

                if (stressRes.stress_grid?.length) fitToBounds(stressRes.stress_grid);
                updateStats({ avgStress: calculateAvgStress(stressRes.stress_grid) });
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

    // Build set of anomaly flight IDs for red highlighting
    const anomalyFlightIds = new Set();
    if (state.anomalies) {
        state.anomalies.forEach(a => anomalyFlightIds.add(a.flight_id));
    }

    timeTravelInterval = setInterval(() => {
        let pos = parseInt(elements.timeSlider.value) + 1;

        if (pos > 100) {
            pos = 0;
            setState({ isPlaying: false });
            elements.playBtn.textContent = '▶ Play';
            stopTimeTravel();  // Exit time travel mode
            clearInterval(timeTravelInterval);
            return;
        }

        elements.timeSlider.value = pos;

        // Render planes at the current time position (don't call setState to avoid layer clearing)
        renderPlanesAtTime(pos, anomalyFlightIds);

        // Update time display
        updateTimeTravelDisplay(pos);
    }, 200);  // 200ms per frame for smooth animation
}

function updateTimeTravelDisplay(position) {
    // Show actual UTC time based on trajectory data range
    const time = getTimeAtPosition(position);
    if (time && !isNaN(time.getTime())) {
        elements.predTime.textContent = time.toISOString().substr(11, 8);
    } else {
        const minutes = Math.round(position * 0.05 * 60);
        elements.predTime.textContent = `+${minutes}s`;
    }
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

/**
 * Render list of models
 */
function renderModelsList(models) {
    const container = document.getElementById('models-list');
    if (!container) return;

    if (!models || models.length === 0) {
        container.innerHTML = '<p style="color: #9ca3af; grid-column: 1/-1;">No models found in registry.</p>';
        return;
    }

    container.innerHTML = models.map(m => `
        <div class="model-card" style="
            background: #16213e; 
            border-left: 4px solid #3b82f6; 
            padding: 20px; 
            border-radius: 4px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        ">
            <h3 style="color: #fff; margin: 0 0 10px 0;">${(m.type || 'Unknown').toUpperCase()}</h3>
            <div style="font-family: monospace; color: #9ca3af; margin-bottom: 15px; font-size: 0.9em;">ID: ${m.model_id}</div>
            <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #333; padding-top: 10px;">
                <span style="color: #60a5fa; font-size: 0.9em;">v${m.version}</span>
                <span style="color: #6b7280; font-size: 0.8em;">${m.trained_at ? new Date(Number(m.trained_at) * 1000).toLocaleDateString() : 'N/A'}</span>
            </div>
            <div style="margin-top: 15px;">
                <span style="
                    display: inline-block; 
                    padding: 4px 8px; 
                    border-radius: 12px; 
                    background: rgba(16, 185, 129, 0.2); 
                    color: #10b981; 
                    font-size: 0.8em;
                ">Active</span>
            </div>
        </div>
    `).join('');
}

// Start application
init().catch(console.error);
