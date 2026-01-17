/**
 * SkyFlux AI - Map View Component
 * 
 * Handles Leaflet map initialization and overlays for:
 * - Trajectory visualization
 * - Density heatmaps
 * - Stress index overlays
 * - Anomaly markers
 */

import { state, subscribe, setState } from '../stores/appState.js';

// Leaflet map instance
let map = null;

// Layer groups for different overlays
const layers = {
    trajectories: null,
    density: null,
    stress: null,
    anomalies: null,
    predictions: null,
};

// Color scales
const STRESS_COLORS = {
    low: '#10b981',
    medium: '#f59e0b',
    high: '#ef4444',
    critical: '#dc2626',
};

const DENSITY_GRADIENT = [
    [0.0, '#1a1a2e'],
    [0.2, '#16213e'],
    [0.4, '#0f3460'],
    [0.6, '#3b82f6'],
    [0.8, '#8b5cf6'],
    [1.0, '#f472b6'],
];

/**
 * Initialize Leaflet map
 */
function initMap() {
    if (map) return map;

    // Create map
    map = L.map('map', {
        center: [40.7, -74.0],  // Default to NYC area
        zoom: 6,
        zoomControl: true,
        attributionControl: false,
    });

    // Dark tile layer (CartoDB Dark Matter)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
        attribution: '© OpenStreetMap, © CARTO',
    }).addTo(map);

    // Initialize layer groups
    layers.trajectories = L.layerGroup().addTo(map);
    layers.density = L.layerGroup().addTo(map);
    layers.stress = L.layerGroup().addTo(map);
    layers.anomalies = L.layerGroup().addTo(map);
    layers.predictions = L.layerGroup().addTo(map);

    // Subscribe to state changes
    subscribe((state) => {
        updateLayers(state);
    });

    return map;
}

/**
 * Update map layers based on state
 */
function updateLayers(state) {
    const view = state.currentView;

    // Clear all layers
    Object.values(layers).forEach(layer => layer.clearLayers());

    // Show relevant layers based on view
    switch (view) {
        case 'map':
            renderTrajectories(state.predictions);
            break;
        case 'density':
            renderDensityHeatmap(state.density);
            break;
        case 'stress':
            renderStressOverlay(state.stress);
            break;
        case 'anomalies':
            renderAnomalyMarkers(state.anomalies);
            break;
    }
}

/**
 * Render trajectory polylines
 */
function renderTrajectories(predictions) {
    if (!predictions || predictions.length === 0) return;

    predictions.slice(0, 100).forEach((pred, idx) => {
        // Render predicted path
        if (pred.predicted && pred.predicted.length > 0) {
            const predPath = pred.predicted.map(p => [p.lat, p.lon]);

            // Get actual end point if available
            if (pred.actual && pred.actual.length > 0) {
                // Add line from prediction start to predicted end
                const predictedLine = L.polyline(predPath, {
                    color: '#8b5cf6',
                    weight: 2,
                    opacity: 0.7,
                    dashArray: '5, 5',
                }).addTo(layers.predictions);

                // Add actual path
                const actualPath = pred.actual.map(p => [p.lat, p.lon]);
                L.polyline(actualPath, {
                    color: '#10b981',
                    weight: 2,
                    opacity: 0.7,
                }).addTo(layers.predictions);

                // Add error line
                if (pred.predicted.length > 0 && pred.actual.length > 0) {
                    const predEnd = pred.predicted[pred.predicted.length - 1];
                    const actEnd = pred.actual[pred.actual.length - 1];
                    L.polyline([
                        [predEnd.lat, predEnd.lon],
                        [actEnd.lat, actEnd.lon],
                    ], {
                        color: '#ef4444',
                        weight: 1,
                        opacity: 0.5,
                        dashArray: '2, 4',
                    }).addTo(layers.predictions);
                }
            }
        }
    });

    // Update legend
    updateLegend('predictions');
}

/**
 * Render density heatmap using rectangles (simple approach)
 */
function renderDensityHeatmap(densityData) {
    if (!densityData || densityData.length === 0) return;

    const gridSize = 0.5; // degrees

    densityData.forEach(cell => {
        const bounds = [
            [cell.lat - gridSize / 2, cell.lon - gridSize / 2],
            [cell.lat + gridSize / 2, cell.lon + gridSize / 2],
        ];

        // Apply what-if factor
        let score = cell.score * (state.densityFactor / 100);
        score = Math.min(1, Math.max(0, score));

        // Get color from gradient
        const color = interpolateGradient(DENSITY_GRADIENT, score);

        const rect = L.rectangle(bounds, {
            color: 'transparent',
            fillColor: color,
            fillOpacity: 0.6,
            weight: 0,
        });

        rect.bindPopup(`
      <strong>Density</strong><br>
      Score: ${(score * 100).toFixed(1)}%<br>
      Percentile: ${cell.percentile?.toFixed(1) || '--'}%
    `);

        rect.addTo(layers.density);
    });

    updateLegend('density');
}

/**
 * Render stress index overlay
 */
function renderStressOverlay(stressData) {
    if (!stressData || stressData.length === 0) return;

    const gridSize = 0.5;

    stressData.forEach(cell => {
        const bounds = [
            [cell.lat - gridSize / 2, cell.lon - gridSize / 2],
            [cell.lat + gridSize / 2, cell.lon + gridSize / 2],
        ];

        // Apply what-if factor to stress
        let stressIndex = cell.stress_index * (state.densityFactor / 100);
        stressIndex = Math.min(100, stressIndex);

        // Determine risk level
        let riskLevel = 'low';
        if (stressIndex >= 75) riskLevel = 'critical';
        else if (stressIndex >= 50) riskLevel = 'high';
        else if (stressIndex >= 25) riskLevel = 'medium';

        const color = STRESS_COLORS[riskLevel];

        const rect = L.rectangle(bounds, {
            color: color,
            fillColor: color,
            fillOpacity: 0.5,
            weight: 1,
        });

        const components = cell.components || {};
        rect.bindPopup(`
      <strong>Stress Index: ${stressIndex.toFixed(1)}</strong><br>
      Risk: ${riskLevel.toUpperCase()}<br>
      <hr style="margin: 4px 0; border-color: #333;">
      Density: ${((components.density_score || 0) * 100).toFixed(0)}%<br>
      Alt Variance: ${((components.altitude_variance || 0) * 100).toFixed(0)}%<br>
      Heading Conflict: ${((components.heading_conflict_score || 0) * 100).toFixed(0)}%
    `);

        rect.addTo(layers.stress);
    });

    updateLegend('stress');
}

/**
 * Render anomaly markers
 */
function renderAnomalyMarkers(anomalies) {
    if (!anomalies || anomalies.length === 0) return;

    anomalies.forEach(anomaly => {
        const loc = anomaly.location;
        if (!loc || !loc.lat || !loc.lon) return;

        // Create circle marker
        const marker = L.circleMarker([loc.lat, loc.lon], {
            radius: 8 + (anomaly.score * 8),
            color: '#ef4444',
            fillColor: '#ef4444',
            fillOpacity: 0.6,
            weight: 2,
        });

        const explanation = anomaly.explanation || {};
        marker.bindPopup(`
      <strong>${anomaly.flight_id}</strong><br>
      Type: ${anomaly.type}<br>
      Score: ${(anomaly.score * 100).toFixed(1)}%<br>
      <hr style="margin: 4px 0; border-color: #333;">
      ${explanation.context || 'Unknown anomaly'}
    `);

        marker.addTo(layers.anomalies);
    });

    updateLegend('anomalies');
}

/**
 * Interpolate color from gradient
 */
function interpolateGradient(gradient, value) {
    value = Math.max(0, Math.min(1, value));

    for (let i = 1; i < gradient.length; i++) {
        if (value <= gradient[i][0]) {
            const t = (value - gradient[i - 1][0]) / (gradient[i][0] - gradient[i - 1][0]);
            return lerpColor(gradient[i - 1][1], gradient[i][1], t);
        }
    }

    return gradient[gradient.length - 1][1];
}

/**
 * Linear interpolation between two hex colors
 */
function lerpColor(color1, color2, t) {
    const c1 = hexToRgb(color1);
    const c2 = hexToRgb(color2);

    const r = Math.round(c1.r + (c2.r - c1.r) * t);
    const g = Math.round(c1.g + (c2.g - c1.g) * t);
    const b = Math.round(c1.b + (c2.b - c1.b) * t);

    return `rgb(${r}, ${g}, ${b})`;
}

function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
    } : { r: 0, g: 0, b: 0 };
}

/**
 * Update map legend
 */
function updateLegend(type) {
    const legendItems = document.querySelector('.legend-items');
    if (!legendItems) return;

    legendItems.innerHTML = '';

    if (type === 'predictions') {
        legendItems.innerHTML = `
      <div class="legend-item">
        <div class="legend-color" style="background: #8b5cf6;"></div>
        <span>Predicted Path</span>
      </div>
      <div class="legend-item">
        <div class="legend-color" style="background: #10b981;"></div>
        <span>Actual Path</span>
      </div>
      <div class="legend-item">
        <div class="legend-color" style="background: #ef4444;"></div>
        <span>Prediction Error</span>
      </div>
    `;
    } else if (type === 'density') {
        legendItems.innerHTML = `
      <div class="legend-item">
        <div class="legend-color" style="background: linear-gradient(90deg, #1a1a2e, #3b82f6, #f472b6); width: 60px;"></div>
        <span>Low → High</span>
      </div>
    `;
    } else if (type === 'stress') {
        legendItems.innerHTML = `
      <div class="legend-item">
        <div class="legend-color" style="background: ${STRESS_COLORS.low};"></div>
        <span>Low</span>
      </div>
      <div class="legend-item">
        <div class="legend-color" style="background: ${STRESS_COLORS.medium};"></div>
        <span>Medium</span>
      </div>
      <div class="legend-item">
        <div class="legend-color" style="background: ${STRESS_COLORS.high};"></div>
        <span>High</span>
      </div>
      <div class="legend-item">
        <div class="legend-color" style="background: ${STRESS_COLORS.critical};"></div>
        <span>Critical</span>
      </div>
    `;
    } else if (type === 'anomalies') {
        legendItems.innerHTML = `
      <div class="legend-item">
        <div class="legend-color" style="background: #ef4444; width: 12px; height: 12px; border-radius: 50%;"></div>
        <span>Anomaly (size = severity)</span>
      </div>
    `;
    }
}

/**
 * Fit map to data bounds
 */
function fitToBounds(data) {
    if (!map || !data || data.length === 0) return;

    const lats = data.map(d => d.lat || d.location?.lat).filter(Boolean);
    const lons = data.map(d => d.lon || d.location?.lon).filter(Boolean);

    if (lats.length === 0) return;

    const bounds = [
        [Math.min(...lats), Math.min(...lons)],
        [Math.max(...lats), Math.max(...lons)],
    ];

    map.fitBounds(bounds, { padding: [50, 50] });
}

export { initMap, updateLayers, fitToBounds, layers };
