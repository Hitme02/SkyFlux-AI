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

const layers = {
    trajectories: null,
    density: null,
    stress: null,
    anomalies: null,
    predictions: null,
};

// Time travel state
let trajectoryCache = [];  // Cached trajectory data for animation
let timeRange = { min: 0, max: 0 };  // Timestamp range for current data
let currentTimePosition = 0;  // Current animation time (0-100 slider position)
let isTimeTraveling = false;  // Flag to prevent layer clearing during animation

/**
 * Set trajectories for time travel animation
 */
function setTrajectories(trajectories) {
    trajectoryCache = trajectories || [];
    console.log(`[TimeTravel] setTrajectories called with ${trajectoryCache.length} trajectories`);

    // Calculate time range from all trajectory points
    let minTs = Infinity;
    let maxTs = -Infinity;
    let totalPoints = 0;

    trajectoryCache.forEach((traj, idx) => {
        // Defensive: ensure points is an array
        if (!traj.points || !Array.isArray(traj.points)) {
            if (idx === 0) console.warn('[TimeTravel] First trajectory has no valid points array:', traj);
            return;
        }
        totalPoints += traj.points.length;
        traj.points.forEach(p => {
            const ts = typeof p.ts === 'number' ? p.ts : parseInt(p.ts || 0);
            if (ts > 0) {
                minTs = Math.min(minTs, ts);
                maxTs = Math.max(maxTs, ts);
            }
        });
    });

    console.log(`[TimeTravel] Total points: ${totalPoints}, min: ${minTs}, max: ${maxTs}`);

    if (minTs !== Infinity && maxTs !== -Infinity) {
        // IMPORTANT: Mutate the existing object instead of reassigning
        // This keeps the exported reference valid
        timeRange.min = minTs;
        timeRange.max = maxTs;
        console.log(`[TimeTravel] Range: ${new Date(minTs * 1000).toISOString()} to ${new Date(maxTs * 1000).toISOString()}`);

        // Auto-render first frame at position 0
        renderPlanesAtTime(0, new Set());
    } else {
        console.warn('[TimeTravel] No valid timestamps found in trajectory data');
    }
}


/**
 * Interpolate plane position at a specific timestamp
 */
function interpolatePosition(points, targetTs) {
    if (!points || points.length === 0) return null;

    // Sort points by timestamp
    const sorted = [...points].sort((a, b) => a.ts - b.ts);

    // Find surrounding points
    let before = null;
    let after = null;

    for (let i = 0; i < sorted.length; i++) {
        if (sorted[i].ts <= targetTs) {
            before = sorted[i];
        }
        if (sorted[i].ts >= targetTs && !after) {
            after = sorted[i];
            break;
        }
    }

    // Return exact match or closest
    if (before && after && before.ts !== after.ts) {
        // Interpolate between points
        const t = (targetTs - before.ts) / (after.ts - before.ts);

        // Calculate dynamic heading from the vector between points
        const dLon = (after.lon - before.lon) * Math.PI / 180;
        const lat1 = before.lat * Math.PI / 180;
        const lat2 = after.lat * Math.PI / 180;

        const y = Math.sin(dLon) * Math.cos(lat2);
        const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);

        let bearing = Math.atan2(y, x) * 180 / Math.PI;
        bearing = (bearing + 360) % 360;

        return {
            lat: before.lat + (after.lat - before.lat) * t,
            lon: before.lon + (after.lon - before.lon) * t,
            alt_ft: before.alt_ft + (after.alt_ft - before.alt_ft) * t,
            heading_deg: bearing,  // Use calculated dynamic heading
        };
    }

    return before || after || sorted[0];
}

/**
 * Render planes at a specific time position (0-100)
 */
function renderPlanesAtTime(position, anomalyFlightIds = new Set()) {
    if (!trajectoryCache || trajectoryCache.length === 0) {
        console.log('[TimeTravel] No trajectory cache');
        return;
    }

    if (timeRange.max === 0) {
        console.log('[TimeTravel] Invalid time range');
        return;
    }

    isTimeTraveling = true;  // Set flag to prevent updateLayers from clearing

    // Clear predictions layer for plane icons
    if (layers.predictions) {
        layers.predictions.clearLayers();
    } else {
        console.warn('[TimeTravel] layers.predictions is null. Map might not be initialized.');
        // Attempt to recover if map exists but layers don't (unlikely but possible)
        if (map) {
            console.log('[TimeTravel] Re-initializing prediction layer');
            layers.predictions = L.layerGroup().addTo(map);
        } else {
            return; // Cannot render without map/layer
        }
    }

    // Calculate target timestamp from position
    const range = timeRange.max - timeRange.min;
    const targetTs = timeRange.min + (position / 100) * range;

    let renderedCount = 0;
    const maxRender = 100;

    trajectoryCache.forEach(traj => {
        if (renderedCount >= maxRender) return;

        const pos = interpolatePosition(traj.points, targetTs);
        if (!pos || !pos.lat || !pos.lon) return;

        const isAnomaly = anomalyFlightIds.has(traj.flight_id);

        // Create plane marker at interpolated position
        const marker = L.marker([pos.lat, pos.lon], {
            icon: createPlaneIcon(pos.heading_deg || 0, isAnomaly),
            zIndexOffset: isAnomaly ? 1000 : 0,
        });

        // Format time for display
        const timeStr = new Date(targetTs * 1000).toISOString().substr(11, 8);

        marker.bindPopup(`
            <strong>${traj.flight_id}</strong><br>
            ${traj.callsign ? `Callsign: ${traj.callsign}<br>` : ''}
            ${isAnomaly ? '<span style="color:#ef4444;">⚠️ Route Anomaly</span><br>' : ''}
            Alt: ${pos.alt_ft?.toLocaleString() || '--'} ft<br>
            Time: ${timeStr}
        `);

        marker.addTo(layers.predictions);
        renderedCount++;

        // Draw trail for this plane (last N points before current time)
        const trailPoints = traj.points
            .filter(p => {
                const ts = typeof p.ts === 'number' ? p.ts : parseInt(p.ts || 0);
                return ts <= targetTs && ts > targetTs - 300;
            })
            .map(p => [p.lat, p.lon]);

        if (trailPoints.length > 1) {
            L.polyline(trailPoints, {
                color: isAnomaly ? '#ef4444' : '#3b82f6',
                weight: 2,
                opacity: 0.5,
            }).addTo(layers.predictions);
        }
    });

    currentTimePosition = position;
    console.log(`[TimeTravel] Rendered ${renderedCount} planes at position ${position}`);
}

/**
 * Update time display in UI
 */
function getTimeAtPosition(position) {
    const range = timeRange.max - timeRange.min;
    const targetTs = timeRange.min + (position / 100) * range;
    return new Date(targetTs * 1000);
}

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

    // Don't clear prediction layers during time travel animation
    if (isTimeTraveling && view === 'map') {
        return;  // Skip layer update during animation
    }

    // Clear all layers
    Object.values(layers).forEach(layer => layer.clearLayers());

    // Show relevant layers based on view
    switch (view) {
        case 'map':
            // Only render live planes if not in time travel mode
            if (!isTimeTraveling) {
                renderLivePlanes(state.predictions, state.anomalies);
            }
            break;
        case 'density':
            renderDensityHeatmap(state.density, trajectoryCache);
            break;
        case 'stress':
            renderStressOverlay(state.stress, trajectoryCache);
            break;
        case 'anomalies':
            renderAnomalyMarkers(state.anomalies);
            break;
    }
}

/**
 * Create a rotated plane icon using DivIcon
 */
function createPlaneIcon(heading, isAnomaly = false) {
    const rotation = heading || 0;
    const color = isAnomaly ? '#ef4444' : '#3b82f6';
    const size = isAnomaly ? 24 : 18;

    return L.divIcon({
        className: 'plane-icon',
        html: `<div style="
            font-size: ${size}px;
            color: ${color};
            transform: rotate(${rotation}deg);
            text-shadow: 0 0 3px rgba(0,0,0,0.8);
            filter: drop-shadow(0 2px 2px rgba(0,0,0,0.5));
        ">✈</div>`,
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2],
    });
}

/**
 * Render live planes on the map with positions from predictions
 * Shows route anomaly flights with their predicted vs actual paths
 */
function renderLivePlanes(predictions, anomalies) {
    if (!predictions || predictions.length === 0) return;

    // Build a set of anomalous flight IDs
    const routeAnomalyFlights = new Set();
    const anomalyMap = new Map();

    if (anomalies && anomalies.length > 0) {
        anomalies.forEach(a => {
            if (a.type === 'route') {
                routeAnomalyFlights.add(a.flight_id);
                anomalyMap.set(a.flight_id, a);
            }
        });
    }

    // Group predictions by flight_id
    const flightPredictions = new Map();
    predictions.forEach(pred => {
        const fid = pred.flight_id;
        if (!flightPredictions.has(fid)) {
            flightPredictions.set(fid, []);
        }
        flightPredictions.get(fid).push(pred);
    });

    // Render each unique flight
    let planeCount = 0;
    const maxPlanes = 1500;

    for (const [flightId, preds] of flightPredictions) {
        if (planeCount >= maxPlanes) break;

        const isAnomaly = routeAnomalyFlights.has(flightId);

        // Get the latest prediction for current position
        const latestPred = preds[preds.length - 1];

        // Get current position from actual data (last known)
        let currentPos = null;
        let heading = 0;

        if (latestPred.actual && latestPred.actual.length > 0) {
            const act = latestPred.actual[0];
            currentPos = [act.lat, act.lon];
            // Calculate heading from actual path if multiple points
            heading = calculateHeading(preds);
        } else if (latestPred.predicted && latestPred.predicted.length > 0) {
            const pred = latestPred.predicted[0];
            currentPos = [pred.lat, pred.lon];
        }

        if (!currentPos) continue;

        // Add plane marker
        const planeMarker = L.marker(currentPos, {
            icon: createPlaneIcon(heading, isAnomaly),
            zIndexOffset: isAnomaly ? 1000 : 0,
        });

        // Bind popup with flight info
        planeMarker.bindPopup(`
            <strong>${flightId}</strong><br>
            ${isAnomaly ? '<span style="color:#ef4444;">⚠️ Route Anomaly</span><br>' : ''}
            Alt: ${latestPred.actual?.[0]?.alt_ft?.toLocaleString() || '--'} ft
        `);

        planeMarker.addTo(layers.predictions);
        planeCount++;

        // For route anomaly flights, draw predicted vs actual paths
        if (isAnomaly && preds.length > 0) {
            // Collect all predicted points for this flight
            const predictedPath = [];
            const actualPath = [];

            preds.forEach(p => {
                if (p.predicted && p.predicted.length > 0) {
                    predictedPath.push([p.predicted[0].lat, p.predicted[0].lon]);
                }
                if (p.actual && p.actual.length > 0) {
                    actualPath.push([p.actual[0].lat, p.actual[0].lon]);
                }
            });

            // Draw predicted path (dotted)
            if (predictedPath.length > 1) {
                L.polyline(predictedPath, {
                    color: '#f59e0b',
                    weight: 3,
                    opacity: 0.8,
                    dashArray: '8, 8',
                }).addTo(layers.predictions).bindPopup('Predicted Path');
            }

            // Draw actual path (solid)
            if (actualPath.length > 1) {
                L.polyline(actualPath, {
                    color: '#ef4444',
                    weight: 3,
                    opacity: 0.9,
                }).addTo(layers.predictions).bindPopup('Actual Path (Anomaly)');
            }

            // Draw deviation line between endpoints
            if (predictedPath.length > 0 && actualPath.length > 0) {
                const predEnd = predictedPath[predictedPath.length - 1];
                const actEnd = actualPath[actualPath.length - 1];

                L.polyline([predEnd, actEnd], {
                    color: '#dc2626',
                    weight: 2,
                    opacity: 0.6,
                    dashArray: '4, 4',
                }).addTo(layers.predictions);
            }
        }
    }

    // Update legend for live view
    updateLegendLive();
}

/**
 * Calculate approximate heading from prediction history
 */
function calculateHeading(predictions) {
    if (predictions.length < 2) return 0;

    const first = predictions[0].actual?.[0] || predictions[0].predicted?.[0];
    const last = predictions[predictions.length - 1].actual?.[0] || predictions[predictions.length - 1].predicted?.[0];

    if (!first || !last) return 0;

    const dLon = (last.lon - first.lon) * Math.PI / 180;
    const lat1 = first.lat * Math.PI / 180;
    const lat2 = last.lat * Math.PI / 180;

    const y = Math.sin(dLon) * Math.cos(lat2);
    const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);

    let bearing = Math.atan2(y, x) * 180 / Math.PI;
    return (bearing + 360) % 360;
}

/**
 * Update legend for live plane view
 */
function updateLegendLive() {
    const legendItems = document.querySelector('.legend-items');
    if (!legendItems) return;

    legendItems.innerHTML = `
        <div class="legend-item">
            <div style="font-size: 14px; color: #3b82f6;">✈</div>
            <span>Normal Flight</span>
        </div>
        <div class="legend-item">
            <div style="font-size: 16px; color: #ef4444;">✈</div>
            <span>Route Anomaly</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #f59e0b; width: 30px; height: 3px; border-style: dashed;"></div>
            <span>Predicted Path</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #ef4444; width: 30px; height: 3px;"></div>
            <span>Actual Path</span>
        </div>
    `;
}

/**
 * Render trajectory polylines
 */
function renderTrajectories(predictions) {
    if (!predictions || predictions.length === 0) return;

    predictions.slice(0, 200).forEach((pred, idx) => {
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
 * Get planes inside a grid cell using trajectory history
 */
function getPlanesInCell(cell, gridSize, trajectories) {
    if (!trajectories || trajectories.length === 0) return [];

    // Bounds
    const latMin = cell.lat - gridSize / 2;
    const latMax = cell.lat + gridSize / 2;
    const lonMin = cell.lon - gridSize / 2;
    const lonMax = cell.lon + gridSize / 2;

    const planes = [];

    trajectories.forEach(t => {
        // Check if any point in the trajectory is within bounds
        const isInCell = t.points && t.points.some(p =>
            p.lat >= latMin && p.lat <= latMax &&
            p.lon >= lonMin && p.lon <= lonMax
        );

        if (isInCell) {
            // Get altitude from the first point in bounds (approximate)
            const pointInBounds = t.points.find(p =>
                p.lat >= latMin && p.lat <= latMax &&
                p.lon >= lonMin && p.lon <= lonMax
            );

            planes.push({
                id: t.flight_id,
                alt: pointInBounds ? (pointInBounds.alt_ft || pointInBounds.alt) : 0
            });
        }
    });

    return planes;
}

/**
 * Render density heatmap using rectangles (simple approach)
 */
function renderDensityHeatmap(densityData, trajectories) {
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

        // Find intersecting planes
        const intersectingPlanes = getPlanesInCell(cell, gridSize, trajectories);

        let planeListHtml = '';
        if (intersectingPlanes.length > 0) {
            planeListHtml = `
                <hr style="margin: 4px 0; border-color: #555;">
                <div style="font-size: 11px; max-height: 100px; overflow-y: auto;">
                    <strong>Planes in Sector (${intersectingPlanes.length}):</strong><br>
                    ${intersectingPlanes.map(p =>
                `<span style="color: #60a5fa;">${p.id}</span> <span style="color: #9ca3af;">(${p.alt?.toLocaleString()}ft)</span>`
            ).join('<br>')}
                </div>
            `;
        }

        rect.bindPopup(`
            <strong>Density Sector</strong><br>
            Score: ${(score * 100).toFixed(1)}%<br>
            Percentile: ${cell.percentile?.toFixed(1) || '--'}%
            ${planeListHtml}
        `);

        rect.addTo(layers.density);
    });

    updateLegend('density');
}

/**
 * Render stress index overlay
 */
function renderStressOverlay(stressData, trajectories) {
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

        // Find intersecting planes
        const intersectingPlanes = getPlanesInCell(cell, gridSize, trajectories);

        let planeListHtml = '';
        if (intersectingPlanes.length > 0) {
            planeListHtml = `
                <hr style="margin: 4px 0; border-color: #555;">
                <div style="font-size: 11px; max-height: 100px; overflow-y: auto;">
                    <strong>Contributing Planes (${intersectingPlanes.length}):</strong><br>
                    ${intersectingPlanes.map(p =>
                `<span style="color: #fca5a5;">${p.id}</span> <span style="color: #9ca3af;">(${p.alt?.toLocaleString()}ft)</span>`
            ).join('<br>')}
                </div>
            `;
        }

        rect.bindPopup(`
            <strong>Stress Index: ${stressIndex.toFixed(1)}</strong><br>
            Risk: ${riskLevel.toUpperCase()}<br>
            <hr style="margin: 4px 0; border-color: #333;">
            Density: ${((components.density_score || 0) * 100).toFixed(0)}%<br>
            Alt Variance: ${((components.altitude_variance || 0) * 100).toFixed(0)}%<br>
            Heading Conflict: ${((components.heading_conflict_score || 0) * 100).toFixed(0)}%
            ${planeListHtml}
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

/**
 * Stop time travel mode
 */
function stopTimeTravel() {
    isTimeTraveling = false;
}

export {
    initMap,
    updateLayers,
    fitToBounds,
    layers,
    setTrajectories,
    renderPlanesAtTime,
    getTimeAtPosition,
    timeRange,
    stopTimeTravel,
};
