/**
 * SkyFlux AI - Anomaly Cards Component
 * 
 * Displays anomaly explanation cards as overlays on the map.
 */

import { state, subscribe } from '../stores/appState.js';

let container = null;

/**
 * Initialize anomaly cards container
 */
function initAnomalyCards() {
    container = document.getElementById('overlay-cards');

    subscribe((state) => {
        if (state.currentView === 'anomalies' || state.currentView === 'map') {
            renderCards(state.anomalies);
        } else {
            clearCards();
        }
    });
}

/**
 * Render anomaly cards
 */
function renderCards(anomalies) {
    if (!container) return;

    // Show top 5 anomalies
    const topAnomalies = (anomalies || [])
        .slice(0, 5);

    container.innerHTML = topAnomalies.map((anomaly, idx) => {
        const explanation = anomaly.explanation || {};
        const typeClass = `type-${anomaly.type || 'behavior'}`;

        return `
      <div class="anomaly-card ${typeClass}" data-idx="${idx}">
        <button class="anomaly-card-close" data-close="${idx}">&times;</button>
        <div class="anomaly-card-header">
          <span class="anomaly-card-title">${anomaly.flight_id || 'Unknown'}</span>
          <span class="anomaly-score">${(anomaly.score * 100).toFixed(0)}%</span>
        </div>
        <div class="anomaly-card-body">
          <strong>${capitalizeFirst(anomaly.type || 'behavior')} anomaly</strong><br>
          ${explanation.context || 'Unusual pattern detected in flight behavior.'}
          ${explanation.deviation_pct ? `<br><em>Deviation: ${explanation.deviation_pct.toFixed(1)}%</em>` : ''}
        </div>
      </div>
    `;
    }).join('');

    // Add close button handlers
    container.querySelectorAll('.anomaly-card-close').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const idx = e.target.dataset.close;
            const card = container.querySelector(`[data-idx="${idx}"]`);
            if (card) card.remove();
        });
    });
}

/**
 * Clear all cards
 */
function clearCards() {
    if (container) {
        container.innerHTML = '';
    }
}

/**
 * Capitalize first letter
 */
function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

export { initAnomalyCards, renderCards, clearCards };
