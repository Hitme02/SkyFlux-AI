# SkyFlux AI — Limitations & Failure Modes

This document describes known limitations and failure modes of the SkyFlux AI system. It is intended for engineering review and honest assessment of system boundaries.

---

## 1. Data Source Limitations

### ADS-B Coverage Gaps

- **Oceanic regions**: ADS-B reception depends on ground station coverage. Over oceans, data is sparse or absent.
- **Low-altitude flights**: Aircraft below ~3,000 ft may not be reliably tracked due to line-of-sight constraints.
- **Mountainous terrain**: Terrain blockage reduces coverage in valleys and mountainous regions.
- **Data providers**: System relies on third-party ADS-B aggregators (e.g., `adsblol`). Provider outages affect data quality.

### Non-Cooperative Aircraft

- Military and government aircraft frequently use transponder modes that mask identity.
- General aviation aircraft may fly without ADS-B Out (varies by jurisdiction/regulation).
- **No intent modeling**: The system has no knowledge of flight plans, pilot intentions, or ATC clearances.

### Sampling Bias

- The current dataset is a **single-day snapshot** (Dec 25, 2025).
- Holiday traffic patterns may not represent typical weekday operations.
- Models trained on limited data may not generalize to other dates, seasons, or airspaces.

---

## 2. Processing Limitations

### Batch Processing (Not Real-Time)

- Data flows through a **batch pipeline** (Bronze → Silver → Gold).
- There is **no streaming ingestion** or real-time updates.
- "Predictions" are post-hoc evaluations, not live forecasts.

### Trajectory Reconstruction

- Trajectories are inferred by grouping points by `flight_id` (hex + callsign).
- **ID collisions** may occur if multiple aircraft share callsigns during the same day.
- Missing points or gaps in transmission create fragmented trajectories.

---

## 3. Model Limitations

### Prediction Quality

- Predictions use **constant-velocity assumption** (dead reckoning).
- No turn anticipation, altitude change prediction, or physics-based flight dynamics.
- Error increases with prediction horizon (30–60 seconds is the useful range).

### Anomaly Detection

- Anomalies are statistical deviations from training data distribution.
- **Not collision detection**: A high anomaly score does not indicate imminent danger.
- Model may flag unusual-but-normal operations (e.g., go-arounds, holding patterns).

### Stress Index Semantics

- Stress is a **composite metric**, not a collision probability.
- Components: density, maneuver variance, heading conflict, anomaly presence.
- High stress does not automatically mean unsafe airspace.

---

## 4. System Limitations

### No Automatic Retraining

- Models are retrained **only on explicit trigger** (manual admin action).
- Drift detection is not implemented; stale models may degrade accuracy over time.

### Read-Only Backend

- Backend APIs are thin wrappers over pre-computed Gold artifacts.
- No on-demand compute or ad-hoc queries.

### Frontend Caching

- Frontend caches aggressively based on `model_version`.
- Stale cache entries may persist until version changes or cache is cleared.

---

## 5. Operational Considerations

### This System Is Not

- An operational ATC tool
- A safety-critical system
- A real-time collision avoidance system (TCAS)
- Certified for any aviation operational use

### Intended Use

- Research and demonstration of ML pipelines on flight data.
- Exploratory analytics and visualization.
- Proof-of-concept for medallion architecture patterns.

---

## Summary

This system is an **engineering prototype** demonstrating batch analytics on ADS-B data. It has meaningful limitations in data coverage, model accuracy, and operational scope. Users should interpret outputs with appropriate skepticism and not rely on this system for any safety-critical decisions.
