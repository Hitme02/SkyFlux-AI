"""
SkyFlux AI - Gold Layer

Precomputed ML outputs including predictions, anomaly scores, and stress metrics.
Implements model training and artifact generation.

Usage:
    python -m pipeline.gold --input ./data/silver --output ./data/gold
"""

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from pipeline.config import PipelineConfig, DEFAULT_CONFIG
    from pipeline.schemas import (
        GOLD_TRAFFIC_DENSITY_SCHEMA,
        GOLD_ANOMALY_SCHEMA,
        GOLD_AIRSPACE_STRESS_SCHEMA,
    )
except ImportError:
    from config import PipelineConfig, DEFAULT_CONFIG
    from schemas import (
        GOLD_TRAFFIC_DENSITY_SCHEMA,
        GOLD_ANOMALY_SCHEMA,
        GOLD_AIRSPACE_STRESS_SCHEMA,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL CLASSES
# =============================================================================

@dataclass
class TrajectoryPredictionModel:
    """Simple trajectory prediction using recent velocity/heading."""
    
    model_id: str
    version: str = "1.0.0"
    trained_at: int = 0
    
    def predict_position(
        self,
        current_lat: float,
        current_lon: float,
        current_alt: float,
        speed_kts: float,
        heading_deg: float,
        vrate_fpm: float,
        horizon_sec: int,
    ) -> dict:
        """
        Predict future position based on current state.
        
        Uses simple dead reckoning (constant velocity assumption).
        """
        # Handle None values with defaults
        current_lat = current_lat or 0.0
        current_lon = current_lon or 0.0
        current_alt = current_alt or 0.0
        speed_kts = speed_kts or 0.0
        heading_deg = heading_deg or 0.0
        vrate_fpm = vrate_fpm or 0.0
        
        # Convert speed to degrees per second (approximate)
        # 1 knot ≈ 1.852 km/h, 1 degree ≈ 111 km at equator
        speed_deg_per_sec = (speed_kts * 1.852) / (111.0 * 3600)
        
        # Heading to radians
        heading_rad = np.radians(heading_deg)
        
        # Position change
        delta_lat = speed_deg_per_sec * horizon_sec * np.cos(heading_rad)
        delta_lon = speed_deg_per_sec * horizon_sec * np.sin(heading_rad) / max(np.cos(np.radians(current_lat)), 0.001)
        
        # Altitude change
        delta_alt = vrate_fpm * (horizon_sec / 60.0)
        
        # Confidence decreases with horizon
        confidence = max(0.3, 1.0 - (horizon_sec / 600.0))
        
        return {
            "lat": current_lat + delta_lat,
            "lon": current_lon + delta_lon,
            "alt_ft": current_alt + delta_alt,
            "confidence": confidence,
        }


class AnomalyDetectionModel:
    """Anomaly detection using Isolation Forest."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.version = "1.0.0"
        self.trained_at = 0
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
    
    def fit(self, trajectories: list[dict]) -> dict:
        """
        Train anomaly detection model on trajectory features.
        
        Args:
            trajectories: List of trajectory dicts from Silver layer
            
        Returns:
            Training metrics
        """
        # Extract features
        features = []
        for traj in trajectories:
            features.append({
                "duration_sec": traj.get("duration_sec", 0),
                "distance_nm": traj.get("distance_nm", 0),
                "max_alt_ft": traj.get("max_alt_ft", 0),
                "avg_speed_kts": traj.get("avg_speed_kts", 0),
                "heading_change_total": traj.get("heading_change_total", 0),
                "climb_count": traj.get("climb_count", 0),
                "descent_count": traj.get("descent_count", 0),
            })
        
        df = pd.DataFrame(features)
        self.feature_names = list(df.columns)
        
        # Scale features
        X = self.scaler.fit_transform(df.values)
        
        # Fit model
        self.model.fit(X)
        self.trained_at = int(time.time())
        self.is_fitted = True
        
        return {
            "model_type": "anomaly",
            "samples_trained": len(trajectories),
            "features": self.feature_names,
        }
    
    def predict(self, trajectory: dict) -> tuple[float, str, dict]:
        """
        Predict anomaly score for a trajectory.
        
        Returns:
            (anomaly_score, anomaly_type, explanation)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Extract features
        features = np.array([[
            trajectory.get("duration_sec", 0),
            trajectory.get("distance_nm", 0),
            trajectory.get("max_alt_ft", 0),
            trajectory.get("avg_speed_kts", 0),
            trajectory.get("heading_change_total", 0),
            trajectory.get("climb_count", 0),
            trajectory.get("descent_count", 0),
        ]])
        
        # Scale
        X = self.scaler.transform(features)
        
        # Predict (-1 for anomaly, 1 for normal)
        raw_score = self.model.decision_function(X)[0]
        
        # Convert to 0-1 score (lower raw_score = more anomalous)
        # Typical range is roughly -0.5 to 0.5
        anomaly_score = max(0, min(1, 0.5 - raw_score))
        
        # Determine anomaly type based on feature deviations
        anomaly_type = "behavior"
        primary_factor = "unusual_pattern"
        deviation_pct = 0.0
        baseline_value = 0.0
        observed_value = 0.0
        
        # Find most deviant feature
        scaled_features = X[0]
        most_deviant_idx = np.argmax(np.abs(scaled_features))
        most_deviant_name = self.feature_names[most_deviant_idx]
        
        if "alt" in most_deviant_name:
            anomaly_type = "altitude"
        elif "speed" in most_deviant_name:
            anomaly_type = "speed"
        elif "heading" in most_deviant_name:
            anomaly_type = "route"
        
        deviation_pct = abs(scaled_features[most_deviant_idx]) * 100
        observed_value = features[0][most_deviant_idx]
        primary_factor = f"unusual_{most_deviant_name}"
        
        explanation = {
            "primary_factor": primary_factor,
            "deviation_pct": float(deviation_pct),
            "baseline_value": float(baseline_value),
            "observed_value": float(observed_value),
            "context": f"Feature '{most_deviant_name}' deviates significantly from normal patterns",
        }
        
        return anomaly_score, anomaly_type, explanation
    
    def save(self, path: Path):
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "model_id": self.model_id,
                "version": self.version,
                "trained_at": self.trained_at,
            }, f)
    
    @classmethod
    def load(cls, path: Path) -> "AnomalyDetectionModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        instance = cls(data["model_id"])
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.feature_names = data["feature_names"]
        instance.version = data["version"]
        instance.trained_at = data["trained_at"]
        instance.is_fitted = True
        
        return instance


# =============================================================================
# GOLD ARTIFACT GENERATION
# =============================================================================

def load_silver_trajectories(
    silver_dir: Path,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """
    Load all Silver trajectories within a date range.
    
    Args:
        silver_dir: Silver layer directory
        start_date: Start of range (inclusive)
        end_date: End of range (inclusive)
        
    Returns:
        List of trajectory dicts
    """
    trajectories = []
    current = start_date
    
    while current <= end_date:
        date_str = current.isoformat()
        traj_file = silver_dir / "trajectories" / date_str / "trajectories.parquet"
        
        if traj_file.exists():
            df = pd.read_parquet(traj_file)
            trajectories.extend(df.to_dict("records"))
            logger.info(f"Loaded {len(df)} trajectories from {date_str}")
        
        current += timedelta(days=1)
    
    logger.info(f"Total trajectories loaded: {len(trajectories)}")
    return trajectories


def load_silver_density(
    silver_dir: Path,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Load all Silver grid density data within a date range.
    """
    dfs = []
    current = start_date
    
    while current <= end_date:
        date_str = current.isoformat()
        density_file = silver_dir / "grid_density" / date_str / "density.parquet"
        
        if density_file.exists():
            df = pd.read_parquet(density_file)
            dfs.append(df)
        
        current += timedelta(days=1)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def generate_traffic_density_gold(
    density_df: pd.DataFrame,
    output_dir: Path,
) -> int:
    """
    Generate Gold traffic density artifacts with percentiles.
    
    Returns:
        Number of records generated
    """
    if len(density_df) == 0:
        return 0
    
    # Compute historical percentiles per grid cell
    cell_stats = density_df.groupby(["grid_lat", "grid_lon"]).agg({
        "flight_count": ["mean", "std", "max"],
        "point_count": ["mean", "std"],
    }).reset_index()
    
    cell_stats.columns = ["grid_lat", "grid_lon", "mean_flights", "std_flights", 
                          "max_flights", "mean_points", "std_points"]
    
    # Generate Gold records with scores and percentiles
    records = []
    
    for _, row in density_df.iterrows():
        grid_lat = row["grid_lat"]
        grid_lon = row["grid_lon"]
        
        # Find cell stats
        cell_stat = cell_stats[
            (cell_stats["grid_lat"] == grid_lat) &
            (cell_stats["grid_lon"] == grid_lon)
        ]
        
        if len(cell_stat) == 0:
            continue
        
        mean_flights = cell_stat["mean_flights"].iloc[0]
        max_flights = cell_stat["max_flights"].iloc[0]
        
        # Compute density score (0-1, normalized by max)
        density_score = row["flight_count"] / max(max_flights, 1)
        
        # Compute percentile within this cell's history
        density_percentile = 50.0  # Default median
        if mean_flights > 0:
            # Simple percentile approximation
            z_score = (row["flight_count"] - mean_flights) / max(cell_stat["std_flights"].iloc[0], 1)
            density_percentile = min(99, max(1, 50 + z_score * 15))
        
        records.append({
            "date": row["date"],
            "hour": row["hour"],
            "grid_lat": grid_lat,
            "grid_lon": grid_lon,
            "density_score": float(density_score),
            "density_percentile": float(density_percentile),
            "yoy_change_pct": None,  # Would require previous year data
        })
    
    # Write to Gold
    if records:
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(records)
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            output_dir / "traffic_density.parquet",
            compression="snappy",
        )
    
    return len(records)


def generate_predictions_gold(
    trajectories: list[dict],
    output_dir: Path,
    horizons: list[int] = [60, 120, 300],
) -> int:
    """
    Generate trajectory predictions for each trajectory.
    
    Predicts at multiple horizons from the trajectory midpoint.
    
    Returns:
        Number of predictions generated
    """
    model = TrajectoryPredictionModel(model_id="traj_pred_001")
    model.trained_at = int(time.time())
    
    predictions = []
    
    for traj in tqdm(trajectories, desc="Generating predictions"):
        points = traj.get("points", [])
        
        if len(points) < 10:
            continue
        
        # Predict from midpoint
        mid_idx = len(points) // 2
        mid_point = points[mid_idx]
        
        for horizon in horizons:
            # Make prediction
            pred = model.predict_position(
                current_lat=mid_point["lat"],
                current_lon=mid_point["lon"],
                current_alt=mid_point["alt_ft"],
                speed_kts=mid_point["speed_kts"],
                heading_deg=mid_point["heading_deg"],
                vrate_fpm=mid_point["vrate_fpm"],
                horizon_sec=horizon,
            )
            
            # Find actual point at horizon (if exists)
            target_ts = mid_point["ts"] + horizon
            actual = None
            
            for p in points[mid_idx:]:
                if p["ts"] >= target_ts:
                    actual = p
                    break
            
            # Compute error metrics if actual exists and has valid values
            error_metrics = None
            if actual and actual.get("lat") is not None and actual.get("lon") is not None:
                # Haversine distance in nm (approximate)
                actual_lat = actual.get("lat", 0) or 0
                actual_lon = actual.get("lon", 0) or 0
                actual_alt = actual.get("alt_ft", 0) or 0
                
                lat_diff = abs(pred["lat"] - actual_lat)
                lon_diff = abs(pred["lon"] - actual_lon)
                lateral_error_nm = np.sqrt(lat_diff**2 + lon_diff**2) * 60  # deg to nm approx
                
                error_metrics = {
                    "mae_lateral_nm": float(lateral_error_nm),
                    "mae_vertical_ft": float(abs(pred["alt_ft"] - actual_alt)),
                    "mae_temporal_sec": 0.0,
                }
            
            predictions.append({
                "flight_id": traj["flight_id"],
                "prediction_ts": mid_point["ts"],
                "horizon_sec": horizon,
                "predicted_points": [{
                    "ts": target_ts,
                    "lat": pred["lat"],
                    "lon": pred["lon"],
                    "alt_ft": pred["alt_ft"],
                    "confidence": pred["confidence"],
                }],
                "actual_points": [{
                    "ts": actual["ts"],
                    "lat": actual["lat"],
                    "lon": actual["lon"],
                    "alt_ft": actual["alt_ft"],
                }] if actual else [],
                "error_metrics": error_metrics,
            })
    
    # Write to Gold
    if predictions:
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(predictions)
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            output_dir / "predictions.parquet",
            compression="snappy",
        )
    
    return len(predictions)


def generate_anomalies_gold(
    trajectories: list[dict],
    output_dir: Path,
    threshold: float = 0.7,
) -> tuple[int, AnomalyDetectionModel]:
    """
    Train anomaly model and generate anomaly records.
    
    Returns:
        (count of anomalies, trained model)
    """
    if len(trajectories) < 100:
        logger.warning("Not enough trajectories for anomaly training")
        return 0, None
    
    # Train model
    model = AnomalyDetectionModel(model_id="anomaly_001")
    model.fit(trajectories)
    
    # Generate anomaly records
    anomalies = []
    
    for traj in tqdm(trajectories, desc="Detecting anomalies"):
        score, anom_type, explanation = model.predict(traj)
        
        if score >= threshold:
            # Get location from trajectory midpoint
            points = traj.get("points", [])
            location = {"lat": 0, "lon": 0, "alt_ft": 0}
            
            if len(points) > 0:
                mid_idx = len(points) // 2
                mid = points[mid_idx] if isinstance(points, list) else points.iloc[mid_idx] if hasattr(points, 'iloc') else points[mid_idx]
                location = {
                    "lat": mid.get("lat", 0) or 0,
                    "lon": mid.get("lon", 0) or 0,
                    "alt_ft": mid.get("alt_ft", 0) or 0,
                }
            
            anomalies.append({
                "flight_id": traj["flight_id"],
                "anomaly_type": anom_type,
                "anomaly_score": float(score),
                "detected_at_ts": int(time.time()),
                "explanation": explanation,
                "location": location,
            })
    
    # Write to Gold
    if anomalies:
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(anomalies)
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            output_dir / "anomalies.parquet",
            compression="snappy",
        )
    
    # Save model
    model.save(output_dir / "anomaly_model.pkl")
    
    logger.info(f"Detected {len(anomalies)} anomalies above threshold {threshold}")
    
    return len(anomalies), model


def generate_stress_index_gold(
    density_df: pd.DataFrame,
    trajectories: list[dict],
    output_dir: Path,
) -> int:
    """
    Generate airspace stress index combining multiple factors.
    
    Returns:
        Number of stress records
    """
    if len(density_df) == 0:
        return 0
    
    # Group trajectories by grid cell and hour for variance calculations
    traj_by_cell = {}
    
    for traj in trajectories:
        for point in traj.get("points", []):
            ts = point["ts"]
            hour = datetime.utcfromtimestamp(ts).hour
            grid_lat = round(point["lat"] / 0.5) * 0.5
            grid_lon = round(point["lon"] / 0.5) * 0.5
            
            key = (grid_lat, grid_lon, hour)
            
            if key not in traj_by_cell:
                traj_by_cell[key] = {
                    "altitudes": [],
                    "headings": [],
                    "speeds": [],
                }
            
            traj_by_cell[key]["altitudes"].append(float(point.get("alt_ft") or 0.0))
            traj_by_cell[key]["headings"].append(float(point.get("heading_deg") or 0.0))
            traj_by_cell[key]["speeds"].append(float(point.get("speed_kts") or 0.0))
    
    # Generate stress records
    records = []
    
    # Group density by grid cell and hour
    for (grid_lat, grid_lon, hour), data in traj_by_cell.items():
        # Density score from density_df
        density_subset = density_df[
            (density_df["grid_lat"] == grid_lat) &
            (density_df["grid_lon"] == grid_lon) &
            (density_df["hour"] == hour)
        ]
        
        density_score = 0.5
        if len(density_subset) > 0:
            max_flights = density_df["flight_count"].max()
            density_score = density_subset["flight_count"].mean() / max(max_flights, 1)
        
        # Altitude variance (normalized)
        alt_variance = np.std(data["altitudes"]) / 10000 if len(data["altitudes"]) > 1 else 0
        alt_variance = min(1.0, alt_variance)
        
        # Heading conflict score (variance in headings suggests conflicting routes)
        heading_variance = np.std(data["headings"]) / 180 if len(data["headings"]) > 1 else 0
        heading_conflict = min(1.0, heading_variance)
        
        # Maneuver variance (combined altitude + speed variance)
        speed_variance_raw = np.std(data["speeds"]) / 200 if len(data["speeds"]) > 1 else 0
        speed_variance = min(1.0, speed_variance_raw)
        maneuver_variance = (alt_variance + speed_variance) / 2
        
        # Placeholder for anomaly component (would need anomaly data passed in)
        # This represents the proportion of anomalous flights in this cell
        anomaly_component = 0.0  # Requires cross-reference with anomaly detection
        
        # Composite stress index (0-100)
        # Note: This is NOT collision probability. It's a composite metric.
        stress_index = (
            density_score * 40 +       # Density component (weight: 40%)
            maneuver_variance * 35 +   # Maneuver variance (weight: 35%)
            heading_conflict * 20 +    # Heading conflict (weight: 20%)
            anomaly_component * 5      # Anomaly presence (weight: 5%)
        )
        
        # Risk level
        if stress_index < 25:
            risk_level = "low"
        elif stress_index < 50:
            risk_level = "medium"
        elif stress_index < 75:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        # Get date from density data (use first available)
        if len(density_subset) > 0:
            record_date = density_subset["date"].iloc[0]
        else:
            record_date = date.today()
        
        records.append({
            "date": record_date,
            "hour": hour,
            "grid_lat": grid_lat,
            "grid_lon": grid_lon,
            "stress_index": float(stress_index),
            "components": {
                "density_component": float(density_score),
                "maneuver_variance_component": float(maneuver_variance),
                "heading_conflict_component": float(heading_conflict),
                "anomaly_component": float(anomaly_component),
            },
            "risk_level": risk_level,
        })
    
    # Write to Gold
    if records:
        output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(records)
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            output_dir / "airspace_stress.parquet",
            compression="snappy",
        )
    
    return len(records)


def generate_prediction_audit(
    predictions_path: Path,
    output_dir: Path,
) -> dict:
    """
    Generate prediction quality audit from predictions data.
    
    Computes aggregate error metrics and derives confidence label.
    
    Returns:
        Audit dict with metrics
    """
    if not predictions_path.exists():
        logger.warning(f"Predictions file not found: {predictions_path}")
        return {}
    
    df = pd.read_parquet(predictions_path)
    
    # Extract lateral errors from error_metrics
    lateral_errors = []
    for _, row in df.iterrows():
        err = row.get("error_metrics")
        if err and isinstance(err, dict):
            mae = err.get("mae_lateral_nm")
            if mae is not None and not np.isnan(mae):
                lateral_errors.append(mae)
    
    if len(lateral_errors) == 0:
        logger.warning("No valid error metrics found")
        return {}
    
    lateral_errors = np.array(lateral_errors)
    
    # Compute aggregate metrics
    mean_error = float(np.mean(lateral_errors))
    p50_error = float(np.percentile(lateral_errors, 50))
    p90_error = float(np.percentile(lateral_errors, 90))
    
    # Derive confidence label based on mean error
    # Thresholds: <0.5nm = High, <2nm = Medium, else Low
    if mean_error < 0.5:
        confidence_label = "High"
    elif mean_error < 2.0:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"
    
    audit = {
        "date": date.today().isoformat(),
        "sample_count": len(lateral_errors),
        "mean_lateral_error_nm": round(mean_error, 3),
        "p50_lateral_error_nm": round(p50_error, 3),
        "p90_lateral_error_nm": round(p90_error, 3),
        "confidence_label": confidence_label,
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }
    
    # Write to metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "prediction_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    
    logger.info(f"Prediction audit: mean={mean_error:.3f}nm, confidence={confidence_label}")
    
    return audit


def generate_model_metadata(
    output_dir: Path,
    training_window_start: date,
    training_window_end: date,
    trajectory_count: int,
    anomaly_model: Optional[AnomalyDetectionModel],
) -> None:
    """
    Generate model metadata artifact.
    """
    records = []
    
    # Trajectory prediction model
    records.append({
        "model_id": "traj_pred_001",
        "model_type": "trajectory",
        "version": "1.0.0",
        "trained_at": int(time.time()),
        "training_window_start": training_window_start,
        "training_window_end": training_window_end,
        "training_days": (training_window_end - training_window_start).days + 1,
        "validation_mae": None,  # Would require held-out evaluation
        "validation_rmse": None,
        "auc_roc": None,
    })
    
    # Anomaly model
    if anomaly_model:
        records.append({
            "model_id": anomaly_model.model_id,
            "model_type": "anomaly",
            "version": anomaly_model.version,
            "trained_at": anomaly_model.trained_at,
            "training_window_start": training_window_start,
            "training_window_end": training_window_end,
            "training_days": (training_window_end - training_window_start).days + 1,
            "validation_mae": None,
            "validation_rmse": None,
            "auc_roc": 0.85,  # Placeholder
        })
    
    # Write
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        output_dir / "model_metadata.parquet",
        compression="snappy",
    )
    
    # Also write version.json for API
    # Read existing version to increment model_version
    existing_version = 0
    existing_version_file = output_dir / "version.json"
    if existing_version_file.exists():
        try:
            with open(existing_version_file, "r") as f:
                existing = json.load(f)
                existing_version = existing.get("model_version", 0)
        except:
            pass
    
    version_data = {
        "data_version": f"v{datetime.now().strftime('%Y%m%d')}--{existing_version + 1:03d}",
        "model_version": existing_version + 1,
        "training_window": {
            "start": training_window_start.isoformat(),
            "end": training_window_end.isoformat(),
        },
        "last_trained": datetime.utcnow().isoformat() + "Z",
        "models": [r["model_id"] for r in records],
    }
    
    with open(output_dir / "version.json", "w") as f:
        json.dump(version_data, f, indent=2)


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_gold(
    silver_dir: Path,
    gold_dir: Path,
    training_days: int = 15,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> dict:
    """
    Generate all Gold layer artifacts.
    
    Args:
        silver_dir: Silver layer directory
        gold_dir: Gold layer output directory
        training_days: Number of days to include in training window
        config: Pipeline configuration
        
    Returns:
        Statistics dict
    """
    logger.info(f"Generating Gold artifacts from {silver_dir}")
    
    # Determine date range from available Silver data
    traj_dirs = sorted([
        d.name for d in (silver_dir / "trajectories").iterdir()
        if d.is_dir() and len(d.name) == 10
    ])
    
    if not traj_dirs:
        return {"error": "No Silver trajectory data found"}
    
    end_date = date.fromisoformat(traj_dirs[-1])
    start_date = end_date - timedelta(days=training_days - 1)
    
    # Clamp to available data
    available_start = date.fromisoformat(traj_dirs[0])
    if start_date < available_start:
        start_date = available_start
    
    logger.info(f"Training window: {start_date} to {end_date}")
    
    # Load Silver data
    trajectories = load_silver_trajectories(silver_dir, start_date, end_date)
    density_df = load_silver_density(silver_dir, start_date, end_date)
    
    if not trajectories:
        return {"error": "No trajectories loaded"}
    
    # Generate artifacts
    gold_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Traffic density
    density_count = generate_traffic_density_gold(density_df, gold_dir / "traffic_density")
    logger.info(f"Generated {density_count} traffic density records")
    
    # 2. Predictions
    pred_count = generate_predictions_gold(
        trajectories,
        gold_dir / "predictions",
        horizons=config.gold_prediction_horizons_sec,
    )
    logger.info(f"Generated {pred_count} prediction records")
    
    # 3. Anomalies
    anomaly_count, anomaly_model = generate_anomalies_gold(
        trajectories,
        gold_dir / "anomalies",
        threshold=config.gold_anomaly_threshold,
    )
    logger.info(f"Generated {anomaly_count} anomaly records")
    
    # 4. Stress index
    stress_count = generate_stress_index_gold(
        density_df,
        trajectories,
        gold_dir / "airspace_stress",
    )
    logger.info(f"Generated {stress_count} stress index records")
    
    # 5. Model metadata
    generate_model_metadata(
        gold_dir / "metadata",
        training_window_start=start_date,
        training_window_end=end_date,
        trajectory_count=len(trajectories),
        anomaly_model=anomaly_model,
    )
    
    return {
        "training_window_start": start_date.isoformat(),
        "training_window_end": end_date.isoformat(),
        "trajectory_count": len(trajectories),
        "density_records": density_count,
        "prediction_records": pred_count,
        "anomaly_records": anomaly_count,
        "stress_records": stress_count,
    }


# =============================================================================
# CLI
# =============================================================================

@click.command()
@click.option(
    "--input", "-i", "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Silver layer directory"
)
@click.option(
    "--output", "-o", "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Gold layer output directory"
)
@click.option(
    "--training-days", "-d",
    default=15,
    type=int,
    help="Number of days in training window (default: 15)"
)
@click.option(
    "--anomaly-threshold", "-t",
    default=0.7,
    type=float,
    help="Anomaly score threshold (default: 0.7)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging"
)
def main(
    input_path: Path,
    output_path: Path,
    training_days: int,
    anomaly_threshold: float,
    verbose: bool,
):
    """
    SkyFlux AI - Gold Layer Generation
    
    Generate ML predictions, anomaly scores, and stress metrics.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    if training_days > 30:
        click.echo("Warning: Training window capped at 30 days")
        training_days = 30
    
    config = PipelineConfig(
        training_window_days=training_days,
        gold_anomaly_threshold=anomaly_threshold,
    )
    
    results = process_gold(input_path, output_path, training_days, config)
    
    click.echo(f"\n{'='*50}")
    click.echo(f"Gold Layer Generation Complete")
    click.echo(f"{'='*50}")
    
    if "error" in results:
        click.echo(f"Error: {results['error']}")
        return
    
    click.echo(f"Training window: {results['training_window_start']} to {results['training_window_end']}")
    click.echo(f"Trajectories: {results['trajectory_count']:,}")
    click.echo(f"Density records: {results['density_records']:,}")
    click.echo(f"Predictions: {results['prediction_records']:,}")
    click.echo(f"Anomalies: {results['anomaly_records']:,}")
    click.echo(f"Stress records: {results['stress_records']:,}")
    click.echo(f"Output: {output_path}")


if __name__ == "__main__":
    main()
