"""
SkyFlux AI - Data Schemas

Pydantic models and schema definitions for Medallion Architecture layers.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional
import pyarrow as pa


# =============================================================================
# BRONZE LAYER SCHEMA
# =============================================================================

BRONZE_SCHEMA = pa.schema([
    ("icao24", pa.string()),
    ("callsign", pa.string()),
    ("timestamp", pa.int64()),
    ("latitude", pa.float64()),
    ("longitude", pa.float64()),
    ("altitude", pa.float64()),
    ("velocity", pa.float64()),
    ("heading", pa.float64()),
    ("vertical_rate", pa.float64()),
    ("on_ground", pa.bool_()),
    ("squawk", pa.string()),
    ("_source_file", pa.string()),
    ("_ingest_ts", pa.int64()),
])


# =============================================================================
# SILVER LAYER SCHEMAS
# =============================================================================

# Schema for individual trajectory points
TRAJECTORY_POINT_TYPE = pa.struct([
    ("ts", pa.int64()),
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("alt_ft", pa.float64()),
    ("speed_kts", pa.float64()),
    ("heading_deg", pa.float64()),
    ("vrate_fpm", pa.float64()),
])

SILVER_TRAJECTORY_SCHEMA = pa.schema([
    ("flight_id", pa.string()),
    ("icao24", pa.string()),
    ("callsign", pa.string()),
    ("date", pa.date32()),
    ("points", pa.list_(TRAJECTORY_POINT_TYPE)),
    ("duration_sec", pa.int64()),
    ("distance_nm", pa.float64()),
    ("max_alt_ft", pa.float64()),
    ("avg_speed_kts", pa.float64()),
    ("heading_change_total", pa.float64()),
    ("climb_count", pa.int32()),
    ("descent_count", pa.int32()),
])

SILVER_GRID_DENSITY_SCHEMA = pa.schema([
    ("date", pa.date32()),
    ("hour", pa.int8()),
    ("grid_lat", pa.float64()),
    ("grid_lon", pa.float64()),
    ("flight_count", pa.int64()),
    ("point_count", pa.int64()),
    ("avg_altitude_ft", pa.float64()),
    ("avg_speed_kts", pa.float64()),
])


# =============================================================================
# GOLD LAYER SCHEMAS
# =============================================================================

GOLD_TRAFFIC_DENSITY_SCHEMA = pa.schema([
    ("date", pa.date32()),
    ("hour", pa.int8()),
    ("grid_lat", pa.float64()),
    ("grid_lon", pa.float64()),
    ("density_score", pa.float64()),
    ("density_percentile", pa.float64()),
    ("yoy_change_pct", pa.float64()),
])

PREDICTED_POINT_TYPE = pa.struct([
    ("ts", pa.int64()),
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("alt_ft", pa.float64()),
    ("confidence", pa.float64()),
])

ACTUAL_POINT_TYPE = pa.struct([
    ("ts", pa.int64()),
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("alt_ft", pa.float64()),
])

ERROR_METRICS_TYPE = pa.struct([
    ("mae_lateral_nm", pa.float64()),
    ("mae_vertical_ft", pa.float64()),
    ("mae_temporal_sec", pa.float64()),
])

GOLD_TRAJECTORY_PREDICTION_SCHEMA = pa.schema([
    ("flight_id", pa.string()),
    ("prediction_ts", pa.int64()),
    ("horizon_sec", pa.int32()),
    ("predicted_points", pa.list_(PREDICTED_POINT_TYPE)),
    ("actual_points", pa.list_(ACTUAL_POINT_TYPE)),
    ("error_metrics", ERROR_METRICS_TYPE),
])

EXPLANATION_TYPE = pa.struct([
    ("primary_factor", pa.string()),
    ("deviation_pct", pa.float64()),
    ("baseline_value", pa.float64()),
    ("observed_value", pa.float64()),
    ("context", pa.string()),
])

LOCATION_TYPE = pa.struct([
    ("lat", pa.float64()),
    ("lon", pa.float64()),
    ("alt_ft", pa.float64()),
])

GOLD_ANOMALY_SCHEMA = pa.schema([
    ("flight_id", pa.string()),
    ("anomaly_type", pa.string()),
    ("anomaly_score", pa.float64()),
    ("detected_at_ts", pa.int64()),
    ("explanation", EXPLANATION_TYPE),
    ("location", LOCATION_TYPE),
])

STRESS_COMPONENTS_TYPE = pa.struct([
    ("density_score", pa.float64()),
    ("altitude_variance", pa.float64()),
    ("heading_conflict_score", pa.float64()),
    ("speed_variance", pa.float64()),
])

GOLD_AIRSPACE_STRESS_SCHEMA = pa.schema([
    ("date", pa.date32()),
    ("hour", pa.int8()),
    ("grid_lat", pa.float64()),
    ("grid_lon", pa.float64()),
    ("stress_index", pa.float64()),
    ("components", STRESS_COMPONENTS_TYPE),
    ("risk_level", pa.string()),
])

GOLD_MODEL_METADATA_SCHEMA = pa.schema([
    ("model_id", pa.string()),
    ("model_type", pa.string()),
    ("version", pa.string()),
    ("trained_at", pa.int64()),
    ("training_window_start", pa.date32()),
    ("training_window_end", pa.date32()),
    ("training_days", pa.int32()),
    ("validation_mae", pa.float64()),
    ("validation_rmse", pa.float64()),
    ("auc_roc", pa.float64()),
])


# =============================================================================
# DATACLASSES FOR TYPE HINTS
# =============================================================================

@dataclass
class BronzeRecord:
    """Raw ADS-B record from Bronze layer."""
    icao24: str
    callsign: Optional[str]
    timestamp: int
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float]
    velocity: Optional[float]
    heading: Optional[float]
    vertical_rate: Optional[float]
    on_ground: Optional[bool]
    squawk: Optional[str]
    _source_file: str
    _ingest_ts: int


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""
    ts: int
    lat: float
    lon: float
    alt_ft: float
    speed_kts: float
    heading_deg: float
    vrate_fpm: float


@dataclass
class SilverTrajectory:
    """Cleaned trajectory from Silver layer."""
    flight_id: str
    icao24: str
    callsign: Optional[str]
    date: date
    points: list[TrajectoryPoint]
    duration_sec: int
    distance_nm: float
    max_alt_ft: float
    avg_speed_kts: float
    heading_change_total: float
    climb_count: int
    descent_count: int
