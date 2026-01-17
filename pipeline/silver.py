"""
SkyFlux AI - Silver Layer

Cleaned and normalized trajectories with derived features.
Implements trajectory reconstruction, downsampling, and feature extraction.

Usage:
    python -m pipeline.silver --input ./data/bronze --output ./data/silver
"""

import logging
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Optional

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from pipeline.config import PipelineConfig, DEFAULT_CONFIG
    from pipeline.schemas import SILVER_TRAJECTORY_SCHEMA, SILVER_GRID_DENSITY_SCHEMA
except ImportError:
    from config import PipelineConfig, DEFAULT_CONFIG
    from schemas import SILVER_TRAJECTORY_SCHEMA, SILVER_GRID_DENSITY_SCHEMA

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Earth radius in nautical miles
EARTH_RADIUS_NM = 3440.065

# Valid coordinate ranges
MIN_LAT, MAX_LAT = -90.0, 90.0
MIN_LON, MAX_LON = -180.0, 180.0
MIN_ALT, MAX_ALT = -1000.0, 60000.0  # Feet
MIN_SPEED, MAX_SPEED = 0.0, 700.0  # Knots
MIN_VRATE, MAX_VRATE = -10000.0, 10000.0  # ft/min


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points in nautical miles.
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (
        math.sin(delta_lat / 2) ** 2 +
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return EARTH_RADIUS_NM * c


def heading_difference(h1: float, h2: float) -> float:
    """
    Calculate absolute heading difference (0-180 degrees).
    """
    diff = abs(h1 - h2) % 360
    return min(diff, 360 - diff)


def is_valid_coordinate(lat: Optional[float], lon: Optional[float]) -> bool:
    """Check if coordinates are valid."""
    if lat is None or lon is None:
        return False
    if not (MIN_LAT <= lat <= MAX_LAT):
        return False
    if not (MIN_LON <= lon <= MAX_LON):
        return False
    return True


def is_valid_altitude(alt: Optional[float]) -> bool:
    """Check if altitude is valid."""
    if alt is None:
        return False
    return MIN_ALT <= alt <= MAX_ALT


def is_valid_speed(speed: Optional[float]) -> bool:
    """Check if speed is valid."""
    if speed is None:
        return True  # Allow missing speed
    return MIN_SPEED <= speed <= MAX_SPEED


# =============================================================================
# TRAJECTORY PROCESSING
# =============================================================================

def load_bronze_date(bronze_dir: Path, date_str: str) -> pd.DataFrame:
    """
    Load all Bronze Parquet files for a specific date.
    
    Args:
        bronze_dir: Bronze layer directory
        date_str: Date string in YYYY-MM-DD format
        
    Returns:
        Combined DataFrame with all records for the date
    """
    date_dir = bronze_dir / date_str
    
    if not date_dir.exists():
        logger.warning(f"No Bronze data for {date_str}")
        return pd.DataFrame()
    
    parquet_files = sorted(date_dir.glob("*.parquet"))
    
    if not parquet_files:
        logger.warning(f"No Parquet files in {date_dir}")
        return pd.DataFrame()
    
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined):,} Bronze records for {date_str}")
    
    return combined


def clean_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and filter Bronze records.
    
    Removes:
    - Records with invalid/null coordinates
    - Records with invalid altitude/speed values
    - Duplicate (icao24, timestamp) pairs
    
    Args:
        df: Raw Bronze DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    initial_count = len(df)
    
    # Drop null coordinates
    df = df.dropna(subset=["latitude", "longitude", "timestamp"])
    
    # Filter valid coordinate ranges
    df = df[
        (df["latitude"] >= MIN_LAT) & (df["latitude"] <= MAX_LAT) &
        (df["longitude"] >= MIN_LON) & (df["longitude"] <= MAX_LON)
    ]
    
    # Filter valid altitude (allow null)
    df = df[
        df["altitude"].isna() |
        ((df["altitude"] >= MIN_ALT) & (df["altitude"] <= MAX_ALT))
    ]
    
    # Filter valid speed (allow null)
    df = df[
        df["velocity"].isna() |
        ((df["velocity"] >= MIN_SPEED) & (df["velocity"] <= MAX_SPEED))
    ]
    
    # Deduplicate by (icao24, timestamp), keep first
    df = df.drop_duplicates(subset=["icao24", "timestamp"], keep="first")
    
    # Sort by icao24 and timestamp
    df = df.sort_values(["icao24", "timestamp"]).reset_index(drop=True)
    
    logger.info(f"Cleaned: {initial_count:,} → {len(df):,} records ({100*len(df)/initial_count:.1f}% retained)")
    
    return df


def segment_trajectories(
    df: pd.DataFrame,
    max_gap_sec: int = 300,
) -> list[pd.DataFrame]:
    """
    Segment a single aircraft's data into separate trajectories.
    
    A new trajectory starts when there's a gap > max_gap_sec seconds.
    
    Args:
        df: DataFrame with records for a single aircraft (sorted by timestamp)
        max_gap_sec: Maximum gap before starting new trajectory
        
    Returns:
        List of DataFrame segments, each representing one trajectory
    """
    if len(df) < 2:
        return [df] if len(df) > 0 else []
    
    # Calculate time gaps
    timestamps = df["timestamp"].values
    gaps = np.diff(timestamps)
    
    # Find segment boundaries
    break_indices = np.where(gaps > max_gap_sec)[0] + 1
    
    # Split into segments
    segments = []
    prev_idx = 0
    
    for break_idx in break_indices:
        segment = df.iloc[prev_idx:break_idx].copy()
        if len(segment) > 1:  # Only keep trajectories with > 1 point
            segments.append(segment)
        prev_idx = break_idx
    
    # Last segment
    segment = df.iloc[prev_idx:].copy()
    if len(segment) > 1:
        segments.append(segment)
    
    return segments


def downsample_trajectory(
    df: pd.DataFrame,
    interval_sec: int = 10,
) -> pd.DataFrame:
    """
    Downsample trajectory to regular time intervals.
    
    Uses linear interpolation for position, altitude.
    
    Args:
        df: Trajectory DataFrame (sorted by timestamp)
        interval_sec: Target interval in seconds
        
    Returns:
        Downsampled DataFrame
    """
    if len(df) < 2:
        return df
    
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    
    # Generate regular timestamps
    regular_ts = np.arange(min_ts, max_ts + 1, interval_sec)
    
    if len(regular_ts) < 2:
        return df
    
    # Set timestamp as index for interpolation
    df_indexed = df.set_index("timestamp").sort_index()
    
    # Interpolate numeric columns
    numeric_cols = ["latitude", "longitude", "altitude", "velocity", "heading", "vertical_rate"]
    
    # Create output frame with regular timestamps
    result_df = pd.DataFrame({"timestamp": regular_ts})
    
    for col in numeric_cols:
        if col in df_indexed.columns:
            # Linear interpolation
            values = np.interp(
                regular_ts,
                df_indexed.index.values,
                df_indexed[col].fillna(method="ffill").fillna(method="bfill").values,
                left=np.nan,
                right=np.nan,
            )
            result_df[col] = values
    
    # Copy non-numeric columns from nearest original point
    result_df["icao24"] = df["icao24"].iloc[0]
    result_df["callsign"] = df["callsign"].iloc[0]
    result_df["on_ground"] = df["on_ground"].iloc[0]
    
    return result_df


def compute_trajectory_features(
    df: pd.DataFrame,
    flight_id: str,
    target_date: date,
) -> dict:
    """
    Compute aggregated features for a trajectory.
    
    Args:
        df: Trajectory DataFrame (downsampled)
        flight_id: Unique flight identifier
        target_date: Date for this trajectory
        
    Returns:
        Dictionary with trajectory record matching Silver schema
    """
    # Build points list
    points = []
    for _, row in df.iterrows():
        points.append({
            "ts": int(row["timestamp"]),
            "lat": row["latitude"],
            "lon": row["longitude"],
            "alt_ft": row.get("altitude", 0) or 0,
            "speed_kts": row.get("velocity", 0) or 0,
            "heading_deg": row.get("heading", 0) or 0,
            "vrate_fpm": row.get("vertical_rate", 0) or 0,
        })
    
    # Duration
    duration_sec = int(df["timestamp"].max() - df["timestamp"].min())
    
    # Distance (sum of segments)
    distance_nm = 0.0
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        distance_nm += haversine_nm(
            prev["latitude"], prev["longitude"],
            curr["latitude"], curr["longitude"]
        )
    
    # Altitude stats
    altitudes = df["altitude"].dropna()
    max_alt_ft = float(altitudes.max()) if len(altitudes) > 0 else 0.0
    
    # Speed stats
    speeds = df["velocity"].dropna()
    avg_speed_kts = float(speeds.mean()) if len(speeds) > 0 else 0.0
    
    # Heading change stats
    headings = df["heading"].dropna()
    heading_change_total = 0.0
    for i in range(1, len(headings)):
        heading_change_total += heading_difference(
            headings.iloc[i - 1], headings.iloc[i]
        )
    
    # Climb/descent count (vertical rate transitions)
    vrates = df["vertical_rate"].fillna(0).values
    climb_count = 0
    descent_count = 0
    
    for i in range(1, len(vrates)):
        if vrates[i - 1] <= 0 and vrates[i] > 500:  # Started climbing
            climb_count += 1
        elif vrates[i - 1] >= 0 and vrates[i] < -500:  # Started descending
            descent_count += 1
    
    return {
        "flight_id": flight_id,
        "icao24": df["icao24"].iloc[0],
        "callsign": str(df["callsign"].iloc[0]) if pd.notna(df["callsign"].iloc[0]) else "",
        "date": target_date,
        "points": points,
        "duration_sec": duration_sec,
        "distance_nm": distance_nm,
        "max_alt_ft": max_alt_ft,
        "avg_speed_kts": avg_speed_kts,
        "heading_change_total": heading_change_total,
        "climb_count": climb_count,
        "descent_count": descent_count,
    }


def compute_grid_density(
    trajectories: list[dict],
    target_date: date,
    grid_resolution: float = 0.5,
) -> list[dict]:
    """
    Compute hourly grid density from trajectories.
    
    Args:
        trajectories: List of trajectory dicts
        target_date: Date for density calculation
        grid_resolution: Grid cell size in degrees
        
    Returns:
        List of grid density records
    """
    # Accumulator: (hour, grid_lat, grid_lon) -> {flights, points, alt_sum, speed_sum}
    grid_data = defaultdict(lambda: {
        "flights": set(),
        "point_count": 0,
        "alt_sum": 0.0,
        "speed_sum": 0.0,
    })
    
    for traj in trajectories:
        flight_id = traj["flight_id"]
        
        for point in traj["points"]:
            ts = point["ts"]
            hour = datetime.utcfromtimestamp(ts).hour
            
            # Grid cell center
            grid_lat = round(point["lat"] / grid_resolution) * grid_resolution
            grid_lon = round(point["lon"] / grid_resolution) * grid_resolution
            
            key = (hour, grid_lat, grid_lon)
            grid_data[key]["flights"].add(flight_id)
            grid_data[key]["point_count"] += 1
            grid_data[key]["alt_sum"] += point["alt_ft"]
            grid_data[key]["speed_sum"] += point["speed_kts"]
    
    # Convert to records
    records = []
    for (hour, grid_lat, grid_lon), data in grid_data.items():
        point_count = data["point_count"]
        records.append({
            "date": target_date,
            "hour": hour,
            "grid_lat": grid_lat,
            "grid_lon": grid_lon,
            "flight_count": len(data["flights"]),
            "point_count": point_count,
            "avg_altitude_ft": data["alt_sum"] / point_count if point_count > 0 else 0,
            "avg_speed_kts": data["speed_sum"] / point_count if point_count > 0 else 0,
        })
    
    return records


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_date(
    bronze_dir: Path,
    silver_dir: Path,
    date_str: str,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> dict:
    """
    Process a single date's data from Bronze to Silver.
    
    Args:
        bronze_dir: Bronze layer directory
        silver_dir: Silver layer output directory
        date_str: Date string in YYYY-MM-DD format
        config: Pipeline configuration
        
    Returns:
        Statistics dict
    """
    logger.info(f"Processing Silver layer for {date_str}")
    
    # Load Bronze data
    bronze_df = load_bronze_date(bronze_dir, date_str)
    
    if len(bronze_df) == 0:
        return {"date": date_str, "error": "No Bronze data"}
    
    # Clean records
    clean_df = clean_records(bronze_df)
    
    if len(clean_df) == 0:
        return {"date": date_str, "error": "No valid records after cleaning"}
    
    # Parse target date
    target_date = date.fromisoformat(date_str)
    
    # Group by aircraft and process trajectories
    trajectories = []
    traj_seq = 0
    
    aircraft_groups = clean_df.groupby("icao24")
    
    for icao24, aircraft_df in tqdm(aircraft_groups, desc="Processing aircraft"):
        # Segment into separate trajectories
        segments = segment_trajectories(
            aircraft_df.sort_values("timestamp"),
            max_gap_sec=config.silver_max_gap_sec,
        )
        
        for segment in segments:
            # Downsample
            downsampled = downsample_trajectory(
                segment,
                interval_sec=config.silver_downsample_interval_sec,
            )
            
            if len(downsampled) < 2:
                continue
            
            # Generate flight ID
            flight_id = f"{icao24}_{date_str}_{traj_seq:04d}"
            traj_seq += 1
            
            # Compute features
            traj_record = compute_trajectory_features(
                downsampled,
                flight_id=flight_id,
                target_date=target_date,
            )
            trajectories.append(traj_record)
    
    logger.info(f"Generated {len(trajectories):,} trajectories")
    
    # Compute grid density
    density_records = compute_grid_density(
        trajectories,
        target_date=target_date,
        grid_resolution=config.silver_grid_resolution_deg,
    )
    
    logger.info(f"Generated {len(density_records):,} grid density records")
    
    # Write trajectories
    traj_output_dir = silver_dir / "trajectories" / date_str
    traj_output_dir.mkdir(parents=True, exist_ok=True)
    
    traj_df = pd.DataFrame(trajectories)
    traj_table = pa.Table.from_pandas(traj_df, preserve_index=False)
    pq.write_table(
        traj_table,
        traj_output_dir / "trajectories.parquet",
        compression="snappy",
    )
    
    # Write grid density
    density_output_dir = silver_dir / "grid_density" / date_str
    density_output_dir.mkdir(parents=True, exist_ok=True)
    
    density_df = pd.DataFrame(density_records)
    density_table = pa.Table.from_pandas(density_df, preserve_index=False)
    pq.write_table(
        density_table,
        density_output_dir / "density.parquet",
        compression="snappy",
    )
    
    return {
        "date": date_str,
        "trajectory_count": len(trajectories),
        "density_cells": len(density_records),
        "bronze_records": len(bronze_df),
        "clean_records": len(clean_df),
    }


def process_all_dates(
    bronze_dir: Path,
    silver_dir: Path,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> list[dict]:
    """
    Process all available Bronze dates to Silver.
    
    Args:
        bronze_dir: Bronze layer directory
        silver_dir: Silver layer output directory
        config: Pipeline configuration
        
    Returns:
        List of statistics dicts
    """
    # Find all date directories in Bronze
    date_dirs = sorted([
        d.name for d in bronze_dir.iterdir()
        if d.is_dir() and len(d.name) == 10  # YYYY-MM-DD format
    ])
    
    if not date_dirs:
        logger.warning(f"No date directories found in {bronze_dir}")
        return []
    
    logger.info(f"Found {len(date_dirs)} dates to process")
    
    results = []
    for date_str in date_dirs:
        try:
            stats = process_date(bronze_dir, silver_dir, date_str, config)
            results.append(stats)
        except Exception as e:
            logger.error(f"Failed to process {date_str}: {e}")
            results.append({"date": date_str, "error": str(e)})
    
    return results


# =============================================================================
# CLI
# =============================================================================

@click.command()
@click.option(
    "--input", "-i", "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Bronze layer directory"
)
@click.option(
    "--output", "-o", "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Silver layer output directory"
)
@click.option(
    "--date", "-d", "target_date",
    default=None,
    help="Specific date to process (YYYY-MM-DD), or all if not specified"
)
@click.option(
    "--downsample-interval", "-s",
    default=10,
    type=int,
    help="Downsample interval in seconds"
)
@click.option(
    "--max-gap", "-g",
    default=300,
    type=int,
    help="Maximum gap before trajectory split (seconds)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging"
)
def main(
    input_path: Path,
    output_path: Path,
    target_date: Optional[str],
    downsample_interval: int,
    max_gap: int,
    verbose: bool,
):
    """
    SkyFlux AI - Silver Layer Processing
    
    Clean, normalize, and extract features from Bronze layer data.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    config = PipelineConfig(
        silver_downsample_interval_sec=downsample_interval,
        silver_max_gap_sec=max_gap,
    )
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    if target_date:
        results = [process_date(input_path, output_path, target_date, config)]
    else:
        results = process_all_dates(input_path, output_path, config)
    
    # Summary
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    total_trajectories = sum(r.get("trajectory_count", 0) for r in successful)
    
    click.echo(f"\n{'='*50}")
    click.echo(f"Silver Processing Complete")
    click.echo(f"{'='*50}")
    click.echo(f"Dates processed: {len(successful)}")
    click.echo(f"Dates failed: {len(failed)}")
    click.echo(f"Total trajectories: {total_trajectories:,}")
    click.echo(f"Output: {output_path}")
    
    if failed:
        click.echo(f"\nFailed dates:")
        for r in failed:
            click.echo(f"  - {r['date']}: {r['error']}")


if __name__ == "__main__":
    main()
