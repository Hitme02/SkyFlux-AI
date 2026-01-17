"""
SkyFlux AI - Bronze Layer

Raw ADS-B data ingestion with minimal transformation.
Handles archive extraction and conversion to immutable Parquet files.

Usage:
    python -m pipeline.bronze --input /path/to/archives --output ./data/bronze
"""

import gzip
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .config import PipelineConfig, DEFAULT_CONFIG
from .schemas import BRONZE_SCHEMA

logger = logging.getLogger(__name__)


# =============================================================================
# COLUMN MAPPINGS FOR DIFFERENT ADS-B DATA SOURCES
# =============================================================================

# OpenSky Network state vectors format
OPENSKY_COLUMN_MAP = {
    "icao24": "icao24",
    "callsign": "callsign", 
    "time": "timestamp",
    "lat": "latitude",
    "lon": "longitude",
    "baroaltitude": "altitude",
    "velocity": "velocity",
    "heading": "heading",
    "vertrate": "vertical_rate",
    "onground": "on_ground",
    "squawk": "squawk",
}

# Alternative column names that might appear in datasets
ALTERNATIVE_COLUMN_MAP = {
    "icao24": ["icao24", "icao", "hex", "aircraft_id"],
    "callsign": ["callsign", "call", "flight"],
    "timestamp": ["time", "timestamp", "ts", "epoch"],
    "latitude": ["lat", "latitude", "position_lat"],
    "longitude": ["lon", "longitude", "lng", "position_lon"],
    "altitude": ["baroaltitude", "altitude", "alt", "geoaltitude", "baro_altitude"],
    "velocity": ["velocity", "speed", "groundspeed", "gs"],
    "heading": ["heading", "track", "trk"],
    "vertical_rate": ["vertrate", "vertical_rate", "vs", "vert_rate", "vrate"],
    "on_ground": ["onground", "on_ground", "ground"],
    "squawk": ["squawk", "transponder"],
}


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Detect column mappings from DataFrame columns.
    
    Args:
        df: Input DataFrame with raw columns
        
    Returns:
        Mapping from standard column names to actual DataFrame column names
    """
    column_map = {}
    df_columns_lower = {c.lower(): c for c in df.columns}
    
    for standard_name, alternatives in ALTERNATIVE_COLUMN_MAP.items():
        for alt in alternatives:
            if alt.lower() in df_columns_lower:
                column_map[standard_name] = df_columns_lower[alt.lower()]
                break
    
    return column_map


def parse_csv_chunk(
    chunk: pd.DataFrame,
    source_file: str,
    ingest_ts: int,
    column_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Parse a chunk of raw CSV data into Bronze schema.
    
    Args:
        chunk: Raw DataFrame chunk
        source_file: Original source filename
        ingest_ts: Ingestion timestamp
        column_map: Optional pre-detected column mapping
        
    Returns:
        DataFrame conforming to Bronze schema
    """
    if column_map is None:
        column_map = detect_columns(chunk)
    
    # Create output DataFrame with Bronze schema columns
    bronze_df = pd.DataFrame()
    
    # Map columns, using None for missing
    for standard_col in ["icao24", "callsign", "timestamp", "latitude", "longitude",
                         "altitude", "velocity", "heading", "vertical_rate", 
                         "on_ground", "squawk"]:
        if standard_col in column_map:
            bronze_df[standard_col] = chunk[column_map[standard_col]]
        else:
            bronze_df[standard_col] = None
    
    # Add metadata columns
    bronze_df["_source_file"] = source_file
    bronze_df["_ingest_ts"] = ingest_ts
    
    # Type conversions
    bronze_df["icao24"] = bronze_df["icao24"].astype(str)
    bronze_df["callsign"] = bronze_df["callsign"].fillna("").astype(str)
    bronze_df["timestamp"] = pd.to_numeric(bronze_df["timestamp"], errors="coerce").astype("Int64")
    
    for col in ["latitude", "longitude", "altitude", "velocity", "heading", "vertical_rate"]:
        bronze_df[col] = pd.to_numeric(bronze_df[col], errors="coerce")
    
    if "on_ground" in bronze_df.columns:
        bronze_df["on_ground"] = bronze_df["on_ground"].map(
            lambda x: True if str(x).lower() in ("true", "1", "t") else 
                      (False if str(x).lower() in ("false", "0", "f") else None)
        )
    
    bronze_df["squawk"] = bronze_df["squawk"].fillna("").astype(str)
    
    return bronze_df


def read_archive(file_path: Path, chunk_size: int = 100_000) -> Iterator[pd.DataFrame]:
    """
    Read an ADS-B archive file in chunks.
    
    Supports:
    - .csv
    - .csv.gz
    - .parquet
    
    Args:
        file_path: Path to archive file
        chunk_size: Number of rows per chunk
        
    Yields:
        DataFrame chunks
    """
    suffix = "".join(file_path.suffixes).lower()
    
    if suffix in (".csv.gz", ".gz"):
        with gzip.open(file_path, "rt") as f:
            for chunk in pd.read_csv(f, chunksize=chunk_size, low_memory=False):
                yield chunk
    elif suffix == ".csv":
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
            yield chunk
    elif suffix == ".parquet":
        # Read Parquet in one go (already columnar)
        yield pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Extract date from archive filename.
    
    Common patterns:
    - 2026-01-15.csv.gz
    - states_2026-01-15.csv
    - adsb_20260115.csv.gz
    
    Returns:
        Date string in YYYY-MM-DD format or None
    """
    import re
    
    # Pattern 1: YYYY-MM-DD
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return match.group(1)
    
    # Pattern 2: YYYYMMDD
    match = re.search(r"(\d{8})", filename)
    if match:
        date_str = match.group(1)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    return None


def ingest_file(
    file_path: Path,
    output_dir: Path,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> dict:
    """
    Ingest a single ADS-B archive file into Bronze layer.
    
    Args:
        file_path: Path to input archive
        output_dir: Bronze output directory
        config: Pipeline configuration
        
    Returns:
        Statistics dict with row counts and timing
    """
    logger.info(f"Ingesting {file_path.name}")
    
    # Determine output date partition
    date_str = extract_date_from_filename(file_path.name)
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        logger.warning(f"Could not extract date from filename, using today: {date_str}")
    
    # Create output directory
    output_date_dir = output_dir / date_str
    output_date_dir.mkdir(parents=True, exist_ok=True)
    
    # Ingest timestamp
    ingest_ts = int(time.time())
    
    # Process in chunks
    total_rows = 0
    chunk_num = 0
    column_map = None  # Detect once, reuse
    
    for chunk in read_archive(file_path, config.bronze_chunk_size):
        if column_map is None:
            column_map = detect_columns(chunk)
            logger.info(f"Detected columns: {column_map}")
        
        bronze_df = parse_csv_chunk(
            chunk, 
            source_file=file_path.name,
            ingest_ts=ingest_ts,
            column_map=column_map,
        )
        
        # Write as Parquet
        output_file = output_date_dir / f"part-{chunk_num:04d}.parquet"
        table = pa.Table.from_pandas(bronze_df, schema=BRONZE_SCHEMA, preserve_index=False)
        pq.write_table(table, output_file, compression="snappy")
        
        total_rows += len(bronze_df)
        chunk_num += 1
    
    logger.info(f"Ingested {total_rows:,} rows into {chunk_num} files")
    
    return {
        "source_file": file_path.name,
        "date": date_str,
        "total_rows": total_rows,
        "num_files": chunk_num,
        "ingest_ts": ingest_ts,
    }


def ingest_directory(
    input_dir: Path,
    output_dir: Path,
    config: PipelineConfig = DEFAULT_CONFIG,
    pattern: str = "*.csv*",
) -> list[dict]:
    """
    Ingest all archive files from a directory.
    
    Args:
        input_dir: Directory containing archive files
        output_dir: Bronze output directory
        config: Pipeline configuration
        pattern: Glob pattern for files
        
    Returns:
        List of statistics dicts for each file
    """
    files = sorted(input_dir.glob(pattern))
    
    if not files:
        logger.warning(f"No files matching '{pattern}' in {input_dir}")
        return []
    
    logger.info(f"Found {len(files)} files to ingest")
    
    results = []
    for file_path in tqdm(files, desc="Ingesting files"):
        try:
            stats = ingest_file(file_path, output_dir, config)
            results.append(stats)
        except Exception as e:
            logger.error(f"Failed to ingest {file_path.name}: {e}")
            results.append({
                "source_file": file_path.name,
                "error": str(e),
            })
    
    return results


# =============================================================================
# CLI
# =============================================================================

@click.command()
@click.option(
    "--input", "-i", "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input file or directory containing ADS-B archives"
)
@click.option(
    "--output", "-o", "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for Bronze layer Parquet files"
)
@click.option(
    "--pattern", "-p",
    default="*.csv*",
    help="Glob pattern for input files (if input is directory)"
)
@click.option(
    "--chunk-size", "-c",
    default=100_000,
    type=int,
    help="Rows per chunk during ingestion"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging"
)
def main(
    input_path: Path,
    output_path: Path,
    pattern: str,
    chunk_size: int,
    verbose: bool,
):
    """
    SkyFlux AI - Bronze Layer Ingestion
    
    Ingest raw ADS-B archive files into Bronze layer Parquet format.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    config = PipelineConfig(bronze_chunk_size=chunk_size)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        results = [ingest_file(input_path, output_path, config)]
    else:
        results = ingest_directory(input_path, output_path, config, pattern)
    
    # Summary
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    total_rows = sum(r.get("total_rows", 0) for r in successful)
    
    click.echo(f"\n{'='*50}")
    click.echo(f"Bronze Ingestion Complete")
    click.echo(f"{'='*50}")
    click.echo(f"Files processed: {len(successful)}")
    click.echo(f"Files failed: {len(failed)}")
    click.echo(f"Total rows: {total_rows:,}")
    click.echo(f"Output: {output_path}")
    
    if failed:
        click.echo(f"\nFailed files:")
        for r in failed:
            click.echo(f"  - {r['source_file']}: {r['error']}")


if __name__ == "__main__":
    main()
