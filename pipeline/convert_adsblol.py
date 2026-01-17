"""
SkyFlux AI - ADSB.lol Data Converter

Converts adsblol globe_history trace files to Bronze layer format.
The adsblol format uses gzip-compressed JSON files with trace arrays.

Usage:
    python -m pipeline.convert_adsblol --input ./data/raw/extracted --output ./data/bronze
"""

import gzip
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Iterator, Optional

import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from pipeline.schemas import BRONZE_SCHEMA
except ImportError:
    from schemas import BRONZE_SCHEMA

logger = logging.getLogger(__name__)


# ADSB.lol trace array indices
# [0] timestamp (seconds since midnight or epoch offset)
# [1] latitude
# [2] longitude  
# [3] altitude (feet, barometric)
# [4] ground speed (knots)
# [5] track/heading (degrees)
# [6] on_ground flag
# [7] vertical rate (ft/min)
# [8] metadata object (nullable, contains callsign, squawk, etc.)
# [9] source type (e.g., "adsb_icao")
# [10] geometric altitude (nullable)
# [11+] additional fields vary


def parse_trace_file(file_path: Path, target_date: date) -> list[dict]:
    """
    Parse a single adsblol trace file.
    
    Args:
        file_path: Path to trace_full_*.json file
        target_date: Date for timestamp calculation
        
    Returns:
        List of Bronze-compatible records
    """
    records = []
    
    try:
        # Read and decompress
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Try gzip decompress
        try:
            data = gzip.decompress(data)
        except gzip.BadGzipFile:
            pass  # File might not be compressed
        
        trace_data = json.loads(data)
    except Exception as e:
        logger.warning(f"Failed to parse {file_path.name}: {e}")
        return []
    
    # Extract ICAO from filename or data
    icao24 = trace_data.get('icao', file_path.stem.replace('trace_full_', '').replace('~', ''))
    
    # Get trace points
    traces = trace_data.get('trace', [])
    
    if not traces:
        return []
    
    # Calculate base timestamp (midnight of target date)
    midnight = datetime(target_date.year, target_date.month, target_date.day)
    midnight_epoch = int(midnight.timestamp())
    
    # First trace timestamp - check if it's relative or absolute
    first_ts = traces[0][0] if traces else 0
    
    # If timestamp < 100000, it's likely seconds since midnight
    # Otherwise it might be epoch offset
    is_relative = first_ts < 100000
    
    for point in traces:
        if len(point) < 8:
            continue
        
        try:
            # Extract fields
            ts_raw = point[0]
            lat = point[1]
            lon = point[2]
            alt = point[3]
            speed = point[4]
            heading = point[5]
            on_ground = bool(point[6]) if point[6] is not None else None
            vrate = point[7]
            
            # Calculate absolute timestamp
            if is_relative:
                timestamp = midnight_epoch + int(ts_raw)
            else:
                timestamp = int(ts_raw)
            
            # Extract callsign and squawk from metadata if present
            callsign = None
            squawk = None
            
            if len(point) > 8 and isinstance(point[8], dict):
                metadata = point[8]
                callsign = metadata.get('flight', '').strip()
                squawk = metadata.get('squawk', '')
            
            records.append({
                'icao24': icao24,
                'callsign': callsign or '',
                'timestamp': timestamp,
                'latitude': float(lat) if lat is not None else None,
                'longitude': float(lon) if lon is not None else None,
                'altitude': float(alt) if alt is not None else None,
                'velocity': float(speed) if speed is not None else None,
                'heading': float(heading) if heading is not None else None,
                'vertical_rate': float(vrate) if vrate is not None else None,
                'on_ground': on_ground,
                'squawk': squawk or '',
                '_source_file': file_path.name,
                '_ingest_ts': int(datetime.now().timestamp()),
            })
            
        except (ValueError, TypeError, IndexError) as e:
            continue
    
    return records


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    target_date: Optional[date] = None,
    batch_size: int = 100000,
) -> dict:
    """
    Convert all adsblol trace files to Bronze format.
    
    Args:
        input_dir: Directory containing extracted traces/ folder
        output_dir: Bronze output directory
        target_date: Date for the data (parsed from dir name if not provided)
        batch_size: Records per output file
        
    Returns:
        Statistics dict
    """
    # Find traces directory
    traces_dir = input_dir / 'traces'
    if not traces_dir.exists():
        traces_dir = input_dir
    
    # Infer date from input directory name if not provided
    if target_date is None:
        # Try to extract date from path
        for part in input_dir.parts[::-1]:
            if len(part) == 10 and '-' in part:
                try:
                    target_date = date.fromisoformat(part)
                    break
                except ValueError:
                    pass
        
        if target_date is None:
            target_date = date.today()
            logger.warning(f"Could not infer date, using today: {target_date}")
    
    logger.info(f"Converting traces for {target_date}")
    
    # Find all trace files
    trace_files = list(traces_dir.rglob('trace_full_*.json'))
    
    if not trace_files:
        logger.warning(f"No trace files found in {traces_dir}")
        return {'error': 'No trace files found'}
    
    logger.info(f"Found {len(trace_files):,} trace files")
    
    # Create output directory
    date_str = target_date.isoformat()
    output_date_dir = output_dir / date_str
    output_date_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files and collect records
    all_records = []
    files_processed = 0
    files_failed = 0
    
    for trace_file in tqdm(trace_files, desc="Processing traces"):
        records = parse_trace_file(trace_file, target_date)
        if records:
            all_records.extend(records)
            files_processed += 1
        else:
            files_failed += 1
        
        # Write batch when large enough
        if len(all_records) >= batch_size:
            part_num = len(list(output_date_dir.glob('part-*.parquet')))
            write_batch(all_records[:batch_size], output_date_dir, part_num)
            all_records = all_records[batch_size:]
    
    # Write remaining records
    if all_records:
        part_num = len(list(output_date_dir.glob('part-*.parquet')))
        write_batch(all_records, output_date_dir, part_num)
    
    total_records = sum(
        len(pd.read_parquet(f)) 
        for f in output_date_dir.glob('part-*.parquet')
    )
    
    logger.info(f"Converted {total_records:,} records from {files_processed:,} files")
    
    return {
        'date': date_str,
        'files_processed': files_processed,
        'files_failed': files_failed,
        'total_records': total_records,
        'output_files': len(list(output_date_dir.glob('part-*.parquet'))),
    }


def write_batch(records: list[dict], output_dir: Path, part_num: int):
    """Write a batch of records to Parquet."""
    df = pd.DataFrame(records)
    
    # Ensure column types
    df['icao24'] = df['icao24'].astype(str)
    df['callsign'] = df['callsign'].fillna('').astype(str)
    df['squawk'] = df['squawk'].fillna('').astype(str)
    df['_source_file'] = df['_source_file'].astype(str)
    
    output_file = output_dir / f'part-{part_num:04d}.parquet'
    
    table = pa.Table.from_pandas(df, schema=BRONZE_SCHEMA, preserve_index=False)
    pq.write_table(table, output_file, compression='snappy')
    
    logger.debug(f"Wrote {len(records):,} records to {output_file.name}")


# =============================================================================
# CLI
# =============================================================================

@click.command()
@click.option(
    '--input', '-i', 'input_path',
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help='Extracted traces directory'
)
@click.option(
    '--output', '-o', 'output_path',
    required=True,
    type=click.Path(path_type=Path),
    help='Bronze output directory'
)
@click.option(
    '--date', '-d', 'target_date',
    default=None,
    help='Target date (YYYY-MM-DD), auto-detected if not specified'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
def main(
    input_path: Path,
    output_path: Path,
    target_date: Optional[str],
    verbose: bool,
):
    """
    Convert adsblol trace files to Bronze layer format.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    
    date_obj = None
    if target_date:
        date_obj = date.fromisoformat(target_date)
    
    results = convert_directory(input_path, output_path, date_obj)
    
    click.echo(f"\n{'='*50}")
    click.echo(f"ADSB.lol Conversion Complete")
    click.echo(f"{'='*50}")
    
    if 'error' in results:
        click.echo(f"Error: {results['error']}")
        return
    
    click.echo(f"Date: {results['date']}")
    click.echo(f"Files processed: {results['files_processed']:,}")
    click.echo(f"Files failed: {results['files_failed']:,}")
    click.echo(f"Total records: {results['total_records']:,}")
    click.echo(f"Output: {output_path}")


if __name__ == '__main__':
    main()
