#!/usr/bin/env python3
"""
Generate a sampled trajectories file for time travel animation.

Reads the large silver trajectories file locally and samples points
to create a much smaller file suitable for API serving.
"""
import os
import sys
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

def main():
    # Read the large trajectories file
    input_file = "data/silver/trajectories/2025-12-25/trajectories.parquet"
    output_file = "data/gold/sampled_trajectories/sampled_trajectories.parquet"
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"Reading {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} trajectories")
    
    # Sample parameters
    sample_rate = 20  # Keep every Nth point
    max_trajectories = 200  # Limit number of trajectories
    
    # Process trajectories
    sampled_trajectories = []
    
    # Uniformly sample trajectories across the entire dataset
    total_rows = len(df)
    step = max(1, total_rows // max_trajectories)
    
    print(f"Sampling every {step}th trajectory from {total_rows} total...")
    
    # Select indices for uniform sampling and create sampled dataframe
    indices = range(0, total_rows, step)
    df_sampled = df.iloc[indices]
    
    for idx, row in df_sampled.iterrows():
        flight_id = row.get("flight_id", f"traj_{idx}")
        points_raw = row.get("points", [])
        
        # Handle numpy arrays
        if hasattr(points_raw, 'tolist'):
            points_raw = points_raw.tolist()
        
        if not points_raw or len(points_raw) < 5:
            continue
        
        # Sample points
        sampled_points = []
        for i, pt in enumerate(points_raw):
            if i % sample_rate == 0 or i == len(points_raw) - 1:
                sampled_points.append({
                    "ts": int(pt.get("ts", 0)),
                    "lat": float(pt.get("lat", 0) or 0),
                    "lon": float(pt.get("lon", 0) or 0),
                    "alt_ft": float(pt.get("alt_ft", 0) or 0),
                    "speed_kts": float(pt.get("speed_kts", 0) or 0),
                    "heading_deg": float(pt.get("heading_deg", 0) or 0),
                })
        
        if len(sampled_points) >= 3:
            sampled_trajectories.append({
                "flight_id": flight_id,
                "points": sampled_points,
            })
    
    print(f"Sampled {len(sampled_trajectories)} trajectories")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write to parquet
    df_out = pd.DataFrame(sampled_trajectories)
    df_out.to_parquet(output_file, compression="snappy")
    
    # Calculate file sizes
    input_size = os.path.getsize(input_file) / (1024 * 1024)
    output_size = os.path.getsize(output_file) / (1024 * 1024)
    
    print(f"\nOriginal file: {input_size:.1f} MB")
    print(f"Sampled file: {output_size:.2f} MB")
    print(f"Compression ratio: {input_size/output_size:.1f}x")
    print(f"\nOutput saved to: {output_file}")

if __name__ == "__main__":
    main()
