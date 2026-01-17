
import json
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, date

def generate_metadata(gold_dir: Path):
    print(f"Generating metadata in {gold_dir}")
    gold_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = gold_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 1. Version JSON
    version_data = {
        "data_version": f"v{datetime.now().strftime('%Y%m%d')}-001",
        "last_trained": datetime.utcnow().isoformat() + "Z",
        "models": ["traj_pred_001", "anomaly_001"],
    }
    
    with open(gold_dir / "version.json", "w") as f:
        json.dump(version_data, f, indent=2)
    # Also save in metadata dir to be safe (backend looks in gold/metadata/version.json?)
    # Backend code: read_json_from_blob(container, "gold/metadata/version.json")
    with open(metadata_dir / "version.json", "w") as f:
        json.dump(version_data, f, indent=2)
        
    print("Generated version.json")

    # 2. Model Metadata Parquet
    records = []
    records.append({
        "model_id": "traj_pred_001",
        "model_type": "trajectory",
        "version": "1.0.0",
        "trained_at": int(time.time()),
        "training_window_start": date(2025, 12, 25),
        "training_window_end": date(2025, 12, 25),
        "training_days": 1,
        "validation_mae": 0.5,
        "validation_rmse": 0.7,
        "auc_roc": None,
    })
    records.append({
        "model_id": "anomaly_001",
        "model_type": "anomaly",
        "version": "1.0.0",
        "trained_at": int(time.time()),
        "training_window_start": date(2025, 12, 25),
        "training_window_end": date(2025, 12, 25),
        "training_days": 1,
        "validation_mae": None,
        "validation_rmse": None,
        "auc_roc": 0.85,
    })
    
    df = pd.DataFrame(records)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        metadata_dir / "model_metadata.parquet",
        compression="snappy",
    )
    print("Generated model_metadata.parquet")

if __name__ == "__main__":
    generate_metadata(Path("./data/gold"))
