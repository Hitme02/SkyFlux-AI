
import os
import json
import logging
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
CONTAINER_NAME = os.environ.get("AZURE_CONTAINER_NAME", "skyflux")
CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
RETRAIN_SECRET = os.environ.get("RETRAIN_SECRET", "skyflux-admin-change-me")

# Helper Functions
def get_blob_service():
    """Get authenticated BlobServiceClient."""
    if not CONNECTION_STRING:
        logger.error("AZURE_STORAGE_CONNECTION_STRING not found")
        return None
    try:
        return BlobServiceClient.from_connection_string(CONNECTION_STRING)
    except Exception as e:
        logger.error(f"Failed to create BlobServiceClient: {e}")
        return None

def read_parquet_from_blob(container_client, blob_path):
    """Read Parquet file from blob storage into DataFrame."""
    try:
        blob_client = container_client.get_blob_client(blob_path)
        if not blob_client.exists():
            return None
        
        stream = blob_client.download_blob()
        data = stream.readall()
        return pd.read_parquet(io.BytesIO(data))
    except Exception as e:
        logger.error(f"Error reading {blob_path}: {e}")
        return None

def read_json_from_blob(container_client, blob_path):
    """Read JSON file from blob storage."""
    try:
        blob_client = container_client.get_blob_client(blob_path)
        if not blob_client.exists():
            return None
        
        stream = blob_client.download_blob()
        data = stream.readall()
        return json.loads(data)
    except Exception as e:
        logger.error(f"Error reading {blob_path}: {e}")
        return None

def get_version_info(container_client):
    """Get data version information."""
    version_data = read_json_from_blob(container_client, "gold/metadata/version.json")
    if version_data:
        return version_data
    return {
        "data_version": "unknown",
        "last_trained": None
    }

# Routes
@app.route("/api/health", methods=["GET"])
def health():
    blob_service = get_blob_service()
    storage_ok = blob_service is not None
    
    return jsonify({
        "status": "ok" if storage_ok else "degraded",
        "version": "1.0.0-local",
        "storage_connected": storage_ok,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })

@app.route("/api/metadata", methods=["GET"])
def metadata():
    blob_service = get_blob_service()
    if not blob_service:
        return jsonify({"error": "Storage not configured"}), 503
    
    container = blob_service.get_container_client(CONTAINER_NAME)
    version_info = get_version_info(container)
    
    try:
        model_df = read_parquet_from_blob(container, "gold/metadata/model_metadata.parquet")
        models = []
        if model_df is not None:
            for _, row in model_df.iterrows():
                models.append({
                    "model_id": row.get("model_id"),
                    "type": row.get("model_type"),
                    "version": row.get("version"),
                    "trained_at": str(row.get("trained_at")),
                })
        
        date_index = read_json_from_blob(container, "gold/date_index.json") or {}
        
        return jsonify({
            "data_version": version_info.get("data_version"),
            "last_trained": version_info.get("last_trained"),
            "date_range": date_index.get("date_range", {}),
            "models": models,
        })
    except Exception as e:
        logger.error(f"Metadata error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/density", methods=["GET"])
def density():
    blob_service = get_blob_service()
    if not blob_service:
        return jsonify({"error": "Storage not configured"}), 503
    
    container = blob_service.get_container_client(CONTAINER_NAME)
    target_date = request.args.get("date")
    target_hour = request.args.get("hour")
    
    df = read_parquet_from_blob(container, "gold/traffic_density/traffic_density.parquet")
    
    if df is None:
        return jsonify({"error": "Density data not found"}), 404
    
    if target_date:
        df = df[df["date"].astype(str) == target_date]
    
    if target_hour is not None:
        df = df[df["hour"] == int(target_hour)]
    
    grid = []
    for _, row in df.iterrows():
        grid.append({
            "lat": row["grid_lat"],
            "lon": row["grid_lon"],
            "hour": int(row["hour"]),
            "score": round(row["density_score"], 4),
            "percentile": round(row["density_percentile"], 1),
        })
    
    version_info = get_version_info(container)
    
    return jsonify({
        "date": target_date,
        "hour": target_hour,
        "grid": grid,
        "data_version": version_info.get("data_version"),
    })

@app.route("/api/predictions", methods=["GET"])
def predictions():
    blob_service = get_blob_service()
    if not blob_service:
        return jsonify({"error": "Storage not configured"}), 503
    
    container = blob_service.get_container_client(CONTAINER_NAME)
    target_date = request.args.get("date")
    flight_id = request.args.get("flight_id")
    horizon = request.args.get("horizon")
    limit = int(request.args.get("limit", 100))
    
    df = read_parquet_from_blob(container, "gold/predictions/predictions.parquet")
    
    if df is None:
        return jsonify({"error": "Predictions not found"}), 404
    
    if flight_id:
        df = df[df["flight_id"] == flight_id]
    
    if target_date:
        df = df[df["flight_id"].str.contains(target_date)]
    
    if horizon:
        df = df[df["horizon_sec"] == int(horizon)]
    
    df = df.head(limit)
    
    predictions = []
    for _, row in df.iterrows():
        # Handle numpy arrays for JSON serialization
        actual = row.get("actual_points")
        if hasattr(actual, 'tolist'):
            actual = actual.tolist()
        elif actual is None:
            actual = []
            
        predicted = row["predicted_points"]
        if hasattr(predicted, 'tolist'):
            predicted = predicted.tolist()
            
        pred = {
            "flight_id": row["flight_id"],
            "prediction_ts": row["prediction_ts"],
            "horizon_sec": row["horizon_sec"],
            "predicted": predicted,
            "actual": actual,
            "error": row.get("error_metrics"),
        }
        predictions.append(pred)
    
    version_info = get_version_info(container)
    
    return jsonify({
        "date": target_date,
        "predictions": predictions,
        "count": len(predictions),
        "data_version": version_info.get("data_version"),
    })

@app.route("/api/anomalies", methods=["GET"])
def anomalies():
    blob_service = get_blob_service()
    if not blob_service:
        return jsonify({"error": "Storage not configured"}), 503
    
    container = blob_service.get_container_client(CONTAINER_NAME)
    target_date = request.args.get("date")
    anomaly_type = request.args.get("type")
    min_score = float(request.args.get("min_score", 0))
    limit = int(request.args.get("limit", 100))
    
    df = read_parquet_from_blob(container, "gold/anomalies/anomalies.parquet")
    
    if df is None:
        return jsonify({"error": "Anomalies not found"}), 404
    
    if target_date:
        df = df[df["flight_id"].str.contains(target_date)]
    
    if anomaly_type:
        df = df[df["anomaly_type"] == anomaly_type]
    
    df = df[df["anomaly_score"] >= min_score]
    df = df.sort_values("anomaly_score", ascending=False).head(limit)
    
    anomalies = []
    for _, row in df.iterrows():
        anomalies.append({
            "flight_id": row["flight_id"],
            "type": row["anomaly_type"],
            "score": round(row["anomaly_score"], 3),
            "explanation": row["explanation"],
            "location": row["location"],
        })
    
    version_info = get_version_info(container)
    
    return jsonify({
        "date": target_date,
        "anomalies": anomalies,
        "count": len(anomalies),
        "data_version": version_info.get("data_version"),
    })

@app.route("/api/stress", methods=["GET"])
def stress():
    blob_service = get_blob_service()
    if not blob_service:
        return jsonify({"error": "Storage not configured"}), 503
    
    container = blob_service.get_container_client(CONTAINER_NAME)
    target_date = request.args.get("date")
    target_hour = request.args.get("hour")
    
    # Filename check: standard gold pipeline writes to gold/airspace_stress/airspace_stress.parquet
    df = read_parquet_from_blob(container, "gold/airspace_stress/airspace_stress.parquet")
    
    if df is None:
        return jsonify({"error": "Stress data not found"}), 404
    
    if target_date:
        df = df[df["date"].astype(str) == target_date]
    
    if target_hour is not None:
        # Handle hour filtering safely
        df = df[df["hour"] == int(target_hour)]
    
    stress_grid = []
    for _, row in df.iterrows():
        # Components are stored as a struct/dict in the 'components' column
        comps = row.get("components", {})
        if not isinstance(comps, dict):
            comps = {} # Handle cases where it might be missing or None
            
        stress_grid.append({
            "lat": row["grid_lat"],
            "lon": row["grid_lon"],
            "stress_index": round(row["stress_index"], 2),
            "risk_level": row["risk_level"],
            "components": {
                "density": round(comps.get("density_score", 0), 2),
                "variance": round(comps.get("altitude_variance", 0), 2),
                "conflict": round(comps.get("heading_conflict_score", 0), 2)
            }
        })
    
    return jsonify({
        "date": target_date,
        "hour": target_hour,
        "stress_grid": stress_grid,
    })

if __name__ == "__main__":
    print(f"Starting Local SkyFlux API on port 5001...")
    print(f"Connecting to container: {CONTAINER_NAME}")
    app.run(host="0.0.0.0", port=5001, debug=True)
