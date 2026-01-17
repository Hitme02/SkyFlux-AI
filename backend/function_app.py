"""
SkyFlux AI - Azure Functions Backend

Thin read-only API layer serving precomputed Gold artifacts.
All endpoints are cache-friendly with appropriate Cache-Control headers.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional

import azure.functions as func
import pandas as pd

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_STORAGE_AVAILABLE = True
except ImportError:
    AZURE_STORAGE_AVAILABLE = False

app = func.FunctionApp()

# =============================================================================
# CONFIGURATION
# =============================================================================

CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "skyflux")
CACHE_MAX_AGE = 86400  # 24 hours
RETRAIN_SECRET = os.getenv("RETRAIN_SECRET", "change-me-in-production")

# In-memory cache for version info (refreshed periodically)
_version_cache = {"data": None, "expires": 0}


# =============================================================================
# HELPERS
# =============================================================================

def get_blob_service() -> Optional[BlobServiceClient]:
    """Get Azure Blob Service client."""
    if not AZURE_STORAGE_AVAILABLE:
        return None
    
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        return None
    
    return BlobServiceClient.from_connection_string(conn_str)


def read_parquet_from_blob(container_client, blob_path: str) -> Optional[pd.DataFrame]:
    """Read a Parquet file from blob storage."""
    try:
        blob_client = container_client.get_blob_client(blob_path)
        data = blob_client.download_blob().readall()
        return pd.read_parquet(BytesIO(data))
    except Exception as e:
        logging.error(f"Error reading {blob_path}: {e}")
        return None


def read_json_from_blob(container_client, blob_path: str) -> Optional[dict]:
    """Read a JSON file from blob storage."""
    try:
        blob_client = container_client.get_blob_client(blob_path)
        data = blob_client.download_blob().readall()
        return json.loads(data)
    except Exception as e:
        logging.error(f"Error reading {blob_path}: {e}")
        return None


def cache_response(data: dict, max_age: int = CACHE_MAX_AGE) -> func.HttpResponse:
    """Create a cached JSON response."""
    return func.HttpResponse(
        json.dumps(data, default=str),
        mimetype="application/json",
        headers={
            "Cache-Control": f"public, max-age={max_age}",
            "Access-Control-Allow-Origin": "*",
        }
    )


def error_response(message: str, status_code: int = 500) -> func.HttpResponse:
    """Create an error response."""
    return func.HttpResponse(
        json.dumps({"error": message}),
        mimetype="application/json",
        status_code=status_code,
        headers={"Access-Control-Allow-Origin": "*"}
    )


def get_version_info(container_client) -> dict:
    """Get current data version info (cached)."""
    global _version_cache
    
    now = datetime.utcnow().timestamp()
    
    if _version_cache["data"] and _version_cache["expires"] > now:
        return _version_cache["data"]
    
    version_data = read_json_from_blob(container_client, "gold/metadata/version.json")
    
    if version_data:
        _version_cache["data"] = version_data
        _version_cache["expires"] = now + 3600  # Cache for 1 hour
        return version_data
    
    return {"data_version": "unknown", "last_trained": None}


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.function_name(name="health")
@app.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint."""
    blob_service = get_blob_service()
    storage_ok = blob_service is not None
    
    return cache_response({
        "status": "ok" if storage_ok else "degraded",
        "version": "1.0.0",
        "storage_connected": storage_ok,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }, max_age=60)


@app.function_name(name="metadata")
@app.route(route="metadata", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def metadata(req: func.HttpRequest) -> func.HttpResponse:
    """Get current data version and model metadata."""
    blob_service = get_blob_service()
    if not blob_service:
        return error_response("Storage not configured", 503)
    
    container = blob_service.get_container_client(CONTAINER_NAME)
    
    # Get version info
    version_info = get_version_info(container)
    
    # Get model metadata
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
    
    # Get date index
    date_index = read_json_from_blob(container, "gold/date_index.json") or {}
    
    return cache_response({
        "data_version": version_info.get("data_version"),
        "last_trained": version_info.get("last_trained"),
        "date_range": date_index.get("date_range", {}),
        "models": models,
    })


@app.function_name(name="density")
@app.route(route="density", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def density(req: func.HttpRequest) -> func.HttpResponse:
    """Get traffic density grid for a date/hour."""
    blob_service = get_blob_service()
    if not blob_service:
        return error_response("Storage not configured", 503)
    
    container = blob_service.get_container_client(CONTAINER_NAME)
    
    # Query parameters
    target_date = req.params.get("date")
    target_hour = req.params.get("hour")
    
    # Load density data
    df = read_parquet_from_blob(container, "gold/traffic_density/traffic_density.parquet")
    
    if df is None:
        return error_response("Density data not found", 404)
    
    # Filter by date if provided
    if target_date:
        df = df[df["date"].astype(str) == target_date]
    
    # Filter by hour if provided
    if target_hour is not None:
        df = df[df["hour"] == int(target_hour)]
    
    # Convert to records
    grid = []
    for _, row in df.iterrows():
        grid.append({
            "lat": row["grid_lat"],
            "lon": row["grid_lon"],
            "hour": int(row["hour"]),
            "score": round(row["density_score"], 4),
            "percentile": round(row["density_percentile"], 1),
        })
    
    # Get version for cache invalidation
    version_info = get_version_info(container)
    
    return cache_response({
        "date": target_date,
        "hour": target_hour,
        "grid": grid,
        "data_version": version_info.get("data_version"),
        "cache_until": (datetime.utcnow() + timedelta(days=1)).isoformat() + "Z",
    })


@app.function_name(name="predictions")
@app.route(route="predictions", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def predictions(req: func.HttpRequest) -> func.HttpResponse:
    """Get trajectory predictions."""
    blob_service = get_blob_service()
    if not blob_service:
        return error_response("Storage not configured", 503)
    
    container = blob_service.get_container_client(CONTAINER_NAME)
    
    # Query parameters
    target_date = req.params.get("date")
    flight_id = req.params.get("flight_id")
    horizon = req.params.get("horizon")
    limit = int(req.params.get("limit", 100))
    
    # Load predictions
    df = read_parquet_from_blob(container, "gold/predictions/predictions.parquet")
    
    if df is None:
        return error_response("Predictions not found", 404)
    
    # Filter by flight_id if provided
    if flight_id:
        df = df[df["flight_id"] == flight_id]
    
    # Filter by date (extracted from flight_id)
    if target_date:
        df = df[df["flight_id"].str.contains(target_date)]
    
    # Filter by horizon if provided
    if horizon:
        df = df[df["horizon_sec"] == int(horizon)]
    
    # Limit results
    df = df.head(limit)
    
    # Convert to response format
    predictions = []
    for _, row in df.iterrows():
        pred = {
            "flight_id": row["flight_id"],
            "prediction_ts": row["prediction_ts"],
            "horizon_sec": row["horizon_sec"],
            "predicted": row["predicted_points"],
            "actual": row.get("actual_points") or [],
            "error": row.get("error_metrics"),
        }
        predictions.append(pred)
    
    version_info = get_version_info(container)
    
    return cache_response({
        "date": target_date,
        "predictions": predictions,
        "count": len(predictions),
        "data_version": version_info.get("data_version"),
    })


@app.function_name(name="anomalies")
@app.route(route="anomalies", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def anomalies(req: func.HttpRequest) -> func.HttpResponse:
    """Get anomaly records."""
    blob_service = get_blob_service()
    if not blob_service:
        return error_response("Storage not configured", 503)
    
    container = blob_service.get_container_client(CONTAINER_NAME)
    
    # Query parameters
    target_date = req.params.get("date")
    anomaly_type = req.params.get("type")
    min_score = float(req.params.get("min_score", 0))
    limit = int(req.params.get("limit", 100))
    
    # Load anomalies
    df = read_parquet_from_blob(container, "gold/anomalies/anomalies.parquet")
    
    if df is None:
        return error_response("Anomalies not found", 404)
    
    # Filter by date (from flight_id)
    if target_date:
        df = df[df["flight_id"].str.contains(target_date)]
    
    # Filter by type
    if anomaly_type:
        df = df[df["anomaly_type"] == anomaly_type]
    
    # Filter by min score
    df = df[df["anomaly_score"] >= min_score]
    
    # Sort by score descending
    df = df.sort_values("anomaly_score", ascending=False).head(limit)
    
    # Convert to response format
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
    
    return cache_response({
        "date": target_date,
        "anomalies": anomalies,
        "count": len(anomalies),
        "data_version": version_info.get("data_version"),
    })


@app.function_name(name="stress")
@app.route(route="stress", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def stress(req: func.HttpRequest) -> func.HttpResponse:
    """Get airspace stress index."""
    blob_service = get_blob_service()
    if not blob_service:
        return error_response("Storage not configured", 503)
    
    container = blob_service.get_container_client(CONTAINER_NAME)
    
    # Query parameters
    target_date = req.params.get("date")
    target_hour = req.params.get("hour")
    min_stress = float(req.params.get("min_stress", 0))
    
    # Load stress data
    df = read_parquet_from_blob(container, "gold/airspace_stress/airspace_stress.parquet")
    
    if df is None:
        return error_response("Stress data not found", 404)
    
    # Filter by date
    if target_date:
        df = df[df["date"].astype(str) == target_date]
    
    # Filter by hour
    if target_hour is not None:
        df = df[df["hour"] == int(target_hour)]
    
    # Filter by min stress
    df = df[df["stress_index"] >= min_stress]
    
    # Convert to response format
    stress_grid = []
    for _, row in df.iterrows():
        stress_grid.append({
            "lat": row["grid_lat"],
            "lon": row["grid_lon"],
            "hour": int(row["hour"]),
            "stress_index": round(row["stress_index"], 2),
            "risk_level": row["risk_level"],
            "components": row["components"],
        })
    
    version_info = get_version_info(container)
    
    return cache_response({
        "date": target_date,
        "hour": target_hour,
        "stress_grid": stress_grid,
        "count": len(stress_grid),
        "data_version": version_info.get("data_version"),
    })


@app.function_name(name="retrain")
@app.route(route="admin/retrain", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def retrain(req: func.HttpRequest) -> func.HttpResponse:
    """Manual retraining trigger (protected)."""
    # Validate authorization
    auth_header = req.headers.get("Authorization", "")
    
    if not auth_header.startswith("Bearer "):
        return error_response("Unauthorized", 401)
    
    token = auth_header.replace("Bearer ", "")
    
    if token != RETRAIN_SECRET:
        return error_response("Invalid token", 403)
    
    # Parse request body
    try:
        body = req.get_json()
    except Exception:
        body = {}
    
    training_days = body.get("training_window_days", 15)
    models = body.get("models", ["trajectory", "anomaly"])
    
    # In a real implementation, this would trigger an Azure ML pipeline
    # or Azure Container Instance to run the local pipeline
    
    job_id = f"retrain-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    # For now, return a placeholder response
    return func.HttpResponse(
        json.dumps({
            "job_id": job_id,
            "status": "queued",
            "training_window_days": training_days,
            "models": models,
            "estimated_completion": (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z",
            "message": "Retraining job queued. In production, this would trigger an Azure ML pipeline.",
        }),
        mimetype="application/json",
        status_code=202,
        headers={"Access-Control-Allow-Origin": "*"}
    )


@app.function_name(name="retrain_status")
@app.route(route="admin/retrain/{job_id}", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def retrain_status(req: func.HttpRequest) -> func.HttpResponse:
    """Check retraining job status."""
    job_id = req.route_params.get("job_id")
    
    # In a real implementation, this would check Azure ML job status
    
    return cache_response({
        "job_id": job_id,
        "status": "completed",  # Placeholder
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "message": "In production, this would return actual job status.",
    }, max_age=60)
