
import os
import json
import time
import logging
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load config
load_dotenv()
CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.environ.get("AZURE_CONTAINER_NAME", "skyflux")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SkyFluxRetrain")

def simulate_retraining(job_id):
    """
    Simulates a retraining job by:
    1. Waiting (simulating work)
    2. Updating the version.json in Azure Blob Storage
    """
    if not CONNECTION_STRING:
        logger.error("No connection string found.")
        return

    blob_service = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service.get_container_client(CONTAINER_NAME)

    logger.info(f"Starting retraining job: {job_id}")
    logger.info("Simulating model training (Isolation Forest + Trajectory Regressor)...")
    
    # Simulate processing time
    time.sleep(5)
    
    # Generate new version
    new_version = f"v{datetime.now().strftime('%Y%m%d')}-{job_id[-4:]}"
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    version_data = {
        "data_version": new_version,
        "last_trained": timestamp,
        "job_id": job_id,
        "status": "completed",
        "models": ["traj_pred_v2", "anomaly_v2"]
    }
    
    # Upload new version.json
    logger.info(f"Deploying new model version: {new_version}")
    
    # Update root version
    blob_client = container_client.get_blob_client("gold/version.json")
    blob_client.upload_blob(json.dumps(version_data, indent=2), overwrite=True)
    
    # Update metadata version
    blob_client_meta = container_client.get_blob_client("gold/metadata/version.json")
    blob_client_meta.upload_blob(json.dumps(version_data, indent=2), overwrite=True)
    
    logger.info("Retraining complete. Frontend will detect new version on next poll.")

if __name__ == "__main__":
    import sys
    # Accept job_id from args or generate one
    job_id = sys.argv[1] if len(sys.argv) > 1 else f"job-{int(time.time())}"
    simulate_retraining(job_id)
