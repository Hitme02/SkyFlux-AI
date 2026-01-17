"""
SkyFlux AI - Azure Upload Utility

Upload curated Gold artifacts to Azure Blob Storage.

Usage:
    python -m pipeline.upload --input ./data/gold --account skyfluxstorage
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    from pipeline.config import PipelineConfig, DEFAULT_CONFIG
except ImportError:
    from config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def upload_to_azure(
    gold_dir: Path,
    connection_string: Optional[str] = None,
    account_name: Optional[str] = None,
    account_key: Optional[str] = None,
    container_name: str = "skyflux",
    dry_run: bool = False,
) -> dict:
    """
    Upload Gold artifacts to Azure Blob Storage.
    
    Args:
        gold_dir: Local Gold layer directory
        connection_string: Azure Storage connection string (preferred)
        account_name: Storage account name (if not using connection string)
        account_key: Storage account key (if not using connection string)
        container_name: Target container name
        dry_run: If True, only list files without uploading
        
    Returns:
        Upload statistics
    """
    if not AZURE_AVAILABLE:
        raise ImportError("azure-storage-blob not installed. Run: pip install azure-storage-blob")
    
    # Build connection
    if connection_string:
        blob_service = BlobServiceClient.from_connection_string(connection_string)
    elif account_name and account_key:
        account_url = f"https://{account_name}.blob.core.windows.net"
        blob_service = BlobServiceClient(account_url=account_url, credential=account_key)
    else:
        raise ValueError("Must provide either connection_string or (account_name, account_key)")
    
    # Ensure container exists
    container_client = blob_service.get_container_client(container_name)
    
    if not dry_run:
        try:
            container_client.create_container()
            logger.info(f"Created container: {container_name}")
        except Exception:
            logger.info(f"Container already exists: {container_name}")
    
    # Find all files to upload
    files_to_upload = []
    
    for parquet_file in gold_dir.rglob("*.parquet"):
        # Construct blob path: gold/{artifact_type}/file.parquet
        relative_path = parquet_file.relative_to(gold_dir)
        blob_path = f"gold/{relative_path}"
        files_to_upload.append((parquet_file, blob_path))
    
    # Also upload JSON files (version.json, etc.)
    for json_file in gold_dir.rglob("*.json"):
        relative_path = json_file.relative_to(gold_dir)
        blob_path = f"gold/{relative_path}"
        files_to_upload.append((json_file, blob_path))
    
    # Also upload model files
    for pkl_file in gold_dir.rglob("*.pkl"):
        relative_path = pkl_file.relative_to(gold_dir)
        blob_path = f"gold/{relative_path}"
        files_to_upload.append((pkl_file, blob_path))
    
    logger.info(f"Found {len(files_to_upload)} files to upload")
    
    if dry_run:
        for local_path, blob_path in files_to_upload:
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(f"Would upload: {blob_path} ({size_mb:.2f} MB)")
        return {"files_found": len(files_to_upload), "uploaded": 0, "dry_run": True}
    
    # Upload files
    uploaded = 0
    failed = []
    
    for local_path, blob_path in files_to_upload:
        try:
            blob_client = container_client.get_blob_client(blob_path)
            
            # Determine content type
            content_type = "application/octet-stream"
            if blob_path.endswith(".parquet"):
                content_type = "application/x-parquet"
            elif blob_path.endswith(".json"):
                content_type = "application/json"
            
            with open(local_path, "rb") as data:
                blob_client.upload_blob(
                    data,
                    overwrite=True,
                    content_settings=ContentSettings(content_type=content_type),
                )
            
            logger.info(f"Uploaded: {blob_path}")
            uploaded += 1
            
        except Exception as e:
            logger.error(f"Failed to upload {blob_path}: {e}")
            failed.append({"path": blob_path, "error": str(e)})
    
    # Generate and upload date index
    date_index = generate_date_index(gold_dir)
    if not dry_run and date_index:
        try:
            index_blob = container_client.get_blob_client("gold/date_index.json")
            index_blob.upload_blob(
                json.dumps(date_index, indent=2),
                overwrite=True,
                content_settings=ContentSettings(content_type="application/json"),
            )
            logger.info("Uploaded date_index.json")
            uploaded += 1
        except Exception as e:
            logger.error(f"Failed to upload date_index.json: {e}")
    
    return {
        "files_found": len(files_to_upload),
        "uploaded": uploaded,
        "failed": len(failed),
        "failed_files": failed,
    }


def generate_date_index(gold_dir: Path) -> dict:
    """
    Generate an index of available dates in Gold data.
    
    Returns:
        Date index dict
    """
    dates = set()
    
    # Check predictions directory for dates
    pred_dir = gold_dir / "predictions"
    if pred_dir.exists():
        # Load parquet and extract dates
        for pf in pred_dir.glob("*.parquet"):
            try:
                import pandas as pd
                df = pd.read_parquet(pf)
                # Extract dates from flight_id (format: icao24_YYYY-MM-DD_seq)
                for flight_id in df["flight_id"].unique():
                    parts = flight_id.split("_")
                    if len(parts) >= 2:
                        dates.add(parts[1])
            except Exception:
                pass
    
    # Sort dates
    sorted_dates = sorted(dates)
    
    return {
        "available_dates": sorted_dates,
        "date_range": {
            "start": sorted_dates[0] if sorted_dates else None,
            "end": sorted_dates[-1] if sorted_dates else None,
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# =============================================================================
# CLI
# =============================================================================

@click.command()
@click.option(
    "--input", "-i", "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Gold layer directory to upload"
)
@click.option(
    "--connection-string", "-c",
    envvar="AZURE_STORAGE_CONNECTION_STRING",
    help="Azure Storage connection string"
)
@click.option(
    "--account", "-a",
    envvar="AZURE_STORAGE_ACCOUNT",
    help="Azure Storage account name"
)
@click.option(
    "--key", "-k",
    envvar="AZURE_STORAGE_KEY",
    help="Azure Storage account key"
)
@click.option(
    "--container", "-n",
    default="skyflux",
    help="Container name (default: skyflux)"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="List files without uploading"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging"
)
def main(
    input_path: Path,
    connection_string: Optional[str],
    account: Optional[str],
    key: Optional[str],
    container: str,
    dry_run: bool,
    verbose: bool,
):
    """
    SkyFlux AI - Azure Upload
    
    Upload Gold layer artifacts to Azure Blob Storage.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    if not AZURE_AVAILABLE:
        click.echo("Error: azure-storage-blob not installed")
        click.echo("Run: pip install azure-storage-blob")
        return
    
    if not connection_string and not (account and key):
        click.echo("Error: Must provide --connection-string or (--account and --key)")
        click.echo("You can also set AZURE_STORAGE_CONNECTION_STRING environment variable")
        return
    
    try:
        results = upload_to_azure(
            input_path,
            connection_string=connection_string,
            account_name=account,
            account_key=key,
            container_name=container,
            dry_run=dry_run,
        )
        
        click.echo(f"\n{'='*50}")
        click.echo(f"Azure Upload {'(Dry Run) ' if dry_run else ''}Complete")
        click.echo(f"{'='*50}")
        click.echo(f"Files found: {results['files_found']}")
        click.echo(f"Uploaded: {results['uploaded']}")
        
        if results.get("failed", 0) > 0:
            click.echo(f"Failed: {results['failed']}")
            for f in results.get("failed_files", []):
                click.echo(f"  - {f['path']}: {f['error']}")
        
    except Exception as e:
        click.echo(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
