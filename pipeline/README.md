# SkyFlux AI - Pipeline

Local data processing pipeline implementing the Medallion Architecture.

## Layers

- **Bronze**: Raw ADS-B data ingestion
- **Silver**: Cleaned and normalized trajectories  
- **Gold**: ML predictions and derived metrics

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

See main README for pipeline commands.
