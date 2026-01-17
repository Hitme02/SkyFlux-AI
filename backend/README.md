# SkyFlux AI - Azure Backend

Thin Azure Functions backend serving read-only APIs for Gold layer artifacts.

## Endpoints

- `GET /api/health` - Health check
- `GET /api/metadata` - Data version and model info
- `GET /api/density` - Traffic density grid
- `GET /api/predictions` - Trajectory predictions
- `GET /api/anomalies` - Anomaly records
- `GET /api/stress` - Airspace stress index
- `POST /api/admin/retrain` - Manual retraining trigger

## Deployment

```bash
# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4

# Run locally
func start

# Deploy to Azure
func azure functionapp publish <app-name>
```

## Configuration

Set these environment variables:

- `AZURE_STORAGE_CONNECTION_STRING` - Blob storage connection
- `RETRAIN_SECRET` - Secret for admin endpoints
