"""
SkyFlux AI - Configuration

Central configuration for pipeline processing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""
    
    # Data directories
    data_root: Path = field(default_factory=lambda: Path("./data"))
    bronze_dir: Optional[Path] = None
    silver_dir: Optional[Path] = None
    gold_dir: Optional[Path] = None
    
    # Training configuration
    training_window_days: int = 15
    max_training_days: int = 30
    
    # Bronze layer settings
    bronze_chunk_size: int = 100_000  # Rows per chunk during ingestion
    
    # Silver layer settings
    silver_downsample_interval_sec: int = 10  # Downsample to 10s intervals
    silver_max_gap_sec: int = 300  # Drop flights with gaps > 5 min
    silver_interpolate_gap_sec: int = 60  # Interpolate gaps < 1 min
    silver_grid_resolution_deg: float = 0.5  # 0.5° grid cells
    
    # Gold layer settings
    gold_prediction_horizons_sec: list[int] = field(
        default_factory=lambda: [60, 120, 300]  # 1, 2, 5 minute predictions
    )
    gold_anomaly_threshold: float = 0.7  # Anomaly score threshold
    
    # Azure settings
    azure_storage_account: Optional[str] = None
    azure_container_name: str = "skyflux"
    
    def __post_init__(self):
        """Set default subdirectories if not specified."""
        if self.bronze_dir is None:
            self.bronze_dir = self.data_root / "bronze"
        if self.silver_dir is None:
            self.silver_dir = self.data_root / "silver"
        if self.gold_dir is None:
            self.gold_dir = self.data_root / "gold"
    
    def ensure_directories(self):
        """Create data directories if they don't exist."""
        for dir_path in [self.bronze_dir, self.silver_dir, self.gold_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load configuration from environment variables."""
        return cls(
            data_root=Path(os.getenv("SKYFLUX_DATA_ROOT", "./data")),
            training_window_days=int(os.getenv("SKYFLUX_TRAINING_DAYS", "15")),
            azure_storage_account=os.getenv("AZURE_STORAGE_ACCOUNT"),
            azure_container_name=os.getenv("AZURE_CONTAINER_NAME", "skyflux"),
        )


# Global default configuration
DEFAULT_CONFIG = PipelineConfig()
