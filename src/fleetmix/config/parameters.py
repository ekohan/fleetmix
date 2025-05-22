from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml

from fleetmix.utils import PROJECT_ROOT

logger = logging.getLogger(__name__)

@dataclass
class Parameters:
    """Configuration parameters for the optimization"""
    vehicles: Dict
    variable_cost_per_hour: float
    avg_speed: float
    max_route_time: float
    service_time: float
    depot: Dict
    goods: List[str]
    clustering: Dict
    demand_file: str
    light_load_penalty: float
    light_load_threshold: float
    compartment_setup_cost: float
    format: str
    post_optimization: bool = True
    expected_vehicles: int = -1
    small_cluster_size: int = 7
    nearest_merge_candidates: int = 10
    max_improvement_iterations: int = 4
    prune_tsp: bool = False

    config_file_path: Optional[Path] = field(default=None, repr=False)
    results_dir: Path = field(default_factory=lambda: Path(os.environ.get("PROJECT_RESULTS_DIR", PROJECT_ROOT / "results")))

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path, None] = None) -> Parameters:
        """Load parameters from YAML file"""
        resolved_config_path: Optional[Path] = None
        if config_path is None:
            default_config_paths = [
                PROJECT_ROOT / "config.yaml",
                PROJECT_ROOT / "src" / "config" / "default_config.yaml",
                Path(__file__).parent / "default_config.yaml"
            ]
            for p in default_config_paths:
                if p.exists():
                    resolved_config_path = p
                    logger.info(f"No config file provided, using default: {resolved_config_path}")
                    break
            if resolved_config_path is None:
                raise FileNotFoundError(
                    "No configuration file provided and no default_config.yaml found in standard locations."
                )
        else:
            resolved_config_path = Path(config_path)
            if not resolved_config_path.exists():
                raise FileNotFoundError(f"Config file not found: {resolved_config_path}")
        
        with open(resolved_config_path) as f:
            data = yaml.safe_load(f)
        
        data['config_file_path'] = resolved_config_path

        try:
            instance =  cls(**data)
        except TypeError as e:
            logger.error(f"Error creating Parameters instance from {resolved_config_path}. Missing or mismatched keys in YAML? Error: {e}")
            raise
        return instance

    def __post_init__(self):
        """Validate parameters after initialization"""
        if not isinstance(self.small_cluster_size, int) or self.small_cluster_size <= 0:
            raise ValueError(
                f"small_cluster_size must be a positive integer. Got: {self.small_cluster_size}"
            )
        
        if not isinstance(self.nearest_merge_candidates, int) or self.nearest_merge_candidates <= 0:
            raise ValueError(
                f"nearest_merge_candidates must be a positive integer. Got: {self.nearest_merge_candidates}"
            )
        
        if not isinstance(self.max_improvement_iterations, int) or self.max_improvement_iterations < 0:
            raise ValueError(
                f"max_improvement_iterations must be a non-negative integer. Got: {self.max_improvement_iterations}"
            )
        
        if not self.results_dir.is_absolute():
            self.results_dir = (PROJECT_ROOT / self.results_dir).resolve()
        
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Results directory set to: {self.results_dir}")
        except Exception as e:
            logger.error(f"Failed to create results directory {self.results_dir} in __post_init__: {e}")

        geo_weight = self.clustering.get('geo_weight', 0.7)
        demand_weight = self.clustering.get('demand_weight', 0.3)
        
        if abs(geo_weight + demand_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Clustering weights must sum to 1.0. Got: "
                f"geo_weight={geo_weight}, demand_weight={demand_weight}"
            )

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Saves the current parameters to a YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data_to_save = {f.name: getattr(self, f.name) for f in fields(self) if f.name != 'config_file_path'}
        for key, value in data_to_save.items():
            if isinstance(value, Path):
                data_to_save[key] = str(value)
                
        with open(output_path, 'w') as f:
            yaml.dump(data_to_save, f, sort_keys=False) 