from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from fleetmix.core_types import DepotLocation, VehicleSpec
from fleetmix.utils import PROJECT_ROOT
from fleetmix.utils.logging import FleetmixLogger

logger = FleetmixLogger.get_logger(__name__)


@dataclass
class Parameters:
    """Configuration parameters for the optimization"""

    vehicles: dict[str, VehicleSpec]
    variable_cost_per_hour: float
    depot: DepotLocation
    goods: list[str]
    clustering: dict
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
    allow_split_stops: bool = False
    pre_small_cluster_size: int = 5
    pre_nearest_merge_candidates: int = 3

    config_file_path: Path | None = field(default=None, repr=False)
    results_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("PROJECT_RESULTS_DIR", PROJECT_ROOT / "results")
        )
    )

    @classmethod
    def from_yaml(cls, config_path: str | Path | None = None) -> Parameters:
        """Load parameters from YAML file"""
        resolved_config_path: Path | None = None
        if config_path is None:
            default_config_paths = [
                PROJECT_ROOT / "config.yaml",
                PROJECT_ROOT / "src" / "config" / "default_config.yaml",
                Path(__file__).parent / "default_config.yaml",
            ]
            for p in default_config_paths:
                if p.exists():
                    resolved_config_path = p
                    logger.info(
                        f"No config file provided, using default: {resolved_config_path}"
                    )
                    break
            if resolved_config_path is None:
                raise FileNotFoundError(
                    "No configuration file provided and no default_config.yaml found in standard locations."
                )
        else:
            resolved_config_path = Path(config_path)
            if not resolved_config_path.exists():
                raise FileNotFoundError(
                    f"Config file not found: {resolved_config_path}"
                )

        try:
            with open(resolved_config_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Error parsing YAML file {resolved_config_path}:\n{e!s}\n"
                f"Please check the YAML syntax (indentation, quotes, etc.)"
            )
        except Exception as e:
            raise ValueError(f"Error reading config file {resolved_config_path}: {e!s}")

        raw_vehicles_data = data.pop("vehicles")
        parsed_vehicles = {}
        for v_name, v_details in raw_vehicles_data.items():
            spec_kwargs = {
                "capacity": v_details.pop("capacity"),
                "fixed_cost": v_details.pop("fixed_cost"),
                "avg_speed": v_details.pop("avg_speed"),
                "service_time": v_details.pop("service_time"),
                "max_route_time": v_details.pop("max_route_time"),
            }

            # Remaining items go into extra
            spec_kwargs["extra"] = v_details
            parsed_vehicles[v_name] = VehicleSpec(**spec_kwargs)

        vehicles = parsed_vehicles

        raw_depot = data.pop("depot")
        depot = DepotLocation(**raw_depot)

        data["config_file_path"] = resolved_config_path

        # Convert results_dir string back to Path if it exists
        if "results_dir" in data and isinstance(data["results_dir"], str):
            data["results_dir"] = Path(data["results_dir"])

        required_fields = ["goods", "demand_file", "clustering"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            raise ValueError(
                f"Missing required fields in config file {resolved_config_path}:\n"
                f"  {', '.join(missing_fields)}\n"
                f"Please ensure all required fields are present in the YAML file."
            )

        try:
            instance = cls(vehicles=vehicles, depot=depot, **data)
        except TypeError as e:
            error_str = str(e)
            if "missing" in error_str:
                raise ValueError(
                    f"Missing required configuration fields in {resolved_config_path}:\n"
                    f"  {error_str}\n"
                    f"Please check the YAML file structure matches the expected format."
                )
            elif "unexpected keyword" in error_str:
                raise ValueError(
                    f"Unknown configuration fields in {resolved_config_path}:\n"
                    f"  {error_str}\n"
                    f"Please check for typos in field names."
                )
            else:
                raise ValueError(
                    f"Error creating Parameters from {resolved_config_path}: {error_str}"
                )
        return instance

    def __post_init__(self):
        """Validate parameters after initialization"""
        if not isinstance(self.small_cluster_size, int) or self.small_cluster_size <= 0:
            raise ValueError(
                f"small_cluster_size must be a positive integer. Got: {self.small_cluster_size}"
            )

        if (
            not isinstance(self.nearest_merge_candidates, int)
            or self.nearest_merge_candidates <= 0
        ):
            raise ValueError(
                f"nearest_merge_candidates must be a positive integer. Got: {self.nearest_merge_candidates}"
            )

        if (
            not isinstance(self.max_improvement_iterations, int)
            or self.max_improvement_iterations < 0
        ):
            raise ValueError(
                f"max_improvement_iterations must be a non-negative integer. Got: {self.max_improvement_iterations}"
            )

        if not self.results_dir.is_absolute():
            self.results_dir = (PROJECT_ROOT / self.results_dir).resolve()

        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Results directory set to: {self.results_dir}")
        except Exception as e:
            logger.error(
                f"Failed to create results directory {self.results_dir} in __post_init__: {e}"
            )

        geo_weight = self.clustering.get("geo_weight", 0.7)
        demand_weight = self.clustering.get("demand_weight", 0.3)

        if abs(geo_weight + demand_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Clustering weights must sum to 1.0. Got: "
                f"geo_weight={geo_weight}, demand_weight={demand_weight}"
            )

    def to_yaml(self, output_path: str | Path) -> None:
        """Saves the current parameters to a YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data_to_save = asdict(self)
        del data_to_save["config_file_path"]

        for key, value in data_to_save.items():
            if isinstance(value, Path):
                data_to_save[key] = str(value)
            elif isinstance(value, DepotLocation):
                data_to_save[key] = value.to_dict()
            elif isinstance(value, dict) and all(
                isinstance(v, VehicleSpec) for v in value.values()
            ):
                data_to_save[key] = {k: v.to_dict() for k, v in value.items()}

        with open(output_path, "w") as f:
            yaml.dump(data_to_save, f, sort_keys=False)
