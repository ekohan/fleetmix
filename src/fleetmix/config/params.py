"""Parameter container dataclasses for the FleetMix configuration system.

The parameter organisation separates problem definition, algorithm settings, runtime settings and
I/O related options into individual immutable dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:  # pragma: no cover – for static typing only
    from fleetmix.benchmarking.models import InstanceSpec

from fleetmix.core_types import DepotLocation, VehicleSpec

__all__ = [
    "ProblemParams",
    "AlgorithmParams",
    "IOParams",
    "RuntimeParams",
    "FleetmixParams",
]


# ---------------------------------------------------------------------------
# Problem definition parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProblemParams:
    """Capture the business problem independent from the algorithm used."""

    vehicles: Dict[str, VehicleSpec]
    depot: DepotLocation
    goods: List[str]
    variable_cost_per_hour: float
    light_load_penalty: float = 0.0
    light_load_threshold: float = 0.0
    compartment_setup_cost: float = 0.0
    allow_split_stops: bool = False
    expected_vehicles: int = (
        -1
    )  # Only relevant for benchmarking with CVRP and MCVRP instances

    # Basic validation to surface common configuration errors early.
    def __post_init__(self):
        if not self.vehicles:
            raise ValueError("ProblemParams.vehicles cannot be empty.")

        # Validate goods are unique
        if len(set(self.goods)) != len(self.goods):
            raise ValueError("ProblemParams.goods contains duplicate entries.")

        # Validate allowed_goods of vehicles reference global goods only
        global_goods = set(self.goods)
        for name, spec in self.vehicles.items():
            if spec.allowed_goods is None:
                continue
            invalid = set(spec.allowed_goods) - global_goods
            if invalid:
                raise ValueError(
                    f"Vehicle '{name}': allowed_goods contains goods not in global list: {sorted(invalid)}"
                )


# ---------------------------------------------------------------------------
# Algorithm parameters – things that influence heuristic/solver behaviour
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AlgorithmParams:
    """Algorithm configuration options."""

    # Clustering
    clustering_max_depth: int = 20
    clustering_method: str = "combine"
    clustering_distance: str = "euclidean"
    geo_weight: float = 0.7
    demand_weight: float = 0.3
    route_time_estimation: str = "BHH"
    prune_tsp: bool = False

    # Merge phase / improvement
    small_cluster_size: int = 7
    nearest_merge_candidates: int = 10
    max_improvement_iterations: int = 4
    pre_small_cluster_size: int = 5
    pre_nearest_merge_candidates: int = 3
    post_optimization: bool = True

    def __post_init__(self):
        # Ensure clustering weights sum to 1.
        if abs(self.geo_weight + self.demand_weight - 1.0) > 1e-6:
            raise ValueError(
                "AlgorithmParams.geo_weight and demand_weight must add up to 1.0"
            )

        if self.clustering_max_depth <= 0:
            raise ValueError("AlgorithmParams.clustering_max_depth must be positive.")

        for field_name in (
            "small_cluster_size",
            "nearest_merge_candidates",
            "max_improvement_iterations",
            "pre_small_cluster_size",
            "pre_nearest_merge_candidates",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"AlgorithmParams.{field_name} must be non-negative.")


# ---------------------------------------------------------------------------
# IO parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IOParams:
    """Settings for data input and output pathways."""

    demand_file: str
    results_dir: Path
    format: str = "json"  # One of: xlsx, json, csv

    def __post_init__(self):
        if self.format not in {"xlsx", "json", "csv"}:
            raise ValueError("IOParams.format must be 'xlsx', 'json' or 'csv'.")

        # Ensure results_dir is absolute
        if not self.results_dir.is_absolute():
            object.__setattr__(
                self, "results_dir", (Path.cwd() / self.results_dir).resolve()
            )

        self.results_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Runtime parameters
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RuntimeParams:
    config: Path
    verbose: bool = False
    debug: bool = False
    gap_rel: float = 0.0  # Renamed from solver_gap_rel
    solver: str = "auto"  # "auto" | "gurobi" | "cbc"
    time_limit: int | None = None  # seconds; None = no limit


# ---------------------------------------------------------------------------
# Aggregate container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FleetmixParams:
    """Aggregate parameter object passed throughout the codebase."""

    problem: ProblemParams
    algorithm: AlgorithmParams
    io: IOParams
    runtime: RuntimeParams

    # Make the object picklable when using joblib (loky backend)
    def __getstate__(self):
        return {
            "problem": self.problem,
            "algorithm": self.algorithm,
            "io": self.io,
            "runtime": self.runtime,
        }

    def __setstate__(self, state):  # noqa: D401  (simple setter)
        object.__setattr__(self, "problem", state["problem"])
        object.__setattr__(self, "algorithm", state["algorithm"])
        object.__setattr__(self, "io", state["io"])
        object.__setattr__(self, "runtime", state["runtime"])

    def apply_instance_spec(self, spec: "InstanceSpec") -> "FleetmixParams":
        """
        Return a copy with ProblemParams fields overridden by the provided
        InstanceSpec.
        """
        import dataclasses as _dc

        new_problem = _dc.replace(
            self.problem,
            vehicles=spec.vehicles,
            depot=spec.depot,
            goods=spec.goods,
            expected_vehicles=spec.expected_vehicles,
        )
        return _dc.replace(self, problem=new_problem)
