from __future__ import annotations

"""Parameter container dataclasses for the new FleetMix configuration system.

The new parameter organisation separates problem definition, algorithm settings and
I/O related options into individual immutable dataclasses.  A small mutable
`RuntimeParams` bucket captures flags that are never serialised to YAML but can
be toggled programmatically.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

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
    expected_vehicles: int = -1
    # TODO: agregar expected_vehicles aca, tal vez con un parametro que indique es un benchmark

    # Basic validation to surface common configuration errors early.
    def __post_init__(self):  # type: ignore[override]
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
                    f"Vehicle '{name}': allowed_goods contains goods not present in global goods list: {sorted(invalid)}"
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

    def __post_init__(self):  # type: ignore[override]
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

    def __post_init__(self):  # type: ignore[override]
        if self.format not in {"xlsx", "json", "csv"}:
            raise ValueError("IOParams.format must be 'xlsx', 'json' or 'csv'.")

        # Ensure results_dir is absolute
        if not self.results_dir.is_absolute():
            object.__setattr__(
                self, "results_dir", (Path.cwd() / self.results_dir).resolve()
            )

        self.results_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Runtime parameters – toggles that are never serialized to yaml
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RuntimeParams:
    verbose: bool = False
    debug: bool = False
    solver_gap_rel: float = 0.0


# ---------------------------------------------------------------------------
# Aggregate container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FleetmixParams:
    """Aggregate parameter object passed throughout the codebase."""

    problem: ProblemParams
    algorithm: AlgorithmParams
    io: IOParams
    runtime: RuntimeParams = field(default_factory=RuntimeParams)

    # Convenience accessors so calling code can use `params.X` like before.
    # TODO: eliminar este metodo para que el llamado sea mas explicito
    def __getattr__(self, item):
        # Delegate lookup to contained dataclasses.
        for section in (self.problem, self.algorithm, self.io, self.runtime):
            if hasattr(section, item):
                return getattr(section, item)
        raise AttributeError(item)

    # ------------------------------------------------------------------
    # Legacy helpers – provide backward-compatibility for code that still
    # expects the old flat/dict style attributes.
    # ------------------------------------------------------------------

    @property
    def clustering(self) -> dict[str, object]:
        """Return a dict equivalent of the old `clustering` section.

        This helper lets existing modules that access `params.clustering[...]`
        continue to function until they are fully migrated to the structured
        `AlgorithmParams` API.
        """

        return {
            "max_depth": self.algorithm.clustering_max_depth,
            "method": self.algorithm.clustering_method,
            "distance": self.algorithm.clustering_distance,
            "geo_weight": self.algorithm.geo_weight,
            "demand_weight": self.algorithm.demand_weight,
            "route_time_estimation": self.algorithm.route_time_estimation,
        }

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