import pandas as pd
import pytest

from fleetmix.benchmarking.converters.cvrp import CVRPBenchmarkType, convert_cvrp_to_fsm
from fleetmix.benchmarking.parsers.cvrp import CVRPParser


@pytest.mark.parametrize(
    "btype, extra_kwargs, mult",
    [
        (CVRPBenchmarkType.NORMAL, {}, 1),
        (CVRPBenchmarkType.SPLIT, {"split_ratios": {"dry": 0.6, "chilled": 0.4}}, 1),
        (CVRPBenchmarkType.SCALED, {"num_goods": 2}, 2),
        (
            CVRPBenchmarkType.COMBINED,
            {"instance_names": ["X-n101-k25", "X-n101-k25"]},
            2,
        ),
    ],
)
def test_convert_cvrp_to_fsm_and_expected(btype, extra_kwargs, mult, small_vrp_path):
    # Parse baseline instance
    parser = CVRPParser(str(small_vrp_path))
    inst = parser.parse()

    # Determine instance input
    if btype == CVRPBenchmarkType.COMBINED:
        names = extra_kwargs["instance_names"]
    else:
        names = small_vrp_path.stem

    # Call converter with correct parameters
    if btype == CVRPBenchmarkType.SPLIT:
        df, params = convert_cvrp_to_fsm(
            names, btype, split_ratios=extra_kwargs["split_ratios"]
        )
    elif btype == CVRPBenchmarkType.SCALED:
        df, params = convert_cvrp_to_fsm(
            names, btype, num_goods=extra_kwargs["num_goods"]
        )
    else:
        # NORMAL and COMBINED
        df, params = convert_cvrp_to_fsm(names, btype)

    # Expected vehicles match parser num_vehicles * multiplier
    assert params.expected_vehicles == inst.num_vehicles * mult

    # DataFrame rows: for combined, rows = (dimension-1)*mult; otherwise rows = dimension-1
    if btype == CVRPBenchmarkType.COMBINED:
        assert len(df) == (inst.dimension - 1) * mult
    else:
        assert len(df) == (inst.dimension - 1)

    # Core columns exist
    for col in [
        "Latitude",
        "Longitude",
        "Dry_Demand",
        "Chilled_Demand",
        "Frozen_Demand",
    ]:
        assert col in df.columns

    # Types
    assert isinstance(df, pd.DataFrame)
    assert hasattr(params, "expected_vehicles")

    # Check expected_vehicles is positive and integer
    assert isinstance(params.expected_vehicles, int)
    assert params.expected_vehicles >= mult
