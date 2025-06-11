from pathlib import Path

import pytest

from fleetmix.benchmarking import parse_mcvrp
from fleetmix.benchmarking.models import MCVRPInstance


def test_parse_sample():
    # Happy path: sample instance from assets
    sample = Path(__file__).parent.parent / "_assets" / "mcvrp" / "sample.dat"
    inst = parse_mcvrp(sample)
    assert isinstance(inst, MCVRPInstance)
    assert inst.name == sample.stem
    assert inst.dimension == 11
    assert inst.capacity == 1000
    assert inst.vehicles == 2
    assert inst.depot_id == 1
    assert len(inst.coords) == inst.dimension
    assert len(inst.demands) == inst.dimension
    # Depot has zero demand
    assert inst.demands[1] == (0, 0, 0)
    # Spot checks
    assert inst.coords[2] == (25.44, 95.54)
    assert inst.demands[4] == (0, 0, 112)


@pytest.mark.parametrize(
    "bad_header, bad_value, message",
    [
        ("PRODUCT TYPES", "2", "Expected 3 product types"),
        ("COMPARTMENTS", "2", "Expected 3 compartments"),
    ],
)
def test_invalid_product_or_compartments(bad_header, bad_value, message, tmp_path):
    sample = Path(__file__).parent.parent / "_assets" / "mcvrp" / "sample.dat"
    data = sample.read_text().splitlines()
    # Modify the specified header to a bad value
    new_data = []
    for line in data:
        if line.startswith(f"{bad_header}"):
            key = line.split(":")[0]
            new_data.append(f"{key}: {bad_value}")
        else:
            new_data.append(line)
    bad_file = tmp_path / "bad1.dat"
    bad_file.write_text("\n".join(new_data))
    with pytest.raises(ValueError) as ei:
        parse_mcvrp(bad_file)
    assert message in str(ei.value)


def test_missing_header(tmp_path):
    sample = Path(__file__).parent.parent / "_assets" / "mcvrp" / "sample.dat"
    data = sample.read_text().splitlines()
    # Remove DIMENSION header
    new_data = [line for line in data if not line.startswith("DIMENSION")]
    bad_file = tmp_path / "bad2.dat"
    bad_file.write_text("\n".join(new_data))
    with pytest.raises(ValueError) as ei:
        parse_mcvrp(bad_file)
    assert "Missing required header: DIMENSION" in str(ei.value)


def test_mismatched_counts(tmp_path):
    sample = Path(__file__).parent.parent / "_assets" / "mcvrp" / "sample.dat"
    data = sample.read_text().splitlines()
    new_data = []
    in_coord = False
    skipped = False
    for line in data:
        if line == "NODE_COORD_SECTION":
            new_data.append(line)
            in_coord = True
            continue
        if in_coord and not skipped and line and line.split()[0].isdigit():
            # Skip first coordinate entry to create mismatch
            skipped = True
            continue
        new_data.append(line)
    bad_file = tmp_path / "bad3.dat"
    bad_file.write_text("\n".join(new_data))
    with pytest.raises(ValueError) as ei:
        parse_mcvrp(bad_file)
    assert "does not match dimension" in str(ei.value)
