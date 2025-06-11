import pytest

from fleetmix.core_types import BenchmarkType


@pytest.mark.parametrize(
    "enum_member,expected_value",
    [
        (BenchmarkType.SINGLE_COMPARTMENT, "single_compartment"),
        (BenchmarkType.MULTI_COMPARTMENT, "multi_compartment"),
    ],
)
def test_benchmark_type_value_and_str(enum_member, expected_value):
    # Enum .value matches
    assert enum_member.value == expected_value
    # str(Enum) uses EnumClass.member
    assert str(enum_member) == f"BenchmarkType.{enum_member.name}"


def test_benchmark_type_name_lookup():
    # Reverse lookup by name returns same member
    for member in BenchmarkType:
        assert BenchmarkType[member.name] is member


def test_repr_contains_name_and_value():
    for member in BenchmarkType:
        rep = repr(member)
        # Should include the class name and the value
        assert member.name in rep
        assert member.value in rep
