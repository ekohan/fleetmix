from fleetmix.benchmarking.parsers.cvrp import CVRPParser


def test_parse_instance_and_solution(small_vrp_path):
    # Initialize parser
    parser = CVRPParser(str(small_vrp_path))
    instance = parser.parse()
    # Basic instance properties
    assert instance.name == small_vrp_path.stem
    assert instance.dimension > 0
    assert instance.capacity > 0
    assert isinstance(instance.coordinates, dict)
    assert isinstance(instance.demands, dict)

    # Parse solution
    parser_sol = CVRPParser(str(small_vrp_path))
    solution = parser_sol.parse_solution()
    assert hasattr(solution, "routes")
    assert hasattr(solution, "cost")
    assert hasattr(solution, "num_vehicles")
    # When parsing BKS solution, num_vehicles should match instance.num_vehicles (both are BKS)
    assert solution.num_vehicles == instance.num_vehicles
    # actual number of routes may differ; ensure it's a non-negative integer and matches len(routes)
    assert isinstance(solution.num_vehicles, int) and solution.num_vehicles >= 0
    assert isinstance(solution.routes, list)
    assert len(solution.routes) == solution.num_vehicles
