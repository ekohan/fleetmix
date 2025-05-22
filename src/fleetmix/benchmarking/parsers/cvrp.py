"""Parser for CVRP instance files."""

import logging
from pathlib import Path
import vrplib

from ..models import CVRPInstance, CVRPSolution

logger = logging.getLogger(__name__)

class CVRPParser:
    """Parser for standard CVRP instance files using VRPLIB."""
    
    def __init__(self, file_path: str):
        """Initialize parser with instance file path."""
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"CVRP instance file not found: {file_path}")
        
        self.instance_name = self.file_path.stem
    
    def parse(self) -> CVRPInstance:
        """Parse CVRP instance file using VRPLIB."""
        # Extract instance details from name
        name = self.instance_name
        num_vehicles = int(name.split('-k')[-1]) if '-k' in name else None
        
        # Use VRPLIB to parse the instance
        instance_data = vrplib.read_instance(str(self.file_path))
        
        # Convert node coordinates to our format
        # VRPLIB uses 0-based indexing, we need 1-based
        coordinates = {
            i + 1: tuple(coord) 
            for i, coord in enumerate(instance_data['node_coord'])
        }
        
        # Convert demands to our format
        demands = {
            i + 1: float(demand)
            for i, demand in enumerate(instance_data['demand'])
        } if 'demand' in instance_data else {}
        
        # Get depot ID (VRPLIB uses 0-based indexing)
        depot_id = instance_data.get('depot', [0])[0] + 1
        
        logger.info(
            f"Parsed CVRP instance {self.instance_name}: "
            f"{len(coordinates)} nodes, capacity={instance_data['capacity']}"
        )
        
        return CVRPInstance(
            name=name,
            dimension=len(coordinates),
            capacity=instance_data['capacity'],
            depot_id=depot_id,
            coordinates=coordinates,
            demands=demands,
            edge_weight_type=instance_data.get('edge_weight_type', 'EUC_2D'),
            num_vehicles=num_vehicles
        )
    
    def parse_solution(self) -> CVRPSolution:
        """Parse corresponding .sol file using VRPLIB."""
        solution_path = self.file_path.with_suffix('.sol')
        if not solution_path.exists():
            raise FileNotFoundError(f"Solution file not found: {solution_path}")
        
        # Use VRPLIB to parse the solution
        solution_data = vrplib.read_solution(str(solution_path))
        
        # Convert to 1-based indexing if necessary
        routes = [
            [node + 1 for node in route]
            for route in solution_data['routes']
        ]
        
        actual_vehicles = len(routes)
        expected_vehicles = int(self.instance_name.split('-k')[-1])
        
        if actual_vehicles != expected_vehicles:
            logger.warning(
                f"Number of routes ({actual_vehicles}) "
                f"differs from instance name k{expected_vehicles}"
            )
        
        return CVRPSolution(
            routes=routes,
            cost=solution_data['cost'],
            num_vehicles=actual_vehicles,
            expected_vehicles=expected_vehicles
        )

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Parse CVRP instance and solution files')
    parser.add_argument('--instance', 
                       default='X-n106-k14',
                       help='Name of the instance file (without extension)')
    
    args = parser.parse_args()
    
    # Construct file path
    instance_path = Path(__file__).parent.parent / 'datasets' / 'cvrp' / f'{args.instance}.vrp'
    
    # Parse instance and solution
    try:
        parser = CVRPParser(str(instance_path))
        instance = parser.parse()
        solution = parser.parse_solution()
        
        print(f"\nInstance details:")
        print(f"Name: {instance.name}")
        print(f"Dimension: {instance.dimension}")
        print(f"Capacity: {instance.capacity}")
        print(f"Number of vehicles: {instance.num_vehicles}")
        
        print(f"\nSolution details:")
        print(f"Number of routes: {solution.num_vehicles}")
        print(f"Total cost: {solution.cost}")
        print(f"Routes:")
        for i, route in enumerate(solution.routes, 1):
            print(f"Route #{i}: {route}")
            
    except Exception as e:
        print(f"Error: {str(e)}") 