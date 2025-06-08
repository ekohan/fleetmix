"""
Bidirectional projection between VRP Euclidean and lat/lon space.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np


@dataclass
class GeoBounds:
    """Geographic boundaries for the projection"""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """Returns the center point of the bounded area"""
        return (
            (self.min_lat + self.max_lat) / 2,
            (self.min_lon + self.max_lon) / 2
        )
    
    @property
    def lat_span(self) -> float:
        """Returns the latitude span"""
        return self.max_lat - self.min_lat
    
    @property
    def lon_span(self) -> float:
        """Returns the longitude span"""
        return self.max_lon - self.min_lon


class CoordinateConverter:
    """
    Converts coordinates between CVRP Euclidean space and geographic space.
    
    The conversion maintains relative distances as much as possible by:
    1. Centering the CVRP coordinates around (0,0)
    2. Scaling them to fit within the specified geographic bounds
    3. Applying a cosine correction for longitude distances at the target latitude
    """
    
    def __init__(
        self,
        cvrp_coords: Dict[int, Tuple[float, float]],
        geo_bounds: Optional[GeoBounds] = None
    ):
        """
        Initialize the converter with CVRP coordinates and optional geographic bounds.
        
        Args:
            cvrp_coords: Dictionary of node_id -> (x, y) coordinates from CVRP instance
            geo_bounds: Geographic boundaries for projection. If None, uses default area.
        """
        # Set default bounds if none provided (Bogota Metro Area bounding box)
        if geo_bounds is None:
            geo_bounds = GeoBounds(
                min_lat=4.3333,
                max_lat=4.9167,
                min_lon=-74.3500,
                max_lon=-73.9167
            )
        self.geo_bounds = geo_bounds
        
        # Calculate CVRP coordinate bounds
        coords = np.array(list(cvrp_coords.values()))
        self.min_x = coords[:, 0].min()
        self.max_x = coords[:, 0].max()
        self.min_y = coords[:, 1].min()
        self.max_y = coords[:, 1].max()
        
        # Calculate scaling factors
        # Use cosine correction for longitude distances at the center latitude
        cos_lat = np.cos(np.radians(geo_bounds.center[0]))
        self.x_scale = geo_bounds.lon_span * cos_lat / (self.max_x - self.min_x)
        self.y_scale = geo_bounds.lat_span / (self.max_y - self.min_y)
        
        # Use the smaller scale to maintain aspect ratio
        self.scale = min(self.x_scale, self.y_scale)
    
    def to_geographic(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert CVRP coordinates to geographic coordinates.
        
        Args:
            x: CVRP x-coordinate
            y: CVRP y-coordinate
            
        Returns:
            Tuple of (latitude, longitude)
        """
        # Center the coordinates
        x_centered = x - (self.max_x + self.min_x) / 2
        y_centered = y - (self.max_y + self.min_y) / 2
        
        # Scale and convert to geographic coordinates
        # Use the same scale factor for both dimensions to maintain aspect ratio
        lat = self.geo_bounds.center[0] + (y_centered * self.scale)
        # Apply cosine correction only to longitude conversion
        cos_lat = np.cos(np.radians(self.geo_bounds.center[0]))
        lon = self.geo_bounds.center[1] + (x_centered * self.scale / cos_lat)
        
        return lat, lon
    
    def to_cvrp(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert geographic coordinates to CVRP coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Tuple of (x, y) coordinates in CVRP space
        """
        # Convert from geographic to centered coordinates
        y_centered = (lat - self.geo_bounds.center[0]) / self.scale
        x_centered = (lon - self.geo_bounds.center[1]) * np.cos(np.radians(lat)) / self.scale
        
        # Un-center the coordinates
        x = x_centered + (self.max_x + self.min_x) / 2
        y = y_centered + (self.max_y + self.min_y) / 2
        
        return x, y
    
    def convert_all_coordinates(self, coords: Dict[int, Tuple[float, float]], to_geographic: bool = True) -> Dict[int, Tuple[float, float]]:
        """
        Convert all coordinates in a dictionary.
        
        Args:
            coords: Dictionary of node_id -> (x, y) or (lat, lon) coordinates
            to_geographic: If True, converts from CVRP to geographic. If False, converts from geographic to CVRP.
            
        Returns:
            Dictionary of converted coordinates
        """
        converted = {}
        for node_id, (x, y) in coords.items():
            if to_geographic:
                converted[node_id] = self.to_geographic(x, y)
            else:
                converted[node_id] = self.to_cvrp(x, y)
        return converted


def validate_conversion(converter: CoordinateConverter, original_coords: Dict[int, Tuple[float, float]]) -> None:
    """
    Validate the coordinate conversion by checking if relative distances are preserved.
    
    Args:
        converter: Initialized CoordinateConverter
        original_coords: Original CVRP coordinates
    """
    from haversine import haversine
    
    # Convert to geographic coordinates
    geo_coords = converter.convert_all_coordinates(original_coords, to_geographic=True)
    
    # Check a few random pairs of points
    import random
    nodes = list(original_coords.keys())
    pairs = [(nodes[i], nodes[j]) 
             for i in range(len(nodes)) 
             for j in range(i+1, len(nodes))]
    
    sample_size = min(10, len(pairs))
    random_pairs = random.sample(pairs, sample_size)
    
    print("\nValidating coordinate conversion:")
    print("Comparing relative distances between random pairs of points")
    print("-" * 60)
    
    for node1, node2 in random_pairs:
        # Calculate Euclidean distance in CVRP space
        coord1 = np.array(original_coords[node1])
        coord2 = np.array(original_coords[node2])
        euclidean_dist = np.linalg.norm(coord1 - coord2)
        
        # Calculate haversine distance in geographic space
        geo1 = geo_coords[node1]
        geo2 = geo_coords[node2]
        haversine_dist = haversine(geo1, geo2)
        
        # Calculate the ratio between distances
        ratio = haversine_dist / euclidean_dist if euclidean_dist > 0 else 0
        
        print(f"Nodes {node1}-{node2}:")
        print(f"  CVRP distance: {euclidean_dist:.2f}")
        print(f"  Geographic distance (km): {haversine_dist:.2f}")
        print(f"  Ratio: {ratio:.4f}")
        print()


if __name__ == "__main__":
    # Example usage
    example_coords = {
        1: (0.0, 0.0),    # Depot
        2: (100.0, 100.0),
        3: (200.0, 50.0),
        4: (150.0, 150.0)
    }
    
    # Create converter with default bounds (Boston area)
    converter = CoordinateConverter(example_coords)
    
    # Convert coordinates
    geo_coords = converter.convert_all_coordinates(example_coords)
    
    print("Original CVRP coordinates:")
    for node, coord in example_coords.items():
        print(f"Node {node}: {coord}")
    
    print("\nConverted geographic coordinates:")
    for node, coord in geo_coords.items():
        print(f"Node {node}: {coord}")
    
    # Validate the conversion
    validate_conversion(converter, example_coords) 