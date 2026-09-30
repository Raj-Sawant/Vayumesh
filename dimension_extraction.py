"""
Building Dimension Extraction Module for VayuMesh
Extracts width, height, length from metric-scaled mesh/point cloud.
"""

import numpy as np
import trimesh
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial import ConvexHull
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BuildingDimensions:
    """Dimensions of a building/structure."""
    width: float      # X extent (East-West) in meters
    length: float     # Y extent (North-South) in meters
    height: float     # Z extent (Ground to roof) in meters
    volume: float     # Approximate volume in m³
    footprint_area: float  # Ground footprint area in m²
    centroid: np.ndarray   # (3,) center point [x, y, z]
    bbox_corners: np.ndarray  # (8, 3) bounding box corners
    oriented_bbox: trimesh.primitives.Box  # Oriented bounding box


def compute_aabb(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Axis-Aligned Bounding Box (AABB).
    
    Args:
        points: (N, 3) point cloud
    
    Returns:
        min_bounds: (3,) minimum corner
        max_bounds: (3,) maximum corner
        centroid: (3,) center
    """
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    centroid = (min_bounds + max_bounds) / 2
    return min_bounds, max_bounds, centroid


def compute_obb(points: np.ndarray) -> trimesh.primitives.Box:
    """
    Compute Oriented Bounding Box (OBB) using PCA/trimesh.
    
    Args:
        points: (N, 3) point cloud
    
    Returns:
        trimesh Box primitive (oriented)
    """
    # Use trimesh's OBB computation
    mesh = trimesh.PointCloud(points)
    obb = mesh.bounding_box_oriented
    return obb


def extract_building_dimensions(
    mesh: trimesh.Trimesh,
    ground_z: Optional[float] = None
) -> BuildingDimensions:
    """
    Extract building dimensions from a metric-scaled mesh.
    
    Args:
        mesh: Watertight mesh in metric scale (meters), Z-up
        ground_z: Optional ground plane Z coordinate (if known)
    
    Returns:
        BuildingDimensions object
    """
    # Get vertices
    vertices = mesh.vertices
    
    # If ground_z not provided, estimate from lowest vertices
    if ground_z is None:
        # Use 5th percentile to avoid outliers
        ground_z = np.percentile(vertices[:, 2], 5)
    
    # Height = max Z - ground Z
    height = vertices[:, 2].max() - ground_z
    
    # Project to ground plane for footprint
    ground_vertices = vertices[vertices[:, 2] < ground_z + 0.5]  # Within 0.5m of ground
    if len(ground_vertices) < 10:
        ground_vertices = vertices
    
    # Footprint using 2D convex hull
    footprint_2d = ground_vertices[:, :2]
    hull = ConvexHull(footprint_2d)
    footprint_area = hull.volume  # 2D volume = area
    
    # OBB for oriented dimensions
    obb = compute_obb(vertices)
    
    # OBB extents (width, length, height) - sorted
    extents = obb.extents.copy()
    # Ensure Z is height (already Z-up from metric scaling)
    # extents[0], extents[1] are horizontal, extents[2] is vertical
    width = extents[0]
    length = extents[1]
    
    # Ensure width <= length (convention)
    if width > length:
        width, length = length, width
    
    # Volume
    volume = mesh.volume if mesh.is_watertight else width * length * height
    
    # Bounding box corners
    bbox_corners = obb.vertices
    
    # Centroid
    centroid = obb.centroid
    
    dims = BuildingDimensions(
        width=float(width),
        length=float(length),
        height=float(height),
        volume=float(volume),
        footprint_area=float(footprint_area),
        centroid=centroid,
        bbox_corners=bbox_corners,
        oriented_bbox=obb
    )
    
    logger.info(f"Building Dimensions:")
    logger.info(f"  Width (X):  {dims.width:.2f} m")
    logger.info(f"  Length (Y): {dims.length:.2f} m")
    logger.info(f"  Height (Z): {dims.height:.2f} m")
    logger.info(f"  Footprint:  {dims.footprint_area:.2f} m²")
    logger.info(f"  Volume:     {dims.volume:.2f} m³")
    logger.info(f"  Centroid:   [{dims.centroid[0]:.2f}, {dims.centroid[1]:.2f}, {dims.centroid[2]:.2f}]")
    
    return dims


def extract_multi_building_dimensions(
    mesh: trimesh.Trimesh,
    ground_z: Optional[float] = None,
    min_volume: float = 10.0,
    cluster_tolerance: float = 2.0
) -> List[BuildingDimensions]:
    """
    Extract dimensions for multiple buildings by clustering.
    
    Args:
        mesh: Input mesh (may contain multiple buildings)
        ground_z: Ground plane Z coordinate
        min_volume: Minimum volume to consider as building
        cluster_tolerance: Distance threshold for clustering (meters)
    
    Returns:
        List of BuildingDimensions for each detected building
    """
    # Split mesh into connected components
    components = mesh.split(only_watertight=False)
    
    buildings = []
    for comp in components:
        if comp.volume < min_volume:
            continue
        
        try:
            dims = extract_building_dimensions(comp, ground_z)
            buildings.append(dims)
        except Exception as e:
            logger.warning(f"Failed to extract dimensions for component: {e}")
    
    # Sort by volume (largest first)
    buildings.sort(key=lambda b: b.volume, reverse=True)
    
    logger.info(f"Detected {len(buildings)} buildings")
    for i, b in enumerate(buildings):
        logger.info(f"  Building {i+1}: {b.width:.1f}x{b.length:.1f}x{b.height:.1f}m, Vol={b.volume:.1f}m³")
    
    return buildings


def compute_roof_height_profile(
    mesh: trimesh.Trimesh,
    ground_z: float,
    grid_resolution: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute height map / roof profile on a regular grid.
    
    Args:
        mesh: Watertight mesh
        ground_z: Ground plane Z
        grid_resolution: Grid cell size in meters
    
    Returns:
        x_coords, y_coords, height_grid
    """
    bounds = mesh.bounds
    x_min, y_min = bounds[0, :2]
    x_max, y_max = bounds[1, :2]
    
    # Create grid
    x_coords = np.arange(x_min, x_max + grid_resolution, grid_resolution)
    y_coords = np.arange(y_min, y_max + grid_resolution, grid_resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Ray casting from above to find roof height
    height_grid = np.full_like(X, ground_z, dtype=float)
    
    # Sample points on grid and cast rays downward
    origins = np.column_stack([X.ravel(), Y.ravel(), np.full_like(X.ravel(), bounds[1, 2] + 10)])
    directions = np.tile([0, 0, -1], (len(origins), 1))
    
    # Use trimesh ray intersection
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        origins, directions, multiple_hits=False
    )
    
    if len(locations) > 0:
        hit_heights = locations[:, 2]
        height_grid.ravel()[index_ray] = hit_heights
    
    # Height relative to ground
    height_grid = height_grid - ground_z
    height_grid[height_grid < 0] = 0
    
    return x_coords, y_coords, height_grid


def export_dimensions_report(
    dimensions: BuildingDimensions,
    output_path: str,
    include_geojson: bool = True
) -> str:
    """
    Export dimensions report as JSON and optionally GeoJSON.
    
    Args:
        dimensions: BuildingDimensions object
        output_path: Output file path (without extension)
        include_geojson: Also export GeoJSON footprint
    
    Returns:
        Path to JSON report
    """
    import json
    
    report = {
        "width_m": dimensions.width,
        "length_m": dimensions.length,
        "height_m": dimensions.height,
        "volume_m3": dimensions.volume,
        "footprint_area_m2": dimensions.footprint_area,
        "centroid": dimensions.centroid.tolist(),
        "bbox_corners": dimensions.bbox_corners.tolist(),
        "oriented_bbox": {
            "transform": dimensions.oriented_bbox.transform.tolist() if hasattr(dimensions.oriented_bbox, 'transform') else None,
            "extents": dimensions.oriented_bbox.extents.tolist() if hasattr(dimensions.oriented_bbox, 'extents') else None
        }
    }
    
    json_path = f"{output_path}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Dimensions report saved to {json_path}")
    
    if include_geojson:
        # Export footprint as GeoJSON polygon
        footprint_coords = dimensions.bbox_corners[dimensions.oriented_bbox.faces[0]][:, :2].tolist()
        # Close the polygon
        footprint_coords.append(footprint_coords[0])
        
        geojson = {
            "type": "Feature",
            "properties": {
                "width_m": dimensions.width,
                "length_m": dimensions.length,
                "height_m": dimensions.height,
                "volume_m3": dimensions.volume,
                "footprint_area_m2": dimensions.footprint_area
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [footprint_coords]
            }
        }
        
        geojson_path = f"{output_path}.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        logger.info(f"GeoJSON footprint saved to {geojson_path}")
    
    return json_path


def create_dimension_visualization(
    mesh: trimesh.Trimesh,
    dimensions: BuildingDimensions,
    output_path: str
) -> trimesh.Scene:
    """
    Create a visualization scene with dimension annotations.
    
    Args:
        mesh: Original mesh
        dimensions: BuildingDimensions
        output_path: Output GLB path
    
    Returns:
        trimesh.Scene with annotations
    """
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    
    # Add oriented bounding box wireframe
    obb = dimensions.oriented_bbox
    obb_mesh = trimesh.creation.box(extents=obb.extents, transform=obb.transform)
    obb_mesh = trimesh.Trimesh(
        vertices=obb_mesh.vertices,
        faces=obb_mesh.edges_unique,
        process=False
    )
    obb_mesh.visual.vertex_colors = [0, 255, 0, 255]  # Green
    scene.add_geometry(obb_mesh)
    
    # Add dimension lines
    corners = dimensions.bbox_corners
    # Height line (vertical at corner 0)
    height_line = trimesh.load_path(np.array([corners[0], corners[0] + [0, 0, dimensions.height]]))
    height_line.colors = [255, 0, 0, 255]  # Red
    scene.add_geometry(height_line)
    
    # Width line (X direction)
    width_line = trimesh.load_path(np.array([corners[0], corners[1]]))
    width_line.colors = [0, 0, 255, 255]  # Blue
    scene.add_geometry(width_line)
    
    # Length line (Y direction)
    length_line = trimesh.load_path(np.array([corners[0], corners[3]]))
    length_line.colors = [255, 255, 0, 255]  # Yellow
    scene.add_geometry(length_line)
    
    scene.export(output_path)
    logger.info(f"Dimension visualization saved to {output_path}")
    
    return scene


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract building dimensions from mesh")
    parser.add_argument("--mesh", required=True, help="Input mesh file (.obj, .ply, .glb)")
    parser.add_argument("--ground-z", type=float, help="Ground plane Z coordinate")
    parser.add_argument("--output", default="dimensions", help="Output prefix")
    parser.add_argument("--multi", action="store_true", help="Extract multiple buildings")
    
    args = parser.parse_args()
    
    mesh = trimesh.load(args.mesh)
    
    if args.multi:
        buildings = extract_multi_building_dimensions(mesh, args.ground_z)
        for i, dims in enumerate(buildings):
            export_dimensions_report(dims, f"{args.output}_building_{i}")
            create_dimension_visualization(mesh, dims, f"{args.output}_building_{i}_viz.glb")
    else:
        dims = extract_building_dimensions(mesh, args.ground_z)
        export_dimensions_report(dims, args.output)
        create_dimension_visualization(mesh, dims, f"{args.output}_viz.glb")