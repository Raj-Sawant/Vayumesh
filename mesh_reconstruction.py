"""
Poisson Surface Reconstruction Module for VayuMesh
Converts VGGT point cloud to watertight mesh with normals.
"""

import numpy as np
import open3d as o3d
import trimesh
from typing import Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def denoise_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply Statistical Outlier Removal (SOR) to clean floating artifacts.
    
    Args:
        points: (N, 3) point coordinates
        colors: (N, 3) optional RGB colors [0,1]
        nb_neighbors: Number of neighbors to analyze for each point
        std_ratio: Standard deviation ratio threshold
    
    Returns:
        Cleaned points and colors
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    logger.info(f"Denoising point cloud: {len(points)} points")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    cleaned_pcd = pcd.select_by_index(ind)
    
    cleaned_points = np.asarray(cleaned_pcd.points)
    cleaned_colors = np.asarray(cleaned_pcd.colors) if colors is not None else None
    
    logger.info(f"After denoising: {len(cleaned_points)} points (removed {len(points) - len(cleaned_points)})")
    return cleaned_points, cleaned_colors


def estimate_normals(
    points: np.ndarray,
    radius: float = 0.1,
    max_nn: int = 30
) -> np.ndarray:
    """
    Estimate normals for point cloud using Open3D.
    
    Args:
        points: (N, 3) point coordinates
        radius: Search radius for normal estimation
        max_nn: Maximum number of neighbors
    
    Returns:
        (N, 3) normal vectors
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_consistent_tangent_plane(k=10)
    
    return np.asarray(pcd.normals)


def poisson_reconstruction(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    depth: int = 9,
    width: int = 0,
    scale: float = 1.1,
    linear_fit: bool = False,
    density_threshold: float = 0.1
) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """
    Run Poisson Surface Reconstruction to generate watertight mesh.
    
    Args:
        points: (N, 3) point coordinates
        normals: (N, 3) optional pre-computed normals
        depth: Octree depth (8-10 typical, higher = more detail but more memory)
        width: Gauss-Seidel solver width (0 = auto)
        scale: Scale factor for reconstruction bounding box
        linear_fit: Use linear interpolation for better accuracy
        density_threshold: Remove low-density vertices (0.1 = remove bottom 10%)
    
    Returns:
        Triangle mesh and vertex densities
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        logger.info("Estimating normals for Poisson reconstruction...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=10)
    
    logger.info(f"Running Poisson reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )
    
    densities = np.asarray(densities)
    
    if density_threshold > 0:
        threshold = np.quantile(densities, density_threshold)
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        logger.info(f"Removed {vertices_to_remove.sum()} low-density vertices")
    
    mesh.compute_vertex_normals()
    logger.info(f"Poisson mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    return mesh, densities


def mesh_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    """Convert Open3D mesh to trimesh for further processing."""
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    vertex_normals = np.asarray(o3d_mesh.vertex_normals) if o3d_mesh.has_vertex_normals() else None
    
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        process=False
    )
    return mesh


def trimesh_to_o3d(tri_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """Convert trimesh to Open3D mesh."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    if hasattr(tri_mesh, 'vertex_normals') and tri_mesh.vertex_normals is not None:
        mesh.vertex_normals = o3d.utility.Vector3dVector(tri_mesh.vertex_normals)
    else:
        mesh.compute_vertex_normals()
    return mesh


def clean_mesh(
    mesh: trimesh.Trimesh,
    min_component_size: int = 100
) -> trimesh.Trimesh:
    """
    Clean mesh: remove disconnected components, degenerate faces, duplicate vertices.
    
    Args:
        mesh: Input trimesh
        min_component_size: Minimum faces for a component to keep
    
    Returns:
        Cleaned mesh
    """
    logger.info("Cleaning mesh...")
    
    # Remove degenerate faces (trimesh >= 4.x renamed the method)
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    else:
        # trimesh 4.x+: keep only non-degenerate faces via mask
        mask = mesh.nondegenerate_faces()
        mesh.update_faces(mask)
    
    # Remove duplicate vertices
    if hasattr(mesh, 'merge_vertices'):
        mesh.merge_vertices()
    else:
        mesh = mesh.process(merge_tex=True, merge_norm=True)
    
    # Remove small disconnected components
    if not mesh.is_watertight:
        logger.warning("Mesh is not watertight")
    
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        logger.info(f"Found {len(components)} components, keeping largest")
        sizes = [len(c.faces) for c in components]
        largest_idx = np.argmax(sizes)
        mesh = components[largest_idx]
    
    # Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()
    
    logger.info(f"Cleaned mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def export_mesh(
    mesh: trimesh.Trimesh,
    output_path: str,
    file_type: str = "obj"
) -> str:
    """Export mesh to file."""
    if file_type == "obj":
        mesh.export(output_path)
    elif file_type == "ply":
        mesh.export(output_path)
    elif file_type == "glb":
        mesh.export(output_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    logger.info(f"Mesh exported to {output_path}")
    return output_path


def full_reconstruction_pipeline(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    denoise: bool = True,
    poisson_depth: int = 9,
    density_threshold: float = 0.1,
    clean: bool = True
) -> trimesh.Trimesh:
    """
    Complete pipeline: denoise -> estimate normals -> Poisson -> clean.
    
    Args:
        points: (N, 3) raw point cloud from VGGT
        colors: (N, 3) optional colors [0,1]
        denoise: Apply statistical outlier removal
        poisson_depth: Octree depth for Poisson
        density_threshold: Low-density vertex removal threshold
        clean: Apply mesh cleaning
    
    Returns:
        Watertight trimesh mesh
    """
    # Step 1: Denoise
    if denoise:
        points, colors = denoise_point_cloud(points, colors)
    
    # Step 2: Estimate normals
    normals = estimate_normals(points)
    
    # Step 3: Poisson reconstruction
    o3d_mesh, densities = poisson_reconstruction(
        points, normals, 
        depth=poisson_depth, 
        density_threshold=density_threshold
    )
    
    # Step 4: Convert to trimesh
    mesh = mesh_to_trimesh(o3d_mesh)
    
    # Step 5: Clean mesh
    if clean:
        mesh = clean_mesh(mesh)
    
    return mesh


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Poisson reconstruction from point cloud")
    parser.add_argument("--input", required=True, help="Input .npz or .ply point cloud")
    parser.add_argument("--output", default="reconstruction_mesh.obj", help="Output mesh file")
    parser.add_argument("--depth", type=int, default=9, help="Poisson depth")
    parser.add_argument("--no-denoise", action="store_true", help="Skip denoising")
    parser.add_argument("--density-threshold", type=float, default=0.1, help="Density threshold quantile")
    
    args = parser.parse_args()
    
    # Load point cloud
    if args.input.endswith(".npz"):
        data = np.load(args.input)
        points = data["world_points_from_depth"].reshape(-1, 3)
        colors = None
        if "images" in data:
            imgs = data["images"]
            if imgs.ndim == 4 and imgs.shape[1] == 3:
                imgs = imgs.transpose(0, 2, 3, 1)
            colors = imgs.reshape(-1, 3)
    elif args.input.endswith(".ply"):
        pcd = o3d.io.read_point_cloud(args.input)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    else:
        raise ValueError("Input must be .npz or .ply")
    
    # Run pipeline
    mesh = full_reconstruction_pipeline(
        points, colors,
        denoise=not args.no_denoise,
        poisson_depth=args.depth,
        density_threshold=args.density_threshold
    )
    
    export_mesh(mesh, args.output)