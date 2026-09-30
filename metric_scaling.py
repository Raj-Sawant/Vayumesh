"""
Metric Scale Recovery & Coordinate Alignment Module for VayuMesh
Handles GPS/IMU fusion, ground plane detection, and axis alignment for metric-scale 3D reconstruction.
"""

import numpy as np
import trimesh
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GPSData:
    """GPS/IMU metadata from drone."""
    latitude: float        # degrees
    longitude: float       # degrees
    altitude: float        # meters (AMSL - Above Mean Sea Level)
    roll: float            # degrees
    pitch: float           # degrees
    yaw: float             # degrees (heading, 0=North)
    rtcm_available: bool = False  # True if RTK correction available
    horizontal_accuracy: float = 1.0  # meters
    vertical_accuracy: float = 1.0    # meters


@dataclass
class CameraPose:
    """Camera extrinsic + GPS metadata for one frame."""
    extrinsic: np.ndarray    # (4, 4) world-to-camera matrix
    intrinsic: np.ndarray    # (3, 3) camera intrinsics
    gps: Optional[GPSData] = None
    frame_idx: int = 0


def extract_translation_rotation(extrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract translation and rotation from 4x4 extrinsic matrix."""
    translation = extrinsic[:3, 3].copy()
    rotation_matrix = extrinsic[:3, :3].copy()
    return translation, rotation_matrix


def build_extrinsic(translation: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Build 4x4 extrinsic matrix from translation and rotation."""
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = translation
    return extrinsic


def gps_to_enu(lat: float, lon: float, alt: float, ref_lat: float, ref_lon: float, ref_alt: float) -> np.ndarray:
    """
    Convert GPS (lat, lon, alt) to local ENU (East, North, Up) coordinates.
    
    Uses WGS84 ellipsoid approximation. Good for local areas (<10km).
    
    Args:
        lat, lon, alt: Target GPS coordinates (degrees, meters)
        ref_lat, ref_lon, ref_alt: Reference/origin GPS coordinates
    
    Returns:
        ENU coordinates in meters [East, North, Up]
    """
    # WGS84 constants
    a = 6378137.0           # semi-major axis (meters)
    f = 1 / 298.257223563   # flattening
    e2 = 2*f - f*f          # eccentricity squared
    
    # Convert to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    ref_lat_rad = np.deg2rad(ref_lat)
    ref_lon_rad = np.deg2rad(ref_lon)
    
    # Radius of curvature in prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    
    # ECEF coordinates
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt) * np.sin(lat_rad)
    
    # Reference ECEF
    N_ref = a / np.sqrt(1 - e2 * np.sin(ref_lat_rad)**2)
    x_ref = (N_ref + ref_alt) * np.cos(ref_lat_rad) * np.cos(ref_lon_rad)
    y_ref = (N_ref + ref_alt) * np.cos(ref_lat_rad) * np.sin(ref_lon_rad)
    z_ref = (N_ref * (1 - e2) + ref_alt) * np.sin(ref_lat_rad)
    
    # Delta ECEF
    dx = x - x_ref
    dy = y - y_ref
    dz = z - z_ref
    
    # ECEF to ENU rotation
    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)
    
    R_ecef_enu = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])
    
    enu = R_ecef_enu @ np.array([dx, dy, dz])
    return enu  # [East, North, Up]


def enu_to_gps(east: float, north: float, up: float, ref_lat: float, ref_lon: float, ref_alt: float) -> Tuple[float, float, float]:
    """Convert local ENU back to GPS coordinates (inverse of gps_to_enu)."""
    # For small areas, use linear approximation
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2*f - f*f
    
    ref_lat_rad = np.deg2rad(ref_lat)
    ref_lon_rad = np.deg2rad(ref_lon)
    N_ref = a / np.sqrt(1 - e2 * np.sin(ref_lat_rad)**2)
    
    # Meters per degree
    m_per_deg_lat = 111132.92 - 559.82 * np.cos(2*ref_lat_rad) + 1.175 * np.cos(4*ref_lat_rad)
    m_per_deg_lon = 111412.84 * np.cos(ref_lat_rad) - 93.5 * np.cos(3*ref_lat_rad)
    
    dlat = north / m_per_deg_lat
    dlon = east / m_per_deg_lon
    dalt = up
    
    return ref_lat + dlat, ref_lon + dlon, ref_alt + dalt


def compute_scale_from_gps(camera_poses: List[CameraPose]) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute metric scale factor from GPS positions of camera centers.
    
    Args:
        camera_poses: List of CameraPose with GPS data
    
    Returns:
        scale_factor: Multiplier to convert VGGT units to meters
        gps_positions_enu: (N, 3) GPS positions in ENU frame
        vggt_positions: (N, 3) VGGT camera centers in reconstruction frame
    """
    valid_poses = [p for p in camera_poses if p.gps is not None]
    if len(valid_poses) < 2:
        raise ValueError("Need at least 2 poses with GPS data for scale estimation")
    
    # Use first GPS as reference
    ref_gps = valid_poses[0].gps
    
    gps_positions = []
    vggt_positions = []
    
    for pose in valid_poses:
        # GPS to ENU
        enu = gps_to_enu(
            pose.gps.latitude, pose.gps.longitude, pose.gps.altitude,
            ref_gps.latitude, ref_gps.longitude, ref_gps.altitude
        )
        gps_positions.append(enu)
        
        # VGGT camera center (translation from extrinsic)
        # Note: VGGT extrinsic is world-to-camera, so camera center = -R^T @ t
        R_wc = pose.extrinsic[:3, :3]
        t_wc = pose.extrinsic[:3, 3]
        cam_center = -R_wc.T @ t_wc
        vggt_positions.append(cam_center)
    
    gps_positions = np.array(gps_positions)  # (N, 3) ENU
    vggt_positions = np.array(vggt_positions)  # (N, 3)
    
    # Align VGGT positions to GPS using Procrustes analysis (similarity transform)
    scale, R_align, t_align = similarity_transform(vggt_positions, gps_positions)
    
    logger.info(f"GPS-based scale factor: {scale:.4f}")
    logger.info(f"GPS positions (ENU): {gps_positions}")
    logger.info(f"VGGT positions (aligned): {(scale * vggt_positions @ R_align.T + t_align)}")
    
    return scale, gps_positions, vggt_positions


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute similarity transform (scale, rotation, translation) to align src to dst.
    Uses Umeyama's method for Procrustes analysis with scaling.
    
    Args:
        src: (N, 3) source points
        dst: (N, 3) destination points
    
    Returns:
        scale: Scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    assert src.shape == dst.shape
    n = src.shape[0]
    
    # Centroids
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    
    # Centered points
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    
    # Covariance matrix
    H = src_centered.T @ dst_centered
    
    # SVD
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Scale
    src_var = np.sum(src_centered**2)
    dst_var = np.sum(dst_centered**2)
    scale = np.sqrt(dst_var / src_var) if src_var > 0 else 1.0
    
    # Translation
    t = dst_centroid - scale * (R @ src_centroid)
    
    return scale, R, t


def detect_ground_plane(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    distance_threshold: float = 0.05,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect ground plane using RANSAC (open3d) or PCA fallback.
    
    Args:
        points: (N, 3) point cloud
        colors: (N, 3) optional colors
        distance_threshold: Max distance for inlier (in VGGT units)
        ransac_n: Points to sample for plane fitting
        num_iterations: RANSAC iterations
        min_points: Minimum inliers for valid plane
    
    Returns:
        plane_model: (4,) [A, B, C, D] for Ax + By + Cz + D = 0
        inlier_indices: Indices of ground points
        ground_points: (M, 3) ground point coordinates
    """
    # Try open3d RANSAC first
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        logger.info(f"Detecting ground plane from {len(points)} points (open3d RANSAC)...")
        
        plane_model, inlier_indices = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        inlier_indices = np.array(inlier_indices)
        
        if len(inlier_indices) < min_points:
            logger.warning(f"Ground plane has only {len(inlier_indices)} inliers (min: {min_points})")
        
        ground_points = points[inlier_indices]
        logger.info(f"Ground plane detected: {len(inlier_indices)} inliers (open3d)")
        logger.info(f"Plane equation: {plane_model[0]:.4f}x + {plane_model[1]:.4f}y + {plane_model[2]:.4f}z + {plane_model[3]:.4f} = 0")
        
        return plane_model, inlier_indices, ground_points
        
    except ImportError:
        logger.warning("open3d not available, using PCA fallback for ground plane")
    except Exception as e:
        logger.warning(f"open3d ground plane failed: {e}, using PCA fallback")
    
    # PCA fallback: use lowest 20% points by Z, fit plane via PCA
    z_vals = points[:, 2]
    threshold = np.percentile(z_vals, 20)
    low_points = points[z_vals <= threshold]
    
    if len(low_points) < 10:
        # Ultimate fallback: horizontal plane at 5th percentile Z
        ground_z = np.percentile(z_vals, 5)
        plane_model = np.array([0.0, 0.0, 1.0, -ground_z])
        inlier_indices = np.where(z_vals <= ground_z + 0.1)[0]
        ground_points = points[inlier_indices]
        logger.info(f"Using horizontal fallback plane at Z={ground_z:.3f}")
        return plane_model, inlier_indices, ground_points
    
    # Center and PCA
    centroid = low_points.mean(axis=0)
    centered = low_points - centroid
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]  # smallest eigenvalue
    
    # Ensure normal points up
    if normal[2] < 0:
        normal = -normal
    
    D = -np.dot(normal, centroid)
    plane_model = np.array([normal[0], normal[1], normal[2], D])
    
    # Inliers: points within distance_threshold of plane
    distances = np.abs(points @ normal + D)
    inlier_indices = np.where(distances <= distance_threshold)[0]
    ground_points = points[inlier_indices]
    
    logger.info(f"Ground plane detected (PCA fallback): {len(inlier_indices)} inliers")
    logger.info(f"Plane equation: {plane_model[0]:.4f}x + {plane_model[1]:.4f}y + {plane_model[2]:.4f}z + {plane_model[3]:.4f} = 0")
    
    return plane_model, inlier_indices, ground_points


def align_to_ground_plane(
    points: np.ndarray,
    plane_model: np.ndarray,
    target_up: np.ndarray = np.array([0, 0, 1])
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align point cloud so ground plane is horizontal (Z-up).
    
    Args:
        points: (N, 3) point cloud
        plane_model: (4,) [A, B, C, D] plane equation
        target_up: Target up direction (default Z-up)
    
    Returns:
        aligned_points: (N, 3) rotated points
        rotation_matrix: (3, 3) rotation applied
    """
    # Plane normal (A, B, C)
    plane_normal = plane_model[:3]
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Ensure normal points "up" (positive Z in target frame)
    if np.dot(plane_normal, target_up) < 0:
        plane_normal = -plane_normal
    
    # Compute rotation to align plane_normal to target_up
    rotation_matrix = rotation_between_vectors(plane_normal, target_up)
    
    # Apply rotation
    aligned_points = points @ rotation_matrix.T
    
    logger.info(f"Ground plane normal: {plane_normal}")
    logger.info(f"Aligned to: {target_up}")
    
    return aligned_points, rotation_matrix


def rotation_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute rotation matrix to align v1 to v2."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    if np.allclose(v1, v2):
        return np.eye(3)
    
    if np.allclose(v1, -v2):
        # 180-degree rotation around any perpendicular axis
        axis = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
        axis = axis - np.dot(axis, v1) * v1
        axis = axis / np.linalg.norm(axis)
        return R.from_rotvec(np.pi * axis).as_matrix()
    
    # General case
    axis = np.cross(v1, v2)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
    return R.from_rotvec(angle * axis).as_matrix()


def align_axes_to_cardinal(
    points: np.ndarray,
    ground_points: np.ndarray,
    camera_poses: Optional[List[CameraPose]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align horizontal axes to cardinal directions (North/East) using GPS yaw.
    If no GPS, align to principal component axes of ground plane.
    
    Args:
        points: (N, 3) full point cloud
        ground_points: (M, 3) ground plane points
        camera_poses: Optional camera poses with GPS yaw
    
    Returns:
        aligned_points: (N, 3) points with X=East, Y=North, Z=Up
        rotation_matrix: (3, 3) horizontal rotation applied
    """
    # Project ground points to XY plane
    ground_xy = ground_points[:, :2]
    
    if len(ground_xy) < 10:
        logger.warning("Insufficient ground points for axis alignment")
        return points, np.eye(3)
    
    # PCA on ground plane to find principal directions
    centroid = ground_xy.mean(axis=0)
    centered = ground_xy - centroid
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Principal axes (sorted by eigenvalue)
    principal_axes = eigvecs[:, np.argsort(eigvals)[::-1]]  # (2, 2)
    
    # First principal axis = longest extent of building
    # We want to align this to either X or Y axis
    # Use GPS yaw if available to determine which is North
    
    if camera_poses is not None:
        # Average GPS yaw from camera poses
        yaws = [np.deg2rad(p.gps.yaw) for p in camera_poses if p.gps is not None]
        if yaws:
            mean_yaw = np.mean(yaws)
            # GPS yaw: 0=North, 90=East (clockwise from North)
            # We want Y=North, X=East
            north_dir = np.array([np.sin(mean_yaw), np.cos(mean_yaw)])
            east_dir = np.array([np.cos(mean_yaw), -np.sin(mean_yaw)])
            
            # Project principal axes to GPS directions
            # Align first principal axis to closest cardinal direction
            axis1 = principal_axes[:, 0]
            dot_north = np.dot(axis1, north_dir)
            dot_east = np.dot(axis1, east_dir)
            
            if abs(dot_north) > abs(dot_east):
                # Align to North
                target_x = east_dir if dot_north > 0 else -east_dir
                target_y = north_dir if dot_north > 0 else -north_dir
            else:
                # Align to East
                target_x = east_dir if dot_east > 0 else -east_dir
                target_y = north_dir if dot_east > 0 else -north_dir
        else:
            # No GPS yaw, align to principal axes
            target_x = principal_axes[:, 0]
            target_y = principal_axes[:, 1]
    else:
        # No GPS, align to principal axes
        target_x = principal_axes[:, 0]
        target_y = principal_axes[:, 1]
    
    # Build 2D rotation
    current_x = principal_axes[:, 0]
    current_y = principal_axes[:, 1]
    
    # Rotation from current to target
    cos_theta = np.dot(current_x, target_x)
    sin_theta = np.dot(current_x, target_y)
    theta = np.arctan2(sin_theta, cos_theta)
    
    R_2d = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
    
    # Extend to 3D
    R_3d = np.eye(3)
    R_3d[:2, :2] = R_2d
    
    aligned_points = points @ R_3d.T
    
    logger.info(f"Horizontal rotation angle: {np.rad2deg(theta):.2f} deg")
    
    return aligned_points, R_3d


def apply_metric_scale(
    mesh: trimesh.Trimesh,
    scale_factor: float,
    translation: Optional[np.ndarray] = None
) -> trimesh.Trimesh:
    """
    Apply metric scale and translation to mesh.
    
    Args:
        mesh: Input mesh (in VGGT units)
        scale_factor: Scale multiplier (VGGT units -> meters)
        translation: Optional translation to apply after scaling
    
    Returns:
        Scaled mesh (in meters)
    """
    scaled_mesh = mesh.copy()
    transform = np.eye(4)
    transform[:3, :3] *= scale_factor
    if translation is not None:
        transform[:3, 3] = translation
    scaled_mesh.apply_transform(transform)
    
    logger.info(f"Applied metric scale: {scale_factor:.4f} (VGGT units -> meters)")
    return scaled_mesh


def full_metric_pipeline(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    camera_poses: List[CameraPose],
    mesh: Optional[trimesh.Trimesh] = None
) -> Dict:
    """
    Complete metric pipeline: GPS scale -> ground plane -> axis alignment.
    
    Args:
        points: (N, 3) point cloud from VGGT
        colors: (N, 3) colors
        camera_poses: List of camera poses with GPS
        mesh: Optional mesh to transform
    
    Returns:
        Dictionary with scaled points, mesh, transformation info
    """
    # Step 1: Compute GPS scale
    scale_factor, gps_positions, vggt_positions = compute_scale_from_gps(camera_poses)
    
    # Step 2: Detect ground plane
    plane_model, inlier_indices, ground_points = detect_ground_plane(points, colors)
    
    # Step 3: Align to ground plane (Z-up)
    aligned_points, R_ground = align_to_ground_plane(points, plane_model)
    aligned_ground_points = ground_points @ R_ground.T
    
    # Step 4: Align horizontal axes to cardinal directions
    final_points, R_horizontal = align_axes_to_cardinal(aligned_points, aligned_ground_points, camera_poses)
    
    # Combined rotation
    R_total = R_horizontal @ R_ground
    
    # Step 5: Apply metric scale
    final_points = final_points * scale_factor
    
    # Transform mesh if provided
    final_mesh = None
    if mesh is not None:
        final_mesh = mesh.copy()
        # Apply rotation
        R_4x4 = np.eye(4)
        R_4x4[:3, :3] = R_total
        final_mesh.apply_transform(R_4x4)
        # Apply scale
        final_mesh = apply_metric_scale(final_mesh, scale_factor)
    
    # Transform camera poses
    transformed_poses = []
    for pose in camera_poses:
        new_pose = CameraPose(
            extrinsic=pose.extrinsic.copy(),
            intrinsic=pose.intrinsic.copy(),
            gps=pose.gps,
            frame_idx=pose.frame_idx
        )
        # Apply same transformation to camera centers
        R_wc = pose.extrinsic[:3, :3]
        t_wc = pose.extrinsic[:3, 3]
        cam_center = -R_wc.T @ t_wc
        new_cam_center = (cam_center @ R_total.T) * scale_factor
        new_R_wc = R_wc @ R_total.T
        new_t_wc = -new_R_wc @ new_cam_center
        new_pose.extrinsic[:3, :3] = new_R_wc
        new_pose.extrinsic[:3, 3] = new_t_wc
        transformed_poses.append(new_pose)
    
    return {
        "points": final_points,
        "colors": colors,
        "mesh": final_mesh,
        "camera_poses": transformed_poses,
        "scale_factor": scale_factor,
        "ground_plane_model": plane_model,
        "ground_rotation": R_ground,
        "horizontal_rotation": R_horizontal,
        "total_rotation": R_total,
        "gps_positions_enu": gps_positions,
        "vggt_positions": vggt_positions
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Metric scaling from GPS")
    parser.add_argument("--npz", required=True, help="VGGT predictions .npz")
    parser.add_argument("--gps", help="GPS data JSON file")
    parser.add_argument("--output", default="scaled_points.npz", help="Output scaled points")
    
    args = parser.parse_args()
    
    data = np.load(args.npz)
    points = data["world_points_from_depth"].reshape(-1, 3)
    colors = None
    if "images" in data:
        imgs = data["images"]
        if imgs.ndim == 4 and imgs.shape[1] == 3:
            imgs = imgs.transpose(0, 2, 3, 1)
        colors = imgs.reshape(-1, 3)
    
    # TODO: Load GPS data and run pipeline
    print("Load GPS data and camera poses to run metric scaling")