"""
VayuMesh: Video → Metric 3D Solid Pipeline
Stage 1: VGGT Inference (Point Cloud)
Stage 2: Poisson Reconstruction (Watertight Mesh)
Stage 3: Metric Scaling + Dimension Extraction (Solid with Dimensions)
"""

import os
import sys
import gc
import argparse
import json
from typing import Optional, List

import cv2
import numpy as np
import torch

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# New pipeline modules (with optional mesh_reconstruction)
try:
    from mesh_reconstruction import full_reconstruction_pipeline, export_mesh
    MESH_AVAILABLE = True
except ImportError as e:
    print("WARNING: Mesh reconstruction unavailable: " + str(e))
    MESH_AVAILABLE = False
    full_reconstruction_pipeline = None
    export_mesh = None

from metric_scaling import (
    CameraPose, GPSData, full_metric_pipeline, compute_scale_from_gps,
    detect_ground_plane, align_to_ground_plane, align_axes_to_cardinal
)
from dimension_extraction import (
    extract_building_dimensions, export_dimensions_report,
    create_dimension_visualization, compute_roof_height_profile
)


# ---------------------------------------------------------------------------
# Step 1 – Extract frames from video
# ---------------------------------------------------------------------------
def extract_frames(video_path: str, output_dir: str, frame_every_n_sec: float = 1.0):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    frame_interval = max(1, int(round(fps * frame_every_n_sec)))

    print(f"Video  : {video_path}")
    print(f"  FPS={fps:.2f}, frames={total_frames}, duration={duration_sec:.1f}s")
    print(f"  Sampling 1 frame every {frame_every_n_sec}s ({frame_interval} raw frames apart)")

    saved, count, idx = [], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            path = os.path.join(output_dir, f"{idx:06d}.png")
            cv2.imwrite(path, frame)
            saved.append(path)
            idx += 1
        count += 1

    cap.release()
    print(f"  Saved {len(saved)} frames → {output_dir}")
    return sorted(saved)


# ---------------------------------------------------------------------------
# Step 2 – Run VGGT inference
# ---------------------------------------------------------------------------
def run_inference(image_paths, model, device="cpu"):
    model.eval()

    print(f"Loading {len(image_paths)} images …")
    images = load_and_preprocess_images(image_paths).to(device)
    print(f"  Shape: {images.shape}")

    print("Running VGGT inference … (this takes several minutes on CPU)")
    with torch.no_grad():
        predictions = model(images)

    print("Converting pose encodings …")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions["pose_enc_list"] = None

    print("Unprojecting depth → world points …")
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    )

    gc.collect()
    return predictions


# ---------------------------------------------------------------------------
# Step 3 – Build CameraPose list from predictions
# ---------------------------------------------------------------------------
def build_camera_poses(predictions, gps_data_list: Optional[List[dict]] = None) -> List[CameraPose]:
    """Create CameraPose objects from VGGT predictions and optional GPS data."""
    poses = []
    extrinsics = predictions["extrinsic"]
    intrinsics = predictions["intrinsic"]
    n_frames = len(extrinsics)
    
    for i in range(n_frames):
        gps = None
        if gps_data_list and i < len(gps_data_list):
            gd = gps_data_list[i]
            gps = GPSData(
                latitude=gd["latitude"],
                longitude=gd["longitude"],
                altitude=gd["altitude"],
                roll=gd.get("roll", 0.0),
                pitch=gd.get("pitch", 0.0),
                yaw=gd.get("yaw", 0.0),
                rtcm_available=gd.get("rtcm_available", False),
                horizontal_accuracy=gd.get("horizontal_accuracy", 1.0),
                vertical_accuracy=gd.get("vertical_accuracy", 1.0)
            )
        
        pose = CameraPose(
            extrinsic=extrinsics[i],
            intrinsic=intrinsics[i],
            gps=gps,
            frame_idx=i
        )
        poses.append(pose)
    
    return poses


# ---------------------------------------------------------------------------
# Step 4 – Load GPS data from JSON (optional)
# ---------------------------------------------------------------------------
def load_gps_data(gps_json_path: str) -> List[dict]:
    """Load GPS/IMU data from JSON file."""
    with open(gps_json_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and "frames" in data:
        return data["frames"]
    return data  # Assume list of frame GPS data


# ---------------------------------------------------------------------------
# Step 5 – Export GLB (original point cloud visualization)
# ---------------------------------------------------------------------------
def export_glb(predictions, output_dir, conf_thres=50.0, show_cam=True):
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, "reconstruction.glb")
    print("Building GLB scene (point cloud) …")
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames="All",
        mask_black_bg=False,
        mask_white_bg=False,
        show_cam=show_cam,
        mask_sky=False,
        target_dir=output_dir,
        prediction_mode="Depthmap and Camera Branch",
    )
    glbscene.export(file_obj=glb_path)
    print(f"  GLB saved → {glb_path}")
    return glb_path


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="VayuMesh: Video → Metric 3D Solid with Dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages:
  1. VGGT Inference      → Point cloud from video frames
  2. Poisson Reconstruction → Watertight mesh from point cloud
  3. Metric Scaling       → GPS/IMU fusion, ground plane alignment, axis alignment
  4. Dimension Extraction → Width, length, height, volume, footprint

GPS JSON Format:
  [
    {"latitude": 28.6139, "longitude": 77.2090, "altitude": 216.0, "yaw": 45.0, "pitch": 0.0, "roll": 0.0},
    ...
  ]
  Or: {"frames": [...]}
        """
    )
    parser.add_argument("--video", default="187401-880363244.mp4", help="Input video file")
    parser.add_argument("--fps", type=float, default=1.0, help="Sample 1 frame every N seconds")
    parser.add_argument("--output", default="output_3d", help="Output directory")
    parser.add_argument("--conf_thres", type=float, default=50.0, help="Confidence percentile threshold")
    parser.add_argument("--no_cam", action="store_true", help="Omit camera frustums from GLB")
    
    # Stage 2: Poisson Reconstruction
    parser.add_argument("--poisson_depth", type=int, default=9, help="Poisson octree depth (8-10)")
    parser.add_argument("--no_denoise", action="store_true", help="Skip statistical outlier removal")
    parser.add_argument("--density_threshold", type=float, default=0.1, help="Poisson density threshold quantile")
    parser.add_argument("--skip_mesh", action="store_true", help="Skip mesh reconstruction (point cloud only)")
    
    # Stage 3: Metric Scaling
    parser.add_argument("--gps", help="GPS/IMU data JSON file for metric scaling")
    parser.add_argument("--skip_scaling", action="store_true", help="Skip metric scaling (use VGGT units)")
    parser.add_argument("--ground_z", type=float, help="Manual ground Z override (meters after scaling)")
    
    # Stage 4: Dimension Extraction
    parser.add_argument("--skip_dims", action="store_true", help="Skip dimension extraction")
    parser.add_argument("--multi_building", action="store_true", help="Extract multiple buildings")
    parser.add_argument("--min_volume", type=float, default=10.0, help="Min volume for multi-building (m³)")
    
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"ERROR: Video not found: {args.video}"); sys.exit(1)

    frames_dir = os.path.join(args.output, "images")
    os.makedirs(args.output, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # STAGE 1: VGGT Inference
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("STAGE 1: VGGT Inference")
    print("="*60)

    image_paths = extract_frames(args.video, frames_dir, frame_every_n_sec=args.fps)
    if not image_paths:
        print("ERROR: No frames extracted."); sys.exit(1)

    device = "cpu"
    print(f"\nUsing device: {device}")
    print("Loading VGGT model from cache …")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, map_location="cpu"))
    model.eval()
    print("Model loaded.")

    predictions = run_inference(image_paths, model, device=device)

    # Save raw predictions
    npz_path = os.path.join(args.output, "predictions.npz")
    np.savez(npz_path, **{k: v for k, v in predictions.items() if v is not None})
    print(f"Predictions → {npz_path}")

    # Export point cloud GLB
    glb_path = export_glb(predictions, args.output, conf_thres=args.conf_thres, show_cam=not args.no_cam)

    # ═══════════════════════════════════════════════════════════════
    # STAGE 2: Poisson Surface Reconstruction
    # ═══════════════════════════════════════════════════════════════
    mesh = None
    if not args.skip_mesh:
        print("\n" + "="*60)
        print("STAGE 2: Poisson Surface Reconstruction")
        print("="*60)

        # Extract points and colors
        points = predictions["world_points_from_depth"].reshape(-1, 3)
        colors = None
        if "images" in predictions:
            imgs = predictions["images"]
            if imgs.ndim == 4 and imgs.shape[1] == 3:
                imgs = imgs.transpose(0, 2, 3, 1)
            colors = imgs.reshape(-1, 3)
            # Filter colors by confidence if available
            if "depth_conf" in predictions:
                conf = predictions["depth_conf"].reshape(-1)
                threshold = np.percentile(conf, args.conf_thres)
                mask = conf >= threshold
                points = points[mask]
                colors = colors[mask]

        print(f"Input point cloud: {len(points)} points")

        mesh = full_reconstruction_pipeline(
            points, colors,
            denoise=not args.no_denoise,
            poisson_depth=args.poisson_depth,
            density_threshold=args.density_threshold,
            clean=True
        )

        # Save mesh
        mesh_path = os.path.join(args.output, "reconstruction_mesh.obj")
        export_mesh(mesh, mesh_path, "obj")
        print(f"Mesh saved → {mesh_path}")

        # Also export as GLB for viewing
        mesh_glb_path = os.path.join(args.output, "reconstruction_mesh.glb")
        export_mesh(mesh, mesh_glb_path, "glb")
        print(f"Mesh GLB → {mesh_glb_path}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 3: Metric Scaling & Alignment
    # ═══════════════════════════════════════════════════════════════
    metric_result = None
    if not args.skip_scaling:
        print("\n" + "="*60)
        print("STAGE 3: Metric Scaling & Coordinate Alignment")
        print("="*60)

        # Load GPS data if provided
        gps_data_list = None
        if args.gps:
            if os.path.isfile(args.gps):
                gps_data_list = load_gps_data(args.gps)
                print(f"Loaded GPS data for {len(gps_data_list)} frames")
            else:
                print(f"WARNING: GPS file not found: {args.gps}")

        # Build camera poses
        camera_poses = build_camera_poses(predictions, gps_data_list)

        # Extract points and colors for scaling
        points = predictions["world_points_from_depth"].reshape(-1, 3)
        colors = None
        if "images" in predictions:
            imgs = predictions["images"]
            if imgs.ndim == 4 and imgs.shape[1] == 3:
                imgs = imgs.transpose(0, 2, 3, 1)
            colors = imgs.reshape(-1, 3)
            # Filter by confidence
            if "depth_conf" in predictions:
                conf = predictions["depth_conf"].reshape(-1)
                threshold = np.percentile(conf, args.conf_thres)
                mask = conf >= threshold
                points = points[mask]
                colors = colors[mask]

        # Run full metric pipeline
        metric_result = full_metric_pipeline(points, colors, camera_poses, mesh)

        # Save scaled point cloud
        scaled_npz = os.path.join(args.output, "predictions_scaled.npz")
        np.savez(scaled_npz,
                 points=metric_result["points"],
                 colors=metric_result["colors"],
                 camera_poses=[{
                     "extrinsic": p.extrinsic.tolist(),
                     "intrinsic": p.intrinsic.tolist(),
                     "gps": {
                         "latitude": p.gps.latitude, "longitude": p.gps.longitude,
                         "altitude": p.gps.altitude, "yaw": p.gps.yaw,
                         "pitch": p.gps.pitch, "roll": p.gps.roll
                     } if p.gps else None
                 } for p in metric_result["camera_poses"]],
                 scale_factor=metric_result["scale_factor"],
                 ground_plane_model=metric_result["ground_plane_model"].tolist(),
                 total_rotation=metric_result["total_rotation"].tolist()
        )
        print(f"Scaled predictions → {scaled_npz}")

        # Save scaled mesh
        if metric_result["mesh"] is not None:
            scaled_mesh_path = os.path.join(args.output, "reconstruction_scaled.obj")
            export_mesh(metric_result["mesh"], scaled_mesh_path, "obj")
            print(f"Scaled mesh → {scaled_mesh_path}")

            scaled_mesh_glb = os.path.join(args.output, "reconstruction_scaled.glb")
            export_mesh(metric_result["mesh"], scaled_mesh_glb, "glb")
            print(f"Scaled mesh GLB → {scaled_mesh_glb}")

        # Save transformation metadata
        meta_path = os.path.join(args.output, "metric_transform.json")
        with open(meta_path, 'w') as f:
            json.dump({
                "scale_factor": float(metric_result["scale_factor"]),
                "ground_plane_model": metric_result["ground_plane_model"].tolist(),
                "ground_rotation": metric_result["ground_rotation"].tolist(),
                "horizontal_rotation": metric_result["horizontal_rotation"].tolist(),
                "total_rotation": metric_result["total_rotation"].tolist()
            }, f, indent=2)
        print(f"Transform metadata → {meta_path}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 4: Dimension Extraction
    # ═══════════════════════════════════════════════════════════════
    if not args.skip_dims and metric_result is not None and metric_result["mesh"] is not None:
        print("\n" + "="*60)
        print("STAGE 4: Dimension Extraction")
        print("="*60)

        scaled_mesh = metric_result["mesh"]
        
        # Determine ground Z
        ground_z = args.ground_z
        if ground_z is None:
            # Estimate from scaled mesh (5th percentile of Z)
            ground_z = float(np.percentile(scaled_mesh.vertices[:, 2], 5))
        print(f"Ground plane Z: {ground_z:.3f} m")

        if args.multi_building:
            buildings = extract_multi_building_dimensions(
                scaled_mesh, ground_z, min_volume=args.min_volume
            )
            for i, dims in enumerate(buildings):
                report_path = export_dimensions_report(dims, os.path.join(args.output, f"dimensions_building_{i}"))
                viz_path = os.path.join(args.output, f"dimensions_building_{i}_viz.glb")
                create_dimension_visualization(scaled_mesh, dims, viz_path)
        else:
            dims = extract_building_dimensions(scaled_mesh, ground_z)
            report_path = export_dimensions_report(dims, os.path.join(args.output, "dimensions"))
            viz_path = os.path.join(args.output, "dimensions_viz.glb")
            create_dimension_visualization(scaled_mesh, dims, viz_path)
            
            # Also compute roof height profile
            x_coords, y_coords, height_grid = compute_roof_height_profile(scaled_mesh, ground_z)
            profile_path = os.path.join(args.output, "roof_profile.npz")
            np.savez(profile_path, x=x_coords, y=y_coords, height=height_grid)
            print(f"Roof height profile → {profile_path}")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("✅ VayuMesh Pipeline Complete!")
    print("="*60)
    print(f"  Frames           : {frames_dir}")
    print(f"  Raw predictions  : {npz_path}")
    print(f"  Point cloud GLB  : {glb_path}")
    
    if mesh is not None:
        print(f"  Mesh (VGGT units): {os.path.join(args.output, 'reconstruction_mesh.obj')}")
    
    if metric_result is not None:
        print(f"  Scaled mesh      : {os.path.join(args.output, 'reconstruction_scaled.obj')}")
        print(f"  Scale factor     : {metric_result['scale_factor']:.4f} (VGGT units → meters)")
    
    if not args.skip_dims and metric_result is not None:
        print(f"  Dimensions report: {os.path.join(args.output, 'dimensions.json')}")
        print(f"  Dimension viz    : {os.path.join(args.output, 'dimensions_viz.glb')}")
        print(f"  Roof profile     : {os.path.join(args.output, 'roof_profile.npz')}")
    
    print("\nOpen .glb files at https://gltf-viewer.donmccurdy.com")


if __name__ == "__main__":
    main()