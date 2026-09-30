"""
VayuMesh Dashboard - SIH 26158 (Minimal Version)
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import gradio as gr
import numpy as np
import torch

sys.path.append("vggt/")

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# Pipeline imports
from video_to_3d import (
    extract_frames, run_inference, build_camera_poses,
    full_metric_pipeline, extract_building_dimensions,
    export_dimensions_report, create_dimension_visualization
)
from visual_util import predictions_to_glb

# Optional mesh reconstruction (requires open3d)
try:
    from mesh_reconstruction import full_reconstruction_pipeline, export_mesh
    MESH_AVAILABLE = True
except ImportError as e:
    print("WARNING: Mesh reconstruction unavailable: " + str(e))
    MESH_AVAILABLE = False
    full_reconstruction_pipeline = None
    export_mesh = None

# Global model cache
_model_cache = None
_device = "cpu"


def load_model_cached(device: str = "cpu") -> VGGT:
    global _model_cache, _device
    if _model_cache is None or _device != device:
        print("Loading VGGT model on " + device + "...")
        model = VGGT()
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location=device))
        model.eval()
        _model_cache = model
        _device = device
    return _model_cache


def process_video_pipeline(
    video_file,
    gps_file,
    fps_sampling: float,
    poisson_depth: int,
    conf_threshold: float,
    enable_mesh: bool,
    enable_scaling: bool,
    enable_dimensions: bool,
    multi_building: bool,
    ground_z_override: Optional[float],
    progress=gr.Progress()
) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str], str, str]:
    start_time = time.time()
    logs = []
    
    def log(msg: str):
        logs.append("[{:.1f}s] {}".format(time.time() - start_time, msg))
        print(msg)
    
    if video_file is None:
        return "ERROR: No video uploaded", None, None, None, None, "{}", "\n".join(logs)
    
    with tempfile.TemporaryDirectory(prefix="vayumesh_") as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        frames_dir = output_dir / "images"
        
        progress(0.05, desc="Extracting frames...")
        log("Stage 1: Frame extraction")
        
        video_path = video_file.name if hasattr(video_file, 'name') else str(video_file)
        image_paths = extract_frames(video_path, str(frames_dir), frame_every_n_sec=fps_sampling)
        
        if not image_paths:
            return "ERROR: No frames extracted", None, None, None, None, "{}", "\n".join(logs)
        
        log("Extracted {} frames".format(len(image_paths)))
        
        progress(0.1, desc="Loading VGGT model...")
        model = load_model_cached("cpu")
        
        progress(0.15, desc="Running VGGT inference...")
        log("Stage 2: VGGT inference")
        
        predictions = run_inference(image_paths, model, device="cpu")
        
        progress(0.35, desc="Building point cloud GLB...")
        log("Stage 3: Point cloud export")
        
        pc_glb_path = output_dir / "pointcloud.glb"
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_threshold,
            filter_by_frames="All",
            show_cam=True,
            mask_sky=False,
            target_dir=str(output_dir),
            prediction_mode="Depthmap and Camera Branch",
        )
        glbscene.export(file_obj=str(pc_glb_path))
        
        mesh_glb_path = None
        scaled_mesh_glb_path = None
        dim_viz_glb_path = None
        dims_json = {}
        
        if enable_mesh:
            if not MESH_AVAILABLE:
                log("Stage 4 skipped: open3d not available")
            else:
                progress(0.45, desc="Poisson surface reconstruction...")
                log("Stage 4: Poisson reconstruction")
                
                points = predictions["world_points_from_depth"].reshape(-1, 3)
                colors = None
                if "images" in predictions:
                    imgs = predictions["images"]
                    if imgs.ndim == 4 and imgs.shape[1] == 3:
                        imgs = imgs.transpose(0, 2, 3, 1)
                    colors = imgs.reshape(-1, 3)
                    if "depth_conf" in predictions:
                        conf = predictions["depth_conf"].reshape(-1)
                        threshold = np.percentile(conf, conf_threshold)
                        mask = conf >= threshold
                        points = points[mask]
                        colors = colors[mask]
                
                mesh = full_reconstruction_pipeline(
                    points, colors,
                    denoise=True,
                    poisson_depth=poisson_depth,
                    density_threshold=0.1,
                    clean=True
                )
                
                mesh_glb_path = output_dir / "mesh.glb"
                export_mesh(mesh, str(mesh_glb_path), "glb")
                log("Mesh: {} vertices, {} faces".format(len(mesh.vertices), len(mesh.faces)))
        
        metric_result = None
        if enable_scaling:
            progress(0.65, desc="Metric scaling & alignment...")
            log("Stage 5: Metric scaling")
            
            gps_data_list = None
            if gps_file:
                gps_path = gps_file.name if hasattr(gps_file, 'name') else str(gps_file)
                with open(gps_path, 'r') as f:
                    gps_data_list = json.load(f)
                if isinstance(gps_data_list, dict) and "frames" in gps_data_list:
                    gps_data_list = gps_data_list["frames"]
                log("Loaded GPS for {} frames".format(len(gps_data_list)))
            
            camera_poses = build_camera_poses(predictions, gps_data_list)
            
            points = predictions["world_points_from_depth"].reshape(-1, 3)
            colors = None
            if "images" in predictions:
                imgs = predictions["images"]
                if imgs.ndim == 4 and imgs.shape[1] == 3:
                    imgs = imgs.transpose(0, 2, 3, 1)
                colors = imgs.reshape(-1, 3)
                if "depth_conf" in predictions:
                    conf = predictions["depth_conf"].reshape(-1)
                    threshold = np.percentile(conf, conf_threshold)
                    mask = conf >= threshold
                    points = points[mask]
                    colors = colors[mask]
            
            mesh_for_scaling = mesh if (enable_mesh and MESH_AVAILABLE and mesh_glb_path) else None
            metric_result = full_metric_pipeline(points, colors, camera_poses, mesh_for_scaling)
            
            if metric_result["mesh"] is not None:
                scaled_mesh_glb_path = output_dir / "mesh_scaled.glb"
                export_mesh(metric_result["mesh"], str(scaled_mesh_glb_path), "glb")
                log("Scale factor: {:.4f} (VGGT units -> meters)".format(metric_result['scale_factor']))
        
        if enable_dimensions and metric_result is not None and metric_result["mesh"] is not None:
            progress(0.85, desc="Extracting dimensions...")
            log("Stage 6: Dimension extraction")
            
            scaled_mesh = metric_result["mesh"]
            ground_z = ground_z_override
            if ground_z is None:
                ground_z = float(np.percentile(scaled_mesh.vertices[:, 2], 5))
            
            dims = extract_building_dimensions(scaled_mesh, ground_z)
            dims_json = {
                "width_m": dims.width,
                "length_m": dims.length,
                "height_m": dims.height,
                "volume_m3": dims.volume,
                "footprint_area_m2": dims.footprint_area,
                "centroid": dims.centroid.tolist(),
            }
            
            dim_viz_glb_path = output_dir / "dimensions_viz.glb"
            create_dimension_visualization(scaled_mesh, dims, str(dim_viz_glb_path))
            
            log("Dimensions: {:.1f} x {:.1f} x {:.1f} m".format(dims.width, dims.length, dims.height))
        elif enable_dimensions:
            log("Stage 6 skipped: Requires mesh and metric scaling")
        
        progress(0.95, desc="Finalizing...")
        
        persistent_dir = Path("dashboard_outputs")
        persistent_dir.mkdir(exist_ok=True)
        
        import shutil
        final_files = {}
        for name, path in [
            ("pointcloud", pc_glb_path),
            ("mesh", mesh_glb_path),
            ("mesh_scaled", scaled_mesh_glb_path),
            ("dimensions_viz", dim_viz_glb_path),
        ]:
            if path and path.exists():
                dst = persistent_dir / "{}_{}.glb".format(name, int(time.time()))
                shutil.copy2(path, dst)
                final_files[name] = str(dst)
        
        elapsed = time.time() - start_time
        status = "Complete in {:.1f}s ({} frames)".format(elapsed, len(image_paths))
        if metric_result:
            status += " | Scale: {:.4f}".format(metric_result['scale_factor'])
        if dims_json:
            status += " | {:.1f}x{:.1f}x{:.1f}m".format(dims_json['width_m'], dims_json['length_m'], dims_json['height_m'])
        
        return (
            status,
            final_files.get("pointcloud"),
            final_files.get("mesh"),
            final_files.get("mesh_scaled"),
            final_files.get("dimensions_viz"),
            json.dumps(dims_json, indent=2),
            "\n".join(logs)
        )


def create_dashboard():
    with gr.Blocks(title="VayuMesh - SIH 26158", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # VayuMesh - SIH 26158
        **Video to Metric 3D Solid Pipeline**
        
        | Core Value | Target |
        |------------|--------|
        | Speed | <15 min for 10-min video |
        | Accuracy | <=1m spatial without GCPs |
        | Trust | Per-region confidence heatmap |
        | Security | Fully offline, air-gapped |
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Configuration")
                
                video_input = gr.Video(label="Drone Video (MP4/MOV)", sources=["upload"], format="mp4")
                gps_input = gr.File(label="GPS/IMU Telemetry (JSON)", file_types=[".json"])
                
                with gr.Accordion("Pipeline Settings", open=True):
                    fps_sampling = gr.Slider(0.5, 5.0, value=1.0, step=0.5, label="Frame Sampling (1 frame per N seconds)")
                    poisson_depth = gr.Slider(8, 10, value=9, step=1, label="Poisson Depth (8=fast, 10=detail)")
                    conf_threshold = gr.Slider(0, 90, value=50, step=5, label="Confidence Percentile Threshold")
                
                with gr.Accordion("Stage Toggles", open=False):
                    enable_mesh = gr.Checkbox(True, label="Stage 4: Poisson Mesh")
                    enable_scaling = gr.Checkbox(True, label="Stage 5: Metric Scaling (needs GPS)")
                    enable_dimensions = gr.Checkbox(True, label="Stage 6: Dimension Extraction")
                    multi_building = gr.Checkbox(False, label="Multi-building detection")
                    ground_z_override = gr.Number(label="Ground Z Override (meters, optional)", value=None, precision=3)
                
                run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")
                status_output = gr.Textbox(label="Status", interactive=False, lines=2)
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("3D Visualization"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Point Cloud (VGGT Raw)**")
                                pc_viewer = gr.Model3D(label="Point Cloud", height=400, clear_color=[0.1, 0.1, 0.1, 1])
                            with gr.Column():
                                gr.Markdown("**Watertight Mesh (Poisson)**")
                                mesh_viewer = gr.Model3D(label="Mesh", height=400, clear_color=[0.1, 0.1, 0.1, 1])
                        
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Metric-Scaled Mesh (Meters)**")
                                scaled_mesh_viewer = gr.Model3D(label="Scaled Mesh", height=400, clear_color=[0.1, 0.1, 0.1, 1])
                            with gr.Column():
                                gr.Markdown("**Dimensions Visualization**")
                                dim_viz_viewer = gr.Model3D(label="Dimensions", height=400, clear_color=[0.1, 0.1, 0.1, 1])
                    
                    with gr.TabItem("Dimensions & Metrics"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Building Dimensions")
                                dims_json = gr.Textbox(label="Dimensions Report", lines=15)
                            with gr.Column():
                                gr.Markdown("### Pipeline Metrics")
                                metrics_json = gr.Textbox(label="Processing Metrics", lines=15)
                    
                    with gr.TabItem("Processing Logs"):
                        log_output = gr.Textbox(label="Pipeline Logs", lines=25, max_lines=30, show_copy_button=True)
                    
                    with gr.TabItem("Export Info"):
                        gr.Markdown("""
                        ### Download Results
                        Use the 3D viewers above to inspect, then download GLB files.
                        
                        **Available exports (in dashboard_outputs/):**
                        - pointcloud_*.glb - Raw VGGT point cloud with cameras
                        - mesh_*.glb - Watertight Poisson mesh (VGGT units)
                        - mesh_scaled_*.glb - Metric-scaled mesh (meters, Z-up, X=East/Y=North)
                        - dimensions_viz_*.glb - Mesh with dimension annotations
                        - dimensions.json - Machine-readable dimensions
                        - dimensions.geojson - GIS-compatible footprint
                        """)
        
        run_btn.click(
            fn=process_video_pipeline,
            inputs=[
                video_input, gps_input,
                fps_sampling, poisson_depth, conf_threshold,
                enable_mesh, enable_scaling, enable_dimensions,
                multi_building, ground_z_override
            ],
            outputs=[
                status_output,
                pc_viewer, mesh_viewer, scaled_mesh_viewer, dim_viz_viewer,
                dims_json, log_output
            ],
            show_progress=True
        )
        
        gr.Markdown("""
        ---
        ### GPS JSON Template
        ```json
        [
          {"latitude": 28.6139, "longitude": 77.2090, "altitude": 216.0, "yaw": 45.0, "pitch": -2.5, "roll": 0.8},
          {"latitude": 28.6140, "longitude": 77.2091, "altitude": 215.8, "yaw": 47.0, "pitch": -2.3, "roll": 0.5}
        ]
        ```
        **Fields:** latitude/longitude (WGS84°), altitude (AMSL m), yaw/pitch/roll (deg), rtcm_available (bool), horizontal_accuracy/vertical_accuracy (m)
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,  # Required for Gradio 5.x in some environments
        show_error=True,
        quiet=False,
        inbrowser=True
    )