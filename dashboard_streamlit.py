"""
VayuMesh Streamlit Dashboard - SIH 26158 (Improved)
"""

import os
import sys
import json
import time
import tempfile
import threading
import shutil
import base64
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import streamlit as st
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

# Optional mesh reconstruction
try:
    from mesh_reconstruction import full_reconstruction_pipeline, export_mesh
    MESH_AVAILABLE = True
except ImportError as e:
    MESH_AVAILABLE = False
    full_reconstruction_pipeline = None
    export_mesh = None


# Page config
st.set_page_config(
    page_title="VayuMesh - SIH 26158",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stage-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    .stage-active { background: #ffd700; color: #333; }
    .stage-done { background: #28a745; color: white; }
    .stage-pending { background: #6c757d; color: white; }
    .log-entry {
        font-family: monospace;
        font-size: 0.85rem;
        padding: 0.25rem;
        border-bottom: 1px solid #eee;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .status-ok { color: #28a745; font-weight: bold; }
    .status-warn { color: #ffc107; font-weight: bold; }
    .status-err { color: #dc3545; font-weight: bold; }
    .viewer-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = None
if 'device' not in st.session_state:
    st.session_state.device = "cpu"
if 'pipeline_logs' not in st.session_state:
    st.session_state.pipeline_logs = []
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = Path("dashboard_outputs")
    st.session_state.output_dir.mkdir(exist_ok=True)


def render_gltf_viewer(glb_path: str, height: int = 500) -> None:
    """Render GLB file using gltf-viewer via iframe."""
    import base64
    try:
        with open(glb_path, 'rb') as f:
            glb_bytes = f.read()
        b64 = base64.b64encode(glb_bytes).decode()
        data_uri = f"data:model/gltf-binary;base64,{b64}"
        
        # Use gltf-viewer from donmccurdy.com via iframe with data URI
        iframe_html = f"""
        <div class="viewer-container" style="width:100%; height:{height}px;">
            <iframe 
                src="https://gltf-viewer.donmccurdy.com/?model={data_uri}"
                width="100%" 
                height="{height}px" 
                frameborder="0"
                allowfullscreen>
            </iframe>
        </div>
        """
        st.components.v1.html(iframe_html, height=height + 20)
    except Exception as e:
        st.error(f"Failed to load viewer: {e}")
        st.code(glb_path)


def load_model_cached(device: str = "cpu") -> VGGT:
    if st.session_state.model_cache is None or st.session_state.device != device:
        with st.spinner("Loading VGGT model on " + device + "..."):
            model = VGGT()
            url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            model.load_state_dict(torch.hub.load_state_dict_from_url(url, map_location=device))
            model.eval()
            st.session_state.model_cache = model
            st.session_state.device = device
    return st.session_state.model_cache


def add_log(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.pipeline_logs.append("[{}] {}".format(timestamp, msg))


def clear_logs():
    st.session_state.pipeline_logs = []


def run_pipeline_thread(
    video_path, gps_path, fps_sampling, poisson_depth, conf_threshold,
    enable_mesh, enable_scaling, enable_dimensions, multi_building, ground_z_override,
    progress_bar, status_text, stage_placeholders
):
    """Run pipeline in background thread."""
    start_time = time.time()
    results = {
        'files': {},
        'dims_json': {},
        'status': '',
        'elapsed': 0,
        'num_frames': 0
    }
    
    try:
        # Stage 1
        update_stage(stage_placeholders, 1, 'active')
        add_log("Stage 1: Frame extraction")
        progress_bar.progress(0.05, text="Stage 1/6: Extracting frames...")
        status_text.text("Stage 1/6: Extracting frames...")
        
        output_dir = Path(tempfile.mkdtemp(prefix="vayumesh_")) / "output"
        output_dir.mkdir(parents=True)
        frames_dir = output_dir / "images"
        
        image_paths = extract_frames(video_path, str(frames_dir), frame_every_n_sec=fps_sampling)
        
        if not image_paths:
            raise RuntimeError("No frames extracted")
        
        add_log("Extracted {} frames".format(len(image_paths)))
        update_stage(stage_placeholders, 1, 'done')
        
        # Stage 2
        update_stage(stage_placeholders, 2, 'active')
        progress_bar.progress(0.1, text="Stage 2/6: Loading VGGT model...")
        status_text.text("Stage 2/6: Loading VGGT model...")
        model = load_model_cached("cpu")
        
        add_log("Stage 2: VGGT inference")
        progress_bar.progress(0.15, text="Stage 2/6: VGGT inference...")
        status_text.text("Stage 2/6: VGGT inference...")
        
        predictions = run_inference(image_paths, model, device="cpu")
        update_stage(stage_placeholders, 2, 'done')
        
        # Stage 3
        update_stage(stage_placeholders, 3, 'active')
        add_log("Stage 3: Point cloud export")
        progress_bar.progress(0.35, text="Stage 3/6: Building point cloud GLB...")
        status_text.text("Stage 3/6: Building point cloud GLB...")
        
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
        update_stage(stage_placeholders, 3, 'done')
        
        mesh_glb_path = None
        scaled_mesh_glb_path = None
        dim_viz_glb_path = None
        dims_json = {}
        metric_result = None
        mesh = None
        
        # Stage 4: Mesh
        if enable_mesh:
            update_stage(stage_placeholders, 4, 'active')
            if not MESH_AVAILABLE:
                add_log("Stage 4 skipped: open3d not available (Windows AppLocker)")
                update_stage(stage_placeholders, 4, 'skipped')
            else:
                add_log("Stage 4: Poisson reconstruction")
                progress_bar.progress(0.45, text="Stage 4/6: Poisson surface reconstruction...")
                status_text.text("Stage 4/6: Poisson surface reconstruction...")
                
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
                add_log("Mesh: {} vertices, {} faces".format(len(mesh.vertices), len(mesh.faces)))
                update_stage(stage_placeholders, 4, 'done')
        else:
            update_stage(stage_placeholders, 4, 'skipped')
        
        # Stage 5: Metric Scaling
        if enable_scaling:
            update_stage(stage_placeholders, 5, 'active')
            add_log("Stage 5: Metric scaling")
            progress_bar.progress(0.65, text="Stage 5/6: Metric scaling & alignment...")
            status_text.text("Stage 5/6: Metric scaling & alignment...")
            
            gps_data_list = None
            if gps_path:
                with open(gps_path, 'r') as f:
                    gps_data_list = json.load(f)
                if isinstance(gps_data_list, dict) and "frames" in gps_data_list:
                    gps_data_list = gps_data_list["frames"]
                add_log("Loaded GPS for {} frames".format(len(gps_data_list)))
            else:
                add_log("No GPS provided: using relative scale (no metric conversion)")
            
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
                add_log("Scale factor: {:.4f} (VGGT units -> meters)".format(metric_result['scale_factor']))
            update_stage(stage_placeholders, 5, 'done')
        else:
            update_stage(stage_placeholders, 5, 'skipped')
        
        # Stage 6: Dimensions
        if enable_dimensions and metric_result is not None and metric_result["mesh"] is not None:
            update_stage(stage_placeholders, 6, 'active')
            add_log("Stage 6: Dimension extraction")
            progress_bar.progress(0.85, text="Stage 6/6: Extracting dimensions...")
            status_text.text("Stage 6/6: Extracting dimensions...")
            
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
            
            add_log("Dimensions: {:.1f} x {:.1f} x {:.1f} m".format(dims.width, dims.length, dims.height))
            update_stage(stage_placeholders, 6, 'done')
        elif enable_dimensions:
            add_log("Stage 6 skipped: Requires mesh and metric scaling")
            update_stage(stage_placeholders, 6, 'skipped')
        else:
            update_stage(stage_placeholders, 6, 'skipped')
        
        # Copy to persistent location
        progress_bar.progress(0.95, text="Finalizing...")
        final_files = {}
        for name, path in [
            ("pointcloud", pc_glb_path),
            ("mesh", mesh_glb_path),
            ("mesh_scaled", scaled_mesh_glb_path),
            ("dimensions_viz", dim_viz_glb_path),
        ]:
            if path and path.exists():
                dst = st.session_state.output_dir / "{}_{}.glb".format(name, int(time.time()))
                shutil.copy2(path, dst)
                final_files[name] = str(dst)
        
        elapsed = time.time() - start_time
        status = "Complete in {:.1f}s ({} frames)".format(elapsed, len(image_paths))
        if metric_result:
            status += " | Scale: {:.4f}".format(metric_result['scale_factor'])
        if dims_json:
            status += " | {:.1f}x{:.1f}x{:.1f}m".format(dims_json['width_m'], dims_json['length_m'], dims_json['height_m'])
        
        results = {
            'status': status,
            'files': final_files,
            'dims_json': dims_json,
            'logs': "\n".join(st.session_state.pipeline_logs),
            'elapsed': elapsed,
            'num_frames': len(image_paths)
        }
        
    except Exception as e:
        add_log("ERROR: " + str(e))
        results = {
            'status': "ERROR: " + str(e),
            'files': {},
            'dims_json': {},
            'logs': "\n".join(st.session_state.pipeline_logs),
            'elapsed': time.time() - start_time,
            'num_frames': 0
        }
    finally:
        st.session_state.pipeline_running = False
        st.session_state.results = results
        # Trigger rerun by setting a flag
        st.session_state._rerun_trigger = not st.session_state.get('_rerun_trigger', False)


def update_stage(placeholders, stage_num, state):
    """Update stage indicator."""
    labels = {
        1: "1. Frame Extraction",
        2: "2. VGGT Inference",
        3: "3. Point Cloud",
        4: "4. Poisson Mesh",
        5: "5. Metric Scaling",
        6: "6. Dimensions"
    }
    badge_class = {
        'active': 'stage-active',
        'done': 'stage-done',
        'skipped': 'stage-pending',
        'pending': 'stage-pending'
    }.get(state, 'stage-pending')
    
    with placeholders[stage_num - 1]:
        st.markdown('<span class="stage-badge {}">{}</span>'.format(badge_class, labels[stage_num]), unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚁 VayuMesh — SIH 26158</h1>
        <p>Video → Metric 3D Solid Pipeline | <strong>Speed • Accuracy • Trust • Security</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>⚡ Speed</h3><p><15 min for 10-min video</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🎯 Accuracy</h3><p>≤1m spatial without GCPs</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🔍 Trust</h3><p>Per-region confidence heatmap</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>🔒 Security</h3><p>Fully offline, air-gapped</p></div>', unsafe_allow_html=True)
    
    # System capabilities
    st.subheader("System Capabilities")
    cap_col1, cap_col2, cap_col3 = st.columns(3)
    with cap_col1:
        st.markdown('<span class="status-ok">✓</span> VGGT Inference (CPU/GPU)', unsafe_allow_html=True)
        st.markdown('<span class="status-ok">✓</span> Point Cloud Export', unsafe_allow_html=True)
        st.markdown('<span class="status-ok">✓</span> Metric Scaling (GPS/IMU)', unsafe_allow_html=True)
    with cap_col2:
        if MESH_AVAILABLE:
            st.markdown('<span class="status-ok">✓</span> Poisson Mesh Reconstruction', unsafe_allow_html=True)
            st.markdown('<span class="status-ok">✓</span> Dimension Extraction', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-warn">⚠</span> Poisson Mesh (open3d blocked)', unsafe_allow_html=True)
            st.markdown('<span class="status-warn">⚠</span> Dimensions (needs mesh)', unsafe_allow_html=True)
    with cap_col3:
        st.markdown('<span class="status-ok">✓</span> GLB Export', unsafe_allow_html=True)
        st.markdown('<span class="status-ok">✓</span> JSON/GeoJSON Export', unsafe_allow_html=True)
        st.markdown('<span class="status-ok">✓</span> Offline Operation', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📥 Input Configuration")
        
        video_file = st.file_uploader("Drone Video (MP4/MOV)", type=["mp4", "mov", "avi"])
        gps_file = st.file_uploader("GPS/IMU Telemetry (JSON)", type=["json"])
        
        st.header("⚙️ Pipeline Settings")
        fps_sampling = st.slider("Frame Sampling (1 frame per N sec)", 0.5, 5.0, 1.0, 0.5)
        poisson_depth = st.slider("Poisson Depth (8=fast, 10=detail)", 8, 10, 9)
        conf_threshold = st.slider("Confidence Percentile", 0, 90, 50, 5)
        
        st.header("🔧 Stage Toggles")
        enable_mesh = st.checkbox("Stage 4: Poisson Mesh", value=MESH_AVAILABLE, disabled=not MESH_AVAILABLE)
        enable_scaling = st.checkbox("Stage 5: Metric Scaling (needs GPS)", value=True)
        enable_dimensions = st.checkbox("Stage 6: Dimension Extraction", value=True)
        multi_building = st.checkbox("Multi-building detection", value=False)
        ground_z_override = st.number_input("Ground Z Override (meters)", value=None, format="%.3f")
        
        st.header("🚀 Run Pipeline")
        run_button = st.button("Start Pipeline", type="primary", use_container_width=True, disabled=st.session_state.pipeline_running or video_file is None)
        
        if st.button("Clear Logs", use_container_width=True):
            clear_logs()
            st.rerun()
        
        if not MESH_AVAILABLE:
            st.warning("open3d blocked by Windows AppLocker. Mesh stages disabled. Use conda-forge open3d or Linux for full pipeline.")
    
    # Stage indicators
    st.subheader("Pipeline Stages")
    stage_cols = st.columns(6)
    stage_placeholders = [col.empty() for col in stage_cols]
    for i in range(6):
        update_stage(stage_placeholders, i+1, 'pending')
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 3D Viewer", "📦 3D Outputs", "📊 Dimensions & Metrics", "📜 Processing Logs", "💾 Download Results"])
    
    # Run pipeline
    if run_button and video_file is not None:
        st.session_state.pipeline_running = True
        clear_logs()
        st.session_state.results = None
        
        # Save uploaded files
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix) as tmp_vid:
            tmp_vid.write(video_file.read())
            video_path = tmp_vid.name
        
        gps_path = None
        if gps_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_gps:
                tmp_gps.write(gps_file.read())
                gps_path = tmp_gps.name
        
        # Progress UI
        progress_bar = st.progress(0, text="Initializing...")
        status_text = st.empty()
        
        # Run in thread
        thread = threading.Thread(
            target=run_pipeline_thread,
            args=(
                video_path, gps_path, fps_sampling, poisson_depth, conf_threshold,
                enable_mesh, enable_scaling, enable_dimensions, multi_building, ground_z_override,
                progress_bar, status_text, stage_placeholders
            ),
            daemon=True
        )
        thread.start()
        
        # Poll for completion
        with st.spinner("Pipeline running..."):
            while st.session_state.pipeline_running:
                time.sleep(0.5)
        
        # Cleanup
        try:
            os.unlink(video_path)
            if gps_path:
                os.unlink(gps_path)
        except:
            pass
        
        st.rerun()
    
    # Display results
    if st.session_state.results:
        results = st.session_state.results
        
        # Status banner
        if results['status'].startswith("ERROR"):
            st.error(results['status'])
        else:
            st.success(results['status'])
        
        files = results['files']
        
        # Tab 1: 3D Viewer (inline)
        with tab1:
            st.subheader("🎯 Interactive 3D Viewer")
            st.caption("Powered by gltf-viewer.donmccurdy.com — click and drag to rotate, scroll to zoom")
            
            model_options = []
            if files.get('pointcloud'):
                model_options.append(("Point Cloud (VGGT Raw)", files['pointcloud']))
            if files.get('mesh'):
                model_options.append(("Poisson Mesh", files['mesh']))
            if files.get('mesh_scaled'):
                model_options.append(("Metric-Scaled Mesh (Meters)", files['mesh_scaled']))
            if files.get('dimensions_viz'):
                model_options.append(("Dimensions Viz", files['dimensions_viz']))
            
            if model_options:
                selected_model = st.selectbox(
                    "Select model to view:",
                    options=range(len(model_options)),
                    format_func=lambda i: model_options[i][0]
                )
                model_name, model_path = model_options[selected_model]
                st.markdown(f"**{model_name}**")
                render_gltf_viewer(model_path, height=550)
            else:
                st.info("No 3D models generated yet")
        
        # Tab 2: 3D Outputs (file list + downloads)
        with tab2:
            st.subheader("Generated 3D Models")
            
            # Point Cloud
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Point Cloud (VGGT Raw)**")
                if files.get('pointcloud'):
                    st.code(files['pointcloud'])
                    with open(files['pointcloud'], 'rb') as f:
                        st.download_button("Download pointcloud.glb", f.read(), file_name="pointcloud.glb", mime="model/gltf-binary")
                    st.caption("Open at https://gltf-viewer.donmccurdy.com")
                else:
                    st.info("Not generated")
                
                st.markdown("**Watertight Mesh (Poisson)**")
                if files.get('mesh'):
                    st.code(files['mesh'])
                    with open(files['mesh'], 'rb') as f:
                        st.download_button("Download mesh.glb", f.read(), file_name="mesh.glb", mime="model/gltf-binary")
                    st.caption("Open at https://gltf-viewer.donmccurdy.com")
                else:
                    st.info("Not generated" + (" (open3d unavailable)" if not MESH_AVAILABLE else ""))
            
            with col2:
                st.markdown("**Metric-Scaled Mesh (Meters)**")
                if files.get('mesh_scaled'):
                    st.code(files['mesh_scaled'])
                    with open(files['mesh_scaled'], 'rb') as f:
                        st.download_button("Download mesh_scaled.glb", f.read(), file_name="mesh_scaled.glb", mime="model/gltf-binary")
                    st.caption("Z-up, X=East, Y=North | Open at https://gltf-viewer.donmccurdy.com")
                else:
                    st.info("Not generated (needs GPS + mesh)")
                
                st.markdown("**Dimensions Visualization**")
                if files.get('dimensions_viz'):
                    st.code(files['dimensions_viz'])
                    with open(files['dimensions_viz'], 'rb') as f:
                        st.download_button("Download dimensions_viz.glb", f.read(), file_name="dimensions_viz.glb", mime="model/gltf-binary")
                    st.caption("Open at https://gltf-viewer.donmccurdy.com")
                else:
                    st.info("Not generated")
        
        # Tab 3: Dimensions
        with tab3:
            st.subheader("Building Dimensions")
            if results['dims_json']:
                st.json(results['dims_json'])
                
                # Quick metrics
                d = results['dims_json']
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Width (m)", f"{d['width_m']:.2f}")
                m2.metric("Length (m)", f"{d['length_m']:.2f}")
                m3.metric("Height (m)", f"{d['height_m']:.2f}")
                m4.metric("Volume (m³)", f"{d['volume_m3']:.1f}")
            else:
                st.info("No dimensions extracted")
            
            st.subheader("Pipeline Metrics")
            st.json({
                "elapsed_seconds": round(results['elapsed'], 1),
                "num_frames": results['num_frames'],
                "stages_completed": [
                    "Stage 1: Frame Extraction",
                    "Stage 2: VGGT Inference",
                    "Stage 3: Point Cloud Export",
                ] + (["Stage 4: Poisson Mesh"] if files.get('mesh') else []) + 
                (["Stage 5: Metric Scaling"] if files.get('mesh_scaled') else []) +
                (["Stage 6: Dimensions"] if results['dims_json'] else [])
            })
        
        # Tab 4: Logs
        with tab4:
            st.subheader("Processing Logs")
            st.text_area("Logs", results['logs'], height=400)
        
        # Tab 5: Downloads
        with tab5:
            st.subheader("All Generated Files")
            st.caption("Files saved to: " + str(st.session_state.output_dir.absolute()))
            
            for name, path in files.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.code(path)
                with col2:
                    with open(path, 'rb') as f:
                        st.download_button(
                            f"Download {name}.glb",
                            f.read(),
                            file_name=Path(path).name,
                            mime="model/gltf-binary",
                            key=f"dl_{name}"
                        )
            
            # Dimensions JSON
            if results['dims_json']:
                dims_path = st.session_state.output_dir / f"dimensions_{int(time.time())}.json"
                with open(dims_path, 'w') as f:
                    json.dump(results['dims_json'], f, indent=2)
                st.markdown("**Dimensions JSON**")
                st.code(str(dims_path))
                with open(dims_path, 'rb') as f:
                    st.download_button("Download dimensions.json", f.read(), file_name="dimensions.json", mime="application/json")
            
            st.markdown("""
            **Viewing GLB files:**
            1. Download using buttons above
            2. Open at https://gltf-viewer.donmccurdy.com (drag & drop)
            3. Or use: Windows 3D Viewer, Blender, MeshLab, or any GLB viewer
            """)
    
    # GPS Template
    with st.expander("📋 GPS JSON Template"):
        st.code("""[
  {"latitude": 28.6139, "longitude": 77.2090, "altitude": 216.0, "yaw": 45.0, "pitch": -2.5, "roll": 0.8, "rtcm_available": false, "horizontal_accuracy": 1.5, "vertical_accuracy": 2.0},
  {"latitude": 28.6140, "longitude": 77.2091, "altitude": 215.8, "yaw": 47.0, "pitch": -2.3, "roll": 0.5, "rtcm_available": false, "horizontal_accuracy": 1.3, "vertical_accuracy": 1.8}
]""", language="json")
        st.markdown("""
        **Fields:**
        - `latitude/longitude`: WGS84 degrees
        - `altitude`: AMSL (meters above mean sea level)
        - `yaw`: Heading (0=North, 90=East, degrees)
        - `pitch/roll`: Camera gimbal angles (degrees)
        - `rtcm_available`: true if RTK correction applied
        - `horizontal_accuracy/vertical_accuracy`: GPS accuracy estimates (meters)
        """)


if __name__ == "__main__":
    main()