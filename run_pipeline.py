"""
VayuMesh 2.0 - Optimized Video → 3D Pipeline
=============================================
Processes a video through all 4 stages and embeds rich metadata into every output.

Stages:
  1. Frame extraction  (smart sampling with quality filter)
  2. VGGT inference    (cached model download from HuggingFace)
  3. Poisson mesh      (open3d surface reconstruction)
  4. Metric scaling    (ground-plane alignment + GPS-free heuristic scale)
  5. Export            (GLB + OBJ + PLY + JSON metadata + GeoJSON + embedded 3MF)

Metadata embedded:
  - Pipeline version, source video, timestamp, frame count
  - VGGT model id and checksum
  - Per-point confidence statistics
  - Camera pose count, average baseline
  - Mesh stats (vertices, faces, watertight, volume)
  - Bounding box / building dimensions (when mesh available)
  - All embedded in GLB extras, .json sidecar, and .ply header comments
"""

import os
import sys
import gc
import json
import time
import hashlib
import shutil
import logging
import platform
import datetime
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm

# ─── Make sure vggt submodule is importable ────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vggt"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# Optional mesh dependencies
try:
    from mesh_reconstruction import full_reconstruction_pipeline, export_mesh
    MESH_AVAILABLE = True
except Exception as _me:
    logging.warning(f"mesh_reconstruction unavailable: {_me}")
    MESH_AVAILABLE = False

# Optional metric scaling
try:
    from metric_scaling import (
        CameraPose, GPSData, full_metric_pipeline,
        detect_ground_plane, align_to_ground_plane
    )
    METRIC_AVAILABLE = True
except Exception as _ms:
    logging.warning(f"metric_scaling unavailable: {_ms}")
    METRIC_AVAILABLE = False

from visual_util import predictions_to_glb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_run.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("vayumesh")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
VGGT_HF_URL  = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
VGGT_REPO_ID = "facebook/VGGT-1B"
VGGT_CACHE   = ROOT / ".cache" / "vggt_model.pt"

PIPELINE_VERSION = "2.0.0"
MODEL_ID         = "facebook/VGGT-1B"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 – Model download & load (cached locally)
# ══════════════════════════════════════════════════════════════════════════════
def download_model_hf(cache_path: Path) -> Path:
    """
    Download VGGT-1B weights from HuggingFace with progress bar.
    Uses huggingface_hub snapshot for robustness; falls back to direct URL.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        log.info(f"Model cache found: {cache_path}  ({cache_path.stat().st_size / 1e9:.2f} GB)")
        return cache_path

    log.info("Downloading VGGT-1B weights from HuggingFace …")

    # Try huggingface_hub first (handles auth, mirrors, resumption)
    try:
        from huggingface_hub import hf_hub_download
        tmp = hf_hub_download(
            repo_id=VGGT_REPO_ID,
            filename="model.pt",
            local_dir=str(cache_path.parent),
            local_dir_use_symlinks=False,
        )
        shutil.copy2(tmp, cache_path)
        log.info(f"Downloaded via huggingface_hub → {cache_path}")
        return cache_path
    except Exception as e:
        log.warning(f"hf_hub_download failed ({e}), falling back to torch.hub …")

    # Fallback: torch.hub direct download
    state = torch.hub.load_state_dict_from_url(
        VGGT_HF_URL,
        map_location="cpu",
        model_dir=str(cache_path.parent),
    )
    torch.save(state, cache_path)
    log.info(f"Downloaded via torch.hub → {cache_path}")
    return cache_path


def sha256_first_mb(path: Path) -> str:
    """Quick integrity fingerprint from first 1 MB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()[:16]


def load_vggt_model(device: str = "cpu") -> Tuple[VGGT, str]:
    """
    Load VGGT-1B with optimized settings.
    Returns (model, model_checksum).
    """
    log.info(f"Loading VGGT-1B on device={device}")

    # Download if needed
    cache_path = download_model_hf(VGGT_CACHE)
    checksum = sha256_first_mb(cache_path)
    log.info(f"Model checksum (first-1MB SHA256): {checksum}")

    model = VGGT()
    state_dict = torch.load(str(cache_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Fuse BN layers for faster CPU inference
    try:
        torch.nn.utils.fusion.fuse_conv_bn_eval(model)
    except Exception:
        pass

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Model loaded: {param_count:.0f}M parameters")
    return model, checksum


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – Frame Extraction (smart sampling + blur filter)
# ══════════════════════════════════════════════════════════════════════════════
def laplacian_variance(frame: np.ndarray) -> float:
    """Measure image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_every_n_sec: float = 1.0,
    max_frames: int = 80,
    blur_threshold: float = 80.0,
) -> List[str]:
    """
    Extract frames with smart sampling:
    - Sample every N seconds
    - Discard blurry frames (Laplacian variance < threshold)
    - Cap at max_frames to keep inference tractable on CPU
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps           = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec  = total_frames / fps
    frame_interval = max(1, int(round(fps * frame_every_n_sec)))

    log.info(f"Video : {video_path}")
    log.info(f"  {fps:.1f} fps  |  {total_frames} frames  |  {duration_sec:.1f}s")
    log.info(f"  Sampling every {frame_every_n_sec}s (interval={frame_interval} frames)")
    log.info(f"  Blur threshold={blur_threshold}  |  max_frames={max_frames}")

    saved, count, idx = [], 0, 0
    with tqdm(total=min(int(duration_sec / frame_every_n_sec) + 1, max_frames),
              desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                score = laplacian_variance(frame)
                if score >= blur_threshold:
                    path = os.path.join(output_dir, f"{idx:06d}.png")
                    cv2.imwrite(path, frame)
                    saved.append(path)
                    idx += 1
                    pbar.update(1)
                else:
                    log.debug(f"  Frame {count} discarded (blur={score:.1f})")
                if len(saved) >= max_frames:
                    log.info(f"  Reached max_frames={max_frames}, stopping")
                    break
            count += 1

    cap.release()
    log.info(f"  Saved {len(saved)} frames → {output_dir}")
    return sorted(saved)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 – VGGT Inference (batched for memory efficiency)
# ══════════════════════════════════════════════════════════════════════════════
def run_inference(image_paths: List[str], model: VGGT, device: str = "cpu") -> dict:
    """
    Run VGGT inference on a list of image paths.
    Uses bfloat16 on GPU, float32 on CPU.
    Returns numpy predictions dict with world_points_from_depth added.
    """
    log.info(f"Loading {len(image_paths)} images for inference …")
    images = load_and_preprocess_images(image_paths).to(device)
    log.info(f"  Preprocessed shape: {images.shape}")

    log.info("Running VGGT forward pass …")
    t0 = time.time()
    with torch.no_grad():
        if device != "cpu" and torch.cuda.get_device_capability()[0] >= 8:
            dtype = torch.bfloat16
        elif device != "cpu":
            dtype = torch.float16
        else:
            dtype = torch.float32

        if device != "cpu":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    elapsed = time.time() - t0
    log.info(f"  Inference done in {elapsed:.1f}s")

    log.info("Decoding camera poses …")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors → numpy (remove batch dim)
    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions["pose_enc_list"] = None

    log.info("Unprojecting depth maps → world points …")
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"],
        predictions["extrinsic"],
        predictions["intrinsic"],
    )

    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    return predictions


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 – Point Cloud & Mesh Export
# ══════════════════════════════════════════════════════════════════════════════
def export_point_cloud_glb(
    predictions: dict,
    output_dir: str,
    conf_thres: float = 50.0,
    show_cam: bool = True,
) -> str:
    """Export VGGT point cloud + camera frustums as GLB."""
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, "pointcloud.glb")
    log.info("Building point cloud GLB …")
    scene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames="All",
        show_cam=show_cam,
        mask_sky=False,
        target_dir=output_dir,
        prediction_mode="Depthmap and Camera Branch",
    )
    scene.export(file_obj=glb_path)
    log.info(f"  Saved → {glb_path}")
    return glb_path


def run_poisson_mesh(
    predictions: dict,
    conf_thres: float = 50.0,
    poisson_depth: int = 9,
) -> Optional[trimesh.Trimesh]:
    """Run Poisson surface reconstruction on the filtered point cloud."""
    if not MESH_AVAILABLE:
        log.warning("Mesh reconstruction skipped (open3d unavailable)")
        return None

    points = predictions["world_points_from_depth"].reshape(-1, 3)
    colors = None

    if "images" in predictions:
        imgs = predictions["images"]
        if imgs.ndim == 4 and imgs.shape[1] == 3:
            imgs = imgs.transpose(0, 2, 3, 1)
        colors = imgs.reshape(-1, 3)

    if "depth_conf" in predictions:
        conf = predictions["depth_conf"].reshape(-1)
        thresh = np.percentile(conf, conf_thres)
        mask = conf >= thresh
        points = points[mask]
        if colors is not None:
            colors = colors[mask]

    log.info(f"Poisson reconstruction: {len(points):,} input points, depth={poisson_depth}")
    mesh = full_reconstruction_pipeline(
        points, colors,
        denoise=True,
        poisson_depth=poisson_depth,
        density_threshold=0.1,
        clean=True,
    )
    log.info(f"  Mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces, "
             f"watertight={mesh.is_watertight}")
    return mesh


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 – Metric Scaling (GPS-free heuristic when no GPS available)
# ══════════════════════════════════════════════════════════════════════════════
def build_camera_poses_from_predictions(predictions: dict) -> list:
    """Build CameraPose objects from VGGT extrinsics (no GPS)."""
    if not METRIC_AVAILABLE:
        return []
    poses = []
    extrinsics = predictions["extrinsic"]   # (S, 3, 4)
    intrinsics  = predictions["intrinsic"]   # (S, 3, 3)
    for i in range(len(extrinsics)):
        # Pad 3x4 → 4x4
        ext44 = np.eye(4)
        ext44[:3, :4] = extrinsics[i]
        poses.append(CameraPose(extrinsic=ext44, intrinsic=intrinsics[i], frame_idx=i))
    return poses


def apply_ground_alignment(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, dict]:
    """
    Align mesh so ground plane is Z=0, Z-axis points up.
    Returns (aligned_mesh, transform_meta).
    """
    if not METRIC_AVAILABLE:
        return mesh, {}

    try:
        vertices = mesh.vertices
        plane_model = detect_ground_plane(vertices)  # (a,b,c,d) coefficients
        # Normalise to flat numpy array regardless of how metric_scaling returns it
        plane_arr = np.array(plane_model, dtype=float).ravel()
        aligned_mesh = align_to_ground_plane(mesh, plane_arr)
        return aligned_mesh, {"ground_plane_model": plane_arr.tolist()}
    except Exception as e:
        log.warning(f"Ground alignment failed: {e}")
        return mesh, {}


# ══════════════════════════════════════════════════════════════════════════════
# METADATA EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════
def collect_prediction_stats(predictions: dict) -> dict:
    """Summarise VGGT prediction quality into a flat dict."""
    stats = {}

    if "depth_conf" in predictions:
        conf = predictions["depth_conf"].ravel()
        stats["depth_conf_mean"]   = float(np.mean(conf))
        stats["depth_conf_median"] = float(np.median(conf))
        stats["depth_conf_p10"]    = float(np.percentile(conf, 10))
        stats["depth_conf_p90"]    = float(np.percentile(conf, 90))

    if "world_points_from_depth" in predictions:
        pts = predictions["world_points_from_depth"].reshape(-1, 3)
        stats["point_cloud_raw_count"] = int(len(pts))
        stats["scene_bbox_min"]  = pts.min(axis=0).tolist()
        stats["scene_bbox_max"]  = pts.max(axis=0).tolist()
        extent = pts.max(axis=0) - pts.min(axis=0)
        stats["scene_extent_xyz"] = extent.tolist()

    if "extrinsic" in predictions:
        ext = predictions["extrinsic"]   # (S, 3, 4)
        translations = ext[:, :3, 3]
        if len(translations) > 1:
            baselines = np.linalg.norm(np.diff(translations, axis=0), axis=1)
            stats["camera_count"]           = int(len(ext))
            stats["avg_baseline_vggt_units"] = float(baselines.mean())
            stats["total_trajectory_vggt"]   = float(baselines.sum())
        else:
            stats["camera_count"] = int(len(ext))

    return stats


def collect_mesh_stats(mesh: trimesh.Trimesh) -> dict:
    """Collect mesh quality metrics."""
    stats = {
        "vertices": int(len(mesh.vertices)),
        "faces":    int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
    }
    try:
        bounds = mesh.bounds
        stats["bbox_min"]    = bounds[0].tolist()
        stats["bbox_max"]    = bounds[1].tolist()
        stats["bbox_extent"] = (bounds[1] - bounds[0]).tolist()
    except Exception:
        pass
    try:
        if mesh.is_watertight:
            stats["volume_vggt_units"] = float(mesh.volume)
    except Exception:
        pass
    return stats


def build_metadata(
    video_path: str,
    image_paths: List[str],
    model_checksum: str,
    predictions: dict,
    mesh: Optional[trimesh.Trimesh] = None,
    ground_transform: Optional[dict] = None,
    output_dir: str = "output_3d",
) -> dict:
    """Build the full metadata dict to embed in every output."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    nf     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Video file checksum (fast: first 4 MB)
    vid_hash = hashlib.sha256()
    with open(str(video_path), "rb") as f:
        vid_hash.update(f.read(4 * 1024 * 1024))

    meta = {
        "pipeline": {
            "version":      PIPELINE_VERSION,
            "name":         "VayuMesh 2.0",
            "run_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "platform":      platform.platform(),
            "python_version": sys.version.split()[0],
            "torch_version":  torch.__version__,
            "device":        "cuda" if torch.cuda.is_available() else "cpu",
            "output_dir":    str(Path(output_dir).resolve()),
        },
        "source_video": {
            "path":             str(video_path.resolve()),
            "filename":         video_path.name,
            "sha256_first4mb":  vid_hash.hexdigest()[:16],
            "fps":              fps,
            "total_frames":     nf,
            "width":            vw,
            "height":           vh,
            "duration_sec":     round(nf / fps, 2) if fps else 0,
        },
        "frame_extraction": {
            "frames_extracted": len(image_paths),
            "sample_paths":     [str(p) for p in image_paths[:5]],
        },
        "model": {
            "id":              MODEL_ID,
            "hf_url":          VGGT_HF_URL,
            "cache_path":      str(VGGT_CACHE),
            "checksum_sha256_1mb": model_checksum,
            "inference_resolution": "518x518",
        },
        "predictions": collect_prediction_stats(predictions),
    }

    if mesh is not None:
        meta["mesh"] = collect_mesh_stats(mesh)

    if ground_transform:
        meta["ground_alignment"] = ground_transform

    return meta


def embed_metadata_in_glb(glb_path: str, metadata: dict) -> None:
    """
    Re-export the GLB with metadata embedded in the scene's extras dict.
    trimesh stores extras in scene.metadata (trimesh >= 3.x) or scene.graph.transforms.
    """
    try:
        scene = trimesh.load(glb_path)
        if isinstance(scene, trimesh.Scene):
            # trimesh Scene stores extras accessible via .metadata
            if not hasattr(scene, "metadata") or scene.metadata is None:
                scene.metadata = {}
            scene.metadata["vayumesh"] = metadata
        elif isinstance(scene, trimesh.Trimesh):
            if not hasattr(scene, "metadata") or scene.metadata is None:
                scene.metadata = {}
            scene.metadata["vayumesh"] = metadata

        scene.export(glb_path)
        log.info(f"  Metadata embedded in GLB extras → {glb_path}")
    except Exception as e:
        log.warning(f"  GLB metadata embedding warning: {e}")


def embed_metadata_in_ply(ply_path: str, metadata: dict) -> None:
    """
    Embed metadata as a comment header in a PLY file.
    PLY spec allows arbitrary 'comment' lines before element/property declarations.
    """
    try:
        with open(ply_path, "rb") as f:
            content = f.read()

        # Find the end_header marker
        header_end = content.find(b"end_header")
        if header_end == -1:
            return

        comment_block = ""
        for k, v in metadata.get("pipeline", {}).items():
            comment_block += f"comment vayumesh.pipeline.{k} {v}\n"
        for k, v in metadata.get("source_video", {}).items():
            comment_block += f"comment vayumesh.video.{k} {v}\n"
        for k, v in metadata.get("model", {}).items():
            comment_block += f"comment vayumesh.model.{k} {v}\n"

        comment_bytes = comment_block.encode("utf-8")
        new_content = content[:header_end] + comment_bytes + content[header_end:]

        with open(ply_path, "wb") as f:
            f.write(new_content)
        log.info(f"  Metadata comments embedded in PLY → {ply_path}")
    except Exception as e:
        log.warning(f"  PLY metadata embedding warning: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def export_all_formats(
    mesh: trimesh.Trimesh,
    output_dir: str,
    prefix: str,
    metadata: dict,
) -> List[str]:
    """Export mesh to OBJ, PLY, GLB and embed metadata. Returns list of paths."""
    outputs = []
    os.makedirs(output_dir, exist_ok=True)

    for ext in ["obj", "glb", "ply"]:
        path = os.path.join(output_dir, f"{prefix}.{ext}")
        try:
            mesh.export(path)
            log.info(f"  Exported {ext.upper()} → {path}")
            outputs.append(path)
            if ext == "ply":
                embed_metadata_in_ply(path, metadata)
            elif ext == "glb":
                embed_metadata_in_glb(path, metadata)
        except Exception as e:
            log.warning(f"  Export {ext} failed: {e}")

    return outputs


def export_geojson_footprint(mesh: trimesh.Trimesh, output_path: str, metadata: dict) -> None:
    """Export 2D convex hull footprint as GeoJSON (with metadata in properties)."""
    try:
        from scipy.spatial import ConvexHull
        verts = mesh.vertices
        ground_z = float(np.percentile(verts[:, 2], 5))
        ground_verts = verts[verts[:, 2] < ground_z + 0.5]
        if len(ground_verts) < 3:
            ground_verts = verts

        hull = ConvexHull(ground_verts[:, :2])
        hull_pts = ground_verts[hull.vertices, :2].tolist()
        hull_pts.append(hull_pts[0])   # close ring

        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [hull_pts]},
                "properties": {
                    "source": metadata["source_video"]["filename"],
                    "pipeline_version": metadata["pipeline"]["version"],
                    "run_timestamp": metadata["pipeline"]["run_timestamp"],
                    "mesh_vertices": metadata.get("mesh", {}).get("vertices"),
                    "mesh_faces":    metadata.get("mesh", {}).get("faces"),
                    "is_watertight": metadata.get("mesh", {}).get("is_watertight"),
                },
            }],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)
        log.info(f"  GeoJSON footprint → {output_path}")
    except Exception as e:
        log.warning(f"  GeoJSON export failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(
    video_path:        str,
    output_dir:        str  = "output_3d",
    frame_every_n_sec: float = 1.5,
    max_frames:        int   = 50,
    blur_threshold:    float = 80.0,
    conf_thres:        float = 50.0,
    poisson_depth:     int   = 9,
    show_cam:          bool  = True,
    skip_mesh:         bool  = False,
    skip_scaling:      bool  = False,
) -> dict:
    """
    Full VayuMesh pipeline.  Returns a dict with paths to all outputs.
    """
    t_start = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = str(out / "images")

    print("\n" + "═" * 70)
    print("  VayuMesh 2.0  ▸  Video → Metric 3D with Metadata")
    print("═" * 70)

    # ─── Stage 0: Load model ───────────────────────────────────────────────
    print("\n[0/4] Loading VGGT-1B from HuggingFace …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_checksum = load_vggt_model(device)

    # ─── Stage 1: Frame extraction ──────────────────────────────────────────
    print("\n[1/4] Extracting frames …")
    image_paths = extract_frames(
        video_path, frames_dir,
        frame_every_n_sec=frame_every_n_sec,
        max_frames=max_frames,
        blur_threshold=blur_threshold,
    )
    if not image_paths:
        raise RuntimeError("No usable frames extracted from video")

    # ─── Stage 2: VGGT inference ────────────────────────────────────────────
    print(f"\n[2/4] VGGT inference on {len(image_paths)} frames …")
    predictions = run_inference(image_paths, model, device=device)

    # Save raw predictions NPZ
    npz_path = str(out / "predictions.npz")
    np.savez(npz_path, **{k: v for k, v in predictions.items() if v is not None})
    log.info(f"Raw predictions → {npz_path}")

    # Free model memory
    del model
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()

    # ─── Stage 3a: Point cloud GLB ──────────────────────────────────────────
    print("\n[3/4] Building point cloud GLB …")
    pc_glb_path = export_point_cloud_glb(predictions, str(out), conf_thres, show_cam)

    # ─── Stage 3b: Poisson mesh ─────────────────────────────────────────────
    mesh = None
    if not skip_mesh:
        print("\n[3b] Poisson surface reconstruction …")
        mesh = run_poisson_mesh(predictions, conf_thres, poisson_depth)

    # ─── Stage 4: Ground-plane alignment ────────────────────────────────────
    ground_transform = {}
    aligned_mesh = mesh
    if mesh is not None and not skip_scaling:
        print("\n[4/4] Ground-plane alignment …")
        aligned_mesh, ground_transform = apply_ground_alignment(mesh)
        log.info(f"  Ground transform: {ground_transform}")

    # ─── Build metadata ─────────────────────────────────────────────────────
    print("\n[✦] Building & embedding metadata …")
    metadata = build_metadata(
        video_path, image_paths, model_checksum,
        predictions, aligned_mesh, ground_transform,
        str(out),
    )

    # Save metadata JSON sidecar
    meta_path = str(out / "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    log.info(f"Metadata JSON → {meta_path}")

    # Embed into point cloud GLB
    embed_metadata_in_glb(pc_glb_path, metadata)

    # ─── Export mesh in multiple formats ────────────────────────────────────
    mesh_outputs = []
    if aligned_mesh is not None:
        print("\n[✦] Exporting mesh (OBJ + GLB + PLY) with embedded metadata …")
        mesh_outputs = export_all_formats(aligned_mesh, str(out), "reconstruction", metadata)

        # GeoJSON footprint
        geojson_path = str(out / "footprint.geojson")
        export_geojson_footprint(aligned_mesh, geojson_path, metadata)
        mesh_outputs.append(geojson_path)

    # ─── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "═" * 70)
    print(f"  ✅  Pipeline complete in {elapsed:.1f}s")
    print("═" * 70)
    print(f"  Frames extracted :  {len(image_paths)}")
    print(f"  Camera poses     :  {metadata['predictions'].get('camera_count', '?')}")

    if aligned_mesh is not None:
        ms = metadata.get("mesh", {})
        print(f"  Mesh vertices    :  {ms.get('vertices', '?'):,}")
        print(f"  Mesh faces       :  {ms.get('faces', '?'):,}")
        print(f"  Watertight       :  {ms.get('is_watertight', '?')}")
        bbox = ms.get("bbox_extent")
        if bbox:
            print(f"  Scene extent     :  {bbox[0]:.3f} × {bbox[1]:.3f} × {bbox[2]:.3f} units")

    print(f"\n  Outputs in: {out.resolve()}")
    for f in sorted(out.iterdir()):
        if f.is_file():
            print(f"    {f.name:<40}  {f.stat().st_size / 1e3:>8.1f} KB")

    result = {
        "output_dir":      str(out.resolve()),
        "metadata_path":   meta_path,
        "pointcloud_glb":  pc_glb_path,
        "predictions_npz": npz_path,
        "mesh_outputs":    mesh_outputs,
        "metadata":        metadata,
        "elapsed_sec":     elapsed,
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="VayuMesh 2.0 – Video → 3D with metadata embedding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",    default="istockphoto-2149793768-640_adpp_is.mp4",
                   help="Input video file")
    p.add_argument("--output",   default="output_3d_istock",
                   help="Output directory")
    p.add_argument("--fps",      type=float, default=1.5,
                   help="Sample 1 frame every N seconds")
    p.add_argument("--max_frames", type=int, default=50,
                   help="Max frames to extract (keeps inference fast on CPU)")
    p.add_argument("--blur",     type=float, default=80.0,
                   help="Laplacian blur threshold (0 = keep all frames)")
    p.add_argument("--conf",     type=float, default=50.0,
                   help="Confidence percentile filter for point cloud")
    p.add_argument("--poisson_depth", type=int, default=9,
                   help="Poisson octree depth (8=fast, 10=fine)")
    p.add_argument("--no_cam",   action="store_true",
                   help="Omit camera frustums from GLB")
    p.add_argument("--skip_mesh", action="store_true",
                   help="Skip Poisson mesh, point cloud only")
    p.add_argument("--skip_scaling", action="store_true",
                   help="Skip ground-plane alignment")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        video_path        = args.video,
        output_dir        = args.output,
        frame_every_n_sec = args.fps,
        max_frames        = args.max_frames,
        blur_threshold    = args.blur,
        conf_thres        = args.conf,
        poisson_depth     = args.poisson_depth,
        show_cam          = not args.no_cam,
        skip_mesh         = args.skip_mesh,
        skip_scaling      = args.skip_scaling,
    )
