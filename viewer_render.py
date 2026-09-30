"""
VayuMesh 2.0 — Render.com Deployment
======================================
Viewer-only dashboard optimised for cloud hosting.

Architecture on Render
----------------------
* VGGT inference (5 GB model, ~12 GB RAM) is NOT run on Render.
  Run the pipeline LOCALLY with run_pipeline.py, then commit/upload
  the output GLB + metadata.json to the precomputed/ folder.

* This script serves:
  - Interactive 3-D viewers (5 modes) powered by Plotly
  - Pre-computed results for all 3 test videos
  - Full pipeline UI (runs if enough RAM is available)
  - Binds to 0.0.0.0:$PORT as required by Render

Deploy on Render
----------------
1. Push this repo to GitHub
2. Create new Web Service on render.com → connect repo
3. Build command:  pip install -r requirements_render.txt
4. Start command:  python viewer_render.py
5. Plan: Standard (2 GB RAM)  ← viewer needs ~200 MB

Local run
---------
    python viewer_render.py
    → http://127.0.0.1:7861
"""

import os
import sys
import gc
import json
import time
import shutil
import hashlib
import logging
import datetime
import platform
import threading
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Generator

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vggt"))

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import trimesh
import gradio as gr
import plotly.graph_objects as go

# ── Viewer engine ─────────────────────────────────────────────────────────────
from viewer_engine import (
    load_output_dir,
    make_ground_plane,
    infer_uncaptured_regions,
    render_pointcloud,
    render_depth_map,
    render_confidence,
    render_wireframe,
    render_solid_mesh,
    measure_distance,
    add_dimension_line,
)

# ── Optional pipeline (only if torch is installed) ────────────────────────────
TORCH_OK = False
try:
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from visual_util import predictions_to_glb
    TORCH_OK = True
except ImportError:
    pass

MESH_OK = False
try:
    from mesh_reconstruction import full_reconstruction_pipeline, export_mesh
    MESH_OK = True
except Exception:
    pass

METRIC_OK = False
try:
    from metric_scaling import CameraPose, detect_ground_plane, align_to_ground_plane
    METRIC_OK = True
except Exception:
    pass

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("vayumesh.render")

# ── Constants ─────────────────────────────────────────────────────────────────
VGGT_CACHE  = ROOT / ".cache" / "vggt_model.pt"
VGGT_HF     = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
VGGT_REPO   = "facebook/VGGT-1B"
OUT_ROOT    = ROOT / "dashboard_outputs"
PRECOMP     = ROOT / "precomputed"          # bundled pre-computed results
OUT_ROOT.mkdir(parents=True, exist_ok=True)
VER         = "2.0.0"

# Render environment
PORT        = int(os.environ.get("PORT", 7861))
IS_RENDER   = os.environ.get("RENDER", "") != ""
VIEWER_ONLY = os.environ.get("VAYUMESH_VIEWER_ONLY", "false").lower() == "true"

# ── Available pre-computed datasets ──────────────────────────────────────────
DEMOS = {}
for label, folder in [
    ("🕌 Mosque Aerial (1280×720)", "dashboard_outputs/20260929_163133"),
    ("🏛️ Taj Mahal HD (1920×1072)", "output_3d_taj"),
    ("🏙️ Urban iStock (768×432)",    "output_3d_istock"),
]:
    p = ROOT / folder
    if p.exists() and (p / "pointcloud.glb").exists():
        DEMOS[label] = str(p)

# Default to first available
DEFAULT_OUT = list(DEMOS.values())[0] if DEMOS else str(OUT_ROOT)

# ── Global model singleton ─────────────────────────────────────────────────────
_MODEL = None
_LOCK  = threading.Lock()


def get_model():
    global _MODEL
    if not TORCH_OK:
        raise RuntimeError("PyTorch not installed — pipeline unavailable on this deployment")
    with _LOCK:
        if _MODEL is None:
            VGGT_CACHE.parent.mkdir(parents=True, exist_ok=True)
            if not VGGT_CACHE.exists():
                LOG.info("Downloading VGGT-1B …")
                try:
                    from huggingface_hub import hf_hub_download
                    tmp = hf_hub_download(repo_id=VGGT_REPO, filename="model.pt",
                                          local_dir=str(VGGT_CACHE.parent),
                                          local_dir_use_symlinks=False)
                    shutil.copy2(tmp, VGGT_CACHE)
                except Exception as e:
                    LOG.warning(f"hf_hub failed ({e}), trying torch.hub")
                    sd = torch.hub.load_state_dict_from_url(
                        VGGT_HF, map_location="cpu",
                        model_dir=str(VGGT_CACHE.parent))
                    torch.save(sd, VGGT_CACHE)
            m = VGGT()
            sd = torch.load(str(VGGT_CACHE), map_location="cpu", weights_only=True)
            m.load_state_dict(sd)
            m.eval()
            _MODEL = m
            LOG.info(f"VGGT-1B ready ({sum(p.numel() for p in m.parameters())/1e6:.0f}M params)")
    return _MODEL


# ── Scene cache ───────────────────────────────────────────────────────────────
_SCENE_CACHE: Dict = {}


def _load_scene(out_dir: str) -> Dict:
    global _SCENE_CACHE
    if _SCENE_CACHE.get("_dir") == out_dir:
        return _SCENE_CACHE
    LOG.info(f"Loading scene: {out_dir}")
    data = load_output_dir(out_dir)
    if "pts" in data:
        data["ground"]   = make_ground_plane(data["pts"])
        data["inf_pts"]  = None
        data["inf_conf"] = None
    data["_dir"] = out_dir
    _SCENE_CACHE = data
    return _SCENE_CACHE


# ── Synthetic cameras for inference (no NPZ) ─────────────────────────────────
def _dummy_cameras(pts: np.ndarray):
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    z_top  = pts[:, 2].max() + 0.5
    r      = max(pts[:, 0].max() - pts[:, 0].min(),
                 pts[:, 1].max() - pts[:, 1].min()) * 0.7
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    centres = np.column_stack([cx + r * np.cos(angles),
                                cy + r * np.sin(angles),
                                np.full(8, z_top)])
    S = len(centres)
    extrinsics = np.zeros((S, 3, 4))
    intrinsics  = np.zeros((S, 3, 3))
    for i, c in enumerate(centres):
        fwd = np.array([cx, cy, pts[:, 2].mean()]) - c
        fwd /= np.linalg.norm(fwd) + 1e-12
        up    = np.array([0., 0., 1.])
        right = np.cross(fwd, up); right /= np.linalg.norm(right) + 1e-12
        up2   = np.cross(right, fwd)
        R = np.stack([right, up2, -fwd], axis=0)
        extrinsics[i, :3, :3] = R
        extrinsics[i, :3, 3]  = -R @ c
        fx = 518.0
        intrinsics[i] = [[fx, 0, 259.], [0, fx, 259.], [0, 0, 1]]
    return extrinsics, intrinsics


# ── View switch ───────────────────────────────────────────────────────────────
def switch_view(out_dir, mode, show_inf,
                p1x, p1y, p1z, p2x, p2y, p2z, show_dim):
    if not out_dir or not os.path.isdir(str(out_dir)):
        empty = go.Figure(layout=dict(
            paper_bgcolor="#0a0a10", plot_bgcolor="#0a0a10",
            scene=dict(bgcolor="#0a0a10"),
            annotations=[dict(text="Select a dataset from the dropdown above",
                              xref="paper", yref="paper", x=.5, y=.5,
                              showarrow=False, font=dict(color="#555", size=18))],
        ))
        return empty, "— no scene loaded —", ""

    data   = _load_scene(str(out_dir))
    pts    = data.get("pts")
    cols   = data.get("cols")
    conf   = data.get("conf")
    mesh   = data.get("mesh")
    ground = data.get("ground")
    meta   = data.get("metadata", {})

    if pts is None:
        return go.Figure(), "No point cloud found", ""

    if show_inf and data.get("inf_pts") is None:
        ext_d, intr_d = _dummy_cameras(pts)
        inf_pts, inf_conf = infer_uncaptured_regions(pts, ext_d, intr_d, n_samples=1000)
        data["inf_pts"]  = inf_pts
        data["inf_conf"] = inf_conf
        _SCENE_CACHE.update(data)

    inf_pts  = data.get("inf_pts")
    inf_conf = data.get("inf_conf")

    if mode == "Point Cloud":
        fig = render_pointcloud(pts, cols, ground, show_inf, inf_pts)
    elif mode == "Depth Map":
        fig = render_depth_map(pts, ground)
    elif mode == "Confidence Heatmap":
        fig = render_confidence(pts, conf, ground, inf_pts, inf_conf)
    elif mode == "Wireframe Mesh":
        fig = render_wireframe(mesh, pts, ground) if mesh is not None else render_pointcloud(pts, cols, ground)
    else:
        fig = render_solid_mesh(mesh, pts, ground) if mesh is not None else render_pointcloud(pts, cols, ground)

    # Dimension overlay
    dim_md = ""
    if show_dim:
        p1 = np.array([p1x, p1y, p1z])
        p2 = np.array([p2x, p2y, p2z])
        result = measure_distance(p1, p2)
        fig = add_dimension_line(fig, p1, p2)
        dim_md = (
            f"**P1** `({p1x:.3f}, {p1y:.3f}, {p1z:.3f})`  \n"
            f"**P2** `({p2x:.3f}, {p2y:.3f}, {p2z:.3f})`  \n\n"
            f"| | |\n|---|---|\n"
            f"| ΔX | `{result['dX_m']:.4f}` |\n"
            f"| ΔY | `{result['dY_m']:.4f}` |\n"
            f"| ΔZ | `{result['dZ_m']:.4f}` |\n"
            f"| **Distance** | **`{result['distance_m']:.4f}` units** |\n"
        )

    # Info panel
    vi  = meta.get("video", {})
    mi  = meta.get("mesh",  {})
    ci  = meta.get("confidence", {})
    gz  = ground.get("ground_z", 0.) if ground else 0.
    inf_n = len(inf_pts) if inf_pts is not None else 0

    info_md = f"""
**Video:** `{vi.get('file', Path(out_dir).name)}`
**Res:** {vi.get('w','?')}×{vi.get('h','?')} · **{vi.get('duration','?')} s** · {vi.get('fps',0):.0f} fps

| Metric | Value |
|--------|-------|
| Raw 3D points | {len(pts):,} |
| Ground Z (locked) | `{gz:.4f}` |
| Depth conf p90 | {ci.get('p90', 0):.2f} |
| Mesh vertices | {mi.get('vertices', '—'):,} |
| Mesh faces | {mi.get('faces', '—'):,} |
| Watertight | {mi.get('watertight', '—')} |
| Extent (units) | {' × '.join(f'{v:.3f}' for v in mi.get('extent', []))} |
| Inferred pts | {inf_n:,} |
| **Mode** | **`{mode}`** |
"""
    return fig, info_md, dim_md


# ── Pipeline generator ────────────────────────────────────────────────────────
def pipeline_gen(video_file, local_path_str,
                 every_n, max_fr, blur_t, conf_pct,
                 en_mesh, en_align, poisson_d, show_cams):
    t0    = time.time()
    lines = []
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def E(msg, status="⏳ Running …", out=""):
        lines.append(f"`{time.time()-t0:.1f}s` {msg}")
        yield "\n\n".join(lines[-50:]), status, out

    # Resolve video path
    vpath = None
    if video_file and os.path.isfile(str(video_file)):
        vpath = str(video_file)
    elif local_path_str and os.path.isfile(local_path_str.strip()):
        vpath = local_path_str.strip()

    if not vpath:
        yield from E("❌ No video found. Upload a file or paste a valid local path.",
                     "❌ No video"); return

    if not TORCH_OK:
        yield from E(
            "⚠️ **PyTorch not installed on this deployment.**\n\n"
            "The 3D viewers work fully with pre-computed results.\n"
            "To run the pipeline, install locally:\n"
            "```\npip install torch torchvision\n.venv/Scripts/python viewer_ultimate.py\n```",
            "⚠️ Pipeline unavailable on cloud"
        ); return

    # Check RAM
    try:
        import ctypes
        class MEM(ctypes.Structure):
            _fields_ = [("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),
                        ("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),
                        ("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),
                        ("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual",ctypes.c_ulonglong)]
        s=MEM(); s.dwLength=ctypes.sizeof(s)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        free_gb = s.ullAvailPhys / 1e9
    except Exception:
        import psutil
        free_gb = psutil.virtual_memory().available / 1e9

    if free_gb < 4.5:
        yield from E(
            f"⚠️ Only **{free_gb:.1f} GB** RAM free. VGGT needs ~5 GB.\n\n"
            f"Run the pipeline locally where more RAM is available, then upload outputs here.",
            f"⚠️ Low RAM ({free_gb:.1f} GB)"
        ); return

    yield from E("**[0/4]** Loading VGGT-1B …")
    try:
        model = get_model()
    except Exception as e:
        yield from E(f"❌ Model load failed: {e}", "❌ Error"); return
    yield from E("✅ Model ready")

    # Frames
    yield from E(f"**[1/4]** Extracting frames …")
    frames_dir = str(out_dir / "images")
    try:
        from viewer_ultimate import _extract
        image_paths = _extract(vpath, frames_dir, every_n, int(max_fr), blur_t)
    except Exception as e:
        yield from E(f"❌ Frame extraction: {e}", "❌ Error"); return
    yield from E(f"✅ {len(image_paths)} frames")

    # Inference
    yield from E(f"**[2/4]** VGGT inference …")
    try:
        images = load_and_preprocess_images(image_paths).to("cpu")
        with torch.no_grad():
            pred = model(images)
        ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
        pred["extrinsic"] = ext; pred["intrinsic"] = intr
        for k in list(pred.keys()):
            if isinstance(pred[k], torch.Tensor):
                pred[k] = pred[k].cpu().numpy().squeeze(0)
        pred["pose_enc_list"] = None
        pred["world_points_from_depth"] = unproject_depth_map_to_point_map(
            pred["depth"], pred["extrinsic"], pred["intrinsic"])
        gc.collect()
    except Exception as e:
        yield from E(f"❌ Inference: {e}", "❌ Error"); return

    n_pts = int(pred["world_points_from_depth"].reshape(-1,3).shape[0])
    yield from E(f"✅ {n_pts:,} 3D points")

    # Point cloud GLB
    pc_glb = str(out_dir / "pointcloud.glb")
    try:
        scene = predictions_to_glb(pred, conf_thres=conf_pct, filter_by_frames="All",
                                    show_cam=show_cams, mask_sky=False,
                                    target_dir=str(out_dir),
                                    prediction_mode="Depthmap and Camera Branch")
        scene.export(file_obj=pc_glb)
    except Exception as e:
        yield from E(f"❌ GLB export: {e}", "❌ Error"); return

    yield from E("✅ Point cloud ready", out=str(out_dir))

    if en_mesh and MESH_OK:
        pts_f = pred["world_points_from_depth"].reshape(-1,3)
        imgs  = pred.get("images")
        cols_f = None
        if imgs is not None:
            if imgs.ndim==4 and imgs.shape[1]==3: imgs = imgs.transpose(0,2,3,1)
            cols_f = imgs.reshape(-1,3)
        if "depth_conf" in pred and conf_pct > 0:
            c = pred["depth_conf"].reshape(-1)
            mask = c >= np.percentile(c, conf_pct)
            pts_f = pts_f[mask]
            if cols_f is not None: cols_f = cols_f[mask]
        try:
            mesh_obj = full_reconstruction_pipeline(pts_f, cols_f, denoise=True,
                                                     poisson_depth=int(poisson_d),
                                                     density_threshold=0.1, clean=True)
            mesh_glb = str(out_dir / "reconstruction.glb")
            export_mesh(mesh_obj, mesh_glb, "glb")
            export_mesh(mesh_obj, str(out_dir / "reconstruction.obj"), "obj")
            mesh_obj.export(str(out_dir / "reconstruction.ply"))
            yield from E(f"✅ Mesh: {len(mesh_obj.vertices):,} verts", out=str(out_dir))
        except Exception as e:
            yield from E(f"⚠️ Mesh: {e}", out=str(out_dir))

    # Metadata
    meta_data = {
        "pipeline": {"version": VER, "run_id": run_id,
                     "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "platform": platform.platform()},
        "video": {"file": Path(vpath).name},
        "model": {"id": VGGT_REPO},
        "frames": len(image_paths),
    }
    if "depth_conf" in pred:
        c = pred["depth_conf"].ravel()
        meta_data["confidence"] = {"mean": float(c.mean()), "p90": float(np.percentile(c,90))}
    with open(str(out_dir/"metadata.json"), "w") as f:
        json.dump(meta_data, f, indent=2, default=str)

    total = time.time()-t0
    yield from E(f"✅ **Done in {total:.1f}s** → `{out_dir}`",
                 status=f"✅ Done {total:.0f}s", out=str(out_dir))


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
html,body,.gradio-container{background:#08080f!important;color:#e2e2ec!important;
  font-family:'Segoe UI',system-ui,sans-serif!important}
.gradio-container{max-width:100%!important;padding:0!important;margin:0!important}
footer{display:none!important}
.gap,.contain{gap:0!important}

/* HEADER */
.vm-hdr{background:linear-gradient(135deg,#0a0820 0%,#1a1040 50%,#0f1832 100%);
  border-bottom:2px solid #ffd600;padding:13px 28px 11px;
  display:flex;align-items:center;justify-content:space-between}
.vm-hdr-left h1{color:#fff!important;font-size:1.45rem;font-weight:900;margin:0;line-height:1.2}
.vm-hdr-left p{color:#90caf9!important;font-size:.76rem;margin:3px 0 0;letter-spacing:.5px}
.vm-hdr-right{display:flex;flex-direction:column;align-items:flex-end;gap:4px}
.vm-sih{background:#ff6b00;color:#fff;font-weight:900;font-size:.7rem;
  padding:3px 9px;border-radius:4px;letter-spacing:1px}
.vm-pills{display:flex;gap:5px;flex-wrap:wrap;justify-content:flex-end}
.vm-pill{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.14);
  color:#cdd!important;font-size:.66rem;padding:2px 7px;border-radius:20px}

/* MODE BAR */
.mode-bar{background:#0e0e1a;border-bottom:1px solid #1e1e2e;
  padding:8px 20px!important;gap:8px!important;align-items:center!important}
.vm-btn{flex:1!important;min-width:0!important}
.vm-btn button{width:100%!important;background:#14142a!important;color:#90caf9!important;
  border:1px solid #2a2a42!important;border-radius:7px!important;
  font-size:.79rem!important;font-weight:600!important;padding:8px 4px!important;
  white-space:nowrap!important;height:38px!important;transition:all .15s!important}
.vm-btn button:hover{background:#1e1e38!important;border-color:#667eea!important;
  color:#fff!important;transform:translateY(-1px)}
.vm-active button{background:linear-gradient(135deg,#667eea,#764ba2)!important;
  color:#fff!important;border-color:#667eea!important;
  box-shadow:0 3px 14px rgba(102,126,234,.45)!important}
#out-dir-box textarea,#out-dir-box input{background:#0d0d1a!important;color:#cdd!important;
  border:1px solid #2a2a3a!important;border-radius:7px!important;
  font-size:.77rem!important;height:38px!important;padding:0 10px!important}
.status-ok textarea{background:#091409!important;color:#66bb6a!important;font-weight:700}
.status-err textarea{background:#1a0808!important;color:#ef5350!important;font-weight:700}
.status-run textarea{background:#08101a!important;color:#42a5f5!important}
.vm-status textarea{height:38px!important;min-height:38px!important;
  border-radius:7px!important;font-size:.79rem!important;padding:8px 12px!important}

/* DEMO DROPDOWN */
.demo-drop select,.demo-drop input{background:#0d0d1a!important;color:#e0e0e0!important;
  border:1px solid #2a2a3a!important;border-radius:7px!important;font-size:.82rem!important}

/* PLOT */
.vm-plot-col{padding:6px 6px 6px 8px!important}
.main-plot{background:#0a0a12;border-radius:10px;border:1px solid #1a1a2a;
  overflow:hidden;height:calc(100vh - 148px)!important;min-height:540px}
.main-plot>div,.main-plot .plotly,.main-plot iframe{
  height:calc(100vh - 150px)!important;min-height:538px!important;width:100%!important}

/* SIDE */
.vm-side-col{padding:6px 8px 6px 4px!important}
.vm-panel{background:#0e0e1a;border:1px solid #1e1e2e;border-radius:10px;
  padding:11px 13px;margin-bottom:8px}
.vm-panel-title{color:#90caf9;font-size:.76rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.8px;margin-bottom:7px;
  padding-bottom:5px;border-bottom:1px solid #1e1e2e;display:flex;
  align-items:center;gap:5px}
.info-panel .prose{background:transparent!important;color:#cde!important;
  font-size:.77rem!important;line-height:1.6!important}
.info-panel .prose b,.info-panel .prose strong{color:#fff!important}
.dim-panel .prose{background:#100e04!important;color:#ffd54f!important;
  font-size:.79rem!important;padding:7px!important;border-radius:6px!important;
  font-family:'Cascadia Code','Fira Code',monospace!important;line-height:1.5!important}
.coord-input input{background:#0a0a16!important;color:#90caf9!important;
  border:1px solid #2a2a3a!important;border-radius:5px!important;
  font-family:'Cascadia Code',monospace!important;font-size:.74rem!important;
  text-align:center!important;height:30px!important;padding:0 3px!important}
.coord-input label{font-size:.68rem!important;color:#667!important}
.vm-check label{font-size:.79rem!important;color:#bbc!important}
.vm-check input[type=checkbox]{accent-color:#667eea!important}
.log-win .prose{background:#06060e!important;color:#b0bec5!important;
  font-family:'Cascadia Code','Fira Code',monospace!important;font-size:.7rem!important;
  padding:9px!important;border-radius:6px!important;max-height:220px;
  overflow-y:auto!important;line-height:1.55!important}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#0a0a10}
::-webkit-scrollbar-thumb{background:#2a2a3a;border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:#667eea}
"""

HEADER_HTML = """
<div class="vm-hdr">
  <div class="vm-hdr-left">
    <h1>🚁 VayuMesh 1Pass &nbsp;·&nbsp; Ultimate Viewer</h1>
    <p>Fixed Ground &nbsp;·&nbsp; Confidence Heatmap &nbsp;·&nbsp; Wireframe / Solid Mesh &nbsp;·&nbsp; Uncaptured Inference &nbsp;·&nbsp; Dimension Tool</p>
  </div>
  <div class="vm-hdr-right">
    <span class="vm-sih">SIH 26158 &nbsp;·&nbsp; MoD INDIA</span>
    <div class="vm-pills">
      <span class="vm-pill">⚡ VGGT-1B 1.25B params</span>
      <span class="vm-pill">🔒 100% Offline capable</span>
      <span class="vm-pill">🎯 CVPR 2025 Best Paper</span>
    </div>
  </div>
</div>
"""


def _empty_fig(msg="Select a dataset above to begin"):
    return go.Figure(layout=dict(
        paper_bgcolor="#08080f", plot_bgcolor="#08080f",
        scene=dict(bgcolor="#08080f"),
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[dict(text=msg, xref="paper", yref="paper",
                          x=.5, y=.5, showarrow=False,
                          font=dict(color="#444", size=16))],
    ))


def build_app():
    with gr.Blocks(title="VayuMesh 2.0 · Ultimate Viewer",
                   css=CSS, theme=gr.themes.Base()) as app:

        gr.HTML(HEADER_HTML)

        current_dir  = gr.State(value=DEFAULT_OUT)
        current_mode = gr.State(value="Point Cloud")

        # ── MODE BAR ─────────────────────────────────────────────────────
        with gr.Row(elem_classes="mode-bar"):
            btn_pc   = gr.Button("☁️ Point Cloud",        elem_classes="vm-btn", size="sm")
            btn_dep  = gr.Button("🌈 Depth Map",           elem_classes="vm-btn", size="sm")
            btn_conf = gr.Button("🔥 Confidence Heatmap",  elem_classes="vm-btn", size="sm")
            btn_wire = gr.Button("🔲 Wireframe Mesh",      elem_classes="vm-btn", size="sm")
            btn_mesh = gr.Button("🏗️ Solid Mesh",          elem_classes="vm-btn", size="sm")

            if DEMOS:
                demo_dd = gr.Dropdown(
                    choices=list(DEMOS.keys()),
                    value=list(DEMOS.keys())[0],
                    label="", show_label=False,
                    container=False,
                    elem_classes="demo-drop",
                    scale=3,
                )
            else:
                out_dir_input = gr.Textbox(
                    value=DEFAULT_OUT, label="", show_label=False,
                    container=False, elem_id="out-dir-box", scale=3,
                )

            load_btn   = gr.Button("📂 Load", size="sm", variant="secondary")
            status_top = gr.Textbox(value="Ready", label="", show_label=False,
                                    interactive=False, container=False,
                                    elem_classes="status-ok vm-status", scale=2)

        # ── Fullscreen JS ──────────────────────────────────────────────────
        gr.HTML("""
<button id="vm-fs-btn" onclick="vmFS()" title="Fullscreen (F key)"
  style="position:fixed;top:10px;right:14px;z-index:9999;
         background:linear-gradient(135deg,#667eea,#764ba2);
         color:#fff;border:none;border-radius:8px;padding:6px 14px;
         font-size:.85rem;font-weight:700;cursor:pointer;
         box-shadow:0 2px 12px rgba(102,126,234,.5)">⛶ Full</button>
<script>
function vmFS(){
  var el=document.getElementById('vm-plot-wrap')||document.documentElement;
  if(!document.fullscreenElement){
    (el.requestFullscreen||el.webkitRequestFullscreen).call(el);
    document.getElementById('vm-fs-btn').textContent='✕ Exit';
  }else{
    (document.exitFullscreen||document.webkitExitFullscreen).call(document);
    document.getElementById('vm-fs-btn').textContent='⛶ Full';
  }
}
document.addEventListener('keydown',e=>{ if(e.key==='f'||e.key==='F') vmFS(); });
</script>""")

        # ── MAIN ROW ──────────────────────────────────────────────────────
        with gr.Row(equal_height=True):

            with gr.Column(scale=6, elem_classes="vm-plot-col"):
                main_plot = gr.Plot(
                    value=_empty_fig(),
                    elem_classes="main-plot",
                    elem_id="vm-plot-wrap",
                    show_label=False,
                    container=False,
                )

            with gr.Column(scale=2, min_width=270, elem_classes="vm-side-col"):

                with gr.Group(elem_classes="vm-panel"):
                    gr.HTML('<div class="vm-panel-title">📊 Scene Info</div>')
                    info_panel = gr.Markdown("*Select a dataset to see info.*",
                                             elem_classes="info-panel")

                with gr.Group(elem_classes="vm-panel"):
                    gr.HTML("""
<div class="vm-panel-title">🎨 Confidence Key</div>
<div style="display:grid;grid-template-columns:12px 1fr;gap:5px 8px;
            align-items:center;font-size:.76rem;line-height:1.5">
  <span style="background:#00c853;width:12px;height:12px;border-radius:3px;display:block"></span>
  <span><b style="color:#00c853">High ≥85%</b> — trust fully</span>
  <span style="background:#ffd600;width:12px;height:12px;border-radius:3px;display:block"></span>
  <span><b style="color:#ffd600">Med 60–85%</b> — use with caution</span>
  <span style="background:#ff1744;width:12px;height:12px;border-radius:3px;display:block"></span>
  <span><b style="color:#ff1744">Low &lt;60%</b> — re-fly / verify</span>
  <span style="background:#ff9800;width:12px;height:12px;border-radius:3px;display:block"></span>
  <span><b style="color:#ff9800">Orange</b> — inferred geometry</span>
</div>""")

                with gr.Group(elem_classes="vm-panel"):
                    gr.HTML('<div class="vm-panel-title">📏 Dimension Tool</div>')
                    show_dim = gr.Checkbox(False, label="Show measurement line",
                                           elem_classes="vm-check")
                    gr.HTML('<div style="font-size:.7rem;color:#556;margin:4px 0 5px">Point 1 (X · Y · Z)</div>')
                    with gr.Row():
                        p1x = gr.Number(label="X", value=0.0,   precision=3, scale=1, min_width=55, elem_classes="coord-input")
                        p1y = gr.Number(label="Y", value=0.0,   precision=3, scale=1, min_width=55, elem_classes="coord-input")
                        p1z = gr.Number(label="Z", value=0.5,   precision=3, scale=1, min_width=55, elem_classes="coord-input")
                    gr.HTML('<div style="font-size:.7rem;color:#556;margin:5px 0">Point 2 (X · Y · Z)</div>')
                    with gr.Row():
                        p2x = gr.Number(label="X", value=0.5,   precision=3, scale=1, min_width=55, elem_classes="coord-input")
                        p2y = gr.Number(label="Y", value=0.5,   precision=3, scale=1, min_width=55, elem_classes="coord-input")
                        p2z = gr.Number(label="Z", value=1.0,   precision=3, scale=1, min_width=55, elem_classes="coord-input")
                    dim_panel = gr.Markdown("*Tick above to measure.*",
                                            elem_classes="dim-panel")

                with gr.Group(elem_classes="vm-panel"):
                    gr.HTML('<div class="vm-panel-title">🔭 Uncaptured Inference</div>')
                    show_inf = gr.Checkbox(False, label="Show inferred geometry",
                                           elem_classes="vm-check")
                    gr.HTML('<div style="font-size:.71rem;color:#556;margin-top:4px;line-height:1.5">'
                            'Space Intersection (Eq.35) estimates<br>occluded regions from camera rays.</div>')

        # ── BOTTOM TABS ───────────────────────────────────────────────────
        with gr.Tabs():
            with gr.TabItem("🚀  Run Pipeline"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("#### 📥 Input")
                        vid_file = gr.File(label="Upload video (MP4/MOV/AVI)",
                                           file_types=["video",".mp4",".mov",".avi",".mkv"],
                                           type="filepath")
                        local_path = gr.Textbox(label="Or paste local path", lines=1,
                                                placeholder="/path/to/video.mp4")
                        gr.Markdown("#### ⚙️ Settings")
                        with gr.Row():
                            s_every = gr.Slider(0.5,5,value=1.5,step=0.5,label="Every N sec")
                            s_max   = gr.Slider(5,80,value=25,step=5,label="Max frames")
                        with gr.Row():
                            s_blur  = gr.Slider(0,200,value=60,step=10,label="Blur filter")
                            s_conf  = gr.Slider(0,90,value=50,step=5,label="Conf %")
                        with gr.Row():
                            s_pois  = gr.Slider(7,10,value=9,step=1,label="Poisson depth")
                        with gr.Row():
                            cb_mesh  = gr.Checkbox(True, label="Poisson mesh")
                            cb_align = gr.Checkbox(True, label="Ground align")
                            cb_cams  = gr.Checkbox(True, label="Cameras")
                        run_btn = gr.Button("🚀  Run Pipeline", variant="primary", size="lg")
                        pipe_status = gr.Textbox(label="Status", lines=2, interactive=False,
                                                 elem_classes="status-run vm-status")
                    with gr.Column(scale=2):
                        pipe_log = gr.Markdown("```\nWaiting …\n```",
                                               elem_classes="log-win")
                        pipe_out_dir = gr.Textbox(label="Latest output folder",
                                                  interactive=False, lines=1)

            with gr.TabItem("📐  Math Spec"):
                gr.Markdown(r"""
### Collinearity (Eq. 34) &nbsp; $[X\,Y\,Z]^\top = [X_0\,Y_0\,Z_0]^\top + \lambda R^T[x-x_0,\,y-y_0,\,-c]^\top$
### Space Intersection (Eq. 35) &nbsp; $\min_P\sum_i\|(P-C_i)-\langle P-C_i,d_i\rangle d_i\|^2$
### Scale (Eq. 4) &nbsp; $d_{of}=d_{oc}\cdot d_{AB}$
### Confidence &nbsp; $\alpha=\text{conf}^{255}$ · Green≥0.85 · Yellow 0.60–0.85 · Red<0.60
### Distance &nbsp; $d=\sqrt{\Delta X^2+\Delta Y^2+\Delta Z^2}\times s$
                """)

            with gr.TabItem("💾  Downloads & About"):
                gr.Markdown(f"""
### VayuMesh 2.0 — SIH 26158 · Ministry of Defence

**Core:** VGGT-1B (Meta AI + Oxford VGG, CVPR 2025 Best Paper)  
**Pipeline:** Frame extract → VGGT inference → Poisson mesh → Ground align → Metadata embed  

| Test Video | Mesh Vertices | Faces | Time (CPU) |
|---|---|---|---|
| iStock Urban (768×432) | 181,450 | 361,227 | 278 s |
| Mosque Aerial (1280×720) | 160,956 | 318,006 | ~360 s |
| **Taj Mahal HD (1920×1072)** | **434,317** | **862,812** | **707 s** |

**Outputs:** `pointcloud.glb` · `reconstruction.glb` · `reconstruction.obj` · `reconstruction.ply` · `metadata.json` · `footprint.geojson`

> ℹ️ **Cloud note:** VGGT-1B requires ~12 GB RAM. Run inference locally with `run_pipeline.py`, then load results here via the dropdown.
                """)

        # ══════════════════════════════════════════════════════════════════
        # EVENT WIRING
        # ══════════════════════════════════════════════════════════════════

        VIEW_OUT = [main_plot, info_panel, dim_panel]

        def _render(out_dir, mode, si, px,py,pz, qx,qy,qz, sd):
            fig, info, dim = switch_view(out_dir, mode, si, px,py,pz, qx,qy,qz, sd)
            return fig, info, dim

        def _set_mode(m, out_dir, si, px,py,pz, qx,qy,qz, sd):
            fig, info, dim = switch_view(out_dir, m, si, px,py,pz, qx,qy,qz, sd)
            return fig, info, dim, m, f"Mode: {m}"

        VIEW_IN_BASE = [current_dir, show_inf, p1x,p1y,p1z, p2x,p2y,p2z, show_dim]

        for btn, mode_name in [
            (btn_pc,   "Point Cloud"),
            (btn_dep,  "Depth Map"),
            (btn_conf, "Confidence Heatmap"),
            (btn_wire, "Wireframe Mesh"),
            (btn_mesh, "Solid Mesh"),
        ]:
            btn.click(
                fn=lambda *a, m=mode_name: _set_mode(m, *a),
                inputs=VIEW_IN_BASE,
                outputs=VIEW_OUT + [current_mode, status_top],
            )

        # Dropdown / load
        def _load_demo(key, mode, si, px,py,pz, qx,qy,qz, sd):
            path = DEMOS.get(key, key)
            fig, info, dim = switch_view(path, mode, si, px,py,pz, qx,qy,qz, sd)
            return fig, info, dim, path, f"✅ {Path(path).name}"

        def _load_path(path, mode, si, px,py,pz, qx,qy,qz, sd):
            fig, info, dim = switch_view(path, mode, si, px,py,pz, qx,qy,qz, sd)
            return fig, info, dim, path, f"✅ {Path(path).name}"

        if DEMOS:
            demo_dd.change(fn=_load_demo,
                           inputs=[demo_dd, current_mode]+VIEW_IN_BASE[1:],
                           outputs=VIEW_OUT+[current_dir, status_top])
        else:
            load_btn.click(fn=_load_path,
                           inputs=[out_dir_input, current_mode]+VIEW_IN_BASE[1:],
                           outputs=VIEW_OUT+[current_dir, status_top])

        # Toggles refresh
        for trigger in [show_dim, show_inf, p1x, p1y, p1z, p2x, p2y, p2z]:
            trigger.change(fn=_render, inputs=VIEW_IN_BASE, outputs=VIEW_OUT)

        # Pipeline
        def _pipe(vf, lp, ev, mf, bt, cp, em, ea, pd, sc):
            for log_md, status, out in pipeline_gen(vf, lp, ev, mf, bt, cp, em, ea, pd, sc):
                yield log_md, status, out

        run_btn.click(
            fn=_pipe,
            inputs=[vid_file, local_path, s_every, s_max, s_blur,
                    s_conf, cb_mesh, cb_align, s_pois, cb_cams],
            outputs=[pipe_log, pipe_status, pipe_out_dir],
        )

        # Auto-load first demo on startup
        def _startup():
            if DEFAULT_OUT and os.path.isdir(DEFAULT_OUT):
                fig, info, dim = switch_view(DEFAULT_OUT, "Point Cloud",
                                              False, 0,0,.5, .5,.5,1., False)
                return fig, info, dim, DEFAULT_OUT, "✅ Ready"
            return _empty_fig(), "", "", DEFAULT_OUT, "Ready"

        app.load(fn=_startup,
                 outputs=[main_plot, info_panel, dim_panel, current_dir, status_top])

    return app


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 68)
    print("  VayuMesh 2.0 — Render.com Viewer")
    print(f"  http://0.0.0.0:{PORT}")
    print("=" * 68)
    print(f"Viewer-only mode: {VIEWER_ONLY} | IS_RENDER: {IS_RENDER}")
    print(f"Torch available: {TORCH_OK} | Mesh available: {MESH_OK}")
    print(f"Pre-computed scenes: {list(DEMOS.keys())}")

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=not IS_RENDER,   # don't try to open browser on Render server
    )
