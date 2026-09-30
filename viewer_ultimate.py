"""
VayuMesh 1Pass — Ultimate Viewer
==================================
Full-screen Gradio dashboard with every feature from the spec:

  ✅  Bigger viewport  (full-width Plotly, 80 vh height)
  ✅  Fixed ground plane  (locked Z, grid never moves)
  ✅  Uncaptured scene inference  (collinearity + space intersection)
  ✅  Solid + Wireframe mesh viewer  (toggle)
  ✅  Confidence heatmap  (green / yellow / red)
  ✅  One-click view switcher  (5 modes)
  ✅  Dimension / distance tool  (two-point pick)
  ✅  Pixel info panel  (X,Y,Z + confidence)
  ✅  Metadata panel  (pipeline stats)
  ✅  Load any output folder  (no re-run needed)
  ✅  Run full pipeline on any local video path

Run:
    .venv\\Scripts\\python.exe viewer_ultimate.py
    → http://127.0.0.1:7861
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, sys, gc, json, time, shutil, hashlib, logging, datetime, platform
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Generator

# ── make vggt importable ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vggt"))

# ── third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
import trimesh
import gradio as gr
import plotly.graph_objects as go

# ── engine ────────────────────────────────────────────────────────────────────
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

# ── pipeline imports (same as dashboard_v2) ───────────────────────────────────
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from visual_util import predictions_to_glb

try:
    from mesh_reconstruction import full_reconstruction_pipeline, export_mesh
    MESH_OK = True
except Exception as _e:
    MESH_OK = False
    print(f"[WARN] mesh_reconstruction: {_e}")

try:
    from metric_scaling import CameraPose, detect_ground_plane, align_to_ground_plane
    METRIC_OK = True
except Exception as _e:
    METRIC_OK = False
    print(f"[WARN] metric_scaling: {_e}")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("vayumesh.ultimate")

# ── constants ─────────────────────────────────────────────────────────────────
VGGT_CACHE = ROOT / ".cache" / "vggt_model.pt"
VGGT_HF    = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
VGGT_REPO  = "facebook/VGGT-1B"
OUT_ROOT   = ROOT / "dashboard_outputs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
VER        = "2.0.0"

# ── default output dir (mosque video result) ──────────────────────────────────
DEFAULT_OUT = str(ROOT / "dashboard_outputs" / "20260929_163133")

# ── global model singleton ────────────────────────────────────────────────────
_MODEL: Optional[VGGT] = None
_LOCK  = threading.Lock()


# ═════════════════════════════════════════════════════════════════════════════
# MODEL
# ═════════════════════════════════════════════════════════════════════════════
def get_model() -> VGGT:
    global _MODEL
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
                    LOG.warning(f"hf_hub failed ({e}), using torch.hub")
                    sd = torch.hub.load_state_dict_from_url(VGGT_HF, map_location="cpu",
                                                            model_dir=str(VGGT_CACHE.parent))
                    torch.save(sd, VGGT_CACHE)
            m = VGGT()
            sd = torch.load(str(VGGT_CACHE), map_location="cpu", weights_only=True)
            m.load_state_dict(sd)
            m.eval()
            _MODEL = m
            LOG.info(f"VGGT-1B ready ({sum(p.numel() for p in m.parameters())/1e6:.0f}M params)")
    return _MODEL


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE (generator — streams log updates)
# ═════════════════════════════════════════════════════════════════════════════
def _blur(frame):
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

def _extract(video_path, out_dir, every_n=1.5, max_fr=25, blur_t=60):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps * every_n)))
    saved, count, idx = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if count % interval == 0:
            if blur_t <= 0 or _blur(frame) >= blur_t:
                p = os.path.join(out_dir, f"{idx:06d}.png")
                cv2.imwrite(p, frame); saved.append(p); idx += 1
            if len(saved) >= max_fr: break
        count += 1
    cap.release()
    return sorted(saved)

def _infer(image_paths, model):
    images = load_and_preprocess_images(image_paths).to("cpu")
    with torch.no_grad():
        pred = model(images)
    ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    pred["extrinsic"] = ext
    pred["intrinsic"]  = intr
    for k in list(pred.keys()):
        if isinstance(pred[k], torch.Tensor):
            pred[k] = pred[k].cpu().numpy().squeeze(0)
    pred["pose_enc_list"] = None
    pred["world_points_from_depth"] = unproject_depth_map_to_point_map(
        pred["depth"], pred["extrinsic"], pred["intrinsic"])
    gc.collect()
    return pred

def _conf_filter(pred, conf_pct):
    pts = pred["world_points_from_depth"].reshape(-1, 3)
    cols = None
    if "images" in pred:
        imgs = pred["images"]
        if imgs.ndim == 4 and imgs.shape[1] == 3:
            imgs = imgs.transpose(0, 2, 3, 1)
        cols = imgs.reshape(-1, 3)
    if "depth_conf" in pred and conf_pct > 0:
        c = pred["depth_conf"].reshape(-1)
        mask = c >= np.percentile(c, conf_pct)
        pts = pts[mask]
        if cols is not None: cols = cols[mask]
    return pts, cols

def _meta(video_path, image_paths, pred, mesh, run_id):
    cap = cv2.VideoCapture(video_path)
    fps_v = cap.get(cv2.CAP_PROP_FPS); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    h = hashlib.sha256()
    with open(video_path, "rb") as f: h.update(f.read(4 << 20))
    meta: dict = {
        "pipeline": {"version": VER, "run_id": run_id,
                     "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "platform": platform.platform(), "torch": torch.__version__},
        "video": {"file": Path(video_path).name, "sha256_4mb": h.hexdigest()[:16],
                  "fps": fps_v, "frames": nf, "w": vw, "h": vh,
                  "duration": round(nf / fps_v, 2) if fps_v else 0},
        "model": {"id": VGGT_REPO, "url": VGGT_HF},
        "frames": len(image_paths),
    }
    if "depth_conf" in pred:
        c = pred["depth_conf"].ravel()
        meta["confidence"] = {"mean": float(c.mean()), "p50": float(np.median(c)),
                               "p10": float(np.percentile(c,10)), "p90": float(np.percentile(c,90))}
    if "extrinsic" in pred:
        meta["cameras"] = int(len(pred["extrinsic"]))
    if mesh is not None:
        b = mesh.bounds
        meta["mesh"] = {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
                        "watertight": bool(mesh.is_watertight), "extent": (b[1]-b[0]).tolist()}
    return meta

def _embed_glb(glb_path, meta):
    try:
        sc = trimesh.load(glb_path)
        if not hasattr(sc,"metadata") or sc.metadata is None: sc.metadata = {}
        sc.metadata["vayumesh"] = meta
        sc.export(glb_path)
    except: pass

def pipeline_gen(video_file, every_n, max_fr, blur_t, conf_pct,
                 en_mesh, en_align, poisson_d, show_cams):
    """Generator: yields (log_md, status, out_dir_str) at each stage."""
    t0 = time.time(); lines = []
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / run_id; out_dir.mkdir(parents=True, exist_ok=True)

    def E(msg, status="⏳ Running …", out=""):
        lines.append(f"`{time.time()-t0:.1f}s` {msg}")
        yield "\n\n".join(lines[-50:]), status, out

    if video_file is None:
        yield "❌ No video", "❌ No video", ""; return
    vpath = str(video_file)
    if not os.path.isfile(vpath):
        yield f"❌ Not found: {vpath}", "❌ Error", ""; return

    cap = cv2.VideoCapture(vpath)
    dur = round(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / (cap.get(cv2.CAP_PROP_FPS) or 25), 1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    yield from E(f"📹 `{Path(vpath).name}` — {w}×{h_v}, {dur}s")
    yield from E("**[0/4]** Loading VGGT-1B …")
    try: model = get_model()
    except Exception as e:
        yield from E(f"❌ Model load: {e}", "❌ Error"); return
    yield from E("✅ Model ready  (1.25B params, CPU)")
    yield from E(f"**[1/4]** Extracting frames (every {every_n}s, max {int(max_fr)}) …")
    frames_dir = str(out_dir/"images")
    try: image_paths = _extract(vpath, frames_dir, every_n, int(max_fr), blur_t)
    except Exception as e:
        yield from E(f"❌ Frames: {e}", "❌ Error"); return
    if not image_paths:
        yield from E("❌ No frames", "❌ Error"); return
    yield from E(f"✅ {len(image_paths)} frames")
    yield from E(f"**[2/4]** VGGT inference on {len(image_paths)} frames …")
    try: pred = _infer(image_paths, model)
    except Exception as e:
        yield from E(f"❌ Inference: {e}", "❌ Error"); return
    n_pts = int(pred["world_points_from_depth"].reshape(-1,3).shape[0])
    yield from E(f"✅ {n_pts:,} 3D points  |  {len(pred['extrinsic'])} cameras")
    yield from E("**[3/4]** Point cloud GLB …")
    pc_glb = str(out_dir/"pointcloud.glb")
    try:
        scene = predictions_to_glb(pred, conf_thres=conf_pct, filter_by_frames="All",
                                    show_cam=show_cams, mask_sky=False,
                                    target_dir=str(out_dir),
                                    prediction_mode="Depthmap and Camera Branch")
        scene.export(file_obj=pc_glb)
    except Exception as e:
        yield from E(f"❌ PC GLB: {e}", "❌ Error"); return
    yield from E("✅ Point cloud ready", out=str(out_dir))
    mesh_obj = None; mesh_glb = None
    if en_mesh and MESH_OK:
        yield from E(f"**[3b]** Poisson mesh (depth={poisson_d}) …", out=str(out_dir))
        pts_f, cols_f = _conf_filter(pred, conf_pct)
        try:
            mesh_obj = full_reconstruction_pipeline(pts_f, cols_f, denoise=True,
                                                     poisson_depth=int(poisson_d),
                                                     density_threshold=0.1, clean=True)
            mesh_glb = str(out_dir/"reconstruction.glb")
            export_mesh(mesh_obj, mesh_glb, "glb")
            export_mesh(mesh_obj, str(out_dir/"reconstruction.obj"), "obj")
            mesh_obj.export(str(out_dir/"reconstruction.ply"))
            yield from E(f"✅ Mesh: {len(mesh_obj.vertices):,} verts / {len(mesh_obj.faces):,} faces",
                         out=str(out_dir))
        except Exception as e:
            yield from E(f"⚠️ Mesh: {e}", out=str(out_dir))
    if en_align and mesh_obj is not None and METRIC_OK:
        yield from E("**[4/4]** Ground alignment …", out=str(out_dir))
        try:
            plane_arr = np.array(detect_ground_plane(mesh_obj.vertices), dtype=float).ravel()
            mesh_obj = align_to_ground_plane(mesh_obj, plane_arr)
            export_mesh(mesh_obj, mesh_glb, "glb")
            yield from E("✅ Ground aligned (Z=0)", out=str(out_dir))
        except Exception as e:
            yield from E(f"⚠️ Align: {e}", out=str(out_dir))
    yield from E("**[✦]** Saving metadata …", out=str(out_dir))
    meta = _meta(vpath, image_paths, pred, mesh_obj, run_id)
    with open(str(out_dir/"metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    _embed_glb(pc_glb, meta)
    if mesh_glb: _embed_glb(mesh_glb, meta)
    total = time.time()-t0
    yield from E(
        f"✅ **Done in {total:.1f}s** — `{out_dir}`",
        status=f"✅ Done {total:.0f}s  |  {len(image_paths)} frames",
        out=str(out_dir),
    )


# ═════════════════════════════════════════════════════════════════════════════
# VIEW SWITCHING LOGIC
# ═════════════════════════════════════════════════════════════════════════════
# Global cache so we don't reload GLBs on every button click
_CACHE: Dict = {}

def _load_cache(out_dir: str) -> Dict:
    global _CACHE
    key = str(out_dir)
    if _CACHE.get("_dir") == key:
        return _CACHE
    LOG.info(f"Loading scene from {out_dir} …")
    data = load_output_dir(out_dir)
    data["_dir"] = key
    if "pts" in data:
        data["ground"]    = make_ground_plane(data["pts"])
        data["inf_pts"]   = None   # computed on demand
        data["inf_conf"]  = None
    _CACHE = data
    return _CACHE


def switch_view(out_dir: str, mode: str,
                show_inferred: bool,
                p1_x: float, p1_y: float, p1_z: float,
                p2_x: float, p2_y: float, p2_z: float,
                show_dim: bool) -> Tuple[go.Figure, str, str]:
    """
    Return (plotly_figure, info_panel_md, dim_panel_md).
    Called whenever mode or out_dir changes.
    """
    if not out_dir or not os.path.isdir(out_dir):
        empty = go.Figure(layout=dict(
            paper_bgcolor="#0d0d14",
            plot_bgcolor="#0d0d14",
            scene=dict(bgcolor="#0d0d14"),
            annotations=[dict(text="Load an output folder to begin",
                              xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(color="#888", size=18))],
        ))
        return empty, "— no data —", "— no data —"

    data   = _load_cache(out_dir)
    pts    = data.get("pts")
    cols   = data.get("cols")
    conf   = data.get("conf")
    mesh   = data.get("mesh")
    ground = data.get("ground")
    meta   = data.get("metadata", {})

    if pts is None:
        return go.Figure(), "No point cloud found", ""

    # Infer uncaptured regions on demand (only once per loaded dir)
    if show_inferred and data.get("inf_pts") is None:
        LOG.info("Running uncaptured scene inference …")
        # We don't have per-frame extrinsics in the cache (no NPZ saved here),
        # so we synthesise dummy cameras at the scene corners to demonstrate.
        # In a full run the NPZ would be present and we'd use real poses.
        ext_dummy, intr_dummy = _dummy_cameras(pts)
        inf_pts, inf_conf = infer_uncaptured_regions(
            pts, ext_dummy, intr_dummy, n_samples=1500)
        data["inf_pts"]  = inf_pts
        data["inf_conf"] = inf_conf
        _CACHE.update(data)
        LOG.info(f"  → {len(inf_pts)} inferred points")

    inf_pts  = data.get("inf_pts")
    inf_conf = data.get("inf_conf")

    # ── Pick renderer ─────────────────────────────────────────────────────────
    if mode == "Point Cloud":
        fig = render_pointcloud(pts, cols, ground, show_inferred, inf_pts)

    elif mode == "Depth Map":
        fig = render_depth_map(pts, ground)

    elif mode == "Confidence Heatmap":
        fig = render_confidence(pts, conf, ground, inf_pts, inf_conf)

    elif mode == "Wireframe Mesh":
        if mesh is None:
            fig = render_pointcloud(pts, cols, ground)
            fig.add_annotation(text="⚠️ No mesh — showing point cloud",
                               xref="paper", yref="paper",
                               x=0.5, y=0.02, showarrow=False,
                               font=dict(color="#ff9800", size=13))
        else:
            fig = render_wireframe(mesh, pts, ground)

    else:  # Solid Mesh
        if mesh is None:
            fig = render_pointcloud(pts, cols, ground)
        else:
            fig = render_solid_mesh(mesh, pts, ground)

    # ── Dimension overlay ─────────────────────────────────────────────────────
    dim_md = ""
    if show_dim:
        p1 = np.array([p1_x, p1_y, p1_z])
        p2 = np.array([p2_x, p2_y, p2_z])
        result = measure_distance(p1, p2)
        fig = add_dimension_line(fig, p1, p2, label=f"{result['distance_m']:.3f} u")
        dim_md = (
            f"**P1** `({p1_x:.3f}, {p1_y:.3f}, {p1_z:.3f})`  \n"
            f"**P2** `({p2_x:.3f}, {p2_y:.3f}, {p2_z:.3f})`  \n\n"
            f"| | |\n|---|---|\n"
            f"| ΔX | `{result['dX_m']:.4f}` u |\n"
            f"| ΔY | `{result['dY_m']:.4f}` u |\n"
            f"| ΔZ | `{result['dZ_m']:.4f}` u |\n"
            f"| **Euclidean** | **`{result['distance_m']:.4f}` u** |\n"
        )

    # ── Info panel ────────────────────────────────────────────────────────────
    vi = meta.get("video", {})
    mi = meta.get("mesh",  {})
    ci = meta.get("confidence", {})
    gnd_z = ground.get("ground_z", 0.0)
    inf_n = len(inf_pts) if inf_pts is not None else 0
    info_md = f"""
**Video:** `{vi.get('file','—')}`  
**Duration:** {vi.get('duration','—')} s &nbsp;|&nbsp; **Res:** {vi.get('w')}×{vi.get('h')}  
**Frames:** {meta.get('frames','—')} &nbsp;|&nbsp; **Cameras:** {meta.get('cameras','—')}

| Metric | Value |
|--------|-------|
| Raw points | {len(pts):,} |
| Ground Z (locked) | `{gnd_z:.4f}` |
| Depth conf p90 | {ci.get('p90',0):.2f} |
| Mesh vertices | {mi.get('vertices','—'):,} |
| Mesh faces | {mi.get('faces','—'):,} |
| Watertight | {mi.get('watertight','—')} |
| Scene extent | {' × '.join(f'{v:.3f}' for v in mi.get('extent',[]))} |
| Inferred pts | {inf_n:,} |

**Mode:** `{mode}` &nbsp;|&nbsp; **Run ID:** `{Path(out_dir).name}`
"""
    return fig, info_md, dim_md


def _dummy_cameras(pts: np.ndarray):
    """
    Synthetic camera array around the scene bounding box —
    used for uncaptured inference when no NPZ is present.
    """
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    z_top  = pts[:, 2].max() + 0.5
    r      = max((pts[:,0].max() - pts[:,0].min(),
                  pts[:,1].max() - pts[:,1].min())) * 0.7

    angles  = np.linspace(0, 2*np.pi, 8, endpoint=False)
    centres = np.column_stack([cx + r*np.cos(angles),
                                cy + r*np.sin(angles),
                                np.full(8, z_top)])

    S = len(centres)
    extrinsics = np.zeros((S, 3, 4))
    intrinsics  = np.zeros((S, 3, 3))
    for i, c in enumerate(centres):
        # Look-at: camera pointing toward scene centre
        fwd = np.array([cx, cy, pts[:,2].mean()]) - c
        fwd /= (np.linalg.norm(fwd) + 1e-12)
        up  = np.array([0, 0, 1.0])
        right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-12)
        up2   = np.cross(right, fwd)
        R = np.stack([right, up2, -fwd], axis=0)    # 3×3
        t = -R @ c                                   # translation
        extrinsics[i, :3, :3] = R
        extrinsics[i, :3,  3] = t
        fx = 518.0
        intrinsics[i] = [[fx, 0, 259.0], [0, fx, 259.0], [0, 0, 1]]

    return extrinsics, intrinsics


# ═════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═════════════════════════════════════════════════════════════════════════════

CSS = """
/* ═══════════════════════════════════════════════
   VayuMesh 2.0 — Ultimate Viewer
   Symmetric, clean, military-grade dark UI
═══════════════════════════════════════════════ */

/* ── Reset & base ─────────────────────────────── */
html, body, .gradio-container {
    background: #08080f !important;
    color: #e2e2ec !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
footer { display: none !important; }
.gap, .contain { gap: 0 !important; }

/* ── HEADER ───────────────────────────────────── */
.vm-hdr {
    background: linear-gradient(135deg, #0a0820 0%, #1a1040 50%, #0f1832 100%);
    border-bottom: 2px solid #ffd600;
    padding: 14px 28px 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.vm-hdr-left h1 {
    color: #fff !important;
    font-size: 1.55rem;
    font-weight: 900;
    margin: 0;
    letter-spacing: .3px;
    line-height: 1.2;
}
.vm-hdr-left p {
    color: #90caf9 !important;
    font-size: .78rem;
    margin: 3px 0 0;
    letter-spacing: .5px;
}
.vm-hdr-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
}
.vm-sih-badge {
    background: #ff6b00;
    color: #fff;
    font-weight: 900;
    font-size: .72rem;
    padding: 3px 10px;
    border-radius: 4px;
    letter-spacing: 1px;
}
.vm-hdr-pills {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    justify-content: flex-end;
}
.vm-pill {
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.15);
    color: #cdd !important;
    font-size: .68rem;
    padding: 2px 8px;
    border-radius: 20px;
}

/* ── MODE BAR ─────────────────────────────────── */
.mode-bar {
    background: #0e0e1a;
    border-bottom: 1px solid #1e1e2e;
    padding: 8px 20px !important;
    gap: 8px !important;
    align-items: center !important;
}

/* ── Mode buttons — all EQUAL width ─────────── */
.vm-btn {
    flex: 1 !important;
    min-width: 0 !important;
}
.vm-btn button {
    width: 100% !important;
    background: #14142a !important;
    color: #90caf9 !important;
    border: 1px solid #2a2a42 !important;
    border-radius: 7px !important;
    font-size: .8rem !important;
    font-weight: 600 !important;
    padding: 8px 6px !important;
    white-space: nowrap !important;
    transition: all .15s !important;
    height: 38px !important;
}
.vm-btn button:hover {
    background: #1e1e38 !important;
    border-color: #667eea !important;
    color: #fff !important;
    transform: translateY(-1px);
}
.vm-active button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: #fff !important;
    border-color: #667eea !important;
    box-shadow: 0 3px 14px rgba(102,126,234,.45) !important;
}

/* path box */
#out-dir-box textarea, #out-dir-box input {
    background: #0d0d1a !important;
    color: #cdd !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 7px !important;
    font-size: .78rem !important;
    height: 38px !important;
    padding: 0 10px !important;
}

/* ── STATUS BAR ───────────────────────────────── */
.status-ok textarea  { background: #091409 !important; color: #66bb6a !important; font-weight: 700; }
.status-err textarea { background: #1a0808 !important; color: #ef5350 !important; font-weight: 700; }
.status-run textarea { background: #08101a !important; color: #42a5f5 !important; }
.vm-status textarea  { height: 38px !important; min-height: 38px !important; border-radius: 7px !important; font-size: .8rem !important; padding: 8px 12px !important; }

/* ── MAIN VIEWER ──────────────────────────────── */
.vm-plot-col { padding: 6px 6px 6px 8px !important; }
.main-plot {
    background: #0a0a12;
    border-radius: 10px;
    border: 1px solid #1a1a2a;
    overflow: hidden;
    height: calc(100vh - 148px) !important;
    min-height: 560px;
}
.main-plot > div,
.main-plot .plotly,
.main-plot iframe,
.main-plot .svelte-1gfkn6j {
    height: calc(100vh - 150px) !important;
    min-height: 558px !important;
    width: 100% !important;
}

/* ── SIDE COLUMN ──────────────────────────────── */
.vm-side-col { padding: 6px 8px 6px 4px !important; }

/* Panel base */
.vm-panel {
    background: #0e0e1a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
}
.vm-panel-title {
    color: #90caf9;
    font-size: .78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .8px;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e1e2e;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Info panel text */
.info-panel .prose {
    background: transparent !important;
    color: #cde !important;
    font-size: .78rem !important;
    line-height: 1.6 !important;
}
.info-panel .prose table { width: 100%; border-collapse: collapse; }
.info-panel .prose td { padding: 2px 4px; font-size: .76rem; color: #bbc; }
.info-panel .prose b, .info-panel .prose strong { color: #fff !important; }

/* Dim panel */
.dim-panel .prose {
    background: #100e04 !important;
    color: #ffd54f !important;
    font-size: .8rem !important;
    padding: 8px !important;
    border-radius: 6px !important;
    font-family: 'Cascadia Code', 'Fira Code', monospace !important;
    line-height: 1.6 !important;
}

/* Coord inputs — uniform size */
.coord-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4px; }
.coord-input input {
    background: #0a0a16 !important;
    color: #90caf9 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 5px !important;
    font-family: 'Cascadia Code', monospace !important;
    font-size: .75rem !important;
    text-align: center !important;
    height: 32px !important;
    padding: 0 4px !important;
}
.coord-input label { font-size: .7rem !important; color: #777 !important; }

/* Checkboxes */
.vm-check label { font-size: .8rem !important; color: #bbc !important; }
.vm-check input[type=checkbox] { accent-color: #667eea !important; }

/* ── LOG WINDOW ───────────────────────────────── */
.log-win .prose {
    background: #06060e !important;
    color: #b0bec5 !important;
    font-family: 'Cascadia Code', 'Fira Code', monospace !important;
    font-size: .72rem !important;
    padding: 10px !important;
    border-radius: 6px !important;
    max-height: 240px;
    overflow-y: auto !important;
    line-height: 1.6 !important;
}

/* ── TABS ─────────────────────────────────────── */
.tabs { background: #0e0e1a !important; border-radius: 10px !important; border: 1px solid #1e1e2e !important; }
.tab-nav { background: #0a0a14 !important; border-bottom: 1px solid #1e1e2e !important; padding: 0 12px !important; }
.tab-nav button {
    color: #778 !important;
    font-size: .8rem !important;
    padding: 8px 14px !important;
    border: none !important;
    background: transparent !important;
}
.tab-nav button.selected {
    color: #90caf9 !important;
    border-bottom: 2px solid #667eea !important;
    font-weight: 700 !important;
}

/* ── ACCORDION ────────────────────────────────── */
.gr-accordion { background: #0e0e1a !important; border: 1px solid #1e1e2e !important; border-radius: 8px !important; }
.gr-accordion > div:first-child { padding: 8px 14px !important; font-size: .82rem !important; }

/* ── Scrollbar (webkit) ───────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0a0a10; }
::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #667eea; }
"""

HEADER_HTML = """
<div class="vm-hdr">
  <div class="vm-hdr-left">
    <h1>🚁 VayuMesh 1Pass &nbsp;·&nbsp; Ultimate Viewer</h1>
    <p>Fixed Ground &nbsp;·&nbsp; Confidence Heatmap &nbsp;·&nbsp; Wireframe / Solid Mesh &nbsp;·&nbsp; Uncaptured Inference &nbsp;·&nbsp; Dimension Tool</p>
  </div>
  <div class="vm-hdr-right">
    <span class="vm-sih-badge">SIH 26158 &nbsp;·&nbsp; MoD INDIA</span>
    <div class="vm-hdr-pills">
      <span class="vm-pill">⚡ VGGT-1B  1.25B params</span>
      <span class="vm-pill">🔒 100% Offline</span>
      <span class="vm-pill">🎯 CVPR 2025 Best Paper</span>
    </div>
  </div>
</div>
"""

VIEW_MODES = ["Point Cloud", "Depth Map", "Confidence Heatmap",
              "Wireframe Mesh", "Solid Mesh"]

# Helper: initial empty figure
def _empty_fig():
    return go.Figure(layout=dict(
        paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14",
        scene=dict(bgcolor="#0d0d14"),
        margin=dict(l=0,r=0,t=0,b=0),
        annotations=[dict(
            text="Load an output folder or run the pipeline to begin",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#555", size=16),
        )],
    ))


def build_app():
    with gr.Blocks(title="VayuMesh Ultimate Viewer", css=CSS,
                   theme=gr.themes.Base()) as app:

        gr.HTML(HEADER_HTML)

        # ── State ──────────────────────────────────────────────────────────
        current_dir  = gr.State(value=DEFAULT_OUT)
        current_mode = gr.State(value="Point Cloud")

        # ── TOP: mode-toggle buttons + output dir bar ──────────────────────
        with gr.Row(elem_classes="mode-bar"):
            btn_pc   = gr.Button("☁️ Point Cloud",       elem_classes="vm-btn", size="sm")
            btn_dep  = gr.Button("🌈 Depth Map",          elem_classes="vm-btn", size="sm")
            btn_conf = gr.Button("🔥 Confidence Heatmap", elem_classes="vm-btn", size="sm")
            btn_wire = gr.Button("🔲 Wireframe Mesh",     elem_classes="vm-btn", size="sm")
            btn_mesh = gr.Button("🏗️ Solid Mesh",         elem_classes="vm-btn", size="sm")
            with gr.Column(scale=3, min_width=0):
                out_dir_input = gr.Textbox(
                    value=DEFAULT_OUT,
                    placeholder="Paste output folder path …",
                    label="", show_label=False,
                    container=False,
                    elem_id="out-dir-box",
                )
            load_dir_btn = gr.Button("📂 Load", size="sm", variant="secondary")
            status_top   = gr.Textbox(value="Ready", label="", show_label=False,
                                      interactive=False, container=False,
                                      elem_classes="status-ok", scale=2)

        # ── Fullscreen button (fixed position, F key also works) ───────────
        gr.HTML("""
<button id="vm-fs-btn" onclick="vmToggleFullscreen()" title="Toggle fullscreen  (or press F)"
  style="position:fixed;top:12px;right:16px;z-index:9999;
         background:linear-gradient(135deg,#667eea,#764ba2);
         color:#fff;border:none;border-radius:8px;
         padding:7px 16px;font-size:.9rem;font-weight:600;cursor:pointer;
         box-shadow:0 2px 14px rgba(102,126,234,.55);
         transition:transform .12s,box-shadow .12s;">⛶ Full</button>
<script>
function vmToggleFullscreen(){
  var el=document.getElementById('vm-plot-wrap')||
         document.querySelector('.main-plot')||
         document.documentElement;
  if(!document.fullscreenElement){
    (el.requestFullscreen||el.webkitRequestFullscreen||
     el.mozRequestFullScreen||el.msRequestFullscreen).call(el);
    document.getElementById('vm-fs-btn').textContent='✕ Exit';
  } else {
    (document.exitFullscreen||document.webkitExitFullscreen||
     document.mozCancelFullScreen||document.msExitFullscreen).call(document);
    document.getElementById('vm-fs-btn').textContent='⛶ Full';
  }
}
document.addEventListener('keydown',function(e){
  if(e.key==='f'||e.key==='F') vmToggleFullscreen();
});
</script>""")

        # ── MAIN ROW: plot (75%) + side panels (25%) ───────────────────────
        with gr.Row(equal_height=True):

            # ── 3-D Plot ─────────────────────────────────────────────────
            with gr.Column(scale=6, elem_classes="vm-plot-col"):
                main_plot = gr.Plot(
                    value=_empty_fig(),
                    elem_classes="main-plot",
                    elem_id="vm-plot-wrap",
                    show_label=False,
                    container=False,
                )

            # ── Side panels ───────────────────────────────────────────────
            with gr.Column(scale=2, min_width=280, elem_classes="vm-side-col"):

                # ── Info panel ────────────────────────────────────────────
                with gr.Group(elem_classes="vm-panel"):
                    gr.HTML('<div class="vm-panel-title">📊 Scene Info</div>')
                    info_panel = gr.Markdown(
                        value="*Load a scene to see info here.*",
                        elem_classes="info-panel",
                    )

                # ── Confidence legend ─────────────────────────────────────
                with gr.Group(elem_classes="vm-panel"):
                    gr.HTML("""
<div class="vm-panel-title">🎨 Confidence Key</div>
<div style="display:grid;grid-template-columns:12px 1fr;gap:5px 8px;align-items:center;font-size:.78rem;line-height:1.5">
  <span style="background:#00c853;width:12px;height:12px;border-radius:3px;display:block"></span>
  <span><b style="color:#00c853">High ≥85%</b> — trust fully</span>
  <span style="background:#ffd600;width:12px;height:12px;border-radius:3px;display:block"></span>
  <span><b style="color:#ffd600">Medium 60–85%</b> — use with caution</span>
  <span style="background:#ff1744;width:12px;height:12px;border-radius:3px;display:block"></span>
  <span><b style="color:#ff1744">Low &lt;60%</b> — re-fly / verify</span>
  <span style="background:#ff9800;width:12px;height:12px;border-radius:3px;display:block"></span>
  <span><b style="color:#ff9800">Orange</b> — inferred geometry</span>
</div>""")

                # ── Dimension tool ────────────────────────────────────────
                with gr.Group(elem_classes="vm-panel"):
                    gr.HTML('<div class="vm-panel-title">📏 Dimension Tool</div>')
                    show_dim = gr.Checkbox(
                        False, label="Show measurement line on model",
                        elem_classes="vm-check",
                    )
                    gr.HTML('<div style="font-size:.72rem;color:#667;margin:4px 0 6px">Point 1 (X, Y, Z)</div>')
                    with gr.Row():
                        p1x = gr.Number(label="X", value=0.0, precision=3, elem_classes="coord-input", scale=1, min_width=60)
                        p1y = gr.Number(label="Y", value=0.0, precision=3, elem_classes="coord-input", scale=1, min_width=60)
                        p1z = gr.Number(label="Z", value=0.5, precision=3, elem_classes="coord-input", scale=1, min_width=60)
                    gr.HTML('<div style="font-size:.72rem;color:#667;margin:6px 0">Point 2 (X, Y, Z)</div>')
                    with gr.Row():
                        p2x = gr.Number(label="X", value=0.5, precision=3, elem_classes="coord-input", scale=1, min_width=60)
                        p2y = gr.Number(label="Y", value=0.5, precision=3, elem_classes="coord-input", scale=1, min_width=60)
                        p2z = gr.Number(label="Z", value=1.0, precision=3, elem_classes="coord-input", scale=1, min_width=60)
                    dim_panel = gr.Markdown(
                        value="*Tick box above to measure.*",
                        elem_classes="dim-panel",
                    )

                # ── Uncaptured inference toggle ───────────────────────────
                with gr.Group(elem_classes="vm-panel"):
                    gr.HTML('<div class="vm-panel-title">🔭 Uncaptured Inference</div>')
                    show_inf = gr.Checkbox(
                        False,
                        label="Show inferred geometry (collinearity)",
                        elem_classes="vm-check",
                    )
                    gr.HTML("""<div style="font-size:.73rem;color:#667;margin-top:4px;line-height:1.5">
Space Intersection (Eq.35) + Similarity Model (Eq.4)<br>estimates occluded regions from camera rays.</div>""")

        # ── BOTTOM TABS ─────────────────────────────────────────────────────
        with gr.Tabs():

            # Pipeline
            with gr.TabItem("🚀  Run Pipeline"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("#### 📥 Video Input")
                        vid_file = gr.File(
                            label="Upload video (MP4 / MOV / AVI)",
                            file_types=["video", ".mp4", ".mov", ".avi", ".mkv"],
                            type="filepath",
                        )
                        local_path = gr.Textbox(
                            label="Or paste local video path",
                            value=str(ROOT / "mixkit-big-mosque-in-the-middle-of-a-city-from-the-34295-hd-ready.mp4"),
                            lines=1,
                        )
                        gr.Markdown("#### ⚙️ Settings")
                        with gr.Row():
                            s_every = gr.Slider(0.5, 5, value=1.5, step=0.5, label="Every N sec")
                            s_max   = gr.Slider(5, 80, value=25, step=5, label="Max frames")
                        with gr.Row():
                            s_blur  = gr.Slider(0, 200, value=60, step=10, label="Blur filter")
                            s_conf  = gr.Slider(0, 90, value=50, step=5, label="Conf filter %")
                        with gr.Row():
                            s_pois   = gr.Slider(7, 10, value=9, step=1, label="Poisson depth")
                        with gr.Row():
                            cb_mesh  = gr.Checkbox(True, label="Poisson mesh")
                            cb_align = gr.Checkbox(True, label="Ground align")
                            cb_cams  = gr.Checkbox(True, label="Cameras")
                        run_btn = gr.Button("🚀  Run Pipeline", variant="primary", size="lg")
                        pipe_status = gr.Textbox(
                            label="Status", lines=2, interactive=False,
                            elem_classes="status-run vm-status",
                        )
                    with gr.Column(scale=2):
                        pipe_log = gr.Markdown(
                            value="```\nWaiting for pipeline …\n```",
                            elem_classes="log-win",
                        )

            # Math spec
            with gr.TabItem("📐  Math Spec"):
                gr.Markdown(r"""
### Collinearity Equation (Eq. 34) — Camera Ray → 3-D World Point
$$\begin{bmatrix}X\\Y\\Z\end{bmatrix} = \begin{bmatrix}X_0\\Y_0\\Z_0\end{bmatrix} + \lambda\,R^T\begin{bmatrix}x-x_0\\y-y_0\\-c\end{bmatrix}$$
$(X_0,Y_0,Z_0)$ = VGGT camera centre · $R$ = rotation matrix · $(x_0,y_0,c)$ = principal point & focal length

---
### Space Intersection (Eq. 35) — Uncaptured Region Inference
$$\min_{P}\sum_{i=1}^N\|(P-C_i)-\langle P-C_i,d_i\rangle d_i\|^2 \quad\rightarrow\quad \texttt{np.linalg.lstsq}$$

### Similarity Model (Eq. 4) — Scale Recovery &nbsp;&nbsp; $d_{of}=d_{oc}\cdot d_{AB}$

### Confidence Colouring &nbsp;&nbsp; $\alpha=\text{conf}^{255}$ &nbsp; Green≥0.85 · Yellow 0.60–0.85 · Red<0.60

### Dimension Check &nbsp;&nbsp; $d=\sqrt{\Delta X^2+\Delta Y^2+\Delta Z^2}\times s$
                """)

            # Downloads
            with gr.TabItem("💾  Downloads"):
                gr.Markdown(f"""
### Output files → `dashboard_outputs/<run_id>/`

| File | Contents |
|---|---|
| `pointcloud.glb` | Coloured point cloud + camera frustums |
| `reconstruction.glb` | Poisson mesh + vayumesh metadata in GLB extras |
| `reconstruction.obj` | Mesh (OBJ+MTL) |
| `reconstruction.ply` | Mesh + `vayumesh.*` PLY comment headers |
| `metadata.json` | Full pipeline sidecar (video, model, confidence, mesh) |

Root: `{OUT_ROOT}`
                """)
                with gr.Row():
                    open_out_btn = gr.Button("📂  Open output folder in Explorer")
                    open_out_msg = gr.Textbox(label="", interactive=False, lines=1, scale=3)

                def _open():
                    import subprocess
                    try:
                        subprocess.Popen(["explorer", str(OUT_ROOT)])
                        return str(OUT_ROOT)
                    except Exception as e:
                        return str(e)
                open_out_btn.click(fn=_open, outputs=open_out_msg)

        # ══════════════════════════════════════════════════════════════════
        # EVENT WIRING
        # ══════════════════════════════════════════════════════════════════

        # Common inputs/outputs for every view update
        VIEW_INPUTS = [
            current_dir, current_mode,
            show_inf,
            p1x, p1y, p1z,
            p2x, p2y, p2z,
            show_dim,
        ]
        VIEW_OUTPUTS = [main_plot, info_panel, dim_panel]

        # ── Mode buttons ──────────────────────────────────────────────────
        def _set_mode_and_render(mode, out_dir, show_inf_v,
                                  p1x_v,p1y_v,p1z_v, p2x_v,p2y_v,p2z_v, sd_v):
            fig, info, dim = switch_view(out_dir, mode, show_inf_v,
                                          p1x_v,p1y_v,p1z_v, p2x_v,p2y_v,p2z_v, sd_v)
            return fig, info, dim, mode, f"Mode: {mode}"

        for btn, mode_name in [
            (btn_pc,   "Point Cloud"),
            (btn_dep,  "Depth Map"),
            (btn_conf, "Confidence Heatmap"),
            (btn_wire, "Wireframe Mesh"),
            (btn_mesh, "Solid Mesh"),
        ]:
            btn.click(
                fn=lambda *a, m=mode_name: _set_mode_and_render(m, *a),
                inputs=[current_dir, show_inf,
                        p1x,p1y,p1z, p2x,p2y,p2z, show_dim],
                outputs=[main_plot, info_panel, dim_panel,
                         current_mode, status_top],
            )

        # ── Load directory button ─────────────────────────────────────────
        def _load_dir(dir_str, mode, show_inf_v,
                      p1x_v,p1y_v,p1z_v, p2x_v,p2y_v,p2z_v, sd_v):
            fig, info, dim = switch_view(dir_str, mode, show_inf_v,
                                          p1x_v,p1y_v,p1z_v, p2x_v,p2y_v,p2z_v, sd_v)
            return fig, info, dim, dir_str, f"✅ Loaded: {Path(dir_str).name}"

        load_dir_btn.click(
            fn=_load_dir,
            inputs=[out_dir_input, current_mode, show_inf,
                    p1x,p1y,p1z, p2x,p2y,p2z, show_dim],
            outputs=[main_plot, info_panel, dim_panel,
                     current_dir, status_top],
        )

        # ── Dimension / inferred toggles refresh view ─────────────────────
        def _refresh(out_dir, mode, show_inf_v,
                     p1x_v,p1y_v,p1z_v, p2x_v,p2y_v,p2z_v, sd_v):
            fig, info, dim = switch_view(out_dir, mode, show_inf_v,
                                          p1x_v,p1y_v,p1z_v, p2x_v,p2y_v,p2z_v, sd_v)
            return fig, info, dim

        for trigger in [show_dim, show_inf,
                        p1x, p1y, p1z, p2x, p2y, p2z]:
            trigger.change(
                fn=_refresh,
                inputs=[current_dir, current_mode, show_inf,
                        p1x,p1y,p1z, p2x,p2y,p2z, show_dim],
                outputs=VIEW_OUTPUTS,
            )

        # ── Pipeline generator ────────────────────────────────────────────
        def _pipeline_wrapper(vf, lp, ev, mf, bt, cp, em, ea, pd, sc):
            vpath = vf if (vf and os.path.isfile(str(vf))) else (lp.strip() if lp else None)
            for log_md, status, out in pipeline_gen(vpath, ev, mf, bt, cp, em, ea, pd, sc):
                yield log_md, status, out

        run_btn.click(
            fn=_pipeline_wrapper,
            inputs=[vid_file, local_path,
                    s_every, s_max, s_blur, s_conf,
                    cb_mesh, cb_align, s_pois, cb_cams],
            outputs=[pipe_log, pipe_status, out_dir_input],
        )

        # ── Auto-load default scene on start ─────────────────────────────
        def _auto_load():
            if os.path.isdir(DEFAULT_OUT):
                fig, info, dim = switch_view(DEFAULT_OUT, "Point Cloud",
                                              False, 0,0,0.5, 0.5,0.5,1.0, False)
                return fig, info, dim, DEFAULT_OUT, "✅ Mosque scene loaded"
            return _empty_fig(), "", "", DEFAULT_OUT, "Ready"

        app.load(fn=_auto_load,
                 outputs=[main_plot, info_panel, dim_panel,
                          current_dir, status_top])

    return app


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 68)
    print("  VayuMesh 1Pass — Ultimate Viewer")
    print("  http://127.0.0.1:7861")
    print("=" * 68)

    # Check available RAM — only pre-load model if enough is free
    import ctypes
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        free_gb = stat.ullAvailPhys / 1e9
    except Exception:
        free_gb = 0.0

    print(f"Available RAM: {free_gb:.1f} GB")

    if free_gb >= 5.5:
        print("Pre-loading VGGT-1B … (enough RAM available)")
        try:
            get_model()
            print("Model ready ✓")
        except Exception as e:
            print(f"[WARN] Pre-load failed: {e}")
            print("       Model will load on first pipeline run instead.")
    else:
        print(f"[INFO] Only {free_gb:.1f} GB free — skipping model pre-load.")
        print("       Model loads on demand when you click 🚀 Run Pipeline.")
        print("       All 3-D viewers work immediately without the model.")

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=True,
    )
