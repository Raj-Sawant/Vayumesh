"""
VayuMesh Dashboard v2  —  SIH 26158
=====================================
Upload any video → live progress log → interactive 3D viewers.

Run:
    .venv\\Scripts\\python.exe dashboard_v2.py
    → http://127.0.0.1:7860   (or your LAN IP on port 7860)

Design decisions
----------------
* gr.File  (not gr.Video) for upload — avoids a Gradio 5.17 schema bug
  that crashes /info on the gr.Video component.
* Generator function for the pipeline so the UI streams live log lines
  without waiting for the full run to finish.
* Model loaded once at startup into a module-level singleton.
* Every output GLB has metadata embedded; JSON sidecar saved alongside.
* share=False  — fully offline, no HuggingFace tunnel needed.
"""

# ── stdlib ───────────────────────────────────────────────────────────────────
import os, sys, gc, json, time, shutil, hashlib, logging, datetime, platform, threading
from pathlib import Path
from typing import Optional, List, Generator

# ── make vggt sub-package importable ─────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vggt"))

# ── third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
import trimesh
import gradio as gr

# ── vggt ──────────────────────────────────────────────────────────────────────
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from visual_util import predictions_to_glb

# ── optional mesh / metric modules ───────────────────────────────────────────
try:
    from mesh_reconstruction import full_reconstruction_pipeline, export_mesh
    MESH_OK = True
except Exception as _me:
    MESH_OK = False
    print(f"[WARN] mesh_reconstruction unavailable: {_me}")

try:
    from metric_scaling import (
        CameraPose, detect_ground_plane, align_to_ground_plane,
    )
    METRIC_OK = True
except Exception as _ms:
    METRIC_OK = False
    print(f"[WARN] metric_scaling unavailable: {_ms}")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("vayumesh")

# ── constants ─────────────────────────────────────────────────────────────────
VGGT_CACHE   = ROOT / ".cache" / "vggt_model.pt"
VGGT_HF_URL  = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
VGGT_REPO    = "facebook/VGGT-1B"
OUT_ROOT     = ROOT / "dashboard_outputs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
PIPELINE_VER = "2.0.0"

# ── global model singleton ────────────────────────────────────────────────────
_MODEL: Optional[VGGT] = None
_LOCK  = threading.Lock()


# ═════════════════════════════════════════════════════════════════════════════
# 1.  MODEL MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

def _download_weights() -> Path:
    """Download VGGT-1B to .cache/ if not already there."""
    VGGT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if VGGT_CACHE.exists():
        LOG.info(f"Model cache hit: {VGGT_CACHE.stat().st_size/1e9:.2f} GB")
        return VGGT_CACHE
    LOG.info("Downloading VGGT-1B from HuggingFace …")
    try:
        from huggingface_hub import hf_hub_download
        tmp = hf_hub_download(repo_id=VGGT_REPO, filename="model.pt",
                              local_dir=str(VGGT_CACHE.parent),
                              local_dir_use_symlinks=False)
        shutil.copy2(tmp, VGGT_CACHE)
    except Exception as e:
        LOG.warning(f"hf_hub_download failed ({e}), using torch.hub …")
        sd = torch.hub.load_state_dict_from_url(VGGT_HF_URL, map_location="cpu",
                                                model_dir=str(VGGT_CACHE.parent))
        torch.save(sd, VGGT_CACHE)
    LOG.info(f"Weights saved → {VGGT_CACHE}")
    return VGGT_CACHE


def get_model() -> VGGT:
    """Singleton: load once, reuse forever."""
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            path = _download_weights()
            m = VGGT()
            sd = torch.load(str(path), map_location="cpu", weights_only=True)
            m.load_state_dict(sd)
            m.eval()
            _MODEL = m
            LOG.info(f"VGGT-1B ready  "
                     f"({sum(p.numel() for p in m.parameters())/1e6:.0f}M params)")
    return _MODEL


# ═════════════════════════════════════════════════════════════════════════════
# 2.  PIPELINE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _blur_score(frame: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


def extract_frames(video_path: str, out_dir: str,
                   every_n: float = 1.5, max_fr: int = 40,
                   blur_thresh: float = 60.0) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(round(fps * every_n)))
    saved, count, idx = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if count % interval == 0:
            if blur_thresh <= 0 or _blur_score(frame) >= blur_thresh:
                p = os.path.join(out_dir, f"{idx:06d}.png")
                cv2.imwrite(p, frame)
                saved.append(p)
                idx += 1
            if len(saved) >= max_fr:
                break
        count += 1
    cap.release()
    return sorted(saved)


def run_inference(image_paths: List[str], model: VGGT) -> dict:
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


def _conf_filter(pred: dict, conf_pct: float):
    pts = pred["world_points_from_depth"].reshape(-1, 3)
    colors = None
    if "images" in pred:
        imgs = pred["images"]
        if imgs.ndim == 4 and imgs.shape[1] == 3:
            imgs = imgs.transpose(0, 2, 3, 1)
        colors = imgs.reshape(-1, 3)
    if "depth_conf" in pred and conf_pct > 0:
        c = pred["depth_conf"].reshape(-1)
        mask = c >= np.percentile(c, conf_pct)
        pts = pts[mask]
        if colors is not None:
            colors = colors[mask]
    return pts, colors


def _video_probe(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS),
                frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    info["duration"] = round(info["frames"] / info["fps"], 2) if info["fps"] else 0
    return info


def _make_metadata(video_path, image_paths, pred, mesh, run_id) -> dict:
    vi = _video_probe(video_path)
    h  = hashlib.sha256()
    with open(video_path, "rb") as f:
        h.update(f.read(4 << 20))
    meta = {
        "pipeline": {"version": PIPELINE_VER, "run_id": run_id,
                     "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "platform": platform.platform(), "torch": torch.__version__},
        "video": {"file": Path(video_path).name, "sha256_4mb": h.hexdigest()[:16],
                  **vi},
        "model": {"id": VGGT_REPO, "url": VGGT_HF_URL},
        "frames": len(image_paths),
    }
    if "depth_conf" in pred:
        c = pred["depth_conf"].ravel()
        meta["confidence"] = {"mean": float(c.mean()), "p50": float(np.median(c)),
                               "p10": float(np.percentile(c, 10)),
                               "p90": float(np.percentile(c, 90))}
    if "extrinsic" in pred:
        meta["cameras"] = int(len(pred["extrinsic"]))
    if mesh is not None:
        b = mesh.bounds
        meta["mesh"] = {"vertices": int(len(mesh.vertices)),
                        "faces": int(len(mesh.faces)),
                        "watertight": bool(mesh.is_watertight),
                        "extent": (b[1] - b[0]).tolist()}
    return meta


def _embed_glb(glb_path: str, meta: dict):
    try:
        sc = trimesh.load(glb_path)
        if not hasattr(sc, "metadata") or sc.metadata is None:
            sc.metadata = {}
        sc.metadata["vayumesh"] = meta
        sc.export(glb_path)
    except Exception as e:
        LOG.debug(f"GLB embed: {e}")


def _embed_ply(ply_path: str, meta: dict):
    try:
        with open(ply_path, "rb") as f:
            raw = f.read()
        end = raw.find(b"end_header")
        if end < 0:
            return
        comments = "".join(
            f"comment vayumesh.{section}.{k} {v}\n"
            for section, d in [("pipeline", meta.get("pipeline", {})),
                                ("video",    meta.get("video", {})),
                                ("model",    meta.get("model", {}))]
            for k, v in d.items()
        ).encode()
        with open(ply_path, "wb") as f:
            f.write(raw[:end] + comments + raw[end:])
    except Exception as e:
        LOG.debug(f"PLY embed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MAIN GENERATOR  (streams partial updates back to Gradio)
# ═════════════════════════════════════════════════════════════════════════════
# Outputs order must match the Gradio outputs list:
#   log_md, status_txt, pc_viewer, mesh_viewer, meta_txt, stats_md

def _emit(lines, msg, t0,
          status="⏳ Running …", pc=None, mesh_g=None, meta="", stats=""):
    lines.append(f"`{time.time()-t0:.1f}s` {msg}")
    log_md = "\n\n".join(lines[-50:])
    yield log_md, status, pc, mesh_g, meta, stats


def pipeline(
    video_file,           # str path from gr.File
    every_n:     float,
    max_fr:      float,
    blur_thresh: float,
    conf_pct:    float,
    en_mesh:     bool,
    en_align:    bool,
    poisson_d:   int,
    show_cams:   bool,
) -> Generator:

    t0    = time.time()
    lines = []

    def emit(msg, **kw):
        yield from _emit(lines, msg, t0, **kw)

    # ── validate input ────────────────────────────────────────────────────────
    if video_file is None:
        yield "❌ No video uploaded.", "❌ No video", None, None, "", ""
        return

    vpath = video_file if isinstance(video_file, str) else str(video_file)
    if not os.path.isfile(vpath):
        yield f"❌ File not found: {vpath}", "❌ Error", None, None, "", ""
        return

    vi = _video_probe(vpath)
    run_id  = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    yield from emit(
        f"📹 Video: `{Path(vpath).name}`  "
        f"({vi['w']}×{vi['h']}, {vi['fps']:.0f}fps, {vi['duration']}s)"
    )

    # ── Stage 0: model ────────────────────────────────────────────────────────
    yield from emit("**[0/4]** Loading VGGT-1B …")
    try:
        model = get_model()
    except Exception as e:
        yield from emit(f"❌ Model load failed: {e}", status="❌ Error")
        return
    yield from emit("✅ Model ready (1.25 B params, CPU)")

    # ── Stage 1: frames ───────────────────────────────────────────────────────
    yield from emit(f"**[1/4]** Extracting frames  "
                    f"(every {every_n}s, max {int(max_fr)}, blur≥{blur_thresh}) …")
    try:
        frames_dir = str(out_dir / "images")
        image_paths = extract_frames(vpath, frames_dir,
                                     every_n=every_n, max_fr=int(max_fr),
                                     blur_thresh=blur_thresh)
    except Exception as e:
        yield from emit(f"❌ Frame extraction failed: {e}", status="❌ Error")
        return
    if not image_paths:
        yield from emit("❌ No usable frames (try lowering blur threshold)", status="❌ Error")
        return
    yield from emit(f"✅ {len(image_paths)} frames saved → `{frames_dir}`")

    # ── Stage 2: inference ────────────────────────────────────────────────────
    yield from emit(f"**[2/4]** VGGT inference on {len(image_paths)} frames … "
                    f"*(CPU ~{len(image_paths)*25//10*10}–{len(image_paths)*45//10*10}s)*")
    t_inf = time.time()
    try:
        pred = run_inference(image_paths, model)
    except Exception as e:
        yield from emit(f"❌ Inference failed: {e}", status="❌ Error")
        return
    n_pts = int(pred["world_points_from_depth"].reshape(-1, 3).shape[0])
    n_cam = int(len(pred["extrinsic"]))
    yield from emit(
        f"✅ Inference done in {time.time()-t_inf:.0f}s  |  "
        f"{n_pts:,} 3-D points  |  {n_cam} camera poses"
    )

    # ── Stage 3a: point cloud GLB ──────────────────────────────────────────────
    yield from emit("**[3/4]** Building point cloud GLB …")
    pc_glb = str(out_dir / "pointcloud.glb")
    try:
        scene = predictions_to_glb(
            pred, conf_thres=conf_pct, filter_by_frames="All",
            show_cam=show_cams, mask_sky=False,
            target_dir=str(out_dir),
            prediction_mode="Depthmap and Camera Branch",
        )
        scene.export(file_obj=pc_glb)
    except Exception as e:
        yield from emit(f"❌ Point cloud export failed: {e}", status="❌ Error")
        return
    sz_pc = Path(pc_glb).stat().st_size / 1e6
    yield from emit(
        f"✅ Point cloud GLB ready ({sz_pc:.1f} MB) — **loading in viewer …**",
        status="⏳ Point cloud ready, building mesh …",
        pc=pc_glb,
    )

    # ── Stage 3b: Poisson mesh ─────────────────────────────────────────────────
    mesh_obj  = None
    mesh_glb  = None
    if en_mesh and MESH_OK:
        yield from emit(f"**[3b]** Poisson reconstruction (depth={poisson_d}) …", pc=pc_glb)
        pts, cols = _conf_filter(pred, conf_pct)
        try:
            mesh_obj = full_reconstruction_pipeline(
                pts, cols, denoise=True,
                poisson_depth=int(poisson_d),
                density_threshold=0.1, clean=True,
            )
            mesh_glb = str(out_dir / "reconstruction.glb")
            export_mesh(mesh_obj, mesh_glb, "glb")
            export_mesh(mesh_obj, str(out_dir / "reconstruction.obj"), "obj")
            mesh_obj.export(str(out_dir / "reconstruction.ply"))
            sz_m = Path(mesh_glb).stat().st_size / 1e6
            yield from emit(
                f"✅ Mesh: {len(mesh_obj.vertices):,} verts / "
                f"{len(mesh_obj.faces):,} faces  ({sz_m:.1f} MB) — **loading …**",
                status="⏳ Mesh ready, aligning …",
                pc=pc_glb, mesh_g=mesh_glb,
            )
        except Exception as e:
            yield from emit(f"⚠️ Mesh failed: {e} — continuing with point cloud only",
                            pc=pc_glb)
    elif en_mesh and not MESH_OK:
        yield from emit("⚠️ open3d unavailable — skipping mesh", pc=pc_glb)

    # ── Stage 4: ground-plane alignment ───────────────────────────────────────
    if en_align and mesh_obj is not None and METRIC_OK:
        yield from emit("**[4/4]** Ground-plane alignment …", pc=pc_glb, mesh_g=mesh_glb)
        try:
            plane_arr = np.array(detect_ground_plane(mesh_obj.vertices),
                                 dtype=float).ravel()
            mesh_obj  = align_to_ground_plane(mesh_obj, plane_arr)
            export_mesh(mesh_obj, mesh_glb, "glb")          # overwrite with aligned
            yield from emit("✅ Ground aligned (Z=0 plane, Z-up)",
                            pc=pc_glb, mesh_g=mesh_glb)
        except Exception as e:
            yield from emit(f"⚠️ Alignment skipped: {e}", pc=pc_glb, mesh_g=mesh_glb)

    # ── Metadata embedding ─────────────────────────────────────────────────────
    yield from emit("**[✦]** Building & embedding metadata …",
                    pc=pc_glb, mesh_g=mesh_glb)
    meta = _make_metadata(vpath, image_paths, pred, mesh_obj, run_id)
    meta_path = str(out_dir / "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    _embed_glb(pc_glb, meta)
    if mesh_glb:
        _embed_glb(mesh_glb, meta)
        ply_p = str(out_dir / "reconstruction.ply")
        if os.path.exists(ply_p):
            _embed_ply(ply_p, meta)

    # ── Summary ────────────────────────────────────────────────────────────────
    total = time.time() - t0
    mi    = meta.get("mesh", {})
    ci    = meta.get("confidence", {})

    stats_md = f"""
### 📊 Run `{run_id}`

| | |
|---|---|
| 🎬 Video | `{Path(vpath).name}` |
| ⏱ Duration | {vi['duration']} s  ({vi['fps']:.0f} fps) |
| 📐 Resolution | {vi['w']} × {vi['h']} |
| 🖼 Frames sampled | {len(image_paths)} |
| 📷 Camera poses | {n_cam} |
| 🔵 Raw 3-D points | {n_pts:,} |
| 🔧 Depth conf mean | {ci.get('mean', 0):.2f} |
| 🔧 Depth conf p90 | {ci.get('p90', 0):.2f} |
| 🏗 Mesh vertices | {mi.get('vertices', '—'):,} |
| 🏗 Mesh faces | {mi.get('faces', '—'):,} |
| 💧 Watertight | {mi.get('watertight', '—')} |
| 📦 Extent (units) | {' × '.join(f'{v:.3f}' for v in mi.get('extent', []))} |
| ⏱ Total time | **{total:.1f} s** |
| 📁 Output folder | `{out_dir}` |
"""

    meta_str = json.dumps(meta, indent=2, default=str)
    status_final = (
        f"✅ Done in {total:.0f}s  |  {len(image_paths)} frames  "
        f"|  {mi.get('vertices', n_pts):,} points/verts"
    )

    yield from emit(
        f"✅ **Pipeline complete in {total:.1f}s** — outputs saved to `{out_dir}`",
        status=status_final, pc=pc_glb, mesh_g=mesh_glb,
        meta=meta_str, stats=stats_md,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4.  GRADIO UI
# ═════════════════════════════════════════════════════════════════════════════

CSS = """
/* layout */
.gradio-container { max-width: 1600px !important; margin: 0 auto; }

/* header */
.vm-hdr {
    background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);
    border-radius: 12px; padding: 22px 32px; margin-bottom: 12px;
}
.vm-hdr h1 { color: #fff !important; font-size: 2rem; margin: 0; }
.vm-hdr p  { color: #bbb !important; margin: 4px 0 0; font-size: .95rem; }

/* panels */
.left-panel  { background:#f9fafb; border-radius:10px; padding:14px; }
.right-panel { border-radius:10px; overflow:hidden; }

/* status colours */
.status-ok textarea { background:#e8f5e9!important; color:#2e7d32!important; font-weight:600; }
.status-err textarea { background:#ffebee!important; color:#c62828!important; font-weight:600; }
.status-run textarea { background:#e3f2fd!important; color:#1565c0!important; }

/* run button */
#run-btn {
    background: linear-gradient(135deg,#667eea,#764ba2)!important;
    border:none!important; font-size:1.05rem!important; font-weight:700!important;
    transition: transform .15s, box-shadow .15s;
}
#run-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102,126,234,.45)!important;
}

/* log window dark theme */
.log-window { border-radius:8px; overflow:hidden; }
.log-window .prose { background:#1a1a2e!important; color:#e0e0e0!important;
    font-family:'Cascadia Code','Fira Code',monospace!important; font-size:12px!important;
    padding:12px!important; min-height:200px; max-height:420px; overflow-y:auto!important; }

/* 3-D viewers */
.viewer canvas { border-radius:8px; }

/* stats table */
.stats .prose table { width:100%; border-collapse:collapse; }
.stats .prose td, .stats .prose th {
    padding:5px 10px; border-bottom:1px solid #e0e0e0; font-size:.88rem;
}
"""

HEADER = """
<div class="vm-hdr">
  <h1>🚁 VayuMesh 2.0</h1>
  <p>Video → Metric 3-D Reconstruction &nbsp;·&nbsp; SIH 26158 &nbsp;·&nbsp;
     <strong>Speed</strong> &nbsp;·&nbsp; <strong>Accuracy</strong> &nbsp;·&nbsp;
     <strong>Trust</strong> &nbsp;·&nbsp; <strong>Security (offline)</strong>
  </p>
</div>
"""


def build_app():
    with gr.Blocks(title="VayuMesh 2.0", css=CSS,
                   theme=gr.themes.Soft()) as app:

        gr.HTML(HEADER)

        with gr.Row(equal_height=False):

            # ── LEFT: controls ─────────────────────────────────────────────
            with gr.Column(scale=1, min_width=300, elem_classes="left-panel"):

                gr.Markdown("### 📥 Video input")
                video_in = gr.File(
                    label="Upload any video  (MP4 / MOV / AVI / MKV)",
                    file_types=["video", ".mp4", ".mov", ".avi", ".mkv",
                                ".MP4", ".MOV", ".AVI", ".MKV"],
                    type="filepath",
                )
                with gr.Accordion("📂 Or use a local file path (no upload needed)", open=True):
                    local_path_box = gr.Textbox(
                        label="Absolute video path",
                        placeholder=r"C:\Users\RAJ SAWANT\Desktop\Vayumesh2.0\mixkit-big-mosque-in-the-middle-of-a-city-from-the-34295-hd-ready.mp4",
                        lines=2,
                        value=r"C:\Users\RAJ SAWANT\Desktop\Vayumesh2.0\mixkit-big-mosque-in-the-middle-of-a-city-from-the-34295-hd-ready.mp4",
                    )
                    use_path_btn = gr.Button("▶  Run on this path", variant="secondary", size="sm")

                gr.Markdown("### ⚙️ Extraction")
                every_n   = gr.Slider(0.5, 5.0, value=1.5, step=0.5,
                                      label="Sample every N seconds")
                max_fr    = gr.Slider(5,   80,  value=25,  step=5,
                                      label="Max frames  (↓ = faster)")
                blur_thr  = gr.Slider(0,   200, value=60,  step=10,
                                      label="Blur threshold  (0 = keep all)")

                gr.Markdown("### 🔬 Reconstruction")
                conf_pct  = gr.Slider(0,  90, value=50, step=5,
                                      label="Confidence filter  (percentile)")
                poisson_d = gr.Slider(7,  10, value=9,  step=1,
                                      label="Poisson depth  (8=fast, 10=detail)")

                gr.Markdown("### 🔧 Stages")
                en_mesh   = gr.Checkbox(True,  label="Poisson mesh reconstruction")
                en_align  = gr.Checkbox(True,  label="Ground-plane alignment")
                show_cams = gr.Checkbox(True,  label="Show camera frustums in viewer")

                run_btn = gr.Button("🚀  Run Pipeline", variant="primary",
                                    size="lg", elem_id="run-btn")
                status_box = gr.Textbox(label="Status", lines=2,
                                        interactive=False,
                                        elem_classes="status-run")

            # ── RIGHT: results ──────────────────────────────────────────────
            with gr.Column(scale=3, elem_classes="right-panel"):

                with gr.Tabs():

                    # ─────────────────────────────────────────────────────────
                    with gr.TabItem("🎯  3-D Viewers"):
                        gr.Markdown(
                            "> Viewers load automatically as each stage finishes.  "
                            "**Drag** to rotate · **Scroll** to zoom · **Right-click** to pan."
                        )
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### ☁️ Point Cloud")
                                pc_viewer = gr.Model3D(
                                    label="Point Cloud + Cameras",
                                    height=500,
                                    clear_color=[0.06, 0.06, 0.10, 1.0],
                                    elem_classes="viewer",
                                )
                            with gr.Column():
                                gr.Markdown("#### 🏗️ Mesh  (Poisson)")
                                mesh_viewer = gr.Model3D(
                                    label="Watertight Mesh",
                                    height=500,
                                    clear_color=[0.06, 0.06, 0.10, 1.0],
                                    elem_classes="viewer",
                                )

                    # ─────────────────────────────────────────────────────────
                    with gr.TabItem("📊  Stats & Metadata"):
                        with gr.Row():
                            with gr.Column(scale=1, elem_classes="stats"):
                                stats_out = gr.Markdown(
                                    value="*Run the pipeline to see stats.*",
                                )
                            with gr.Column(scale=1):
                                meta_out = gr.Textbox(
                                    label="metadata.json  (embedded in every output file)",
                                    lines=32, max_lines=60,
                                    interactive=False,
                                    show_copy_button=True,
                                )

                    # ─────────────────────────────────────────────────────────
                    with gr.TabItem("📜  Live Log"):
                        log_out = gr.Markdown(
                            value="```\nWaiting for pipeline …\n```",
                            elem_classes="log-window",
                        )

                    # ─────────────────────────────────────────────────────────
                    with gr.TabItem("💾  Downloads & Load"):
                        gr.Markdown(f"""
### Output files  →  `dashboard_outputs/<run_id>/`  or  `output_3d_*/`

| File | Description |
|------|-------------|
| `pointcloud.glb` | Colored point cloud + camera frustums |
| `reconstruction.glb` | Poisson mesh with embedded metadata |
| `reconstruction.obj` | Mesh in OBJ/MTL format |
| `reconstruction.ply` | Mesh with `vayumesh.*` PLY comment headers |
| `metadata.json` | Full pipeline metadata sidecar |

Root folder: `{OUT_ROOT}`
                        """)
                        with gr.Row():
                            open_btn = gr.Button("📂  Open output folder in Explorer")
                            open_msg = gr.Textbox(label="", interactive=False, lines=1)

                        def _open_folder():
                            import subprocess
                            try:
                                subprocess.Popen(["explorer", str(OUT_ROOT)])
                                return f"Opened: {OUT_ROOT}"
                            except Exception as e:
                                return str(e)

                        open_btn.click(fn=_open_folder, outputs=open_msg)

                        gr.Markdown("---\n### 🔁 Load an existing output directory into viewers")
                        gr.Markdown(
                            "Point to any folder that contains `pointcloud.glb` / `reconstruction.glb` "
                            "(e.g. `output_3d_mosque` or `output_3d_istock`)."
                        )
                        load_dir_box = gr.Textbox(
                            label="Output directory path",
                            placeholder=r"C:\Users\RAJ SAWANT\Desktop\Vayumesh2.0\output_3d_mosque",
                            lines=1,
                        )
                        load_btn = gr.Button("📂  Load into 3-D viewers", variant="secondary")
                        load_msg = gr.Textbox(label="Load status", interactive=False, lines=2)

                        def _load_dir(dir_path: str):
                            p = Path(dir_path.strip())
                            if not p.exists():
                                return (None, None, "", "",
                                        f"❌ Directory not found: {p}",
                                        f"❌ Not found: {p}")
                            pc  = str(p / "pointcloud.glb")     if (p / "pointcloud.glb").exists()     else None
                            msh = str(p / "reconstruction.glb") if (p / "reconstruction.glb").exists() else None
                            meta_str, stats_md = "", ""
                            meta_p = p / "metadata.json"
                            if meta_p.exists():
                                with open(meta_p, encoding="utf-8") as f:
                                    meta_data = json.load(f)
                                meta_str = json.dumps(meta_data, indent=2)
                                mi = meta_data.get("mesh", {})
                                vi = meta_data.get("video", {})
                                ci = meta_data.get("confidence", {})
                                stats_md = f"""
### 📊 Loaded: `{p.name}`

| | |
|---|---|
| 🎬 Video | `{vi.get('file', '?')}` |
| ⏱ Duration | {vi.get('duration', '?')} s |
| 📐 Resolution | {vi.get('w')}×{vi.get('h')} |
| 🖼 Frames sampled | {meta_data.get('frames', '?')} |
| 📷 Camera poses | {meta_data.get('cameras', '?')} |
| 🔧 Depth conf mean | {ci.get('mean', 0):.2f} |
| 🔧 Depth conf p90 | {ci.get('p90', 0):.2f} |
| 🏗 Mesh vertices | {mi.get('vertices', '—'):,} |
| 🏗 Mesh faces | {mi.get('faces', '—'):,} |
| 💧 Watertight | {mi.get('watertight', '—')} |
| 📦 Extent | {' × '.join(f'{v:.3f}' for v in mi.get('extent', []))} |
| 📁 Folder | `{p}` |
"""
                            found = []
                            if pc:  found.append("pointcloud.glb")
                            if msh: found.append("reconstruction.glb")
                            msg = f"✅ Loaded {', '.join(found)} from {p}"
                            return pc, msh, meta_str, stats_md, msg, f"✅ Loaded: {p}"

                        load_btn.click(
                            fn=_load_dir,
                            inputs=[load_dir_box],
                            outputs=[pc_viewer, mesh_viewer, meta_out, stats_out,
                                     status_box, load_msg],
                        )

        # ── Help accordion ──────────────────────────────────────────────────
        with gr.Accordion("ℹ️  Pipeline overview & tips", open=False):
            gr.Markdown(f"""
**Pipeline stages:**
1. **Frame extract** — sample every N s, discard blurry frames (Laplacian filter)
2. **VGGT inference** — 1.25B transformer: depth maps + camera poses in one forward pass
3. **Point cloud** — depth unprojected through estimated cameras → GLB (visible immediately)
4. **Poisson mesh** — open3D screened Poisson → watertight mesh → GLB
5. **Ground align** — RANSAC plane fit → Z=0 ground, Z-axis up
6. **Metadata embed** — pipeline info baked into GLB extras + PLY headers + JSON sidecar

**Performance (CPU, no GPU):**

| Frames | Approx. time |
|--------|--------------|
| 10 | ~4 min |
| 20 | ~8 min |
| 30 | ~12 min |

**Tips:**
- Use 10–20 s clips for fastest turn-around
- Set "Max frames" to 15 for a quick preview run
- Disable "Poisson mesh" to get the point cloud ~2× faster
- Model is cached in `.cache/` after the first ~5 GB download
- All outputs land in `{OUT_ROOT}`
            """)

        # ── Event wiring ────────────────────────────────────────────────────

        def _resolve_video(uploaded, local_path_str):
            """Return the best video path: uploaded file takes priority, else local path."""
            if uploaded and os.path.isfile(str(uploaded)):
                return str(uploaded)
            if local_path_str and os.path.isfile(local_path_str.strip()):
                return local_path_str.strip()
            return uploaded  # may be None, pipeline will catch it

        def _pipeline_from_upload(uploaded, local_path_str,
                                  every_n, max_fr, blur_thr,
                                  conf_pct, en_mesh, en_align, poisson_d, show_cams):
            vpath = _resolve_video(uploaded, local_path_str)
            yield from pipeline(vpath, every_n, max_fr, blur_thr,
                                 conf_pct, en_mesh, en_align, poisson_d, show_cams)

        run_btn.click(
            fn=_pipeline_from_upload,
            inputs=[video_in, local_path_box,
                    every_n, max_fr, blur_thr,
                    conf_pct, en_mesh, en_align, poisson_d, show_cams],
            outputs=[log_out, status_box, pc_viewer, mesh_viewer,
                     meta_out, stats_out],
        )

        use_path_btn.click(
            fn=_pipeline_from_upload,
            inputs=[video_in, local_path_box,
                    every_n, max_fr, blur_thr,
                    conf_pct, en_mesh, en_align, poisson_d, show_cams],
            outputs=[log_out, status_box, pc_viewer, mesh_viewer,
                     meta_out, stats_out],
        )

    return app


# ═════════════════════════════════════════════════════════════════════════════
# 5.  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 68)
    print("  VayuMesh 2.0  —  Dashboard")
    print("  http://127.0.0.1:7860   (or http://<your-LAN-IP>:7860)")
    print("=" * 68)

    # Pre-warm model so first pipeline run is instant
    print("\nPre-loading VGGT-1B …  (uses cache after first download)")
    try:
        get_model()
        print("Model ready ✓\n")
    except Exception as e:
        print(f"[WARN] Pre-load failed: {e} — will retry on first run\n")

    app = build_app()
    app.launch(
        server_name="0.0.0.0",   # listen on all interfaces → accessible at 127.0.0.1:7860
        server_port=7860,
        share=False,              # fully offline, no HuggingFace tunnel
        show_error=True,
        quiet=False,
        inbrowser=True,           # auto-opens browser tab
    )
