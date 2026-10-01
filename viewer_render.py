"""
VayuMesh 2.0 — Render.com Web Service
=======================================
Start command : python viewer_render.py
Build command : pip install -r requirements_render.txt
Plan          : Standard (2 GB RAM)
Region        : Singapore

What runs on Render
-------------------
  - 5-mode interactive 3-D viewer (Point Cloud / Depth / Confidence /
    Wireframe / Solid Mesh)
  - Pre-computed results for Mosque + Taj Mahal (bundled in precomputed/)
  - Dimension tool, confidence heatmap, ground-plane lock
  - Fullscreen button (F key)
  NOTE: VGGT inference is NOT run on Render (model = 5 GB, needs 12 GB RAM).
        Run run_pipeline.py locally, commit outputs to precomputed/, redeploy.

Environment variables set by Render automatically
-------------------------------------------------
  PORT   : Render injects this; we MUST bind to it.
  RENDER : set to "true" by render.yaml so we skip browser-open.
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, sys, gc, json, time, logging, datetime, platform, threading
from pathlib import Path
from typing import Optional, Dict

# ── patch gradio_client schema bug BEFORE importing gradio ────────────────────
try:
    import gradio_client.utils as _gcu

    _orig_get_type = _gcu.get_type
    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "unknown"
        return _orig_get_type(schema)
    _gcu.get_type = _safe_get_type

    _orig_j2p = _gcu._json_schema_to_python_type
    def _safe_j2p(schema, defs):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_j2p(schema, defs)
    _gcu._json_schema_to_python_type = _safe_j2p
except Exception:
    pass  # gradio not yet installed during build phase — fine

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vggt"))

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import trimesh
import gradio as gr
import plotly.graph_objects as go

# ── viewer engine ─────────────────────────────────────────────────────────────
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

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOG = logging.getLogger("vayumesh.render")

# ── environment ───────────────────────────────────────────────────────────────
PORT      = int(os.environ.get("PORT", 7861))
IS_RENDER = os.environ.get("RENDER", "") != ""
VER       = "2.0.0"

LOG.info(f"PORT={PORT}  IS_RENDER={IS_RENDER}")

# ── pre-computed datasets bundled in the repo ─────────────────────────────────
PRECOMP = ROOT / "precomputed"
_CANDIDATES = [
    ("🕌 Mosque Aerial  (1280×720 · 17 frames · 160K verts)",
     PRECOMP / "mosque"),
    ("🏛️ Taj Mahal HD  (1920×1072 · 20 frames · 434K verts)",
     PRECOMP / "taj"),
    # also check top-level output dirs created by run_pipeline.py
    ("🏙️ Urban iStock  (768×432 · 10 frames · 181K verts)",
     ROOT / "output_3d_istock"),
]
DEMOS: Dict[str, str] = {}
for label, folder in _CANDIDATES:
    p = Path(folder)
    if p.exists() and (p / "pointcloud.glb").exists():
        DEMOS[label] = str(p)
        LOG.info(f"Pre-computed scene: {label} → {p}")

DEFAULT_OUT = list(DEMOS.values())[0] if DEMOS else ""
LOG.info(f"Scenes available: {len(DEMOS)}   default: {DEFAULT_OUT}")


# ─────────────────────────────────────────────────────────────────────────────
# SCENE CACHE
# ─────────────────────────────────────────────────────────────────────────────
_CACHE: Dict = {}


def _load_scene(out_dir: str) -> Dict:
    global _CACHE
    if _CACHE.get("_dir") == out_dir:
        return _CACHE
    LOG.info(f"Loading scene: {out_dir}")
    data = load_output_dir(out_dir)
    if "pts" in data:
        data["ground"]   = make_ground_plane(data["pts"])
        data["inf_pts"]  = None
        data["inf_conf"] = None
    data["_dir"] = out_dir
    _CACHE = data
    return _CACHE


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
    ext  = np.zeros((S, 3, 4))
    intr = np.zeros((S, 3, 3))
    for i, c in enumerate(centres):
        fwd = np.array([cx, cy, pts[:, 2].mean()]) - c
        fwd /= np.linalg.norm(fwd) + 1e-12
        up    = np.array([0., 0., 1.])
        right = np.cross(fwd, up); right /= np.linalg.norm(right) + 1e-12
        up2   = np.cross(right, fwd)
        R = np.stack([right, up2, -fwd])
        ext[i, :3, :3] = R
        ext[i, :3, 3]  = -R @ c
        intr[i] = [[518., 0., 259.], [0., 518., 259.], [0., 0., 1.]]
    return ext, intr


# ─────────────────────────────────────────────────────────────────────────────
# VIEW SWITCH
# ─────────────────────────────────────────────────────────────────────────────
def switch_view(out_dir, mode, show_inf,
                p1x, p1y, p1z, p2x, p2y, p2z, show_dim):
    if not out_dir or not os.path.isdir(str(out_dir)):
        fig = go.Figure(layout=dict(
            paper_bgcolor="#08080f", plot_bgcolor="#08080f",
            scene=dict(bgcolor="#08080f"),
            annotations=[dict(
                text="Select a dataset from the dropdown above",
                xref="paper", yref="paper", x=.5, y=.5,
                showarrow=False, font=dict(color="#555", size=18))],
        ))
        return fig, "— no scene —", ""

    data   = _load_scene(str(out_dir))
    pts    = data.get("pts")
    cols   = data.get("cols")
    conf   = data.get("conf")
    mesh   = data.get("mesh")
    ground = data.get("ground")
    meta   = data.get("metadata", {})

    if pts is None:
        return go.Figure(), "No point cloud found", ""

    # Uncaptured inference (computed once per scene)
    if show_inf and data.get("inf_pts") is None:
        ed, id_ = _dummy_cameras(pts)
        inf_pts, inf_conf = infer_uncaptured_regions(pts, ed, id_, n_samples=800)
        data["inf_pts"]  = inf_pts
        data["inf_conf"] = inf_conf
        _CACHE.update(data)

    inf_pts  = data.get("inf_pts")
    inf_conf = data.get("inf_conf")

    if mode == "Point Cloud":
        fig = render_pointcloud(pts, cols, ground, show_inf, inf_pts)
    elif mode == "Depth Map":
        fig = render_depth_map(pts, ground)
    elif mode == "Confidence Heatmap":
        fig = render_confidence(pts, conf, ground, inf_pts, inf_conf)
    elif mode == "Wireframe Mesh":
        fig = (render_wireframe(mesh, pts, ground) if mesh is not None
               else render_pointcloud(pts, cols, ground))
    else:
        fig = (render_solid_mesh(mesh, pts, ground) if mesh is not None
               else render_pointcloud(pts, cols, ground))

    # Dimension overlay
    dim_md = ""
    if show_dim:
        p1 = np.array([float(p1x), float(p1y), float(p1z)])
        p2 = np.array([float(p2x), float(p2y), float(p2z)])
        r  = measure_distance(p1, p2)
        fig = add_dimension_line(fig, p1, p2)
        dim_md = (
            f"**P1** `({p1x:.3f}, {p1y:.3f}, {p1z:.3f})`  \n"
            f"**P2** `({p2x:.3f}, {p2y:.3f}, {p2z:.3f})`  \n\n"
            f"| | |\n|---|---|\n"
            f"| ΔX | `{r['dX_m']:.4f}` |\n"
            f"| ΔY | `{r['dY_m']:.4f}` |\n"
            f"| ΔZ | `{r['dZ_m']:.4f}` |\n"
            f"| **Distance** | **`{r['distance_m']:.4f}` units** |\n"
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
| Extent | {' × '.join(f'{v:.3f}' for v in mi.get('extent', []))} |
| Inferred pts | {inf_n:,} |
| **Mode** | **`{mode}`** |
"""
    return fig, info_md, dim_md


# ─────────────────────────────────────────────────────────────────────────────
# CSS  (identical to viewer_ultimate.py — clean symmetric layout)
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
html,body,.gradio-container{
    background:#08080f!important;
    color:#e2e2ec!important;
    font-family:'Segoe UI',system-ui,sans-serif!important;
}
.gradio-container{max-width:100%!important;padding:0!important;margin:0!important}
footer{display:none!important}
.gap,.contain{gap:0!important}

/* HEADER */
.vm-hdr{background:linear-gradient(135deg,#0a0820 0%,#1a1040 50%,#0f1832 100%);
  border-bottom:2px solid #ffd600;padding:13px 28px 11px;
  display:flex;align-items:center;justify-content:space-between}
.vm-hdr-left h1{color:#fff!important;font-size:1.45rem;font-weight:900;
  margin:0;line-height:1.2;letter-spacing:.2px}
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
.vm-btn button{width:100%!important;background:#14142a!important;
  color:#90caf9!important;border:1px solid #2a2a42!important;
  border-radius:7px!important;font-size:.79rem!important;font-weight:600!important;
  padding:8px 4px!important;white-space:nowrap!important;height:38px!important;
  transition:all .15s!important}
.vm-btn button:hover{background:#1e1e38!important;border-color:#667eea!important;
  color:#fff!important;transform:translateY(-1px)}
.vm-active button{background:linear-gradient(135deg,#667eea,#764ba2)!important;
  color:#fff!important;border-color:#667eea!important;
  box-shadow:0 3px 14px rgba(102,126,234,.45)!important}
.demo-drop select,.demo-drop input{background:#0d0d1a!important;
  color:#e0e0e0!important;border:1px solid #2a2a3a!important;
  border-radius:7px!important;font-size:.82rem!important}
.status-ok textarea{background:#091409!important;color:#66bb6a!important;font-weight:700}
.status-run textarea{background:#08101a!important;color:#42a5f5!important}
.vm-status textarea{height:38px!important;min-height:38px!important;
  border-radius:7px!important;font-size:.79rem!important;padding:8px 12px!important}

/* PLOT — fills viewport */
.vm-plot-col{padding:6px 6px 6px 8px!important}
.main-plot{background:#0a0a12;border-radius:10px;border:1px solid #1a1a2a;
  overflow:hidden;
  height:calc(100vh - 148px)!important;min-height:540px}
.main-plot>div,.main-plot .plotly,.main-plot iframe{
  height:calc(100vh - 150px)!important;min-height:538px!important;width:100%!important}

/* SIDE PANELS */
.vm-side-col{padding:6px 8px 6px 4px!important}
.vm-panel{background:#0e0e1a;border:1px solid #1e1e2e;border-radius:10px;
  padding:11px 13px;margin-bottom:8px}
.vm-panel-title{color:#90caf9;font-size:.76rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.8px;margin-bottom:7px;
  padding-bottom:5px;border-bottom:1px solid #1e1e2e;
  display:flex;align-items:center;gap:5px}
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

/* TABS */
.tab-nav button{color:#778!important;font-size:.8rem!important;
  padding:8px 14px!important;border:none!important;background:transparent!important}
.tab-nav button.selected{color:#90caf9!important;
  border-bottom:2px solid #667eea!important;font-weight:700!important}

::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#0a0a10}
::-webkit-scrollbar-thumb{background:#2a2a3a;border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:#667eea}
"""

HEADER_HTML = """
<div class="vm-hdr">
  <div class="vm-hdr-left">
    <h1>🚁 VayuMesh 1Pass &nbsp;·&nbsp; Ultimate Viewer</h1>
    <p>Fixed Ground &nbsp;·&nbsp; Confidence Heatmap &nbsp;·&nbsp;
       Wireframe / Solid Mesh &nbsp;·&nbsp; Uncaptured Inference &nbsp;·&nbsp;
       Dimension Tool</p>
  </div>
  <div class="vm-hdr-right">
    <span class="vm-sih">SIH 26158 &nbsp;·&nbsp; MoD INDIA</span>
    <div class="vm-pills">
      <span class="vm-pill">⚡ VGGT-1B &nbsp;1.25B params</span>
      <span class="vm-pill">🔒 100% Offline capable</span>
      <span class="vm-pill">🎯 CVPR 2025 Best Paper</span>
    </div>
  </div>
</div>
"""

FULLSCREEN_JS = """
<button id="vm-fs-btn" onclick="vmFS()" title="Fullscreen (F key)"
  style="position:fixed;top:10px;right:14px;z-index:9999;
         background:linear-gradient(135deg,#667eea,#764ba2);
         color:#fff;border:none;border-radius:8px;
         padding:6px 14px;font-size:.85rem;font-weight:700;
         cursor:pointer;box-shadow:0 2px 12px rgba(102,126,234,.5)">
  ⛶ Full
</button>
<script>
function vmFS(){
  var el = document.getElementById('vm-plot-wrap') || document.documentElement;
  if (!document.fullscreenElement) {
    (el.requestFullscreen || el.webkitRequestFullscreen).call(el);
    document.getElementById('vm-fs-btn').textContent = '✕ Exit';
  } else {
    (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    document.getElementById('vm-fs-btn').textContent = '⛶ Full';
  }
}
document.addEventListener('keydown', e => {
  if (e.key === 'f' || e.key === 'F') vmFS();
});
</script>
"""


def _empty_fig():
    return go.Figure(layout=dict(
        paper_bgcolor="#08080f", plot_bgcolor="#08080f",
        scene=dict(bgcolor="#08080f"),
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[dict(
            text="Select a dataset from the dropdown above",
            xref="paper", yref="paper", x=.5, y=.5,
            showarrow=False, font=dict(color="#444", size=16),
        )],
    ))


# ─────────────────────────────────────────────────────────────────────────────
# BUILD APP
# ─────────────────────────────────────────────────────────────────────────────
def build_app():
    with gr.Blocks(
        title="VayuMesh 2.0 · Ultimate Viewer",
        css=CSS,
        theme=gr.themes.Base(),
    ) as app:

        gr.HTML(HEADER_HTML)

        current_dir  = gr.State(value=DEFAULT_OUT)
        current_mode = gr.State(value="Point Cloud")

        # ── MODE BAR ─────────────────────────────────────────────────────────
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
                demo_dd = gr.Textbox(
                    value="",
                    placeholder="No precomputed scenes found — paste output folder path",
                    label="", show_label=False,
                    container=False, scale=3,
                )

            load_btn   = gr.Button("📂 Load", size="sm", variant="secondary")
            status_top = gr.Textbox(
                value="✅ Ready", label="", show_label=False,
                interactive=False, container=False,
                elem_classes="status-ok vm-status", scale=2,
            )

        gr.HTML(FULLSCREEN_JS)

        # ── MAIN ROW: plot (75%) + side panels (25%) ──────────────────────────
        with gr.Row(equal_height=True):

            # ── 3-D Plot ─────────────────────────────────────────────────────
            with gr.Column(scale=6, elem_classes="vm-plot-col"):
                main_plot = gr.Plot(
                    value=_empty_fig(),
                    elem_classes="main-plot",
                    elem_id="vm-plot-wrap",
                    show_label=False,
                    container=False,
                )

            # ── Side panels ───────────────────────────────────────────────────
            with gr.Column(scale=2, min_width=270, elem_classes="vm-side-col"):

                with gr.Group(elem_classes="vm-panel"):
                    gr.HTML('<div class="vm-panel-title">📊 Scene Info</div>')
                    info_panel = gr.Markdown(
                        "*Select a dataset above.*",
                        elem_classes="info-panel",
                    )

                with gr.Group(elem_classes="vm-panel"):
                    gr.HTML("""
<div class="vm-panel-title">🎨 Confidence Key</div>
<div style="display:grid;grid-template-columns:12px 1fr;
            gap:5px 8px;align-items:center;font-size:.76rem;line-height:1.5">
  <span style="background:#00c853;width:12px;height:12px;
               border-radius:3px;display:block"></span>
  <span><b style="color:#00c853">High ≥ 85%</b> — trust fully</span>
  <span style="background:#ffd600;width:12px;height:12px;
               border-radius:3px;display:block"></span>
  <span><b style="color:#ffd600">Med 60–85%</b> — use with caution</span>
  <span style="background:#ff1744;width:12px;height:12px;
               border-radius:3px;display:block"></span>
  <span><b style="color:#ff1744">Low &lt;60%</b> — re-fly / verify</span>
  <span style="background:#ff9800;width:12px;height:12px;
               border-radius:3px;display:block"></span>
  <span><b style="color:#ff9800">Orange</b> — inferred geometry</span>
</div>""")

                with gr.Group(elem_classes="vm-panel"):
                    gr.HTML('<div class="vm-panel-title">📏 Dimension Tool</div>')
                    show_dim = gr.Checkbox(
                        False, label="Show measurement line",
                        elem_classes="vm-check",
                    )
                    gr.HTML('<div style="font-size:.7rem;color:#556;'
                            'margin:4px 0 5px">Point 1 &nbsp;(X · Y · Z)</div>')
                    with gr.Row():
                        p1x = gr.Number(label="X", value=0.0, precision=3,
                                        scale=1, min_width=55, elem_classes="coord-input")
                        p1y = gr.Number(label="Y", value=0.0, precision=3,
                                        scale=1, min_width=55, elem_classes="coord-input")
                        p1z = gr.Number(label="Z", value=0.5, precision=3,
                                        scale=1, min_width=55, elem_classes="coord-input")
                    gr.HTML('<div style="font-size:.7rem;color:#556;'
                            'margin:5px 0">Point 2 &nbsp;(X · Y · Z)</div>')
                    with gr.Row():
                        p2x = gr.Number(label="X", value=0.5, precision=3,
                                        scale=1, min_width=55, elem_classes="coord-input")
                        p2y = gr.Number(label="Y", value=0.5, precision=3,
                                        scale=1, min_width=55, elem_classes="coord-input")
                        p2z = gr.Number(label="Z", value=1.0, precision=3,
                                        scale=1, min_width=55, elem_classes="coord-input")
                    dim_panel = gr.Markdown(
                        "*Tick above to measure.*",
                        elem_classes="dim-panel",
                    )

                with gr.Group(elem_classes="vm-panel"):
                    gr.HTML('<div class="vm-panel-title">🔭 Uncaptured Inference</div>')
                    show_inf = gr.Checkbox(
                        False, label="Show inferred geometry (Eq.35)",
                        elem_classes="vm-check",
                    )
                    gr.HTML('<div style="font-size:.71rem;color:#556;'
                            'margin-top:4px;line-height:1.5">'
                            'Space Intersection estimates<br>'
                            'occluded regions from camera rays.</div>')

        # ── BOTTOM TABS ───────────────────────────────────────────────────────
        with gr.Tabs():

            with gr.TabItem("ℹ️  About & Results"):
                gr.Markdown(f"""
### VayuMesh 2.0 — SIH 26158 · Ministry of Defence, Indian Army

**Core model:** VGGT-1B (Meta AI + Oxford VGG · CVPR 2025 Best Paper)  
**Pipeline:** Frame extract → VGGT inference → Poisson mesh → Ground align → Metadata embed

| Test Video | Resolution | Frames | Mesh Vertices | Mesh Faces | CPU Time |
|---|---|---|---|---|---|
| iStock Urban | 768×432 | 10 | 181,450 | 361,227 | 278 s |
| **Mosque Aerial** | **1280×720** | **17** | **160,956** | **318,006** | **~360 s** |
| **Taj Mahal HD** | **1920×1072** | **20** | **434,317** | **862,812** | **707 s** |

**5 interactive view modes:** ☁️ Point Cloud · 🌈 Depth Map · 🔥 Confidence Heatmap · 🔲 Wireframe · 🏗️ Solid Mesh

**Ground plane is locked** — `dragmode=orbit`, `zaxis.autorange=False`, `camera.up=(0,0,1)`  
**Metadata embedded** in every output GLB (extras), PLY (comment headers), and JSON sidecar.

> ℹ️ **Cloud deployment note:** VGGT-1B requires ~12 GB RAM.  
> Run inference locally with `run_pipeline.py`, commit outputs to `precomputed/`, redeploy.

---
**GitHub:** [github.com/Raj-Sawant/Vayumesh](https://github.com/Raj-Sawant/Vayumesh)  
**Deployed on:** Render.com · Standard plan · Singapore region
                """)

            with gr.TabItem("📐  Math"):
                gr.Markdown(r"""
### Collinearity Eq. 34
$$[X\;Y\;Z]^\top = [X_0\;Y_0\;Z_0]^\top + \lambda R^T[x-x_0,\;y-y_0,\;-c]^\top$$

### Space Intersection Eq. 35 (uncaptured regions)
$$\min_P\sum_i\|(P-C_i)-\langle P-C_i,d_i\rangle d_i\|^2 \;\rightarrow\; \texttt{lstsq}$$

### Scale Eq. 4 &nbsp;&nbsp; $d_{of}=d_{oc}\cdot d_{AB}$

### Confidence &nbsp;&nbsp; $\alpha=\text{conf}^{255}$ · 🟢 ≥0.85 · 🟡 0.60–0.85 · 🔴 <0.60

### Dimension &nbsp;&nbsp; $d=\sqrt{\Delta X^2+\Delta Y^2+\Delta Z^2}\times s$
                """)

        # ─────────────────────────────────────────────────────────────────────
        # EVENT WIRING
        # ─────────────────────────────────────────────────────────────────────
        VIEW_OUT = [main_plot, info_panel, dim_panel]
        VIEW_IN  = [current_dir, show_inf,
                    p1x, p1y, p1z, p2x, p2y, p2z, show_dim]

        def _render(out_dir, si, px, py, pz, qx, qy, qz, sd, mode="Point Cloud"):
            return switch_view(out_dir, mode, si, px, py, pz, qx, qy, qz, sd)

        def _set_mode(m, out_dir, si, px, py, pz, qx, qy, qz, sd):
            fig, info, dim = switch_view(out_dir, m, si, px, py, pz, qx, qy, qz, sd)
            return fig, info, dim, m, f"Mode: {m}"

        for btn, mode_name in [
            (btn_pc,   "Point Cloud"),
            (btn_dep,  "Depth Map"),
            (btn_conf, "Confidence Heatmap"),
            (btn_wire, "Wireframe Mesh"),
            (btn_mesh, "Solid Mesh"),
        ]:
            btn.click(
                fn=lambda *a, m=mode_name: _set_mode(m, *a),
                inputs=VIEW_IN,
                outputs=VIEW_OUT + [current_mode, status_top],
            )

        def _load(key_or_path, mode, si, px, py, pz, qx, qy, qz, sd):
            path = DEMOS.get(key_or_path, key_or_path)
            fig, info, dim = switch_view(path, mode, si, px, py, pz, qx, qy, qz, sd)
            short = Path(path).name if path else "?"
            return fig, info, dim, path, f"✅ {short}"

        demo_dd.change(
            fn=_load,
            inputs=[demo_dd, current_mode] + VIEW_IN[1:],
            outputs=VIEW_OUT + [current_dir, status_top],
        )
        load_btn.click(
            fn=_load,
            inputs=[demo_dd, current_mode] + VIEW_IN[1:],
            outputs=VIEW_OUT + [current_dir, status_top],
        )

        for trigger in [show_dim, show_inf, p1x, p1y, p1z, p2x, p2y, p2z]:
            trigger.change(
                fn=lambda *a: switch_view(*a[:1], "Point Cloud", *a[1:]),
                inputs=VIEW_IN,
                outputs=VIEW_OUT,
            )

        # Auto-load first demo on startup
        def _startup():
            if DEFAULT_OUT and os.path.isdir(DEFAULT_OUT):
                fig, info, dim = switch_view(
                    DEFAULT_OUT, "Point Cloud",
                    False, 0., 0., 0.5, 0.5, 0.5, 1., False,
                )
                return fig, info, dim, DEFAULT_OUT, "✅ Ready"
            return _empty_fig(), "", "", DEFAULT_OUT, "✅ Ready"

        app.load(
            fn=_startup,
            outputs=[main_plot, info_panel, dim_panel, current_dir, status_top],
        )

    return app


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    LOG.info("=" * 60)
    LOG.info(f"  VayuMesh 2.0 · Render Viewer  (port {PORT})")
    LOG.info(f"  Scenes: {list(DEMOS.keys())}")
    LOG.info("=" * 60)

    app = build_app()
    app.launch(
        server_name="0.0.0.0",   # REQUIRED — Render routes traffic to this
        server_port=PORT,         # REQUIRED — Render injects $PORT
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=not IS_RENDER,  # never open browser on server
    )
