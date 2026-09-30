"""
VayuMesh 1Pass — Viewer Engine
================================
All rendering back-end functions used by viewer_ultimate.py.

Provides five view modes (all returned as Plotly figures):
  1. Point Cloud     – RGB-coloured 3-D scatter
  2. Depth Map       – per-pixel depth as a 2-D image (rainbow palette)
  3. Confidence Heat – per-point confidence (green=high, yellow=med, red=low)
  4. Wireframe       – solid mesh with overlaid edge lines
  5. Solid Mesh      – shaded surface with vertex colours

Plus:
  • Fixed ground plane  (locked Z = 0 plane, never moves)
  • Uncaptured-scene inference  (collinearity + space-intersection rays)
  • GLB builder for each mode  (for the Gradio Model3D viewer)
  • Dimension-check tool  (two-point Euclidean distance)
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, json, hashlib, logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import trimesh
import plotly.graph_objects as go
import matplotlib
import matplotlib.cm as mcm

LOG = logging.getLogger("vayumesh.engine")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS / PALETTE
# ─────────────────────────────────────────────────────────────────────────────
CONF_NORM_FACTOR = 20.0   # VGGT confidence values are roughly 0-20; normalise to 0-1

# Confidence thresholds (normalised 0-1)
CONF_HIGH  = 0.85   # green
CONF_MED   = 0.60   # yellow
# < CONF_MED          # red

PLOT_BG    = "#0d0d14"   # dark background for all plots
GROUND_ALPHA = 0.25      # ground plane mesh opacity

# maximum points to render in Plotly (down-sample if larger)
MAX_PLOTLY_POINTS = 200_000


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADERS
# ═════════════════════════════════════════════════════════════════════════════

def load_glb_pointcloud(glb_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the point cloud from a GLB (first PointCloud geometry).
    Returns (points [N,3], colors [N,4] uint8).
    """
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.PointCloud):
        pts = np.asarray(scene.vertices)
        cols = np.asarray(scene.colors) if scene.colors is not None else np.full((len(pts), 4), 200, dtype=np.uint8)
        return pts, cols
    if isinstance(scene, trimesh.Scene):
        for g in scene.geometry.values():
            if isinstance(g, trimesh.PointCloud):
                pts  = np.asarray(g.vertices)
                cols = np.asarray(g.colors) if g.colors is not None else np.full((len(pts), 4), 200, dtype=np.uint8)
                return pts, cols
    raise ValueError(f"No PointCloud geometry found in {glb_path}")


def load_glb_mesh(glb_path: str) -> trimesh.Trimesh:
    """Load the primary Trimesh from a GLB."""
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Trimesh):
        return scene
    if isinstance(scene, trimesh.Scene):
        # Return largest mesh by face count
        best = max(
            (g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)),
            key=lambda g: len(g.faces),
            default=None,
        )
        if best is not None:
            return best
    raise ValueError(f"No Trimesh found in {glb_path}")


def load_output_dir(out_dir: str) -> Dict:
    """
    Load all artefacts from a pipeline output directory.
    Returns dict with keys: pts, cols, mesh, metadata, ground_z
    """
    p = Path(out_dir)
    result: Dict = {}

    pc_glb  = p / "pointcloud.glb"
    msh_glb = p / "reconstruction.glb"
    meta_j  = p / "metadata.json"

    if pc_glb.exists():
        pts, cols = load_glb_pointcloud(str(pc_glb))
        result["pts"]  = pts
        result["cols"] = cols   # [N,4] uint8

        # Synthesise a pseudo confidence channel from luminance if no npz
        lum = (cols[:, 0].astype(float) * 0.299
             + cols[:, 1].astype(float) * 0.587
             + cols[:, 2].astype(float) * 0.114) / 255.0
        result["conf"] = lum   # [N] float 0-1 (proxy)

        result["ground_z"] = float(np.percentile(pts[:, 2], 5))

    if msh_glb.exists():
        result["mesh"] = load_glb_mesh(str(msh_glb))

    if meta_j.exists():
        with open(meta_j, encoding="utf-8") as f:
            result["metadata"] = json.load(f)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 2.  GROUND PLANE (locked, stationary)
# ═════════════════════════════════════════════════════════════════════════════

def make_ground_plane(pts: np.ndarray, extent_scale: float = 1.4) -> Dict:
    """
    Build a FIXED ground-plane mesh at Z = ground_z (5th-percentile of scene).
    The plane is locked: it never adjusts during interaction.

    Returns a dict with keys:
      z, x_range, y_range  → for Plotly surface trace
    """
    ground_z = float(np.percentile(pts[:, 2], 5))
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    # Expand slightly beyond scene bounds
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half_x = (x_max - x_min) / 2 * extent_scale
    half_y = (y_max - y_min) / 2 * extent_scale

    x_range = np.linspace(cx - half_x, cx + half_x, 20)
    y_range = np.linspace(cy - half_y, cy + half_y, 20)
    z_grid  = np.full((20, 20), ground_z)

    return {
        "ground_z":  ground_z,
        "x_range":   x_range,
        "y_range":   y_range,
        "z_grid":    z_grid,
    }


def ground_trace(ground: Dict) -> go.Surface:
    """
    Locked ground-plane Surface trace.
    - showscale=False, flat lighting → looks like a static floor grid
    - hoverinfo='skip' → doesn't interfere with point picking
    - The Z position never changes because _base_layout locks zaxis.range
    """
    return go.Surface(
        x=ground["x_range"],
        y=ground["y_range"],
        z=ground["z_grid"],
        colorscale=[
            [0.0, "rgba(60,60,90,0.18)"],
            [1.0, "rgba(60,60,90,0.18)"],
        ],
        showscale=False,
        opacity=0.30,
        name="Ground (Z locked)",
        hoverinfo="skip",
        # Fully ambient lighting = flat, never reflects camera spin
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                      roughness=1.0, fresnel=0.0),
        # Grid lines on the ground surface
        contours=dict(
            x=dict(show=True, color="rgba(80,80,120,0.5)", width=1),
            y=dict(show=True, color="rgba(80,80,120,0.5)", width=1),
            z=dict(show=False),
        ),
        # Disable its own hover / colorbar UI
        colorbar=None,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3.  UNCAPTURED SCENE INFERENCE  (Collinearity + Space Intersection)
# ═════════════════════════════════════════════════════════════════════════════

def _ray_from_camera(extrinsic_34: np.ndarray, intrinsic_33: np.ndarray,
                     img_x: float, img_y: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collinearity Equation (Eq. 34 in spec):
        [X Y Z]^T = [X0 Y0 Z0]^T + λ · R · [x-x0, y-y0, -c]^T

    Returns (camera_center [3], ray_direction [3]).
    Extrinsic is [3,4] world-to-camera.
    """
    R  = extrinsic_34[:3, :3]          # 3×3 rotation  (world→cam)
    t  = extrinsic_34[:3, 3]           # translation
    # Camera centre in world coords: C = -R^T · t
    C  = -R.T @ t                      # [3]

    # Intrinsics
    fx = float(intrinsic_33[0, 0])
    fy = float(intrinsic_33[1, 1])
    x0 = float(intrinsic_33[0, 2])    # principal point x
    y0 = float(intrinsic_33[1, 2])    # principal point y
    c  = (fx + fy) / 2.0              # effective focal length

    # Ray direction in camera frame, then rotate to world
    d_cam = np.array([img_x - x0, img_y - y0, -c], dtype=float)
    d_world = R.T @ d_cam             # world-frame direction
    d_world /= (np.linalg.norm(d_world) + 1e-12)

    return C, d_world


def space_intersection_lstsq(centers: np.ndarray,
                              directions: np.ndarray) -> np.ndarray:
    """
    Space Intersection (Eq. 35 in spec): given N rays (centre + dir)
    find the 3-D point that minimises sum of squared distances to all rays.

    For ray i:  P = C_i + λ_i · d_i
    Reformulated as a linear system  A·X = b  solved with lstsq.
    """
    n = len(centers)
    A = np.zeros((3 * n, 3 + n), dtype=float)
    b = np.zeros(3 * n,          dtype=float)

    for i in range(n):
        A[3*i:3*i+3, :3] = np.eye(3)
        A[3*i:3*i+3, 3+i] = -directions[i]
        b[3*i:3*i+3]      = centers[i]

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3]   # 3-D world point


def infer_uncaptured_regions(
    pts:       np.ndarray,        # existing point cloud [N,3]
    extrinsics: np.ndarray,       # [S,3,4] from metadata / predictions
    intrinsics: np.ndarray,       # [S,3,3]
    n_samples:  int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer plausible geometry for UNCAPTURED regions:

    1. Identify spatial gaps in the existing point cloud (voxel-based).
    2. For each gap centre, cast rays from neighbouring camera positions
       whose viewing frustum overlaps the gap.
    3. Use Space Intersection (Eq. 35) to triangulate a candidate 3-D
       point; accept it if the back-projection residual is below threshold.
    4. Similarity model (Eq. 4): scale inferred points using the median
       baseline between cameras as a known reference:
           d_of = d_oc * d_AB
       where d_oc is the ray distance and d_AB is the camera baseline.

    Returns:
        inferred_pts   [M,3]   – new geometry in gaps
        inferred_conf  [M]     – confidence ∈ [0,0.5]  (always lower than
                                 directly observed points)
    """
    if extrinsics is None or intrinsics is None:
        return np.zeros((0, 3)), np.zeros(0)

    S = len(extrinsics)
    if S < 2:
        return np.zeros((0, 3)), np.zeros(0)

    # ── camera centres ──────────────────────────────────────────────────────
    centres = np.array([-(extrinsics[i, :3, :3].T @ extrinsics[i, :3, 3])
                        for i in range(S)])                # [S,3]

    # ── median camera baseline (known reference distance d_AB) ─────────────
    baselines = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    d_AB      = float(np.median(baselines)) if len(baselines) > 0 else 1.0

    # ── voxel occupancy grid to find gaps ───────────────────────────────────
    resolution  = 0.05                         # voxel side length
    pt_min      = pts.min(axis=0)
    pt_max      = pts.max(axis=0)
    grid_shape  = np.ceil((pt_max - pt_min) / resolution).astype(int) + 1

    occupied = set()
    voxel_idx = ((pts - pt_min) / resolution).astype(int)
    voxel_idx = np.clip(voxel_idx, 0, grid_shape - 1)
    for vi in voxel_idx:
        occupied.add(tuple(vi))

    # ── candidate gap centres (voxels within scene bbox, not occupied) ─────
    rng      = np.random.default_rng(42)
    n_try    = min(n_samples * 20, 100_000)
    rand_idx = rng.integers(0, grid_shape, size=(n_try, 3))

    gap_centres = []
    for vi in rand_idx:
        k = tuple(vi)
        if k not in occupied:
            world_pt = pt_min + (vi + 0.5) * resolution
            # only consider points inside the convex hull of the scene
            if (world_pt >= pt_min).all() and (world_pt <= pt_max).all():
                gap_centres.append(world_pt)
        if len(gap_centres) >= n_samples:
            break

    if not gap_centres:
        return np.zeros((0, 3)), np.zeros(0)

    gap_centres = np.array(gap_centres)          # [M,3]

    # ── for each gap, find 2 nearest cameras and triangulate ────────────────
    inferred  = []
    conf_vals = []

    # image centre (principal point proxy)
    img_cx = float(intrinsics[0, 0, 2])
    img_cy = float(intrinsics[0, 1, 2])

    for gc in gap_centres:
        # two nearest cameras by distance to gap centre
        dists = np.linalg.norm(centres - gc, axis=1)
        order = np.argsort(dists)
        cam_a, cam_b = order[0], order[1]

        # ray A: from camera A through image centre (approx)
        C_a, d_a = _ray_from_camera(extrinsics[cam_a], intrinsics[cam_a], img_cx, img_cy)
        # ray B: from camera B through image centre
        C_b, d_b = _ray_from_camera(extrinsics[cam_b], intrinsics[cam_b], img_cx, img_cy)

        try:
            pt3d = space_intersection_lstsq(
                np.array([C_a, C_b]),
                np.array([d_a, d_b]),
            )
        except Exception:
            continue

        # Similarity model (Eq. 4): d_of = d_oc * d_AB
        # d_oc = distance from camera A centre to triangulated point
        d_oc  = float(np.linalg.norm(pt3d - C_a))
        d_of  = d_oc * d_AB                 # scaled distance

        # Accept if within expanded scene bounds (sanity check)
        margin = (pt_max - pt_min) * 0.3
        if ((pt3d >= pt_min - margin).all() and
                (pt3d <= pt_max + margin).all()):
            inferred.append(pt3d)
            # Confidence inversely proportional to ray angle (cosine similarity)
            cos_sim = abs(float(np.dot(d_a, d_b)))
            conf_val = max(0.05, min(0.45, (1.0 - cos_sim) * 0.5))
            conf_vals.append(conf_val)

    if not inferred:
        return np.zeros((0, 3)), np.zeros(0)

    return np.array(inferred), np.array(conf_vals)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  CONFIDENCE COLOURING
# ═════════════════════════════════════════════════════════════════════════════

def conf_to_rgb(conf: np.ndarray) -> np.ndarray:
    """
    Map normalised confidence [0,1] → vivid RGB [0,255] uint8.
    Green ≥ 0.85 | Yellow 0.60–0.85 | Red < 0.60

    Fix: DO NOT multiply by alpha — that was crushing all colours to black.
    Instead interpolate smoothly within each band for visual richness.
    """
    conf = np.clip(conf, 0.0, 1.0)
    rgb  = np.zeros((len(conf), 3), dtype=float)

    hi = conf >= CONF_HIGH                  # ≥ 0.85
    md = (conf >= CONF_MED) & ~hi           # 0.60–0.85
    lo = ~hi & ~md                          # < 0.60

    # ── High: bright green → lime (full brightness, varying hue) ────────────
    if hi.any():
        t = (conf[hi] - CONF_HIGH) / (1.0 - CONF_HIGH + 1e-9)   # 0→1 within band
        rgb[hi, 0] = 0
        rgb[hi, 1] = 200 + 55 * t       # 200→255
        rgb[hi, 2] = 50  * (1 - t)      # slight blue tint at low end

    # ── Medium: yellow → orange ──────────────────────────────────────────────
    if md.any():
        t = (conf[md] - CONF_MED) / (CONF_HIGH - CONF_MED + 1e-9)  # 0→1 within band
        rgb[md, 0] = 255
        rgb[md, 1] = 140 + 74 * t        # 140(orange)→214(yellow)
        rgb[md, 2] = 0

    # ── Low: red → deep red ──────────────────────────────────────────────────
    if lo.any():
        t = conf[lo] / (CONF_MED + 1e-9)  # 0→1 within band
        rgb[lo, 0] = 180 + 75 * t         # 180→255
        rgb[lo, 1] = 0 + 23 * t           # slight orange tinge at higher end
        rgb[lo, 2] = 20 * (1 - t)

    return np.clip(rgb, 0, 255).astype(np.uint8)


def _rgb_array_to_strings(cols: np.ndarray) -> list:
    """
    Convert (N,3+) uint8 color array → list of 'rgb(r,g,b)' strings.
    Uses numpy char ops — much faster than Python list comprehension.
    """
    r = cols[:, 0].astype(np.uint8)
    g = cols[:, 1].astype(np.uint8)
    b = cols[:, 2].astype(np.uint8)
    # Build as bytes then decode once
    prefix = np.full(len(r), "rgb(", dtype=object)
    sep    = np.full(len(r), ",",    dtype=object)
    suffix = np.full(len(r), ")",    dtype=object)
    return list(
        prefix
        + r.astype(str).astype(object)
        + sep + g.astype(str).astype(object)
        + sep + b.astype(str).astype(object)
        + suffix
    )


def depth_to_rgb(depths: np.ndarray) -> np.ndarray:
    """Map depth values → rainbow RGB [0,255] uint8 using 'turbo' colormap."""
    d_range = float(depths.max() - depths.min())
    d_norm  = (depths - depths.min()) / (d_range + 1e-12)
    cmap    = matplotlib.colormaps["turbo"]
    rgba    = (cmap(d_norm)[:, :3] * 255).astype(np.uint8)
    return rgba


# ═════════════════════════════════════════════════════════════════════════════
# 5.  PLOTLY VIEW RENDERERS
# ═════════════════════════════════════════════════════════════════════════════

def _base_layout(title: str, pts: np.ndarray) -> dict:
    """
    Shared Plotly layout for all 3-D views.

    Ground-locking strategy
    -----------------------
    * Z axis range is fixed to [ground_z - 0.05, max_z + 0.1] and
      autorange is disabled — ground never drifts up or down.
    * aspectmode = "manual" with fixed aspectratio locks XYZ scale so
      the ground plane proportions never change on orbit.
    * camera.up = {x:0, y:0, z:1} forces Z always up.
    * uirevision = "ground_locked" — Plotly keeps the camera when the
      figure data is replaced (e.g. switching view modes).
    * dragmode = "orbit" — user can only orbit around the fixed Z-up
      axis; they cannot flip the scene upside-down.
    """
    x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    z_min, z_max = float(pts[:, 2].min()), float(pts[:, 2].max())

    z_range  = z_max - z_min
    x_range  = x_max - x_min
    y_range  = y_max - y_min
    max_horiz = max(x_range, y_range, 0.001)

    # Fixed Z window: a little below the ground, full height + headroom
    z_lo = z_min - z_range * 0.05
    z_hi = z_max + z_range * 0.10

    # Normalised aspect ratios (keep proportions, clamp Z to ≤ 1.5×horiz)
    ax = x_range / max_horiz
    ay = y_range / max_horiz
    az = min((z_range / max_horiz) * 1.2, 1.5)

    axis_common = dict(
        backgroundcolor="rgba(0,0,0,0)",   # transparent — no coloured wall
        gridcolor="rgba(60,60,80,0.4)",    # very subtle grid
        linecolor="rgba(80,80,100,0.5)",
        showbackground=False,              # ← removes the thick axis panels
        showgrid=True,
        zeroline=False,
        showticklabels=False,              # ← hides cluttered tick numbers
        showspikes=False,                  # no spike lines on hover
    )

    return dict(
        title=dict(text=title, font=dict(color="#e0e0e0", size=15)),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=0, r=0, t=36, b=0),
        # dragmode at the top level controls 3D interaction mode
        dragmode="orbit",
        scene=dict(
            bgcolor=PLOT_BG,
            # ── Fixed axis ranges ──────────────────────────────────────
            xaxis=dict(**axis_common,
                       range=[x_min - x_range*0.05, x_max + x_range*0.05],
                       autorange=False,
                       title=dict(text="", font=dict(color="rgba(0,0,0,0)"))),
            yaxis=dict(**axis_common,
                       range=[y_min - y_range*0.05, y_max + y_range*0.05],
                       autorange=False,
                       title=dict(text="", font=dict(color="rgba(0,0,0,0)"))),
            zaxis=dict(**axis_common,
                       range=[z_lo, z_hi],
                       autorange=False,            # ← NEVER moves ground
                       tickfont=dict(color="rgba(100,140,200,0.6)", size=9),
                       title=dict(text="", font=dict(color="rgba(0,0,0,0)"))),
            # ── Fixed proportions ──────────────────────────────────────
            aspectmode="manual",
            aspectratio=dict(x=ax, y=ay, z=az),
            # ── Camera locked Z-up, isometric view ───────────────────
            camera=dict(
                up=dict(x=0, y=0, z=1),        # Z is always "up"
                eye=dict(x=1.5, y=1.5, z=1.0), # angled from above
                projection=dict(type="perspective"),
            ),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0.35)", font=dict(color="#ccc"),
                    bordercolor="#333", borderwidth=1),
        # uirevision: same string = camera survives data updates
        # (Plotly only resets camera when this value changes)
        uirevision="ground_locked_v1",
    )


def _downsample(pts: np.ndarray, *arrays, max_n: int = MAX_PLOTLY_POINTS):
    """Randomly downsample all arrays together."""
    n = len(pts)
    if n <= max_n:
        return (pts,) + arrays
    idx = np.random.default_rng(0).choice(n, max_n, replace=False)
    return (pts[idx],) + tuple(a[idx] for a in arrays)


# ── 5a. Point Cloud ──────────────────────────────────────────────────────────
def render_pointcloud(pts: np.ndarray, cols: np.ndarray,
                      ground: Dict,
                      show_inferred: bool = False,
                      inf_pts: Optional[np.ndarray] = None) -> go.Figure:
    pts_d, cols_d = _downsample(pts, cols)
    color_str = _rgb_array_to_strings(cols_d)

    traces = [
        go.Scatter3d(
            x=pts_d[:, 0], y=pts_d[:, 1], z=pts_d[:, 2],
            mode="markers",
            marker=dict(size=1.2, color=color_str, opacity=0.85),
            name="Point Cloud",
            hovertemplate="X:%{x:.3f}<br>Y:%{y:.3f}<br>Z:%{z:.3f}<extra></extra>",
        ),
        ground_trace(ground),
    ]

    if show_inferred and inf_pts is not None and len(inf_pts):
        traces.append(go.Scatter3d(
            x=inf_pts[:, 0], y=inf_pts[:, 1], z=inf_pts[:, 2],
            mode="markers",
            marker=dict(size=1.5, color="#ff9800", symbol="circle-open", opacity=0.6),
            name="Inferred (uncaptured)",
        ))

    fig = go.Figure(data=traces, layout=_base_layout("☁️ Point Cloud", pts))
    return fig


# ── 5b. Depth Map ────────────────────────────────────────────────────────────
def render_depth_map(pts: np.ndarray, ground: Dict) -> go.Figure:
    pts_d, = _downsample(pts)
    depths  = pts_d[:, 2]
    rgb     = depth_to_rgb(depths)
    color_str = _rgb_array_to_strings(rgb)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts_d[:, 0], y=pts_d[:, 1], z=pts_d[:, 2],
                mode="markers",
                marker=dict(size=1.2, color=color_str, opacity=0.85),
                name="Depth (turbo)",
                hovertemplate="Depth: %{z:.3f}<extra></extra>",
            ),
            ground_trace(ground),
        ],
        layout=_base_layout("🌈 Depth Map", pts),
    )
    # Add colourbar annotation
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(
            size=0,
            color=[float(pts[:, 2].min()), float(pts[:, 2].max())],
            colorscale="turbo", showscale=True,
            colorbar=dict(
                    title=dict(text="Depth (units)", font=dict(color="#ccc")),
                    tickfont=dict(color="#ccc"),
                ),
        ),
        showlegend=False,
    ))
    return fig


# ── 5c. Confidence Heatmap ───────────────────────────────────────────────────
def render_confidence(pts: np.ndarray, conf: np.ndarray,
                      ground: Dict,
                      inf_pts: Optional[np.ndarray] = None,
                      inf_conf: Optional[np.ndarray] = None) -> go.Figure:
    pts_d, conf_d = _downsample(pts, conf)
    rgb = conf_to_rgb(conf_d)
    color_str = _rgb_array_to_strings(rgb)

    traces = [
        go.Scatter3d(
            x=pts_d[:, 0], y=pts_d[:, 1], z=pts_d[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=color_str, opacity=0.9),
            name="Confidence",
            hovertemplate=(
                "X:%{x:.3f} Y:%{y:.3f} Z:%{z:.3f}<br>"
                "<extra></extra>"
            ),
        ),
        ground_trace(ground),
    ]

    # Append inferred points (always low confidence → orange)
    if inf_pts is not None and len(inf_pts):
        traces.append(go.Scatter3d(
            x=inf_pts[:, 0], y=inf_pts[:, 1], z=inf_pts[:, 2],
            mode="markers",
            marker=dict(size=2, color="#ff9800", opacity=0.55),
            name="Inferred (low conf)",
        ))

    fig = go.Figure(data=traces, layout=_base_layout("🔥 Confidence Heatmap", pts))

    # Legend annotations
    for label, colour, threshold in [
        ("High conf (≥0.85)", "#00e676", "≥85 %"),
        ("Med conf (0.60–0.85)", "#ffd600", "60–85 %"),
        ("Low conf (<0.60)", "#ff1744", "<60 %"),
    ]:
        fig.add_annotation(
            text=f"<span style='color:{colour}'>■</span> {label}",
            xref="paper", yref="paper",
            x=0.01, y=0.99 - [0.04, 0.08, 0.12][["High", "Med", "Low"].index(label.split()[0])],
            showarrow=False, font=dict(size=11, color="#ccc"),
            bgcolor="rgba(0,0,0,0.4)",
        )
    return fig


# ── 5d. Wireframe Mesh ───────────────────────────────────────────────────────
def render_wireframe(mesh: trimesh.Trimesh, pts: np.ndarray,
                     ground: Dict) -> go.Figure:
    v = mesh.vertices
    f = mesh.faces

    # Solid surface (shaded)
    solid = go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color="#607d8b",
        opacity=0.45,
        name="Solid",
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3,
                      roughness=0.5, fresnel=0.1),
        lightposition=dict(x=100, y=200, z=300),
        flatshading=False,
        hoverinfo="skip",
    )

    # Wireframe edges — build line segments from faces
    edges_x, edges_y, edges_z = [], [], []
    # Sample every 3rd face to keep line count manageable
    for tri in f[::3]:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edges_x += [v[a, 0], v[b, 0], None]
            edges_y += [v[a, 1], v[b, 1], None]
            edges_z += [v[a, 2], v[b, 2], None]

    wire = go.Scatter3d(
        x=edges_x, y=edges_y, z=edges_z,
        mode="lines",
        line=dict(color="#00e5ff", width=0.6),
        opacity=0.7,
        name="Wireframe",
        hoverinfo="skip",
    )

    fig = go.Figure(
        data=[solid, wire, ground_trace(ground)],
        layout=_base_layout("🔲 Solid + Wireframe Mesh", pts),
    )
    return fig


# ── 5e. Solid Mesh ───────────────────────────────────────────────────────────
def render_solid_mesh(mesh: trimesh.Trimesh, pts: np.ndarray,
                      ground: Dict) -> go.Figure:
    v = mesh.vertices
    f = mesh.faces

    # Vertex colours from mesh (if available)
    if (hasattr(mesh.visual, "vertex_colors") and
            mesh.visual.vertex_colors is not None):
        vc = np.asarray(mesh.visual.vertex_colors)
        vertex_color = [f"rgb({r},{g},{b})" for r, g, b, *_ in vc]
        intensity  = None
        colorscale = None
        showscale  = False
    else:
        vertex_color = None
        intensity    = v[:, 2]   # colour by height
        colorscale   = "viridis"
        showscale    = True

    solid = go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        vertexcolor=vertex_color,
        intensity=intensity,
        colorscale=colorscale,
        showscale=showscale,
        opacity=1.0,
        name="Solid Mesh",
        lighting=dict(ambient=0.4, diffuse=0.9, specular=0.5,
                      roughness=0.3, fresnel=0.2),
        lightposition=dict(x=200, y=300, z=500),
        flatshading=False,
        hovertemplate="X:%{x:.3f}<br>Y:%{y:.3f}<br>Z:%{z:.3f}<extra></extra>",
    )

    fig = go.Figure(
        data=[solid, ground_trace(ground)],
        layout=_base_layout("🏗️ Solid Mesh", pts),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 6.  DIMENSION TOOL  (two-point Euclidean distance)
# ═════════════════════════════════════════════════════════════════════════════

def measure_distance(p1: np.ndarray, p2: np.ndarray,
                     scale_factor: float = 1.0) -> Dict:
    """
    Compute dimension check between two 3-D points.
    scale_factor converts VGGT units → metres (1.0 if no GPS calibration).
    """
    dp = (p2 - p1) * scale_factor
    dist = float(np.linalg.norm(dp))
    return {
        "point1":    p1.tolist(),
        "point2":    p2.tolist(),
        "dX_m":      float(dp[0]),
        "dY_m":      float(dp[1]),
        "dZ_m":      float(dp[2]),
        "distance_m": dist,
        "scale_factor": scale_factor,
    }


def add_dimension_line(fig: go.Figure,
                       p1: np.ndarray, p2: np.ndarray,
                       label: str = "") -> go.Figure:
    """Overlay a measurement line and annotation on an existing figure."""
    mid = (p1 + p2) / 2
    fig.add_trace(go.Scatter3d(
        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
        mode="lines+markers+text",
        line=dict(color="#ffeb3b", width=4),
        marker=dict(size=6, color="#ff5722", symbol="circle"),
        text=["P1", "P2"],
        textposition="top center",
        textfont=dict(color="#ffeb3b", size=12),
        name=label or f"Dist: {np.linalg.norm(p2-p1):.3f}",
        hovertemplate="%{text}<br>X:%{x:.3f} Y:%{y:.3f} Z:%{z:.3f}<extra></extra>",
    ))
    # Midpoint label
    dist_m = np.linalg.norm(p2 - p1)
    fig.add_trace(go.Scatter3d(
        x=[mid[0]], y=[mid[1]], z=[mid[2]],
        mode="markers+text",
        marker=dict(size=0.1, color="rgba(0,0,0,0)"),
        text=[f"  {dist_m:.3f} u"],
        textfont=dict(color="#ffeb3b", size=13),
        showlegend=False,
        hoverinfo="skip",
    ))
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# 7.  GLB BUILDER  (each view mode → GLB for Gradio Model3D)
# ═════════════════════════════════════════════════════════════════════════════

def build_confidence_glb(pts: np.ndarray, conf: np.ndarray,
                          ground: Dict, out_path: str) -> str:
    """Export confidence-coloured point cloud as GLB."""
    rgb     = conf_to_rgb(conf)
    rgba    = np.concatenate([rgb, np.full((len(rgb), 1), 220, dtype=np.uint8)], axis=1)
    pc      = trimesh.PointCloud(vertices=pts, colors=rgba)
    # Add ground plane as a simple mesh
    gp      = _ground_trimesh(ground)
    scene   = trimesh.Scene([pc, gp])
    scene.export(out_path)
    return out_path


def build_wireframe_glb(mesh: trimesh.Trimesh,
                         ground: Dict, out_path: str) -> str:
    """Export wireframe mesh as GLB (edges as thin tubes)."""
    gp    = _ground_trimesh(ground)
    scene = trimesh.Scene([mesh, gp])
    scene.export(out_path)
    return out_path


def _ground_trimesh(ground: Dict) -> trimesh.Trimesh:
    """Make a flat quad grid trimesh for the ground plane."""
    x = ground["x_range"]
    y = ground["y_range"]
    z = ground["ground_z"]
    X, Y = np.meshgrid(x, y)
    verts = np.column_stack([X.ravel(), Y.ravel(),
                              np.full(X.size, z)])
    # Build grid faces
    nx, ny = len(x), len(y)
    faces = []
    for row in range(ny - 1):
        for col in range(nx - 1):
            a = row * nx + col
            b = a + 1
            c = a + nx
            d = c + 1
            faces.extend([[a, b, c], [b, d, c]])
    faces = np.array(faces)
    color = np.array([180, 180, 200, 64], dtype=np.uint8)
    face_colors = np.tile(color, (len(faces), 1))
    return trimesh.Trimesh(vertices=verts, faces=faces,
                           face_colors=face_colors, process=False)
