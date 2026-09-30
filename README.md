<div align="center">

<img src="https://img.shields.io/badge/SIH-2026-FF6B00?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHRleHQgeT0iMjAiIGZvbnQtc2l6ZT0iMjAiPvCfmqA8L3RleHQ+PC9zdmc+" />
<img src="https://img.shields.io/badge/Problem_ID-SIH26158-138808?style=for-the-badge" />
<img src="https://img.shields.io/badge/Ministry-MoD_Indian_Army-navy?style=for-the-badge" />

# 🚁 VayuMesh 2.0
### *One Drone Pass · Full Metric 3D Intelligence*

**Video → Depth → Point Cloud → Mesh → Dimensions — in a single AI forward pass**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14_CPU-EE4C2C?logo=pytorch)](https://pytorch.org)
[![VGGT](https://img.shields.io/badge/Model-VGGT--1B_CVPR25-764ba2)](https://huggingface.co/facebook/VGGT-1B)
[![Gradio](https://img.shields.io/badge/UI-Gradio_5.17-F97316)](https://gradio.app)
[![License](https://img.shields.io/badge/License-Non--Commercial-red)](LICENSE.txt)
[![Live Demo](https://img.shields.io/badge/Dashboard-http://127.0.0.1:7861-00C853?logo=googlechrome)](#running-the-dashboard)

</div>

---

## 🎯 Problem Statement

> **SIH 26158 — Indian Army, Ministry of Defence**
>
> Develop an AI-powered system that converts a single drone video pass into a **metric-accurate, dimensioned 3D solid model** of a target structure — without Ground Control Points, without COLMAP bootstrapping, and fully offline/air-gapped.

**Core Requirements:**

| Requirement | Target |
|---|---|
| ⚡ Speed | < 15 min for a 10-min video |
| 🎯 Accuracy | ≤ 1 m spatial error without GCPs |
| 🔍 Trust | Per-region confidence heatmap |
| 🔒 Security | Fully offline, air-gapped deployment |

---

## 💡 What VayuMesh Does

VayuMesh takes **any drone video** and produces a **complete, dimensioned 3D model** with confidence heatmaps — in one automated pipeline, with no manual intervention.

```
Drone Video (MP4)
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1  │  Smart Frame Extraction  (blur filter + sampling)  │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 2  │  VGGT-1B Inference  (1.25B param transformer)      │
│           │  → Depth maps + Camera poses + 3D point cloud      │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 3  │  Poisson Surface Reconstruction  (open3D)          │
│           │  → Watertight mesh (160K–450K vertices)             │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 4  │  Ground-Plane Alignment  (RANSAC + Z-lock)         │
│           │  → Metric-scale, Z-up, ground at Z=0               │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 5  │  Metadata Embedding                                 │
│           │  → Baked into GLB extras + PLY headers + JSON      │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
  5 Output Formats  +  Interactive Dashboard  +  Confidence Heatmap
```

---

## 🏗️ Architecture

### Core Model — VGGT-1B (CVPR 2025 Best Paper)

```
Input: S frames (518×518) ──► DINOv2 ViT-L/14 patch embed
                                        │
                          ┌─────────────┴─────────────┐
                          │   24× Alternating Blocks   │
                          │  ┌───────────────────────┐ │
                          │  │  Frame Attention (FA)  │ │  ← within each image
                          │  ├───────────────────────┤ │
                          │  │  Global Attention (GA) │ │  ← across all frames
                          │  └───────────────────────┘ │
                          └─────────────┬─────────────┘
                                        │  cached features (4 layers)
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
             CameraHead             DPTHead ×2           TrackHead
          (9-DoF iterative)    (depth + pointmap)    (2D pixel tracks)
                    │                   │
                    ▼                   ▼
          Extrinsic [S,3,4]    depth_map [S,H,W,1]
          Intrinsic [S,3,3]    point_map [S,H,W,3]
                    └───────────────────┘
                                │
                  unproject_depth_map_to_point_map()
                                │
                    world_points [S,H,W,3]  ← highest accuracy
```

### VayuMesh Pipeline Modules

```
run_pipeline.py          ← CLI master pipeline
viewer_ultimate.py       ← SIH demo dashboard (http://127.0.0.1:7861)
viewer_engine.py         ← All 5 rendering modes (Plotly)
dashboard_v2.py          ← Alternative Gradio dashboard (port 7860)
video_to_3d.py           ← Video frame extraction + VGGT inference
mesh_reconstruction.py   ← Poisson reconstruction (open3D)
metric_scaling.py        ← GPS fusion + ground alignment
dimension_extraction.py  ← Building dimensions (OBB + ConvexHull)
visual_util.py           ← GLB export + sky segmentation
convert_exports.py       ← OBJ/PLY/GeoJSON/CityGML/3DTiles export
```

---

## 🔬 Technical Stack

| Layer | Technology | Role |
|---|---|---|
| **AI Model** | VGGT-1B (Meta/Oxford, CVPR 2025) | 3D reconstruction from video |
| **Backbone** | DINOv2 ViT-L/14 | Visual feature extraction |
| **Deep Learning** | PyTorch 2.14 (CPU/GPU) | Inference engine |
| **Mesh** | Open3D 0.20 | Poisson surface reconstruction |
| **3D Export** | Trimesh 5.x | GLB/OBJ/PLY export + metadata |
| **Visualization** | Plotly (3D interactive) | 5-mode dashboard viewer |
| **UI** | Gradio 5.17 | Web dashboard |
| **Geometry** | SciPy, NumPy | RANSAC, convex hull, OBB |
| **Video** | OpenCV 5.0 | Frame extraction + blur filter |
| **Metadata** | JSON + GLB extras + PLY headers | Traceability / audit trail |

---

## 📐 Mathematical Foundation

### Collinearity Equation (Eq. 34) — 3D Ray from Camera

```
[X]   [X₀]       [r₁₁ r₁₂ r₁₃] [x - x₀]
[Y] = [Y₀] + λ · [r₂₁ r₂₂ r₂₃] [y - y₀]
[Z]   [Z₀]       [r₃₁ r₃₂ r₃₃] [-c    ]
```

Where `(X₀,Y₀,Z₀)` = VGGT camera centre, `R` = rotation matrix, `(x₀,y₀,c)` = principal point & focal length from VGGT intrinsics.

### Space Intersection (Eq. 35) — Uncaptured Region Inference

For N rays from N cameras through the same gap region, find the best-fit 3D point:

```
min_P  Σᵢ ‖(P - Cᵢ) - ⟨P-Cᵢ, dᵢ⟩·dᵢ‖²     (solved via least-squares)
```

### Similarity Model (Eq. 4) — Scale Recovery

```
d_of = d_oc · d_AB
```

`d_AB` = median inter-camera baseline · `d_oc` = ray distance · `d_of` = real-world scaled distance.

### Confidence Colouring

```
alpha = confidence^255     (VGGT patent formula)

Green  (conf ≥ 0.85) → trust fully
Yellow (conf 0.60–0.85) → use with caution
Red    (conf < 0.60)  → re-fly / verify
Orange → inferred (mathematically estimated from uncaptured regions)
```

### Dimension Check

```
d = √(ΔX² + ΔY² + ΔZ²) × s      (s = metric scale factor from GPS)
```

---

## 🚀 Quick Start

### Install

```bash
git clone <repo>
cd Vayumesh2.0

# Create virtualenv
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac

# Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install einops safetensors huggingface_hub scipy trimesh matplotlib \
            requests opencv-python tqdm open3d onnxruntime gradio plotly
```

### Run the Ultimate Viewer (recommended)

```bash
.venv\Scripts\python.exe viewer_ultimate.py
# Opens http://127.0.0.1:7861
```

### Run the Pipeline on any Video

```bash
.venv\Scripts\python.exe run_pipeline.py \
    --video "your_video.mp4" \
    --output output_3d_result \
    --fps 1.5 \
    --max_frames 25
```

### Load Results into Viewer

1. Open `http://127.0.0.1:7861`
2. Paste `output_3d_result` in the path box
3. Click **📂 Load** — all 5 view modes available instantly

---

## 🎮 Dashboard Features

### 5 View Modes (one-click switch)

| Button | Mode | What you see |
|---|---|---|
| ☁️ | **Point Cloud** | 1.5M+ coloured 3D points + camera frustums |
| 🌈 | **Depth Map** | Turbo rainbow by elevation (near=blue, far=red) |
| 🔥 | **Confidence Heatmap** | Green/Yellow/Red per confidence band |
| 🔲 | **Wireframe Mesh** | Solid surface + cyan edge overlay |
| 🏗️ | **Solid Mesh** | Vertex-coloured shaded surface |

### Fixed Ground Plane

- Z-axis `autorange=False` — ground never drifts
- `dragmode="orbit"` — Z-up axis always locked
- `camera.up = {z:1}` — cannot flip scene

### Dimension Tool

Enter two 3D points → get ΔX, ΔY, ΔZ, Euclidean distance drawn as a yellow line on the model.

### Uncaptured Scene Inference

Toggle to show **orange inferred points** — regions the drone never directly saw, estimated via Space Intersection rays from neighbouring cameras.

### Fullscreen

Press **F** or click **⛶ Full** button (top-right, always visible).

---

## 📊 Results on Test Videos

| Video | Duration | Frames | Inference | Mesh Verts | Mesh Faces |
|---|---|---|---|---|---|
| iStock (urban) | 14.5s | 10 | 278s | 181,450 | 361,227 |
| Mosque aerial | 32.9s | 17 | ~360s | 160,956 | 318,006 |
| Taj Mahal HD | 116.8s | 20 | 358s | **434,317** | **862,812** |

All runs on **CPU only** (no GPU). With GPU (A100), inference drops to ~2–5 seconds.

---

## 🗂️ Output Files

Every pipeline run produces:

```
output_3d_<name>/
├── images/               ← extracted frames (blur-filtered)
├── predictions.npz       ← raw VGGT predictions (depth, cameras, world_points)
├── pointcloud.glb        ← coloured 3D point cloud + camera frustums
├── reconstruction.glb    ← Poisson mesh (metadata in GLB extras)
├── reconstruction.obj    ← Mesh (OBJ + MTL)
├── reconstruction.ply    ← Mesh + vayumesh.* PLY comment headers
├── footprint.geojson     ← 2D convex-hull footprint (GIS-ready)
└── metadata.json         ← full pipeline sidecar (video, model, confidence, mesh)
```

### Embedded Metadata (in every output)

```json
{
  "pipeline": { "version": "2.0.0", "run_id": "20260929_163133", "ts": "..." },
  "video": { "file": "mosque.mp4", "fps": 30, "w": 1280, "h": 720, "duration": 32.87 },
  "model": { "id": "facebook/VGGT-1B", "checksum_sha256_1mb": "307a18dc" },
  "confidence": { "mean": 5.72, "p50": 5.36, "p10": 1.0, "p90": 10.76 },
  "cameras": 17,
  "mesh": { "vertices": 160956, "faces": 318006, "watertight": false, "extent": [...] }
}
```

---

## 🆚 Competitive Analysis

| Feature | VayuMesh 2.0 | SketchBrowse | DJI Terra | Pix4D | ContextCapture | Matterport |
|---|---|---|---|---|---|---|
| **No GCPs needed** | ✅ | ❌ | ⚠️ | ❌ | ❌ | ❌ |
| **Single video pass** | ✅ | ❌ | ⚠️ | ❌ | ❌ | ❌ |
| **Fully offline** | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ |
| **Open source** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Confidence heatmap** | ✅ | ❌ | ❌ | ⚠️ | ❌ | ❌ |
| **Uncaptured inference** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Free / no licence** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Runs on laptop CPU** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Embedded metadata** | ✅ | ❌ | ⚠️ | ✅ | ⚠️ | ❌ |
| **GeoJSON footprint** | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **3D wireframe view** | ✅ | ❌ | ❌ | ⚠️ | ✅ | ✅ |
| **Fixed ground axis** | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| **Military-grade audit** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **India-made** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

> ✅ Full support · ⚠️ Partial · ❌ Not supported

---

## 📁 Project Structure

```
Vayumesh2.0/
│
├── vggt/                        ← VGGT-1B model (Meta AI / Oxford VGG)
│   ├── models/vggt.py           ← Top-level model (aggregator + heads)
│   ├── heads/                   ← CameraHead, DPTHead, TrackHead
│   └── utils/                   ← load_fn, pose_enc, geometry
│
├── run_pipeline.py              ← CLI: video → full 3D + metadata
├── viewer_ultimate.py           ← SIH demo: Ultimate Viewer dashboard
├── viewer_engine.py             ← All rendering backends (5 Plotly modes)
├── dashboard_v2.py              ← Alternative Gradio dashboard
├── video_to_3d.py               ← Core pipeline stages
├── mesh_reconstruction.py       ← Poisson reconstruction
├── metric_scaling.py            ← GPS fusion + RANSAC ground alignment
├── dimension_extraction.py      ← OBB + ConvexHull dimensions
├── visual_util.py               ← GLB export + sky segmentation
├── convert_exports.py           ← Multi-format export
│
├── configs/
│   └── uavff3d_finetune.yaml    ← Fine-tuning config for drone data
│
├── output_3d_istock/            ← iStock urban result
├── output_3d_taj/               ← Taj Mahal result (434K verts)
├── dashboard_outputs/           ← All viewer session outputs
│   └── 20260929_163133/         ← Mosque result (160K verts)
│
├── examples/                    ← Sample image sequences
├── docs/                        ← Package documentation
└── .cache/vggt_model.pt         ← Cached VGGT-1B weights (5 GB)
```

---

## 🔧 Troubleshooting

| Issue | Fix |
|---|---|
| `No module named 'torch'` | Activate venv: `.venv\Scripts\activate` |
| `OOM / memory error` | Reduce `--max_frames` to 10–15 |
| `Cannot open video` | Use full absolute path; avoid emoji in CLI args |
| `Heatmap no colours` | Fixed in viewer_engine.py — restart viewer |
| Dashboard crashes on startup | Gradio 5.17 schema bug patched in `.venv\Lib\site-packages\gradio_client\utils.py` |
| Ground plane moves on orbit | Fixed: `zaxis.autorange=False`, `dragmode="orbit"`, `camera.up={z:1}` |

---

## 📚 References

- Wang et al., *"VGGT: Visual Geometry Grounded Transformer"*, CVPR 2025 Best Paper — [arXiv:2503.11651](https://arxiv.org/abs/2503.11651)
- Meta AI + Oxford VGG — [HuggingFace: facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B)
- Open3D — Screened Poisson Surface Reconstruction
- Kazhdan & Hoppe, *"Screened Poisson Surface Reconstruction"*, ToG 2013

---

## ⚖️ License

- **Code (VayuMesh additions):** MIT License
- **VGGT-1B weights:** Non-commercial research use only ([LICENSE.txt](LICENSE.txt))
- **VGGT-1B-Commercial:** Available at [HuggingFace](https://huggingface.co/facebook/VGGT-1B-Commercial) with approval form

---

<div align="center">

**Made with ❤️ for the Indian Army — Atmanirbhar Bharat in 3D**

*SIH 2026 · Problem Statement 26158 · Theme: Defence & Security*

</div>
#   V a y u m e s h  
 