# VayuMesh: Metric 3D Reconstruction Pipeline for Indian Army

## Overview

VayuMesh is a complete 3D reconstruction pipeline that transforms drone video into **dimensioned, watertight 3D solids with metric accuracy** — designed for the Indian Army's precision intelligence requirements.

### Pipeline Stages

| Stage | Input | Output | Technology |
|-------|-------|--------|------------|
| **1. VGGT Inference** | Video frames | Point cloud + camera poses | VGGT (CVPR 2025) |
| **2. Poisson Reconstruction** | Point cloud | Watertight mesh (OBJ/GLB) | Open3D Poisson |
| **3. Metric Scaling** | Mesh + GPS/IMU | Scaled mesh (meters) | GPS/IMU fusion + Ground plane alignment |
| **4. Dimension Extraction** | Scaled mesh | Dimensions (W×L×H), Volume, Footprint | OBB + Convex Hull |

---

## Quick Start

### 1. Install Dependencies

```bash
# Core + pipeline dependencies
pip install -r requirements.txt
# Or with optional dependencies
pip install -e .[pipeline]
```

### 2. Run Full Pipeline

```bash
# Basic run (point cloud only)
python video_to_3d.py --video your_video.mp4 --output output_3d

# Full pipeline with metric scaling and dimensions
python video_to_3d.py \
  --video your_video.mp4 \
  --output output_3d \
  --gps sample_gps.json \
  --poisson_depth 9 \
  --conf_thres 50
```

### 3. Outputs Generated

```
output_3d/
├── images/                    # Extracted frames
├── predictions.npz            # Raw VGGT predictions
├── reconstruction.glb         # Point cloud visualization
├── reconstruction_mesh.obj    # Watertight mesh (VGGT units)
├── reconstruction_mesh.glb    # Mesh for viewing
├── predictions_scaled.npz     # Metric-scaled point cloud
├── reconstruction_scaled.obj  # Metric-scaled mesh (meters)
├── reconstruction_scaled.glb  # Scaled mesh for viewing
├── metric_transform.json      # Scale/rotation metadata
├── dimensions.json            # Building dimensions report
├── dimensions_viz.glb         # Mesh with dimension annotations
├── roof_profile.npz           # Roof height map
└── dimensions.geojson         # Footprint for GIS
```

---

## GPS/IMU Data Format

Create a JSON file with per-frame GPS data:

```json
[
  {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "altitude": 216.0,
    "yaw": 45.0,
    "pitch": -2.5,
    "roll": 0.8,
    "rtcm_available": false,
    "horizontal_accuracy": 1.5,
    "vertical_accuracy": 2.0
  },
  ...
]
```

**Fields:**
- `latitude/longitude`: WGS84 degrees
- `altitude`: AMSL (meters above mean sea level)
- `yaw`: Heading (0=North, 90=East, degrees)
- `pitch/roll`: Camera gimbal angles (degrees)
- `rtcm_available`: True if RTK correction applied
- `horizontal_accuracy/vertical_accuracy`: GPS accuracy estimates (meters)

---

## Pipeline Details

### Stage 1: VGGT Inference
- Loads pretrained VGGT-1B (4GB weights from HuggingFace)
- Runs on CPU by default (works on any VRAM)
- Outputs: depth, world_points, camera poses, confidence maps

### Stage 2: Poisson Surface Reconstruction
```
Point Cloud → Statistical Outlier Removal → Normal Estimation → Poisson → Mesh Cleaning
```
- **Depth**: 8-10 (9 default, higher = more detail)
- **Denoising**: Statistical Outlier Removal (20 neighbors, 2σ)
- **Density threshold**: Removes bottom 10% low-density vertices
- **Output**: Watertight manifold mesh

### Stage 3: Metric Scaling & Alignment

1. **GPS Scale Recovery**: Procrustes alignment of VGGT camera centers to GPS ENU positions
2. **Ground Plane Detection**: RANSAC plane fitting on point cloud
3. **Z-Up Alignment**: Rotate so ground plane normal = [0, 0, 1]
4. **Cardinal Axis Alignment**: Align X=East, Y=North using GPS yaw or PCA
5. **Apply Scale**: Multiply all coordinates by GPS-derived scale factor

### Stage 4: Dimension Extraction

- **Oriented Bounding Box (OBB)**: PCA-based for rotated buildings
- **Width/Length**: OBB extents in ground plane (X/Y)
- **Height**: Max Z - Ground Z
- **Footprint**: 2D Convex Hull area
- **Volume**: Mesh volume (if watertight) or W×L×H
- **Roof Profile**: Ray-casted height map on regular grid

---

## Fine-tuning on UAVFF3D (for maximum accuracy)

The UAVFF3D paper shows **76% pose error reduction** and **41% Chamfer distance reduction** with fine-tuning.

### 1. Prepare UAVFF3D Dataset
```
Download from: https://github.com/.../UAVFF3D
Structure:
/data/uavff3d/
├── real/
│   ├── sequences/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── syn/
    └── ...
```

### 2. Configure Paths
Edit `configs/uavff3d_finetune.yaml`:
```yaml
data:
  train:
    data_root: "/data/uavff3d"  # YOUR PATH
  val:
    data_root: "/data/uavff3d"
```

### 3. Launch Training
```bash
# Single GPU
python training/launch.py --config configs/uavff3d_finetune.yaml

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 training/launch.py --config configs/uavff3d_finetune.yaml
```

### 4. Use Fine-tuned Model
```bash
# Update video_to_3d.py to load your checkpoint
# model.load_state_dict(torch.load("checkpoints/uavff3d_finetune/checkpoint_50.pt"))
```

---

## Command Reference

### video_to_3d.py Options

```bash
# Input/Output
--video VIDEO.mp4           # Input video
--fps 1.0                   # Sample 1 frame every N seconds
--output output_3d          # Output directory

# Stage 1: VGGT
--conf_thres 50.0           # Confidence percentile (0-100)
--no_cam                    # Hide camera frustums

# Stage 2: Poisson
--poisson_depth 9           # Octree depth (8-10)
--no_denoise                # Skip outlier removal
--density_threshold 0.1     # Density quantile threshold
--skip_mesh                 # Skip mesh reconstruction

# Stage 3: Metric Scaling
--gps gps_data.json         # GPS/IMU JSON file
--skip_scaling              # Skip metric scaling
--ground_z 0.0              # Manual ground Z (meters)

# Stage 4: Dimensions
--skip_dims                 # Skip dimension extraction
--multi_building            # Detect multiple buildings
--min_volume 10.0           # Min building volume (m³)
```

---

## Dimension Report (dimensions.json)

```json
{
  "width_m": 42.3,
  "length_m": 68.7,
  "height_m": 15.2,
  "volume_m3": 43892,
  "footprint_area_m2": 2887,
  "centroid": [102.4, -56.8, 7.6],
  "bbox_corners": [...],
  "oriented_bbox": {
    "transform": [...],
    "extents": [42.3, 68.7, 15.2]
  }
}
```

---

## GeoJSON Footprint (dimensions.geojson)

Compatible with QGIS, ArcGIS, Cesium, etc. for operational mapping.

---

## Validation with UAVFF3D-Real LiDAR

```bash
# After fine-tuning, evaluate on UAVFF3D-Real test set
python evaluation/uavff3d_eval.py \
  --checkpoint checkpoints/uavff3d_finetune/checkpoint_best.pt \
  --data_root /data/uavff3d \
  --split test
```

Metrics reported:
- **Pose ATE** (Absolute Trajectory Error)
- **Ray Error** (Camera ray angular error)
- **Chamfer Distance** (Point cloud accuracy)
- **Scale Error** (Metric scale accuracy)

---

## Architecture

```
video_to_3d.py (main orchestrator)
├── mesh_reconstruction.py      # Poisson reconstruction
├── metric_scaling.py           # GPS fusion, ground alignment
├── dimension_extraction.py     # Building dimensions
└── training/
    ├── trainer.py              # DDP training loop
    ├── loss.py                 # Multi-task loss
    └── data/
        └── uavff3d/            # UAVFF3D dataset loader
```

---

## Performance Notes

| Component | Time (CPU) | GPU Memory |
|-----------|------------|------------|
| VGGT Inference (30 frames) | ~5 min | 4.7 GB |
| Poisson Reconstruction | ~30 sec | 2-4 GB |
| Metric Scaling | <5 sec | <1 GB |
| Dimension Extraction | <10 sec | <1 GB |

**Recommendation**: Use GPU for VGGT inference (edit `device = "cuda"` in video_to_3d.py)

---

## Legal Notice

⚠️ **Parliament/Red Zone Footage**: Drone flights over Indian Parliament (Jantar Mantar area) are **illegal** without Central Government permission. Use public benchmark datasets (UAVFF3D, UrbanScene3D, DroneSplat) for development and validation.

---

## Citation

If you use VayuMesh for research:
```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={CVPR}, year={2025}
}
```
UAVFF3D fine-tuning methodology from UAVFF3D paper (CVPR 2025).