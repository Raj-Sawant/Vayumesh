"""
Semantic‑aware multi‑LOD mesh generation.
Classifies mesh faces (roof, wall, road, vegetation) with a tiny PointNet++ head
fine‑tuned on UAVFF3D, then applies class‑weighted quadric decimation.
"""
import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Tiny PointNet++ classification head (operates on face centroids + normals)
# ----------------------------------------------------------------------
class _SemanticHead(nn.Module):
    def __init__(self, num_classes=4, feat_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, xyz_normal):   # (N,6)  xyz + normal
        return self.mlp(xyz_normal)


# ----------------------------------------------------------------------
# High‑level wrapper used by pipeline.py
# ----------------------------------------------------------------------
CLASS_NAMES = ["roof", "wall", "road", "vegetation"]
# target triangle budget per class for LOD0 / LOD1 / LOD2
LOD_BUDGETS = {
    "roof":        (20000, 50000, 150000),
    "wall":        (15000, 40000, 120000),
    "road":        (10000, 30000, 100000),
    "vegetation":  (5000,  15000, 50000),
}


def _classify_faces(mesh, device=None):
    """Returns per‑face class index."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # compute face centroids + normals
    centroids = mesh.triangles_center            # (F,3)
    normals = mesh.face_normals                  # (F,3)
    feats = np.concatenate([centroids, normals], axis=1).astype(np.float32)

    model = _SemanticHead().to(device).eval()
    # TODO: load fine‑tuned weights from UAVFF3D
    # model.load_state_dict(torch.load(...))
    with torch.no_grad():
        logits = model(torch.from_numpy(feats).to(device))
    pred = logits.argmax(1).cpu().numpy()
    return pred   # (F,) int


def _decimate_class(mesh, face_mask, target_tris):
    """Quadric decimation on a sub‑mesh defined by face_mask."""
    sub = mesh.submesh([face_mask], append=True)[0]
    if len(sub.faces) <= target_tris:
        return sub
    # Open3D style quadric decimation via trimesh (approx)
    sub = sub.simplify_quadratic_decimation(target_tris)
    return sub


def build_semantic_lod(mesh, predictions, device=None):
    """
    Returns dict {lod_name: trimesh.Trimesh} with three LODs.
    `predictions` currently unused but kept for future (e.g., texture).
    """
    face_class = _classify_faces(mesh, device)

    lod_meshes = {}
    for lod_idx, lod_name in enumerate(["LOD0", "LOD1", "LOD2"]):
        combined = trimesh.Trimesh()
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            mask = (face_class == cls_idx)
            if not mask.any():
                continue
            budget = LOD_BUDGETS[cls_name][lod_idx]
            sub = _decimate_class(mesh, mask, budget)
            combined = trimesh.util.concatenate([combined, sub])
        # Ensure watertight-ish
        combined.remove_duplicate_faces()
        combined.remove_unreferenced_vertices()
        lod_meshes[lod_name] = combined

    return lod_meshes