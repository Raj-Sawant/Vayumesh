"""
Uncertainty propagation:
- BA covariance (from factor graph) → pose uncertainty
- DFV aleatoric variance (per‑pixel)
- NeRF epistemic variance (Monte‑Carlo dropout or ensemble)
Combines to per‑vertex σ for the final mesh.
"""
import numpy as np
import torch

def propagate_uncertainty(predictions, mesh):
    """
    predictions dict should contain:
        - "extrinsic_cov": (N,6,6)  (optional, from GTSAM marginalCovariance)
        - "depth_aleatoric": (N,H,W) variance from DFV
    mesh: trimesh.Trimesh of the finest LOD (LOD2)
    Returns per‑vertex standard deviation (N_vertices,).
    """
    V = mesh.vertices.shape[0]
    vertex_sigma = np.full(V, 0.05, dtype=np.float32)   # default 5 cm

    # 1) Pose covariance → project to vertex positions
    if "extrinsic_cov" in predictions:
        # Very rough: average pose variance, assume isotropic 1 cm per metre baseline
        pose_var = predictions["extrinsic_cov"][:, :3, :3].mean()   # scalar
        vertex_sigma += np.sqrt(pose_var) * 100.0   # convert to cm

    # 2) Aleatoric depth variance – rasterise vertices onto key‑frames
    if "depth_aleatoric" in predictions:
        depth_var = predictions["depth_aleatoric"]      # (N,H,W)
        extr = predictions["extrinsic"]                 # (N,4,4)
        intr = predictions["intrinsic"]                 # (N,3,3)
        imgs = predictions["images"]                    # (N,H,W,3)
        N, H, W, _ = imgs.shape

        for i in range(N):
            K = intr[i]
            T_wc = extr[i]
            R = T_wc[:3, :3]
            t = T_wc[:3, 3]
            # project all vertices
            verts_cam = (R @ mesh.vertices.T + t.reshape(3,1)).T   # (V,3)
            z = verts_cam[:, 2]
            valid = z > 0.1
            uv = (K @ verts_cam[valid].T).T
            uv[:, 0] /= uv[:, 2]
            uv[:, 1] /= uv[:, 2]
            u = np.clip(np.round(uv[:, 0]).astype(int), 0, W-1)
            v = np.clip(np.round(uv[:, 1]).astype(int), 0, H-1)
            var_at_vert = depth_var[i, v, u]
            vertex_sigma[valid] = np.maximum(vertex_sigma[valid], np.sqrt(var_at_vert)*100.0)

    # 3) NeRF epistemic – Monte Carlo dropout (5 forward passes)
    if hasattr(predictions, "nerf_model"):
        nerf = predictions["nerf_model"]
        nerf.train()   # enable dropout
        with torch.no_grad():
            samples = []
            for _ in range(5):
                # query density at vertex positions (normalized to [-1,1])
                pts = torch.from_numpy(mesh.vertices).float().to(next(nerf.parameters()).device)
                pts_norm = pts / pts.abs().max()   # crude normalisation
                _, sigma = nerf(pts_norm, torch.zeros_like(pts_norm))
                samples.append(sigma.cpu().numpy())
            sigma_samples = np.stack(samples, axis=0)   # (5, V, 1)
            epistemic_std = sigma_samples.std(0).squeeze() * 100.0
            vertex_sigma = np.maximum(vertex_sigma, epistemic_std)

    return vertex_sigma.astype(np.float32)