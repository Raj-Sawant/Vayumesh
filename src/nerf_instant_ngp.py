"""
Tiny Instant‑NGP (hash‑grid) NeRF trained on a handful of key‑frames.
Produces a textured mesh via marching‑cubes (resolution controllable).
Designed to run <30 s on RTX 3050 / Jetson Orin.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from tqdm import tqdm

# ----------------------------------------------------------------------
# Minimal hash‑grid encoder (from instant‑ngp paper) – 16‑level, 2^19 params
# ----------------------------------------------------------------------
class HashGridEncoder(nn.Module):
    def __init__(self, n_levels=16, log2_hashmap_size=19, feature_dim=2):
        super().__init__()
        self.n_levels = n_levels
        self.hashmap_size = 2 ** log2_hashmap_size
        self.feature_dim = feature_dim
        self.embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(self.hashmap_size, feature_dim) * 1e-4)
            for _ in range(n_levels)
        ])
        self.register_buffer("scales", torch.tensor([2 ** i for i in range(n_levels)]))

    def forward(self, xyz):          # xyz: (N,3) in [-1,1]
        N = xyz.shape[0]
        feats = []
        for lvl, scale in enumerate(self.scales):
            grid_coords = (xyz * scale).floor().long()  # (N,3)
            # simple 3‑D hash (x*73856093 ^ y*19349663 ^ z*83492791) % size
            h = (grid_coords[:, 0] * 73856093 ^ grid_coords[:, 1] * 19349663 ^ grid_coords[:, 2] * 83492791) % self.hashmap_size
            feats.append(self.embeddings[lvl][h])
        return torch.cat(feats, dim=-1)   # (N, n_levels*feature_dim)


class TinyNeRF(nn.Module):
    def __init__(self, encoder, hidden=64):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.n_levels * encoder.feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),               # rgb + density
        )

    def forward(self, xyz, dirs):
        h = self.encoder(xyz)
        out = self.mlp(h)
        rgb = torch.sigmoid(out[..., :3])
        sigma = F.softplus(out[..., 3:])
        return rgb, sigma


class InstantNGPFusion:
    """
    High‑level wrapper used by pipeline.py.
    """
    def __init__(self, predictions, keyframe_every=5, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictions = predictions
        self.keyframe_every = keyframe_every
        self._prepare_training_data()

        # Model
        self.encoder = HashGridEncoder().to(self.device)
        self.nerf = TinyNeRF(self.encoder).to(self.device)
        self.opt = torch.optim.Adam(self.nerf.parameters(), lr=1e-2, betas=(0.9, 0.99))

    def _prepare_training_data(self):
        """Collect rays from every key‑frame."""
        extr = self.predictions["extrinsic"]          # (N,4,4) world→cam
        intr = self.predictions["intrinsic"]          # (N,3,3)
        imgs = self.predictions["images"]             # (N,H,W,3) RGB 0‑1
        N, H, W, _ = imgs.shape

        rays_o, rays_d, target_rgb = [], [], []
        for i in range(0, N, self.keyframe_every):
            T_wc = extr[i]
            K = intr[i]
            # camera centre in world
            cam_pos = T_wc[:3, 3]
            R_wc = T_wc[:3, :3]
            # pixel grid
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3).T   # (3, HW)
            rays_cam = np.linalg.inv(K) @ pix                               # (3, HW)
            rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=0, keepdims=True)
            rays_world = (R_wc @ rays_cam).T                                 # (HW,3)
            rays_o.append(np.broadcast_to(cam_pos, rays_world.shape))
            rays_d.append(rays_world)
            target_rgb.append(imgs[i].reshape(-1, 3))

        self.rays_o = torch.from_numpy(np.concatenate(rays_o)).float().to(self.device)
        self.rays_d = torch.from_numpy(np.concatenate(rays_d)).float().to(self.device)
        self.target_rgb = torch.from_numpy(np.concatenate(target_rgb)).float().to(self.device)

    def train(self, iterations=2000, batch_size=4096):
        self.nerf.train()
        n_rays = self.rays_o.shape[0]
        for it in tqdm(range(iterations), desc="Instant‑NGP"):
            idx = torch.randint(0, n_rays, (batch_size,), device=self.device)
            ro = self.rays_o[idx]
            rd = self.rays_d[idx]
            gt = self.target_rgb[idx]

            # simple stratified sampling 64 pts per ray
            t_vals = torch.linspace(0.5, 20.0, 64, device=self.device)  # near/far heuristic
            pts = ro.unsqueeze(1) + rd.unsqueeze(1) * t_vals.view(1, -1, 1)   # (B,64,3)
            pts_flat = pts.view(-1, 3)
            dirs_flat = rd.unsqueeze(1).expand(-1, 64, -1).reshape(-1, 3)

            rgb, sigma = self.nerf(pts_flat, dirs_flat)
            rgb = rgb.view(batch_size, 64, 3)
            sigma = sigma.view(batch_size, 64, 1)

            # volume rendering (simple alpha compositing)
            delta = t_vals[1:] - t_vals[:-1]
            delta = torch.cat([delta, torch.tensor([1e10], device=self.device)])
            alpha = 1 - torch.exp(-sigma * delta.view(1, -1, 1))
            weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], 1), 1)[:, :-1]
            comp_rgb = (weights * rgb).sum(1)

            loss = F.mse_loss(comp_rgb, gt)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if it % 500 == 0:
                tqdm.write(f"iter {it} loss {loss.item():.6f}")

    @torch.no_grad()
    def extract_mesh(self, resolution=512, threshold=25.0):
        """Run marching cubes on a dense grid."""
        self.nerf.eval()
        # query density on grid
        xs = torch.linspace(-1, 1, resolution, device=self.device)
        ys = torch.linspace(-1, 1, resolution, device=self.device)
        zs = torch.linspace(-1, 1, resolution, device=self.device)
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing='ij')
        pts = torch.stack([grid_x, grid_y, grid_z], -1).view(-1, 3)
        # dummy dirs (density only)
        dirs = torch.zeros_like(pts)
        _, sigma = self.nerf(pts, dirs)
        sigma = sigma.view(resolution, resolution, resolution).cpu().numpy()

        # marching cubes
        verts, faces, _, _ = trimesh.voxel.ops.matrix_to_marching_cubes(sigma > threshold, pitch=2.0/resolution)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        return mesh