"""
Multi‑scale Temporal Depth‑From‑Video (DFV) network.
Wraps a pretrained MiDaS‑style encoder fine‑tuned on UAVFF3D.
Provides depth + aleatoric uncertainty per frame.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms

# ----------------------------------------------------------------------
# Simple encoder‑decoder (MiDaS‑lite) – replace with actual fine‑tuned ckpt
# ----------------------------------------------------------------------
class _MiDaSLite(nn.Module):
    def __init__(self):
        super().__init__()
        # Using a tiny ResNet18 backbone + DPT head (very small)
        import torchvision.models as models
        backbone = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # 512 x H/32 x W/32
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 2, 3, padding=1),   # 2 channels → depth, log‑var
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        depth = F.softplus(out[:, :1])          # positive depth
        logvar = out[:, 1:]                     # aleatoric log‑variance
        return depth, logvar


class DepthFromVideoNet:
    """
    High‑level API used by pipeline.py.
    """
    def __init__(self, pretrained="uavff3d", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = _MiDaSLite().to(self.device).eval()
        # TODO: load fine‑tuned weights from UAVFF3D (download & load_state_dict)
        # self.net.load_state_dict(torch.load(...))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _infer_single(self, img_bgr):
        """img_bgr: HxWx3 uint8"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        depth, logvar = self.net(tensor)
        depth = F.interpolate(depth, size=img_bgr.shape[:2], mode='bilinear', align_corners=False)
        logvar = F.interpolate(logvar, size=img_bgr.shape[:2], mode='bilinear', align_corners=False)
        return depth.squeeze().cpu().numpy(), logvar.squeeze().cpu().numpy()

    def predict(self, image_paths, extrinsics, intrinsics):
        """
        Runs the network on every frame, then enforces temporal consistency
        with a simple photometric + edge‑aware loss (here just a placeholder).
        Returns:
            refined_depth  – list/array (N, H, W)
            aleatoric_var  – list/array (N, H, W)
        """
        N = len(image_paths)
        refined = []
        aleatoric = []
        for i, p in enumerate(image_paths):
            frame = cv2.imread(p)
            d, lv = self._infer_single(frame)
            refined.append(d)
            aleatoric.append(np.exp(lv))   # variance
        refined = np.stack(refined)          # (N, H, W)
        aleatoric = np.stack(aleatoric)

        # ---- Temporal consistency (placeholder) ----
        # In a real implementation you would run a few gradient steps minimising
        # photometric error between adjacent frames using the current poses.
        # Here we just return the raw predictions.
        return refined, aleatoric