"""
Frame‑level enhancement: CLAHE, dark‑channel dehazing, optional 2× Real‑ESRGAN.
All functions are CPU‑only and fast enough for on‑board use.
"""
import cv2
import numpy as np

try:
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    _HAS_REALESRGAN = True
except Exception:
    _HAS_REALESRGAN = False


def _clahe(img, clip=2.0, grid=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def _dark_channel_dehaze(img, patch=15, omega=0.95, t0=0.1):
    """Fast single‑image dehazing (He et al. 2011) – works on BGR uint8."""
    img_f = img.astype(np.float32) / 255.0
    dark = np.min(img_f, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(dark, kernel)
    A = np.max(img_f.reshape(-1, 3), axis=0)
    transmission = 1 - omega * dark
    transmission = np.clip(transmission, t0, 1)
    for c in range(3):
        img_f[:, :, c] = (img_f[:, :, c] - A[c]) / transmission + A[c]
    return np.clip(img_f * 255, 0, 255).astype(np.uint8)


# Lazy‑load Real‑ESRGAN once
_ESRGAN_UPSCALER = None
def _get_esrgan(upscale=2):
    global _ESRGAN_UPSCALER
    if not _HAS_REALESRGAN:
        return None
    if _ESRGAN_UPSCALER is None:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=upscale)
        _ESRGAN_UPSCALER = RealESRGANer(scale=upscale, model_path=None, model=model,
                                       tile=0, tile_pad=10, pre_pad=0, half=False)
    return _ESRGAN_UPSCALER


def enhance_frames(frame_bgr, do_sr=False):
    """
    Apply CLAHE + dehaze (+ optional 2× super‑resolution) on a single BGR frame.
    Returns enhanced BGR uint8 image.
    """
    # 1) CLAHE
    frame = _clahe(frame_bgr)

    # 2) Dehaze
    frame = _dark_channel_dehaze(frame)

    # 3) Optional super‑resolution (only on key‑frames)
    if do_sr:
        upscaler = _get_esrgan()
        if upscaler is not None:
            frame, _ = upscaler.enhance(frame, outscale=2)

    return frame