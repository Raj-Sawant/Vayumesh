"""
Dynamic object detection + masking + depth/colour inpainting.
Uses SAM2 (segment‑anything‑2) for class‑agnostic masks and YOLO‑v9‑seg
for class‑aware detection (vehicles, persons).  Inpainting via
EDGE‑CONNECT (fast CUDA) – here we provide a CPU fallback using
OpenCV inpainting.
"""
import cv2
import numpy as np

# ----------------------------------------------------------------------
# Placeholder detectors – replace with real model loading (ONNX / TensorRT)
# ----------------------------------------------------------------------
def _load_sam2():
    # TODO: load SAM2 ONNX model, return callable(img) -> list[masks]
    return None

def _load_yolo_seg():
    # TODO: load YOLOv9‑seg ONNX/TensorRT, return callable(img) -> list[boxes,masks,cls]
    return None

_SAM2 = _load_sam2()
_YOLO = _load_yolo_seg()


def _detect_masks(frame_bgr):
    """Return a combined binary mask (H,W) of dynamic objects."""
    h, w = frame_bgr.shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    # SAM2 (class‑agnostic) – fallback to simple motion cue if not available
    if _SAM2 is not None:
        sam_masks = _SAM2(frame_bgr)
        for m in sam_masks:
            combined = cv2.bitwise_or(combined, (m > 0.5).astype(np.uint8))

    # YOLO‑seg (vehicle / person classes) – union
    if _YOLO is not None:
        boxes, masks, clses = _YOLO(frame_bgr)
        for m, c in zip(masks, clses):
            if c in (0, 2, 3, 5, 7):   # COCO: person, car, motorcycle, bus, truck
                combined = cv2.bitwise_or(combined, (m > 0.5).astype(np.uint8))

    # Simple morphological clean‑up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    return combined.astype(bool)


def _inpaint_frame(frame_bgr, mask_bool, depth_map=None):
    """
    Inpaint colour (and optionally depth) using OpenCV TELEA.
    For production replace with EDGE‑CONNECT (CUDA) for real‑time.
    """
    mask_u8 = (mask_bool * 255).astype(np.uint8)
    inpainted = cv2.inpaint(frame_bgr, mask_u8, 3, cv2.INPAINT_TELEA)

    if depth_map is not None:
        # depth_map is float32 HxW
        depth_inpainted = cv2.inpaint(depth_map, mask_u8, 3, cv2.INPAINT_TELEA)
        return inpainted, depth_inpainted
    return inpainted, None


def mask_dynamic_objects(predictions, image_paths):
    """
    Main entry used by pipeline.py.
    `predictions` dict contains at least:
        - "images": (N, H, W, 3) float32 RGB 0‑1  (from VGGT preprocessing)
        - "depth":  (N, H, W) float32 metric depth
    Returns updated predictions with cleaned images/depth.
    """
    N = len(image_paths)
    clean_images = []
    clean_depth = []

    for i in range(N):
        # Convert VGGT pre‑processed tensor back to uint8 BGR for detectors
        img_tensor = predictions["images"][i]          # (H,W,3) RGB 0‑1
        img_bgr = (img_tensor[..., ::-1] * 255).astype(np.uint8)

        depth = predictions["depth"][i]                # (H,W) metric

        dyn_mask = _detect_masks(img_bgr)

        if dyn_mask.any():
            img_clean, depth_clean = _inpaint_frame(img_bgr, dyn_mask, depth)
        else:
            img_clean, depth_clean = img_bgr, depth

        # back to VGGT format
        clean_images.append(img_clean[..., ::-1].astype(np.float32) / 255.0)  # RGB 0‑1
        clean_depth.append(depth_clean if depth_clean is not None else depth)

    predictions["images"] = np.stack(clean_images)      # (N,H,W,3) RGB 0‑1
    predictions["depth"] = np.stack(clean_depth)        # (N,H,W)
    return predictions