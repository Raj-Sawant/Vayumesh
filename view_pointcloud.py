"""
View VGGT output as interactive 3D point cloud using Open3D.
Usage: python view_pointcloud.py [--npz output_3d/predictions.npz] [--conf 50]
"""

import argparse
import numpy as np
import open3d as o3d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="output_3d/predictions.npz")
    parser.add_argument("--conf", type=float, default=50.0,
                        help="Filter out lowest N%% confidence points (0=keep all, 50=keep top half)")
    args = parser.parse_args()

    print(f"Loading {args.npz} ...")
    data = np.load(args.npz)

    print("Keys found:", list(data.keys()))

    # ── world points ──
    pts = data["world_points_from_depth"]   # (S, H, W, 3)
    conf = data["world_points_conf"]        # (S, H, W)
    imgs = data["images"]                   # (S, 3, H, W)  values in [0,1]

    S, H, W, _ = pts.shape
    print(f"Frames={S}, H={H}, W={W}  →  {S*H*W:,} raw points")

    pts_flat  = pts.reshape(-1, 3)
    conf_flat = conf.reshape(-1)
    # Convert CHW → HWC and flatten
    cols_flat = imgs.transpose(0, 2, 3, 1).reshape(-1, 3).clip(0, 1)

    # ── confidence filter ──
    if args.conf > 0:
        threshold = np.percentile(conf_flat, args.conf)
        mask = conf_flat > threshold
    else:
        mask = np.ones(len(pts_flat), dtype=bool)

    print(f"Keeping {mask.sum():,} / {len(pts_flat):,} points "
          f"(conf threshold = {args.conf:.0f}th percentile)")

    # ── remove NaN / Inf ──
    finite_mask = np.isfinite(pts_flat[mask]).all(axis=1)
    pts_clean  = pts_flat[mask][finite_mask]
    cols_clean = cols_flat[mask][finite_mask]
    print(f"After NaN removal: {len(pts_clean):,} points")

    # ── build Open3D point cloud ──
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_clean)
    pcd.colors = o3d.utility.Vector3dVector(cols_clean)

    # ── optional: estimate normals for better shading ──
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    print("\nControls:")
    print("  Mouse drag    — rotate")
    print("  Scroll        — zoom")
    print("  Middle drag   — pan")
    print("  R             — reset view")
    print("  Q / Esc       — quit")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="VGGT 3D Point Cloud",
        width=1280,
        height=720,
        point_show_normal=False,
    )

if __name__ == "__main__":
    main()
