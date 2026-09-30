"""
Factor‑graph bundle adjustment fusing:
  • Visual reprojection factors (VGGT / COLMAP key‑frames)
  • IMU pre‑integration factors (high‑rate)
  • GNSS / RTK / PPK absolute pose priors
  • Optional barometric altitude factors
Built on GTSAM (Python bindings).  Returns optimized extrinsics / intrinsics.
"""
import numpy as np

try:
    import gtsam
    _HAS_GTSAM = True
except Exception:
    _HAS_GTSAM = False


def _read_csv(path, expected_cols):
    if not path:
        return None
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    assert data.shape[1] >= expected_cols, f"{path} needs ≥{expected_cols} columns"
    return data


def build_factor_graph(predictions, imu_csv=None, gps_csv=None, rtk_csv=None, cam_json=None):
    """
    Constructs a gtsam.NonlinearFactorGraph and initial Values.
    `predictions` must contain:
        - "extrinsic": (N,4,4) world‑to‑cam
        - "intrinsic": (N,3,3) K
        - "images": list of paths (for optional visual factors)
    Returns (graph, initial_values).
    """
    if not _HAS_GTSAM:
        raise RuntimeError("GTSAM not installed – `pip install gtsam`")

    N = predictions["extrinsic"].shape[0]
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # ---- Camera intrinsics (shared or per‑frame) ----
    # For simplicity assume constant K (first frame)
    K = predictions["intrinsic"][0]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    calib = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

    # ---- Pose symbols ----
    X = lambda i: gtsam.symbol('x', i)

    # Initial poses from VGGT
    for i in range(N):
        T_wc = predictions["extrinsic"][i]          # 4x4 world→cam
        R = gtsam.Rot3(T_wc[:3, :3])
        t = gtsam.Point3(T_wc[:3, 3])
        values.insert(X(i), gtsam.Pose3(R, t))

    # ---- Visual reprojection factors (placeholder) ----
    # In a real system you would add gtsam.GenericProjectionFactor
    # for each tracked 2D‑3D correspondence.  Skipped here for brevity.

    # ---- IMU pre‑integration ----
    imu_data = _read_csv(imu_csv, expected_cols=7)   # t, wx,wy,wz, ax,ay,az
    if imu_data is not None:
        # GTSAM expects a continuous IMU stream; here we just demonstrate
        # creating a CombinedImuFactor between consecutive frames.
        # You must synchronize IMU timestamps to frame timestamps.
        # This is a stub – replace with proper integration.
        pass

    # ---- GNSS / RTK absolute pose priors ----
    for csv_path, sigma in [(gps_csv, 1.0), (rtk_csv, 0.02)]:  # sigma in metres
        data = _read_csv(csv_path, expected_cols=4)  # t, lat, lon, alt
        if data is None:
            continue
        # Convert lat/lon/alt → local ENU (requires origin).  Omitted.
        # Add gtsam.GPSFactor or PriorFactorPose3 for matched frames.
        pass

    # ---- Barometric altitude (optional) ----
    # Similar to GNSS but only Z component.

    return graph, values


def optimize_graph(graph, values, max_iter=50):
    """
    Runs Levenberg‑Marquardt (or iSAM2 for incremental) and returns
    a dict with optimized extrinsics/intrinsics matching predictions keys.
    """
    if not _HAS_GTSAM:
        raise RuntimeError("GTSAM not installed")

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(max_iter)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    result = optimizer.optimize()

    N = max([int(gtsam.symbolChr(k)) for k in result.keys()]) + 1
    opt_extr = np.zeros((N, 4, 4))
    for i in range(N):
        pose = result.atPose3(gtsam.symbol('x', i))
        R = pose.rotation().matrix()
        t = pose.translation()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        opt_extr[i] = T

    return {"extrinsic": opt_extr, "intrinsic": predictions["intrinsic"]}