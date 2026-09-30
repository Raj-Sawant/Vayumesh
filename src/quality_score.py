"""
Flight‑path quality metric:
score = Σ (baseline_i * sin(view_angle_i)) / N
Higher ⇒ better geometry coverage.
Returns scalar in [0,1] (approx).
"""
import numpy as np

def flight_quality_score(predictions):
    """
    predictions must contain:
        - "extrinsic": (N,4,4) world→cam
    """
    extr = predictions["extrinsic"]          # (N,4,4)
    N = extr.shape[0]
    if N < 2:
        return 0.0

    # camera centres
    centres = extr[:, :3, 3]                 # (N,3)

    # baselines between consecutive frames
    baselines = np.linalg.norm(centres[1:] - centres[:-1], axis=1)   # (N-1,)

    # view angles: angle between viewing direction and vertical (z)
    view_dirs = -extr[:, :3, 2]              # camera looks along -Z_cam
    view_angles = np.arccos(np.clip(np.abs(view_dirs[:, 2]), 0, 1))   # 0 = nadir, pi/2 = horizon

    # use sin(view_angle) to reward oblique views
    scores = baselines * np.sin(view_angles[:-1])
    score = scores.sum() / (baselines.sum() + 1e-6)
    # normalise roughly to 0‑1 (empirical max ~ 30 m baseline * 1)
    return float(np.clip(score / 30.0, 0.0, 1.0))