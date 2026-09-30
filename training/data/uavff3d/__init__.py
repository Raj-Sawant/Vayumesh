"""
UAVFF3D Dataset Loader for VGGT Fine-tuning
Supports UAVFF3D-Real and UAVFF3D-Syn datasets with LiDAR ground truth.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class UAVFF3DDataset(Dataset):
    """
    UAVFF3D Dataset for VGGT fine-tuning.
    
    Dataset structure expected:
    uavff3d/
    ├── real/
    │   ├── sequences/
    │   │   ├── seq_001/
    │   │   │   ├── images/
    │   │   │   │   ├── 0000.jpg
    │   │   │   │   └── ...
    │   │   │   ├── poses.txt          # Camera poses (COLMAP format)
    │   │   │   ├── intrinsics.txt     # Camera intrinsics
    │   │   │   ├── depth/             # LiDAR depth maps
    │   │   │   │   ├── 0000.png
    │   │   │   │   └── ...
    │   │   │   └── pointcloud.las     # LiDAR point cloud
    │   │   └── ...
    │   ├── train.txt
    │   ├── val.txt
    │   └── test.txt
    └── syn/
        └── (similar structure)
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (518, 518),
        num_frames: int = 8,
        frame_sampler = None,
        augment: Dict = None,
        load_depth: bool = True,
        load_pose: bool = True,
        load_point_cloud: bool = True,
        normalize_scale: bool = True,
        domain: str = "real"  # "real" or "syn"
    ):
        self.data_root = Path(data_root) / domain
        self.split = split
        self.image_size = image_size
        self.num_frames = num_frames
        self.frame_sampler = frame_sampler
        self.augment = augment or {}
        self.load_depth = load_depth
        self.load_pose = load_pose
        self.load_point_cloud = load_point_cloud
        self.normalize_scale = normalize_scale
        self.domain = domain
        
        # Load sequence list
        self.sequences = self._load_split_file()
        
        # Load all frame metadata
        self.frame_data = self._load_all_frames()
        
        logger.info(f"UAVFF3D {domain} {split}: {len(self.sequences)} sequences, {len(self.frame_data)} frames")
    
    def _load_split_file(self) -> List[str]:
        """Load sequence names from split file."""
        split_file = self.data_root / f"{self.split}.txt"
        if not split_file.exists():
            # Fallback: list all sequences
            seq_dir = self.data_root / "sequences"
            if seq_dir.exists():
                sequences = sorted([d.name for d in seq_dir.iterdir() if d.is_dir()])
                logger.warning(f"Split file not found, using all sequences: {len(sequences)}")
                return sequences
            else:
                raise FileNotFoundError(f"No sequences found in {self.data_root}")
        
        with open(split_file, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
        return sequences
    
    def _load_all_frames(self) -> List[Dict]:
        """Load metadata for all frames in all sequences."""
        all_frames = []
        
        for seq_name in self.sequences:
            seq_path = self.data_root / "sequences" / seq_name
            
            # Load poses
            poses_path = seq_path / "poses.txt"
            if poses_path.exists():
                poses = self._load_poses(poses_path)
            else:
                poses = None
            
            # Load intrinsics
            intr_path = seq_path / "intrinsics.txt"
            if intr_path.exists():
                intrinsics = self._load_intrinsics(intr_path)
            else:
                intrinsics = None
            
            # List images
            img_dir = seq_path / "images"
            if not img_dir.exists():
                logger.warning(f"Image directory not found: {img_dir}")
                continue
            
            image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
            
            # Load depth files if available
            depth_files = {}
            if self.load_depth:
                depth_dir = seq_path / "depth"
                if depth_dir.exists():
                    for df in depth_dir.glob("*.png"):
                        idx = int(df.stem)
                        depth_files[idx] = df
            
            # Load point cloud
            pc_path = None
            if self.load_point_cloud:
                for ext in ['.las', '.laz', '.ply']:
                    pc_path = seq_path / f"pointcloud{ext}"
                    if pc_path.exists():
                        break
            
            for idx, img_file in enumerate(image_files):
                frame_info = {
                    'sequence': seq_name,
                    'frame_idx': idx,
                    'image_path': str(img_file),
                    'pose': poses[idx] if poses is not None and idx < len(poses) else None,
                    'intrinsic': intrinsics[idx] if intrinsics is not None and idx < len(intrinsics) else None,
                    'depth_path': str(depth_files.get(idx)) if idx in depth_files else None,
                    'pointcloud_path': str(pc_path) if pc_path else None
                }
                all_frames.append(frame_info)
        
        return all_frames
    
    def _load_poses(self, path: Path) -> np.ndarray:
        """Load camera poses from COLMAP format."""
        # COLMAP format: image_id qw qx qy qz tx ty tz camera_id name
        poses = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 8:
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    
                    # Quaternion to rotation matrix
                    from scipy.spatial.transform import Rotation as R
                    rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
                    
                    # Build 4x4 extrinsic (world to camera)
                    extrinsic = np.eye(4)
                    extrinsic[:3, :3] = rot
                    extrinsic[:3, 3] = [tx, ty, tz]
                    poses.append(extrinsic)
        return np.array(poses) if poses else None
    
    def _load_intrinsics(self, path: Path) -> np.ndarray:
        """Load camera intrinsics."""
        # Format: camera_id model width height params...
        intrinsics = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    # SIMPLE_PINHOLE: camera_id model width height fx cx cy
                    # PINHOLE: camera_id model width height fx fy cx cy
                    model = parts[1]
                    width, height = int(parts[2]), int(parts[3])
                    params = list(map(float, parts[4:]))
                    
                    if model == 'SIMPLE_PINHOLE':
                        fx = fy = params[0]
                        cx, cy = params[1], params[2]
                    elif model == 'PINHOLE':
                        fx, fy = params[0], params[1]
                        cx, cy = params[2], params[3]
                    else:
                        fx = fy = params[0]
                        cx, cy = width/2, height/2
                    
                    K = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    intrinsics.append(K)
        return np.array(intrinsics) if intrinsics else None
    
    def __len__(self) -> int:
        # Return number of sequence samples (not individual frames)
        return max(1, len(self.frame_data) // self.num_frames)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a batch of frames from a sequence."""
        # Sample a sequence
        seq_idx = idx % len(self.sequences)
        seq_name = self.sequences[seq_idx]
        
        # Get all frames for this sequence
        seq_frames = [f for f in self.frame_data if f['sequence'] == seq_name]
        
        if len(seq_frames) < self.num_frames:
            # Pad by repeating
            selected_frames = seq_frames
            while len(selected_frames) < self.num_frames:
                selected_frames.extend(seq_frames[:self.num_frames - len(selected_frames)])
            selected_frames = selected_frames[:self.num_frames]
        else:
            # Sample frames using sampler
            if self.frame_sampler:
                selected_frames = self.frame_sampler.sample(seq_frames, self.num_frames)
            else:
                # Uniform sampling
                indices = np.linspace(0, len(seq_frames)-1, self.num_frames, dtype=int)
                selected_frames = [seq_frames[i] for i in indices]
        
        # Load data
        return self._load_frames(selected_frames)
    
    def _load_frames(self, frames: List[Dict]) -> Dict:
        """Load images, poses, depths for a list of frames."""
        images = []
        poses = []
        intrinsics = []
        depths = []
        point_masks = []
        
        for f in frames:
            # Load image
            img = self._load_image(f['image_path'])
            images.append(img)
            
            # Load pose
            if self.load_pose and f['pose'] is not None:
                poses.append(f['pose'])
            else:
                poses.append(np.eye(4, dtype=np.float32))
            
            # Load intrinsic
            if self.load_pose and f['intrinsic'] is not None:
                intrinsics.append(f['intrinsic'])
            else:
                # Default intrinsic
                h, w = self.image_size
                intrinsics.append(np.array([
                    [w, 0, w/2],
                    [0, w, h/2],
                    [0, 0, 1]
                ], dtype=np.float32))
            
            # Load depth
            if self.load_depth and f['depth_path']:
                depth = self._load_depth(f['depth_path'])
                depths.append(depth)
                # Create mask from valid depth
                point_masks.append((depth > 0).astype(np.float32))
            else:
                h, w = self.image_size
                depths.append(np.zeros((h, w, 1), dtype=np.float32))
                point_masks.append(np.ones((h, w), dtype=np.float32))
        
        # Stack
        images = torch.from_numpy(np.stack(images)).float() / 255.0  # (S, H, W, 3)
        images = images.permute(0, 3, 1, 2)  # (S, 3, H, W)
        
        poses = torch.from_numpy(np.stack(poses)).float()  # (S, 4, 4)
        intrinsics = torch.from_numpy(np.stack(intrinsics)).float()  # (S, 3, 3)
        depths = torch.from_numpy(np.stack(depths)).float()  # (S, H, W, 1)
        point_masks = torch.from_numpy(np.stack(point_masks)).bool()  # (S, H, W)
        
        # Normalize scale if requested (using LiDAR depth)
        if self.normalize_scale and self.load_depth:
            images, poses, depths, point_masks = self._normalize_metric_scale(
                images, poses, depths, point_masks
            )
        
        return {
            'images': images,
            'extrinsics': poses,
            'intrinsics': intrinsics,
            'depths': depths,
            'point_masks': point_masks,
            'seq_name': frames[0]['sequence'],
            'frame_indices': torch.tensor([f['frame_idx'] for f in frames])
        }
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load and resize image."""
        import cv2
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size[::-1], interpolation=cv2.INTER_AREA)
        return img.astype(np.float32)
    
    def _load_depth(self, path: str) -> np.ndarray:
        """Load depth map."""
        import cv2
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0  # mm to meters
        elif depth.dtype == np.uint8:
            depth = depth.astype(np.float32) / 255.0 * 100  # normalized to meters
        depth = cv2.resize(depth, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
        return depth[..., None].astype(np.float32)
    
    def _normalize_metric_scale(
        self, images, poses, depths, point_masks
    ):
        """Normalize scene to metric scale using LiDAR depth."""
        # Compute scale factor from valid depth points
        valid_mask = point_masks & (depths > 0.1) & (depths < 100.0)
        
        if valid_mask.sum() > 100:
            # Use median depth as reference
            valid_depths = depths[valid_mask]
            median_depth = valid_depths.median()
            
            # Scale poses so median depth ~ 10m (typical UAV altitude)
            target_median = 10.0
            scale = target_median / (median_depth + 1e-6)
            
            # Apply scale to poses (translation only)
            poses = poses.clone()
            poses[:, :3, 3] *= scale
            
            # Scale depths
            depths = depths * scale
        
        return images, poses, depths, point_masks
    
    def get_loader(self, epoch: int = 0, **kwargs) -> DataLoader:
        """Get DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=kwargs.get('batch_size', 1),
            shuffle=kwargs.get('shuffle', self.split == 'train'),
            num_workers=kwargs.get('num_workers', 4),
            pin_memory=kwargs.get('pin_memory', True),
            drop_last=kwargs.get('drop_last', self.split == 'train'),
            worker_init_fn=lambda w: np.random.seed(epoch * 1000 + w)
        )


class UniformFrameSampler:
    """Uniform frame sampler for sequence sampling."""
    
    def __init__(self, num_frames: int, max_skip: int = 5):
        self.num_frames = num_frames
        self.max_skip = max_skip
    
    def sample(self, frames: List[Dict], num_frames: int) -> List[Dict]:
        if len(frames) <= num_frames:
            return frames
        
        # Random start, uniform spacing with some jitter
        max_start = len(frames) - num_frames
        start = np.random.randint(0, max_start + 1)
        
        indices = np.linspace(start, start + num_frames - 1, num_frames, dtype=int)
        # Add small jitter
        jitter = np.random.randint(-self.max_skip, self.max_skip + 1, num_frames)
        indices = np.clip(indices + jitter, 0, len(frames) - 1)
        indices = np.unique(indices)
        
        # Ensure we have enough frames
        while len(indices) < num_frames:
            idx = np.random.randint(0, len(frames))
            if idx not in indices:
                indices = np.append(indices, idx)
        
        return [frames[i] for i in sorted(indices)[:num_frames]]


def create_uavff3d_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create train and val dataloaders from config."""
    train_dataset = UAVFF3DDataset(**config['data']['train'])
    val_dataset = UAVFF3DDataset(**config['data']['val'])
    
    train_loader = train_dataset.get_loader(
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4)
    )
    val_loader = val_dataset.get_loader(
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4),
        shuffle=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_frames", type=int, default=8)
    args = parser.parse_args()
    
    dataset = UAVFF3DDataset(args.data_root, args.split, num_frames=args.num_frames)
    sample = dataset[0]
    print(f"Images: {sample['images'].shape}")
    print(f"Poses: {sample['extrinsics'].shape}")
    print(f"Depths: {sample['depths'].shape}")
    print(f"Masks: {sample['point_masks'].shape}")