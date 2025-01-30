#!/usr/bin/env python3

import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
from pathlib import Path

from .raft_stereo import RAFTStereo
from .utils.utils import InputPadder

DEVICE = 'cuda'

class StereoProcessor:
    def __init__(self, model_path, output_dir="output"):
        self.device = DEVICE
        self.model = self.load_model(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_model(self, path):
        """Load RAFT-Stereo model with default arguments"""
        args = self.get_default_args()
        model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        model.load_state_dict(torch.load(path))
        model = model.module
        model.to(self.device)
        model.eval()
        return model

    def process_stereo_pair(self, left_img_path, right_img_path, output_prefix=None):
        """Process a stereo pair and generate disparity, depth and point cloud"""
        # Load images
        image1 = self.load_image(left_img_path)
        image2 = self.load_image(right_img_path)

        # Get disparity
        with torch.no_grad():
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            _, flow_up = self.model(image1, image2, iters=32, test_mode=True)
            disparity = padder.unpad(flow_up).squeeze()

        # Convert to depth
        depth = self.disparity_to_depth(disparity)

        # Generate point cloud
        pcd = self.generate_pointcloud(depth, Image.open(left_img_path))

        # Save outputs
        if output_prefix:
            self.save_outputs(output_prefix, disparity, depth, pcd)

        return disparity, depth, pcd

    @staticmethod
    def load_image(path):
        """Load and preprocess image"""
        img = np.array(Image.open(path)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    def disparity_to_depth(self, disparity, baseline=0.1, focal_length=1000):
        """Convert disparity to depth"""
        depth = (baseline * focal_length) / (disparity.cpu().numpy() + 1e-6)
        return depth

    def generate_pointcloud(self, depth, color_img):
        """Generate colored point cloud"""
        h, w = depth.shape
        y, x = np.mgrid[0:h, 0:w]
        
        # Create point cloud
        points = np.stack([x, y, depth], axis=-1)
        colors = np.array(color_img).reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

    def save_outputs(self, prefix, disparity, depth, pcd):
        """Save all outputs"""
        # Save disparity
        plt.imsave(self.output_dir / f"{prefix}_disparity.png", 
                  -disparity.cpu().numpy(), cmap='jet')
        
        # Save depth
        plt.imsave(self.output_dir / f"{prefix}_depth.png",
                  depth, cmap='viridis')
        
        # Save point cloud
        o3d.io.write_point_cloud(
            str(self.output_dir / f"{prefix}_cloud.ply"), pcd)

    @staticmethod
    def get_default_args():
        """Get default model arguments"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--mixed_precision', action='store_true')
        parser.add_argument('--valid_iters', type=int, default=32)
        parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
        parser.add_argument('--corr_implementation', default="reg")
        parser.add_argument('--shared_backbone', action='store_true')
        parser.add_argument('--corr_levels', type=int, default=4)
        parser.add_argument('--corr_radius', type=int, default=4)
        parser.add_argument('--n_downsample', type=int, default=2)
        parser.add_argument('--context_norm', type=str, default="batch")
        parser.add_argument('--slow_fast_gru', action='store_true')
        parser.add_argument('--n_gru_layers', type=int, default=3)
        return parser.parse_args([])
