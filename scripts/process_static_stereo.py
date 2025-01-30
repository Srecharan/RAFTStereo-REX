#!/usr/bin/env python3

import sys
import os
import numpy as np
import torch
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
from datetime import datetime
import argparse
import gc

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

from src.raft_stereo_leaf.utils.utils import InputPadder
from src.raft_stereo_leaf.raft_stereo import RAFTStereo

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class StaticStereoProcessor:
    def __init__(self, model_path=None):
        """Initialize the stereo processor"""
        self.output_dir = os.path.join(parent_dir, 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_image_dim = 832  # Max dimension to process
        self.count = 0
        self.args, self.model = self.init_(model_path)
        
    def clear_gpu_cache(self):
        """Clear GPU cache and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def read_calibration(self, calib_path):
        """Read camera calibration file if it exists"""
        try:
            if calib_path and os.path.exists(calib_path):
                with open(calib_path, 'r') as f:
                    content = f.read().strip()
                    # Remove brackets and split by semicolon
                    rows = content.replace('[', '').replace(']', '').split(';')
                    # Parse each row
                    matrix = []
                    for row in rows:
                        # Convert row string to numbers
                        row_values = [float(x) for x in row.strip().split()]
                        matrix.append(row_values)
                    
                    # Convert to numpy array
                    K = np.array(matrix)
                    
                    return {
                        'focal_length': float(K[0,0]),  # Focal length in x
                        'cx': float(K[0,2]),           # Principal point x
                        'cy': float(K[1,2]),           # Principal point y
                        'baseline': 0.1                # Default baseline
                    }
        except Exception as e:
            print(f"Warning: Could not read calibration file: {e}")
        
        # Return default values if calibration file can't be read
        return {
            'focal_length': 3250.816,
            'baseline': 0.1,
            'cx': 398.69,
            'cy': 480.329
        }

    def init_(self, model_path=None):
        """Initialize RAFT model and parameters"""
        parser = argparse.ArgumentParser()
        if model_path is None:
            model_path = os.path.join(parent_dir, 'models/raftstereo-eth3d.pth')
        
        parser.add_argument('--restore_ckpt', help="restore checkpoint",
                          default=model_path)
        parser.add_argument('--mixed_precision', action='store_true')
        parser.add_argument('--valid_iters', type=int, default=32)
        parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3)
        parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], 
                          default="reg", help="correlation volume implementation")
        parser.add_argument('--shared_backbone', action='store_true')
        parser.add_argument('--corr_levels', type=int, default=4)
        parser.add_argument('--corr_radius', type=int, default=4)
        parser.add_argument('--n_downsample', type=int, default=2)
        parser.add_argument('--context_norm', type=str, default="batch")
        parser.add_argument('--slow_fast_gru', action='store_true')
        parser.add_argument('--n_gru_layers', type=int, default=3)

        args = parser.parse_args([])
        
        # Initialize model
        model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        
        # Load weights
        print(f"Loading model from: {args.restore_ckpt}")
        state_dict = torch.load(args.restore_ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model = model.module
        model.to(DEVICE)
        model.eval()
        
        return args, model

    def resize_if_needed(self, img):
        """Resize image if larger than max dimensions while maintaining aspect ratio"""
        h, w = img.shape[:2]
        scale = min(self.max_image_dim / max(h, w), 1.0)
        
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
        
        return img, scale

    def load_image(self, img):
        """Load and preprocess image ensuring full precision"""
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE, dtype=torch.float32)

    def process_stereo_pair(self, left_path, right_path, focal_length=3250.816, baseline=0.1, cx=398.69, cy=480.329):
        """Process a single stereo pair"""
        try:
            print(f"\nProcessing stereo pair:")
            print(f"Left image: {left_path}")
            print(f"Right image: {right_path}")
            
            # Load images
            imgL_orig = cv.imread(left_path)
            imgR_orig = cv.imread(right_path)
            
            if imgL_orig is None or imgR_orig is None:
                raise ValueError(f"Could not load images from {left_path} or {right_path}")

            # Get dimensions
            original_height, original_width = imgL_orig.shape[:2]
            print(f"Image dimensions: {original_width}x{original_height}")

            # Resize if needed
            imgL, scale_L = self.resize_if_needed(imgL_orig)
            imgR, scale_R = self.resize_if_needed(imgR_orig)

            # Process through RAFT
            with torch.no_grad():
                self.clear_gpu_cache()
                
                image1 = self.load_image(imgL)
                image2 = self.load_image(imgR)
                padder = InputPadder(image1.shape, divis_by=32)
                image1, image2 = padder.pad(image1, image2)
                
                _, flow_up = self.model(image1, image2, iters=32, test_mode=True)
                disparity = padder.unpad(flow_up).squeeze().cpu().numpy()

                # Clear GPU memory
                del image1, image2, flow_up
                self.clear_gpu_cache()

                # Scale disparity back to original resolution
                if scale_L < 1.0:
                    disparity = cv.resize(disparity, (original_width, original_height), 
                                    interpolation=cv.INTER_LINEAR)
                    disparity *= 1.0 / scale_L

                # Calculate depth
                z = (focal_length * baseline) / -disparity

                # Generate point cloud
                pcl = []
                rgb = []
                for i in range(original_height):
                    for j in range(original_width):
                        # Adjust depth threshold to be more permissive
                        if 0.1 < z[i][j] < 10.0:  # Accept points between 0.1m and 10m
                            x_ = (z[i][j] / focal_length) * (j - cx)
                            y_ = (z[i][j] / focal_length) * (i - cy)
                            r_ = imgL_orig[i, j, 2] / 255  # OpenCV uses BGR
                            g_ = imgL_orig[i, j, 1] / 255
                            b_ = imgL_orig[i, j, 0] / 255
                            pcl.append([x_, y_, z[i][j]])
                            rgb.append([r_, g_, b_])

                # Create point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcl)
                pcd.colors = o3d.utility.Vector3dVector(rgb)

                # Optional: Filter outliers
                if len(pcl) > 0:
                    pcd, _ = pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)

                # Save all outputs
                print("\nSaving outputs...")
                # Save as PCD
                output_prefix = os.path.splitext(os.path.basename(left_path))[0]
                o3d.io.write_point_cloud(
                    os.path.join(self.output_dir, f"{output_prefix}_cloud.pcd"), pcd)
                
                # Save visualizations and numpy arrays
                plt.imsave(os.path.join(self.output_dir, f"{output_prefix}_disparity.png"), 
                          -disparity, cmap='jet')
                plt.imsave(os.path.join(self.output_dir, f"{output_prefix}_depth.png"), 
                          z, cmap='jet')
                np.save(os.path.join(self.output_dir, f"{output_prefix}_depth.npy"), z)
                np.save(os.path.join(self.output_dir, f"{output_prefix}_disparity.npy"), 
                       disparity)

                print(f"Successfully processed {output_prefix}")
                print(f"Generated point cloud with {len(pcl)} points")
                print(f"Files saved in: {self.output_dir}")
                
                return disparity, z, pcd

        except Exception as e:
            print(f"Error processing stereo pair: {e}")
            raise

def main():
    try:
        processor = StaticStereoProcessor()
        
        # Process example pairs
        example_pairs = [
            ("plants", 
             os.path.join(parent_dir, "examples/images/stereo_pairs/plants/left.png"),
             os.path.join(parent_dir, "examples/images/stereo_pairs/plants/right.png"),
             os.path.join(parent_dir, "examples/images/stereo_pairs/plants/calib.txt")),
            ("indoor", 
             os.path.join(parent_dir, "examples/images/stereo_pairs/indoor/im0.png"),
             os.path.join(parent_dir, "examples/images/stereo_pairs/indoor/im1.png"),
             None),
            ("objects", 
             os.path.join(parent_dir, "examples/images/stereo_pairs/objects/im0.png"),
             os.path.join(parent_dir, "examples/images/stereo_pairs/objects/im1.png"),
             None)
        ]

        for name, left, right, calib in example_pairs:
            print(f"\nProcessing {name} scene...")
            # Read calibration if available
            params = processor.read_calibration(calib) if calib else {}
            processor.process_stereo_pair(left, right, **params)

    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()