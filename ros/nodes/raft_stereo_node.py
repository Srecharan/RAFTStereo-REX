#!/usr/bin/env python3

# RAFTSTERO NODE

import sys
import os

HOME_DIR = os.path.expanduser('~')
sys.path.append(os.path.join(HOME_DIR, 'RAFT-Stereo/sampler'))

import rospy
import argparse
import numpy as np
import torch
import corr_sampler
from datetime import datetime

DEVICE = 'cuda'
print(os.getcwd())
print(HOME_DIR)

os.chdir(HOME_DIR + '/RAFT-Stereo')
sys.path.append(HOME_DIR + '/RAFT-Stereo')
sys.path.append('core')

from tqdm import tqdm
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
import open3d as o3d
import time
import cv2 as cv
from raftstereo.msg import depth
from raft_stereo import RAFTStereo
import gc

class StereoSync:
    def __init__(self):
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(HOME_DIR, '/home/buggspray/ros/catkin_ws/src/raft_stereo_leaf/assets')
        os.makedirs(self.output_dir, exist_ok=True)

        # Memory management parameters
        self.max_image_dim = 832  # Max dimension to process (multiple of 32 for RAFT)
        self.clear_gpu_cache()
        
        # ROS subscribers
        self.image_sub1 = Subscriber('/theia/left/image_rect_color', Image, queue_size=5)
        self.image_sub2 = Subscriber('/theia/right/image_rect_color', Image, queue_size=5)
        self.image_sub3 = rospy.Subscriber("theia/right/camera_info", CameraInfo, self.cam_info_, queue_size=5)
        self.subscriber_count = 0
        self.pub = rospy.Publisher('/depth_image', depth, queue_size=2)

        # Set up stereo synchronization
        self.stereo_ = ApproximateTimeSynchronizer([self.image_sub1, self.image_sub2], queue_size=2, slop=0.1)
        self.stereo_.registerCallback(self.stereo_callback)
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        
        # Initialize model and counter
        self.args, self.model = self.init_()
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            # Disable automatic mixed precision for eval
            torch.cuda.amp.autocast(enabled=False)
        self.count = 0
        rospy.set_param('/raft_iteration', self.count)
        
        # GPU optimization flags
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            
        # Save output flag
        self.save_outputs = True  # Set to False to disable saving intermediate results

    def clear_gpu_cache(self):
        """Clear GPU cache and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def init_(self):
        """Initialize RAFT model and parameters"""
        rospy.set_param('raft_done', False)
        rospy.set_param('SDF_sub', False)

        parser = argparse.ArgumentParser()
        parser.add_argument('--restore_ckpt', help="restore checkpoint",
                          default=HOME_DIR + '/RAFT-Stereo/models/raftstereo-eth3d.pth')
        parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
        parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--valid_iters', type=int, default=32,
                          help='number of flow-field updates during forward pass')

        # Architecture choices
        parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                          help="hidden state and context dimensions")
        parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], 
                          default="reg_cuda", help="correlation volume implementation")
        parser.add_argument('--shared_backbone', action='store_true',
                          help="use a single backbone for the context and feature encoders")
        parser.add_argument('--corr_levels', type=int, default=4, 
                          help="number of levels in the correlation pyramid")
        parser.add_argument('--corr_radius', type=int, default=4, 
                          help="width of the correlation pyramid")
        parser.add_argument('--n_downsample', type=int, default=2, 
                          help="resolution of the disparity field (1/2^K)")
        parser.add_argument('--context_norm', type=str, default="batch", 
                          choices=['group', 'batch', 'instance', 'none'],
                          help="normalization of context encoder")
        parser.add_argument('--slow_fast_gru', action='store_true', 
                          help="iterate the low-res GRUs more frequently")
        parser.add_argument('--n_gru_layers', type=int, default=3, 
                          help="number of hidden GRU levels")

        args = parser.parse_args()
        
        # Initialize model ensuring full precision
        model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        
        # Load weights and ensure they're float32
        state_dict = torch.load(args.restore_ckpt, map_location=torch.device('cpu'))
        # Convert all model parameters to float32
        for k, v in state_dict.items():
            state_dict[k] = v.float()
            
        model.load_state_dict(state_dict)
        model = model.module
        model.to(DEVICE)
        model.eval()
        
        # Ensure all model parameters are float32
        for param in model.parameters():
            param.data = param.data.float()
        
        return args, model

    def resize_if_needed(self, img):
        """Resize image if larger than max dimensions while maintaining aspect ratio"""
        h, w = img.shape[:2]
        scale = min(self.max_image_dim / max(h, w), 1.0)
        
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
        
        return img, scale

    def load_image(self, imfile):
        """Load and preprocess image ensuring full precision"""
        img = np.array(imfile).astype(np.uint8)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return img_tensor[None].to(DEVICE, non_blocking=True, dtype=torch.float32)

    def stereo_callback(self, imageL, imageR):
        print('stereo pair received ...')
        
        try:
            with torch.cuda.device(0):
                self.clear_gpu_cache()
                
                # Get image parameters and calibration
                n_channels = 3 if imageL.encoding == 'rgb8' else 1
                
                # Get original image dimensions
                original_height = imageL.height
                original_width = imageL.width
                
                # Get camera calibration
                calibP = np.array(rospy.get_param("/theia/right/projection_matrix"))
                calibP = np.reshape(calibP, (3, 4))
                
                if not np.any(calibP):
                    print('Waiting for projection matrix...')
                    return

                # Calculate camera parameters
                f_norm = calibP[0, 0]
                baseline = -1 * calibP[0, 3] / calibP[0, 0]
                Cx = calibP[0, 2]
                Cy = calibP[1, 2]

                with torch.no_grad():
                    # Convert ROS images to numpy arrays
                    imgL_orig = np.ndarray(shape=(original_height, original_width, n_channels), 
                                        dtype=np.uint8, buffer=imageL.data)
                    imgR_orig = np.ndarray(shape=(original_height, original_width, n_channels), 
                                        dtype=np.uint8, buffer=imageR.data)

                    # Save original images if needed
                    if self.save_outputs:
                        plt.imsave(os.path.join(self.output_dir, f"left_rect{self.count}.png"), imgL_orig)
                        plt.imsave(os.path.join(self.output_dir, f"right_rect{self.count}.png"), imgR_orig)

                    # Resize images for RAFT processing
                    imgL, scale_L = self.resize_if_needed(imgL_orig)
                    imgR, scale_R = self.resize_if_needed(imgR_orig)

                    # Process images through RAFT
                    image1 = self.load_image(imgL)
                    image2 = self.load_image(imgR)
                    padder = InputPadder(image1.shape, divis_by=32)
                    image1, image2 = padder.pad(image1, image2)
                    
                    _, flow_up = self.model(image1, image2, iters=self.args.valid_iters, test_mode=True)
                    
                    # Clean up intermediate tensors
                    del image1, image2
                    self.clear_gpu_cache()

                    # Post-process disparity
                    flow_up = padder.unpad(flow_up).squeeze()
                    disparity = flow_up.cpu().numpy().squeeze()

                    # Scale disparity back to original resolution
                    if scale_L < 1.0:
                        disparity = cv.resize(disparity, (original_width, original_height), 
                                        interpolation=cv.INTER_LINEAR)
                        disparity *= 1.0 / scale_L

                    # Calculate depth
                    z = (f_norm * baseline) / -disparity

                    # Generate point cloud using original resolution images
                    pcl = []
                    rgb = []
                    for i in range(original_height):
                        for j in range(original_width):
                            if z[i][j] <= 0.5:
                                x_ = (z[i][j] / f_norm) * (j - Cx)
                                y_ = (z[i][j] / f_norm) * (i - Cy)
                                r_ = imgL_orig[i, j, 0] / 255
                                g_ = imgL_orig[i, j, 1] / 255
                                b_ = imgL_orig[i, j, 2] / 255
                                pcl.append([x_, y_, z[i][j]])
                                rgb.append([r_, g_, b_])

                    # Save results if enabled
                    if self.save_outputs:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pcl)
                        pcd.colors = o3d.utility.Vector3dVector(rgb)
                        o3d.io.write_point_cloud(os.path.join(self.output_dir, 
                                                            f"temp_{self.count}.pcd"), pcd)
                        plt.imsave(os.path.join(self.output_dir, f"pcd{self.count}.png"), 
                                 -disparity, cmap='jet')
                        plt.imsave(os.path.join(self.output_dir, f"depth{self.count}.png"), 
                                 z, cmap='jet')
                        np.save(os.path.join(self.output_dir, f"depth{self.count}.npy"), z)
                        np.save(os.path.join(self.output_dir, f"disparity{self.count}.npy"), 
                               disparity)

                    # Update counters and parameters
                    if self.subscriber_count >= 5:
                        self.subscriber_count = 0
                    print('---------------{}-----------------'.format(self.subscriber_count))
                    self.subscriber_count += 1
                    print(f"Current Image Count: {self.count}")
                    self.count += 1
                    rospy.set_param('raft_done', True)

                    # Publish depth message
                    depth_msg = depth()
                    depth_msg.imageData = z.astype('float32').ravel()
                    self.pub.publish(depth_msg)
                    print('Published depth message')

        except RuntimeError as e:
            if "Input type" in str(e):
                rospy.logerr("Tensor type mismatch. Attempting to recover...")
                self.clear_gpu_cache()
                # Reinitialize model with correct precision
                self.args, self.model = self.init_()
            elif "out of memory" in str(e):
                rospy.logerr("GPU OOM error. Attempting to recover...")
                self.clear_gpu_cache()
            else:
                rospy.logerr(f"Error in stereo callback: {str(e)}")
                raise e
        finally:
            self.clear_gpu_cache()
            
    def cam_info_(self, cam_info):
        """Camera info callback"""
        rospy.set_param('/theia/right/projection_matrix', cam_info.P)

def init():
    b = StereoSync()
    rospy.init_node('raft_image_sync', anonymous=False)
    rospy.spin()

if __name__ == '__main__':
    init()