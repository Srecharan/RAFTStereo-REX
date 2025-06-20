# RAFT-Stereo-TREX

An advanced deep learning-based stereo matching system for high-precision depth estimation and 3D scene reconstruction, built on the RAFT-Stereo architecture.

## Architecture Overview
<img src="RAFTStereo.png">

```
[Stereo Pair] ─────┐
                   v
[Feature Extraction (Shared CNN Backbone)]
                   │
                   v
[4D Correlation Volume Computation]
                   │
                   v
[Recurrent GRU Refinement] ───┬──> [Disparity Maps]
                             ├──> [Depth Maps]
                             └──> [3D Point Clouds]
```

### Input: High-Resolution Stereo Image Pairs
<div style="display: flex; justify-content: space-between; align-items: center;">
  <img src="assets/left_rect1.png" width="49%" alt="Left Image"/>
  <img src="assets/right_rect1.png" width="49%" alt="Right Image"/>
</div>
<p align="center"><i>Rectified Stereo Image Pair</i></p>

### Dense Reconstruction Results

#### 1. Depth Maps
<div align="center">
  <img src="assets/depth1.png" width="800" />
  <p align="center"><i>High-Resolution Depth Map (blue = near-field, red = far-field)</i></p>
</div>

#### 2. Point Cloud Visualization
<div align="center">
  <img src="assets/point_cloud_vis.gif" width="800" />
  <p align="center"><i>Dense 3D Point Cloud Reconstruction</i></p>
</div>

## Note on System Integration
This system represents the depth perception pipeline for the REX platform, integrating three key components:

1. **RAFT-Stereo Node (This Repository)**
   - High-precision depth estimation
   - Dense 3D reconstruction
   - Sub-pixel disparity accuracy

2. **LeafGrasp-Vision-ML** ([LeafGrasp-Vision-ML](https://github.com/Srecharan/Leaf-Grasping-Vision-ML.git))
   - Depth-aware grasp point selection
   - 3D point cloud processing
   - Geometric feature extraction

3. **REX Robot Integration** ([REX-Robot](https://github.com/Srecharan/REX-Robot.git))
   - Stereo camera system
   - Calibrated image streams
   - Real-time data acquisition

Each component has its dedicated repository for detailed implementation. This repository focuses on high-precision depth estimation and 3D reconstruction for robotic manipulation.

## Technical Implementation

### 1. Deep Learning Architecture
- **Feature Extraction**:
  - Shared CNN backbone (ResNet-based)
  - Multi-scale feature pyramid (4 levels)
  - Instance normalization for robust feature encoding

- **Correlation Volume**:
  - 4D correlation computation with CUDA optimization
  - Multi-level correlation pyramid for hierarchical matching
  - Memory-efficient implementation for high-resolution processing

- **Iterative Refinement**:
  - Recurrent GRU updates (3-layer architecture)
  - Sub-pixel accuracy through soft argmax
  - Adaptive update steps based on scene complexity

### 2. Training & Evaluation

#### Datasets
- **SceneFlow**: Primary training dataset (FlyingThings3D, Driving, Monkaa)
- **KITTI**: Outdoor driving scenes
- **ETH3D**: Indoor/outdoor scenes
- **Middlebury**: High-resolution stereo pairs

#### Training Configuration
```bash
python train_stereo.py \
    --batch_size 6 \
    --train_iters 16 \
    --valid_iters 32 \
    --spatial_scale -0.2 0.4 \
    --n_downsample 2 \
    --mixed_precision \
    --num_steps 100000
```

#### Training Details
- **Optimizer**: AdamW (lr=0.0002, weight_decay=1e-5)
- **Learning Rate**: OneCycleLR scheduler
- **Loss Function**: Multi-scale sequence loss (gamma=0.9)
- **Validation**: Every 10000 steps
- **GPU Memory Optimization**: 
  - Mixed precision training
  - Gradient clipping at 1.0
  - Optional n_downsample=3 for reduced memory usage

### 3. Depth Computation Pipeline
- **Post-Processing**:
  ```python
  depth = (baseline * focal_length) / disparity
  ```
- **Precision Enhancements**:
  - Sub-pixel interpolation
  - Boundary refinement
  - Statistical outlier filtering

## Performance Optimizations

### CUDA Kernel Acceleration
Custom CUDA kernels for 3D point cloud generation provide significant speedup over CPU implementation:

- **Implementation**: `src/raft_stereo_leaf/cuda_kernels.py`
- **Features**: Parallel processing of 1.5M pixels, memory coalescing optimization
- **Usage**:
```python
pcl, rgb = generate_pointcloud_optimized(
    depth_map=depth, rgb_image=image,
    fx=focal_x, fy=focal_y, cx=center_x, cy=center_y
)
```

### TensorRT Model Optimization
TensorRT compilation optimizes model inference performance:

- **Implementation**: `src/raft_stereo_leaf/tensorrt_optimization.py`
- **Features**: FP16 precision, operator fusion, graph optimization
- **Usage**:
```python
optimized_model = optimize_raft_stereo_pipeline(
    model=raft_stereo_model, enable_tensorrt=True
)
```

### Optimization Features
- **Automatic Fallback**: CPU/PyTorch fallback when optimizations unavailable
- **Memory Management**: Efficient GPU memory usage and cleanup
- **Production Ready**: Robust error handling and caching

## Performance Metrics

1. **Accuracy**:
   - Sub-pixel disparity accuracy < 0.5px
   - Depth error < 1% at 10m range
   - Point cloud density > 90% on textured regions

2. **Processing Speed** (with NVIDIA RTX 3080):
   - 1080p stereo pair: ~100ms
   - Point cloud generation: optimized with CUDA kernels
   - Total pipeline latency: <200ms (real-time capable)

## Prerequisites

- CUDA-capable GPU (NVIDIA GTX 1080 or better)
- PyTorch 1.7+
- CUDA 10.2+ and cuDNN
- Python 3.7+
- **For Optimizations**:
  - PyCUDA 2021.1+
  - TensorRT 8.0+
  - torch-tensorrt 1.1.0+

## Quick Start

1. **Installation**:
```bash
git clone https://github.com/Srecharan/RAFTStereo-TREX.git
cd RAFTStereo-TREX
conda env create -f environment.yaml
conda activate raftstereo

# Install optimization dependencies (optional)
pip install pycuda torch-tensorrt tensorrt
```

2. **Process Example Data**:
```bash
python scripts/process_static_stereo.py
```

3. **Test Optimizations** (optional):
```bash
python scripts/test_optimizations.py
```

## Output Formats

- **Raw Data**:
  - `depth*.npy`: Dense depth maps (float32)
  - `disparity*.npy`: Disparity maps
  - `temp_*.pcd`: Organized point clouds

- **Visualizations**:
  - `depth*.png`: Colored depth visualizations
  - `disparity*.png`: Disparity heat maps
  - Point cloud renderings (interactive)

## Citation

If you use this implementation, please cite:
```
@inproceedings{lipson2021raft,
  title={RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching},
  author={Lipson, Lahav and Teed, Zachary and Deng, Jia},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2021}
}
```
<img src="assets/RAFTStereo.png">

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Based on the [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) implementation by Princeton Vision Lab.
