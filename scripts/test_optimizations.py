#!/usr/bin/env python3

"""
Basic functionality test for RAFT-Stereo optimizations.

This script tests the optimization modules can be imported and initialized
without requiring full CUDA/TensorRT setup, useful for CI/CD or development.
"""

import sys
import os
import numpy as np

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_cuda_kernels_import():
    """Test CUDA kernels module can be imported."""
    print("Testing CUDA kernels import...")
    try:
        from src.raft_stereo_leaf.cuda_kernels import CUDAPointCloudGenerator, generate_pointcloud_optimized
        print("✅ CUDA kernels module imported successfully")
        
        # Test initialization (should work even without CUDA)
        generator = CUDAPointCloudGenerator()
        print(f"✅ CUDA generator initialized (CUDA available: {generator.cuda_initialized})")
        
        return True
    except Exception as e:
        print(f"❌ CUDA kernels import failed: {e}")
        return False

def test_tensorrt_optimization_import():
    """Test TensorRT optimization module can be imported."""
    print("\nTesting TensorRT optimization import...")
    try:
        from src.raft_stereo_leaf.tensorrt_optimization import TensorRTOptimizer, optimize_raft_stereo_pipeline
        print("✅ TensorRT optimization module imported successfully")
        
        # Test initialization (should work even without TensorRT)
        optimizer = TensorRTOptimizer()
        print(f"✅ TensorRT optimizer initialized (TensorRT available: {optimizer.tensorrt_available})")
        
        return True
    except Exception as e:
        print(f"❌ TensorRT optimization import failed: {e}")
        return False

def test_cpu_fallback_functionality():
    """Test CPU fallback implementations work."""
    print("\nTesting CPU fallback functionality...")
    try:
        from src.raft_stereo_leaf.cuda_kernels import CUDAPointCloudGenerator
        
        # Create sample data
        height, width = 100, 150  # Small for quick testing
        depth_map = np.random.uniform(0.1, 2.0, (height, width)).astype(np.float32)
        rgb_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Test CPU fallback
        generator = CUDAPointCloudGenerator()
        points, colors = generator._generate_pointcloud_cpu_fallback(
            depth_map, rgb_image, fx=800.0, fy=800.0, cx=75.0, cy=50.0, depth_threshold=1.5
        )
        
        print(f"✅ CPU fallback generated {len(points)} points and {len(colors)} colors")
        
        # Validate output format
        assert isinstance(points, np.ndarray), "Points should be numpy array"
        assert isinstance(colors, np.ndarray), "Colors should be numpy array"
        assert points.shape[1] == 3, "Points should have 3 coordinates"
        assert colors.shape[1] == 3, "Colors should have 3 channels"
        assert len(points) == len(colors), "Points and colors should have same length"
        
        print("✅ CPU fallback output validation passed")
        return True
        
    except Exception as e:
        print(f"❌ CPU fallback test failed: {e}")
        return False

def test_module_integration():
    """Test that modules can be used together."""
    print("\nTesting module integration...")
    try:
        from src.raft_stereo_leaf.cuda_kernels import generate_pointcloud_optimized
        from src.raft_stereo_leaf.tensorrt_optimization import optimize_raft_stereo_pipeline
        
        # Test that functions exist and are callable
        assert callable(generate_pointcloud_optimized), "generate_pointcloud_optimized should be callable"
        assert callable(optimize_raft_stereo_pipeline), "optimize_raft_stereo_pipeline should be callable"
        
        print("✅ Module integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Module integration test failed: {e}")
        return False

def test_ros_node_imports():
    """Test that ROS node can import optimization modules."""
    print("\nTesting ROS node integration...")
    try:
        # Simulate the imports from raft_stereo_node.py
        import sys
        old_path = sys.path.copy()
        
        # Test the import pattern used in the ROS node
        from src.raft_stereo_leaf.cuda_kernels import generate_pointcloud_optimized
        from src.raft_stereo_leaf.tensorrt_optimization import optimize_raft_stereo_pipeline
        
        print("✅ ROS node imports work correctly")
        
        # Test that OPTIMIZATIONS_AVAILABLE flag would work
        OPTIMIZATIONS_AVAILABLE = True  # Since we can import
        print(f"✅ OPTIMIZATIONS_AVAILABLE would be: {OPTIMIZATIONS_AVAILABLE}")
        
        sys.path = old_path
        return True
        
    except Exception as e:
        print(f"❌ ROS node integration test failed: {e}")
        return False

def main():
    """Run all tests and report results."""
    print("RAFT-STEREO OPTIMIZATION MODULE TESTS")
    print("=" * 50)
    print("Testing basic functionality without requiring CUDA/TensorRT hardware")
    
    tests = [
        test_cuda_kernels_import,
        test_tensorrt_optimization_import,
        test_cpu_fallback_functionality,
        test_module_integration,
        test_ros_node_imports
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! Optimization modules are working correctly.")
        exit_code = 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        exit_code = 1
    
    print("\nNote: Full performance benefits require CUDA/TensorRT hardware.")
    print("These tests only verify the modules can be imported and CPU fallbacks work.")
    
    return exit_code

if __name__ == '__main__':
    exit(main()) 