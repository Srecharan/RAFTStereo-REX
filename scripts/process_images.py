import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.raft_stereo_leaf import StereoProcessor

def main():
    # Initialize processor
    model_path = "models/raftstereo-eth3d.pth"
    processor = StereoProcessor(model_path)
    
    # Process examples
    example_pairs = [
        ("plants", "examples/images/stereo_pairs/plants/left.png", 
                  "examples/images/stereo_pairs/plants/right.png"),
        ("indoor", "examples/images/stereo_pairs/indoor/im0.png",
                  "examples/images/stereo_pairs/indoor/im1.png"),
        ("objects", "examples/images/stereo_pairs/objects/im0.png",
                   "examples/images/stereo_pairs/objects/im1.png")
    ]

    for name, left, right in example_pairs:
        print(f"Processing {name} scene...")
        disparity, depth, pcd = processor.process_stereo_pair(
            left, right, output_prefix=name)

if __name__ == "__main__":
    main()
