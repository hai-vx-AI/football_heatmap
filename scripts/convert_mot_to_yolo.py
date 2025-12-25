"""
Convert SoccerNet MOT format to YOLO format
MOT format: frame_id, track_id, x, y, width, height, conf, -1, -1, -1
YOLO format: class_id, x_center_norm, y_center_norm, width_norm, height_norm
"""
import os
import glob
from tqdm import tqdm

# Configuration
TRACKING_DIR = "input/SoccerNet/tracking-2023"
OUTPUT_DIR = "input/SoccerNet/yolo_format"
SPLITS = ["train", "test"]

# Class mapping: 0 = player, 1 = ball (you may need to adjust based on your needs)
# For now, we'll treat all objects as class 0 (player)
# You'll need to identify ball separately if needed

def convert_mot_to_yolo():
    for split in SPLITS:
        split_path = os.path.join(TRACKING_DIR, split)
        if not os.path.exists(split_path):
            print(f"Split {split} not found, skipping...")
            continue
            
        # Get all sequence directories
        sequences = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        
        print(f"\nProcessing {split} split with {len(sequences)} sequences...")
        
        for seq in tqdm(sequences):
            seq_dir = os.path.join(split_path, seq)
            gt_file = os.path.join(seq_dir, "gt", "gt.txt")
            img_dir = os.path.join(seq_dir, "img1")
            seqinfo_file = os.path.join(seq_dir, "seqinfo.ini")
            
            if not os.path.exists(gt_file) or not os.path.exists(seqinfo_file):
                continue
            
            # Read sequence info
            img_width, img_height = 1920, 1080  # Default
            with open(seqinfo_file, 'r') as f:
                for line in f:
                    if 'imWidth' in line:
                        img_width = int(line.split('=')[1].strip())
                    elif 'imHeight' in line:
                        img_height = int(line.split('=')[1].strip())
            
            # Read annotations
            annotations = {}
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    frame_id = int(parts[0])
                    x, y, w, h = map(float, parts[2:6])
                    
                    # Convert to YOLO format
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    # Assuming all objects are class 0 for now
                    class_id = 0
                    
                    if frame_id not in annotations:
                        annotations[frame_id] = []
                    annotations[frame_id].append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            # Create output directories
            out_img_dir = os.path.join(OUTPUT_DIR, "images", split, seq)
            out_label_dir = os.path.join(OUTPUT_DIR, "labels", split, seq)
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_label_dir, exist_ok=True)
            
            # Copy images and create label files
            for frame_id, labels in annotations.items():
                img_name = f"{frame_id:06d}.jpg"
                src_img = os.path.join(img_dir, img_name)
                dst_img = os.path.join(out_img_dir, img_name)
                
                if os.path.exists(src_img):
                    # Copy image (or create symlink to save space)
                    import shutil
                    if not os.path.exists(dst_img):
                        shutil.copy2(src_img, dst_img)
                    
                    # Write label file
                    label_file = os.path.join(out_label_dir, img_name.replace('.jpg', '.txt'))
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(labels)) 
    
    print("\nConversion complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Update data.yaml to point to this directory")
    print("2. Update prepare_crop_data.py to use the new path")

if __name__ == "__main__":
    convert_mot_to_yolo()
