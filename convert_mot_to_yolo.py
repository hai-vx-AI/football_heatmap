"""
Convert SoccerNet MOT format to YOLO format with proper class labeling
Uses gameinfo.ini to identify object classes (ball, player, goalkeeper, referee)

Classes:
- 0: ball
- 1: player  
- 2: goalkeeper
- 3: referee
"""
import os
import glob
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import configparser

# Configuration
TRACKING_DIR = "input/SoccerNet/tracking-2023"
OUTPUT_DIR = "input/SoccerNet/yolo_format"
SPLITS = ["train", "test"]
MAX_WORKERS = 4  # Parallel processing threads


def parse_gameinfo(gameinfo_file):
    """
    Parse gameinfo.ini to extract track ID to class mapping
    Returns: dict {track_id: class_id}
    """
    config = configparser.ConfigParser()
    config.read(gameinfo_file)
    
    track_to_class = {}
    
    for key in config['Sequence']:
        # ConfigParser converts keys to lowercase
        if key.startswith('trackletid_'):
            track_id = int(key.split('_')[1])
            value = config['Sequence'][key].strip()
            
            # Classify based on label
            if 'ball' in value.lower():
                track_to_class[track_id] = 0  # ball
            elif 'referee' in value.lower():
                track_to_class[track_id] = 3  # referee
            elif 'goalkeeper' in value.lower():
                track_to_class[track_id] = 2  # goalkeeper
            elif 'player' in value.lower():
                track_to_class[track_id] = 1  # player
    
    return track_to_class


def convert_sequence(args):
    """Convert single sequence (for parallel processing)"""
    split, seq, split_path = args
    
    seq_dir = os.path.join(split_path, seq)
    gt_file = os.path.join(seq_dir, "gt", "gt.txt")
    img_dir = os.path.join(seq_dir, "img1")
    seqinfo_file = os.path.join(seq_dir, "seqinfo.ini")
    gameinfo_file = os.path.join(seq_dir, "gameinfo.ini")
    
    # Validate files
    if not all(os.path.exists(f) for f in [gt_file, seqinfo_file, gameinfo_file]):
        return 0, 0
    
    # Read sequence dimensions
    config = configparser.ConfigParser()
    config.read(seqinfo_file)
    img_width = int(config['Sequence']['imWidth'])
    img_height = int(config['Sequence']['imHeight'])
    
    # Parse track ID to class mapping
    track_to_class = parse_gameinfo(gameinfo_file)
    
    # Read and convert annotations
    annotations = {}
    skipped = 0
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            
            # Skip if track_id not in mapping (shouldn't happen)
            if track_id not in track_to_class:
                skipped += 1
                continue
            
            # Convert to YOLO format (normalized)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            
            # Get class from mapping
            class_id = track_to_class[track_id]
            
            if frame_id not in annotations:
                annotations[frame_id] = []
            annotations[frame_id].append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )
    
    # Create output directories
    out_img_dir = os.path.join(OUTPUT_DIR, "images", split, seq)
    out_label_dir = os.path.join(OUTPUT_DIR, "labels", split, seq)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    
    # Write labels and copy images
    processed = 0
    for frame_id, labels in annotations.items():
        img_name = f"{frame_id:06d}.jpg"
        src_img = os.path.join(img_dir, img_name)
        dst_img = os.path.join(out_img_dir, img_name)
        
        if os.path.exists(src_img):
            # Copy image
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)
            
            # Write label
            label_file = os.path.join(out_label_dir, img_name.replace('.jpg', '.txt'))
            with open(label_file, 'w') as f:
                f.write('\n'.join(labels))
            
            processed += 1
    
    return processed, skipped


def convert_mot_to_yolo():
    """Main conversion function with parallel processing"""
    print("=" * 70)
    print("CONVERT SOCCERNET MOT → YOLO FORMAT (WITH PROPER CLASSES)")
    print("=" * 70)
    
    # Prepare tasks
    tasks = []
    for split in SPLITS:
        split_path = os.path.join(TRACKING_DIR, split)
        if not os.path.exists(split_path):
            print(f"⚠️ Split '{split}' not found, skipping...")
            continue
        
        sequences = [d for d in os.listdir(split_path) 
                     if os.path.isdir(os.path.join(split_path, d))]
        
        print(f"\n📁 Found {len(sequences)} sequences in {split}/")
        tasks.extend([(split, seq, split_path) for seq in sequences])
    
    # Process in parallel
    print(f"\n🚀 Processing {len(tasks)} sequences with {MAX_WORKERS} workers...")
    
    total_processed = 0
    total_skipped = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(convert_sequence, tasks),
            total=len(tasks),
            desc="Converting"
        ))
    
    for processed, skipped in results:
        total_processed += processed
        total_skipped += skipped
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"📊 Statistics:")
    print(f"  Frames processed: {total_processed:,}")
    print(f"  Annotations skipped: {total_skipped:,}")
    print(f"\n📁 Output: {OUTPUT_DIR}/")
    print(f"  - images/{split}/SNMOT-XXX/*.jpg")
    print(f"  - labels/{split}/SNMOT-XXX/*.txt")
    print("\n📋 Class mapping:")
    print("  0 = ball")
    print("  1 = player")
    print("  2 = goalkeeper")
    print("  3 = referee")
    print("=" * 70)


if __name__ == "__main__":
    convert_mot_to_yolo()
