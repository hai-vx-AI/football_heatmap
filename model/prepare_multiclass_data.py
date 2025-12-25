"""
Prepare Multi-Class Training Data from YOLO format
Uses ground truth class labels from convert_mot_to_yolo.py

YOLO Classes (from gameinfo.ini):
- 0: ball
- 1: player
- 2: goalkeeper  
- 3: referee

Output Classes (for MultiClass model):
- 0: Background (negative samples)
- 1: Ball
- 2: Team A (player/goalkeeper left side)
- 3: Team B (player/goalkeeper right side)
- 4: Referee
"""
import cv2
import os
import random
import glob
import json
from tqdm import tqdm

# --- CẤU HÌNH ---
DATA_ROOT = r"C:\Users\DungLe\Documents\GitHub\football_heatmap\input\SoccerNet\yolo_format"
SAVE_DIR = "../input/multiclass_dataset"
CROP_SIZE = 64
BACKGROUND_SAMPLES_PER_IMAGE = 2  # Number of background crops per image

def get_team_from_position(cx, w_img):
    """
    Determine team based on field position
    Team A: left side (cx < w_img/2)
    Team B: right side (cx >= w_img/2)
    """
    return 'team_a' if cx < w_img / 2 else 'team_b'


def crop_and_save(img, cx, cy, crop_size, class_label, save_dir, counter):
    """Crop and save image with label"""
    h, w = img.shape[:2]
    
    # Random offset for data augmentation (except background)
    if class_label != 'background':
        offset_x = random.randint(-10, 10)
        offset_y = random.randint(-10, 10)
    else:
        offset_x = offset_y = 0
    
    x1 = int(cx + offset_x - crop_size / 2)
    y1 = int(cy + offset_y - crop_size / 2)
    
    # Boundary check
    if x1 < 0 or y1 < 0 or x1 + crop_size > w or y1 + crop_size > h:
        return None
    
    crop = img[y1:y1 + crop_size, x1:x1 + crop_size]
    
    # Save crop
    filename = f"{class_label}_{counter:06d}.jpg"
    filepath = os.path.join(save_dir, class_label, filename)
    cv2.imwrite(filepath, crop)
    
    return {
        'file': filename,
        'class': class_label,
        'original_center': (cx, cy),
        'crop_offset': (offset_x, offset_y)
    }


def prepare_multiclass_data():
    """Main function - convert YOLO format to crop-based multiclass dataset"""
    print("=" * 70)
    print("PREPARING MULTI-CLASS TRAINING DATA (5 CLASSES)")
    print("Using ground truth labels from YOLO format")
    print("=" * 70)
    
    # Create output directories
    for class_name in ['background', 'ball', 'team_a', 'team_b', 'referee']:
        os.makedirs(os.path.join(SAVE_DIR, class_name), exist_ok=True)
    
    # Get all training images
    img_paths = glob.glob(os.path.join(DATA_ROOT, "images", "train", "**", "*.jpg"), recursive=True)
    print(f"\n📁 Found {len(img_paths):,} training images\n")
    
    counters = {'background': 0, 'ball': 0, 'team_a': 0, 'team_b': 0, 'referee': 0}
    metadata = []
    
    for img_path in tqdm(img_paths, desc="Processing"):
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h_img, w_img = img.shape[:2]
        
        # Load YOLO labels
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        if not os.path.exists(label_path):
            continue
        
        # Parse all objects in this image
        objects = []  # List of (class_id, cx, cy, w, h)
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 5:
                    yolo_class_id = int(parts[0])
                    cx_norm, cy_norm, w_norm, h_norm = parts[1:5]
                    
                    # Convert to pixel coordinates
                    cx = cx_norm * w_img
                    cy = cy_norm * h_img
                    
                    objects.append((yolo_class_id, cx, cy, w_norm, h_norm))
        
        # Process each object based on YOLO class
        for yolo_class_id, cx, cy, w_norm, h_norm in objects:
            # Map YOLO class to output class
            if yolo_class_id == 0:  # ball
                class_label = 'ball'
            elif yolo_class_id == 3:  # referee
                class_label = 'referee'
            elif yolo_class_id in [1, 2]:  # player or goalkeeper
                class_label = get_team_from_position(cx, w_img)
            else:
                continue  # Skip unknown classes
            
            # Crop and save
            result = crop_and_save(img, cx, cy, CROP_SIZE, class_label, SAVE_DIR, counters[class_label])
            if result:
                metadata.append(result)
                counters[class_label] += 1
        
        # --- BACKGROUND CROPS ---
        # Sample random locations far from all objects
        for _ in range(BACKGROUND_SAMPLES_PER_IMAGE):
            attempts = 0
            while attempts < 10:
                rx = random.randint(CROP_SIZE//2, w_img - CROP_SIZE//2)
                ry = random.randint(CROP_SIZE//2, h_img - CROP_SIZE//2)
                
                # Check distance from all objects
                is_clear = True
                for _, ox, oy, _, _ in objects:
                    dist = ((rx - ox)**2 + (ry - oy)**2)**0.5
                    if dist < CROP_SIZE:
                        is_clear = False
                        break
                
                if is_clear:
                    result = crop_and_save(img, rx, ry, CROP_SIZE, 'background', SAVE_DIR, counters['background'])
                    if result:
                        metadata.append(result)
                        counters['background'] += 1
                    break
                
                attempts += 1
    
    # Save metadata
    metadata_file = os.path.join(SAVE_DIR, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            'stats': counters,
            'crop_size': CROP_SIZE,
            'yolo_class_mapping': {
                '0': 'ball',
                '1': 'player', 
                '2': 'goalkeeper',
                '3': 'referee'
            },
            'output_class_mapping': {
                '0': 'background',
                '1': 'ball',
                '2': 'team_a',
                '3': 'team_b', 
                '4': 'referee'
            },
            'samples': metadata
        }, f, indent=2)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("✅ COMPLETED!")
    print("=" * 70)
    print(f"\n📊 Class Distribution:")
    print(f"  Background: {counters['background']:>8,}")
    print(f"  Ball:       {counters['ball']:>8,}")
    print(f"  Team A:     {counters['team_a']:>8,}")
    print(f"  Team B:     {counters['team_b']:>8,}")
    print(f"  Referee:    {counters['referee']:>8,}")
    print(f"  {'─'*30}")
    print(f"  TOTAL:      {sum(counters.values()):>8,}")
    print(f"\n📁 Output: {SAVE_DIR}/")
    print(f"📄 Metadata: {metadata_file}")
    print("=" * 70)

if __name__ == "__main__":
    prepare_multiclass_data()
