"""
Prepare YOLO dataset from SoccerNet tracking data.
Converts SoccerNet gamestate.json annotations to YOLO format.
"""

import json
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import cv2
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare YOLO dataset from SoccerNet')
    parser.add_argument('--soccernet-root', type=str, required=True,
                        help='Root directory of SoccerNet tracking dataset')
    parser.add_argument('--output-dir', type=str, default='input/SoccerNet/yolo_dataset',
                        help='Output directory for YOLO format dataset')
    parser.add_argument('--max-sequences', type=int, default=None,
                        help='Maximum number of sequences to process')
    parser.add_argument('--sample-rate', type=int, default=5,
                        help='Sample every N frames')
    return parser.parse_args()


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert bbox from [x, y, w, h] to YOLO format [x_center, y_center, width, height] normalized.
    """
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height


def process_sequence(seq_path, output_dir, class_map, sample_rate=5):
    """
    Process one SoccerNet sequence and convert to YOLO format.
    """
    # Load ground truth annotations (MOT format)
    gt_path = seq_path / 'gt' / 'gt.txt'
    if not gt_path.exists():
        return 0
    
    # Read all annotations
    annotations_by_frame = {}
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            
            frame_num = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            class_id = int(parts[6])
            
            # Map class ID (1=player, 2=goalkeeper, 3=referee, 4=ball)
            if class_id == 1:
                class_name = 'player'
            elif class_id == 2:
                class_name = 'goalkeeper'
            elif class_id == 3:
                class_name = 'referee'
            elif class_id == 4:
                class_name = 'ball'
            else:
                continue
            
            if frame_num not in annotations_by_frame:
                annotations_by_frame[frame_num] = []
            
            annotations_by_frame[frame_num].append({
                'class': class_name,
                'bbox': [x, y, w, h]
            })
    
    # Get image dimensions from seqinfo
    seqinfo_path = seq_path / 'seqinfo.ini'
    img_width = 1920
    img_height = 1080
    
    if seqinfo_path.exists():
        with open(seqinfo_path, 'r') as f:
            for line in f:
                if 'imWidth' in line:
                    img_width = int(line.split('=')[1].strip())
                elif 'imHeight' in line:
                    img_height = int(line.split('=')[1].strip())
    
    frames_processed = 0
    
    # Get image directory
    img_dir = seq_path / 'img1'
    if not img_dir.exists():
        return 0
    
    # Process frames with annotations
    for frame_num, frame_annotations in annotations_by_frame.items():
        # Sample frames
        if frame_num % sample_rate != 0:
            continue
        
        # Get image path
        img_path = img_dir / f'{frame_num:06d}.jpg'
        if not img_path.exists():
            continue
        
        # Convert annotations to YOLO format
        yolo_annotations = []
        
        for ann in frame_annotations:
            class_name = ann['class']
            if class_name not in class_map:
                continue
            
            class_id = class_map[class_name]
            bbox = ann['bbox']
            
            # Convert to YOLO format
            x_center, y_center, width, height = convert_bbox_to_yolo(bbox, img_width, img_height)
            
            # Validate bbox
            if width <= 0 or height <= 0 or x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
                continue
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Skip frames without valid annotations
        if not yolo_annotations:
            continue
        
        # Copy image
        dest_img_path = output_dir / 'images' / f'{seq_path.name}_{img_path.name}'
        dest_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, dest_img_path)
        
        # Save label
        label_path = output_dir / 'labels' / f'{seq_path.name}_{img_path.stem}.txt'
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        frames_processed += 1
    
    return frames_processed


def main():
    args = parse_args()
    
    soccernet_root = Path(args.soccernet_root)
    output_dir = Path(args.output_dir)
    
    # Class mapping
    class_map = {
        'player': 0,
        'ball': 1,
        'referee': 2,
        'goalkeeper': 3
    }
    
    print("="*80)
    print("YOLO Dataset Preparation from SoccerNet")
    print("="*80)
    print(f"SoccerNet root: {soccernet_root}")
    print(f"Output directory: {output_dir}")
    print(f"Sample rate: {args.sample_rate}")
    print(f"Class mapping: {class_map}")
    
    # Process train and val splits
    for split in ['train', 'val']:
        split_dir = soccernet_root / split
        if not split_dir.exists():
            print(f"\nWarning: {split} directory not found, skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing {split} split")
        print(f"{'='*80}")
        
        # Get sequences
        sequences = sorted(list(split_dir.glob('SNMOT-*')))
        if args.max_sequences:
            sequences = sequences[:args.max_sequences]
        
        print(f"Found {len(sequences)} sequences")
        
        # Output directory for this split
        split_output_dir = output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        total_frames = 0
        
        # Process each sequence
        for seq_path in tqdm(sequences, desc=f"Processing {split}"):
            frames = process_sequence(seq_path, split_output_dir, class_map, args.sample_rate)
            total_frames += frames
        
        print(f"\n{split} split: {total_frames} frames processed")
    
    # Create data.yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'player',
            1: 'ball',
            2: 'referee',
            3: 'goalkeeper'
        },
        'nc': 4
    }
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n{'='*80}")
    print(f"Dataset preparation complete!")
    print(f"Data YAML: {yaml_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
