"""
Multi-class Tracking Script
Sử dụng MultiClassObjectDetector để phân biệt bóng, team A và team B
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Import model
from model.model_multiclass import MultiClassObjectDetector

# ============== CONFIG ==============
# Paths
INPUT_VIDEO = project_root / "input/SoccerNet/tracking-2023/test/SNMOT-116/img1"
GT_FILE = project_root / "input/SoccerNet/tracking-2023/test/SNMOT-116/gt/gt.txt"
OUTPUT_VIDEO = project_root / "output/videos/tracking_multiclass.mp4"
TINY_MODEL_PATH = project_root / "model/multiclass_detector_5class.pth"
YOLO_MODEL = project_root / "input/yolo11x.pt"  # Upgraded to YOLOv11

# Hyper-parameters
CROP_SIZE = 64
MAX_FRAMES = 300
FPS = 10
CONF_THRESHOLD = 0.4  # Threshold cho classification confidence

# Class colors (BGR format for OpenCV)
CLASS_COLORS = {
    0: (128, 128, 128),  # Background - Gray
    1: (0, 255, 255),    # Ball - Yellow
    2: (0, 0, 255),      # Team A - Red
    3: (255, 0, 0),      # Team B - Blue
    4: (0, 255, 0)       # Referee - Green
}

CLASS_NAMES = {
    0: "Background",
    1: "Ball",
    2: "Team A",
    3: "Team B",
    4: "Referee"
}


def load_ground_truth(gt_file):
    """Load ground truth annotations"""
    gt_data = defaultdict(list)
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            x, y, w, h = map(float, parts[2:6])
            gt_data[frame_id].append([x, y, w, h])
    return gt_data


def extract_crop_batch(frame, bboxes, crop_size=64):
    """Extract multiple crops from frame"""
    crops = []
    valid_indices = []
    
    for idx, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        cx, cy = int(x + w/2), int(y + h/2)
        
        # Calculate crop bounds
        x1 = max(0, cx - crop_size // 2)
        y1 = max(0, cy - crop_size // 2)
        x2 = min(frame.shape[1], x1 + crop_size)
        y2 = min(frame.shape[0], y1 + crop_size)
        
        crop = frame[y1:y2, x1:x2]
        
        # Skip if crop too small
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            continue
            
        # Resize to 64x64
        crop_resized = cv2.resize(crop, (crop_size, crop_size))
        
        # Convert to RGB and normalize
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        crop_normalized = crop_rgb.astype(np.float32) / 255.0
        crop_tensor = torch.from_numpy(crop_normalized).permute(2, 0, 1)
        
        crops.append(crop_tensor)
        valid_indices.append(idx)
    
    if len(crops) == 0:
        return None, []
    
    # Stack to batch
    batch = torch.stack(crops)
    return batch, valid_indices


def main():
    print("=" * 60)
    print("🎯 MULTI-CLASS FOOTBALL TRACKING")
    print("=" * 60)
    
    # Load models
    print("\n📦 Loading models...")
    yolo = YOLO(str(YOLO_MODEL))
    
    tiny_model = MultiClassObjectDetector(num_classes=5)  # Updated to 5 classes
    if TINY_MODEL_PATH.exists():
        checkpoint = torch.load(str(TINY_MODEL_PATH), map_location='cpu')
        tiny_model.load_state_dict(checkpoint)
        print(f"✓ Loaded multi-class model (5 classes): {TINY_MODEL_PATH.name}")
    else:
        print(f"❌ Model not found: {TINY_MODEL_PATH}")
        return
    
    tiny_model.eval()
    device = 'cpu'
    tiny_model.to(device)
    
    # Load ground truth
    print(f"\n📊 Loading ground truth from: {GT_FILE.name}")
    gt_data = load_ground_truth(GT_FILE)
    
    # Get video frames
    print(f"\n🎬 Loading video frames from: {INPUT_VIDEO.name}")
    frame_files = sorted(list(INPUT_VIDEO.glob("*.jpg")))[:MAX_FRAMES]
    print(f"✓ Found {len(frame_files)} frames")
    
    # Read first frame for video writer
    sample_frame = cv2.imread(str(frame_files[0]))
    height, width = sample_frame.shape[:2]
    
    # Create output directory
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, FPS, (width, height))
    
    # Statistics
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_detections = 0
    
    print(f"\n🚀 Processing {len(frame_files)} frames...")
    print(f"Output: {OUTPUT_VIDEO}")
    print("-" * 60)
    
    for frame_idx, frame_file in enumerate(frame_files, 1):
        frame = cv2.imread(str(frame_file))
        
        # Get GT boxes for this frame
        gt_boxes = gt_data.get(frame_idx, [])
        
        if len(gt_boxes) == 0:
            out.write(frame)
            continue
        
        # Extract crops
        batch, valid_indices = extract_crop_batch(frame, gt_boxes, CROP_SIZE)
        
        if batch is None:
            out.write(frame)
            continue
        
        # Run multi-class classification
        with torch.no_grad():
            batch = batch.to(device)
            class_probs, bbox_deltas = tiny_model.predict(batch)
        
        # Process detections
        for i, bbox_idx in enumerate(valid_indices):
            x, y, w, h = gt_boxes[bbox_idx]
            
            # Get predicted class
            pred_class = torch.argmax(class_probs[i]).item()
            confidence = class_probs[i][pred_class].item()
            
            # Skip low confidence and background
            if confidence < CONF_THRESHOLD or pred_class == 0:
                continue
            
            # Update statistics
            class_counts[pred_class] += 1
            total_detections += 1
            
            # Draw bbox with class-specific color
            color = CLASS_COLORS[pred_class]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{CLASS_NAMES[pred_class]} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw frame info
        info_text = f"Frame: {frame_idx}/{len(frame_files)}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw class statistics
        y_offset = 60
        for cls_id in [1, 2, 3]:  # Skip background
            cls_text = f"{CLASS_NAMES[cls_id]}: {class_counts[cls_id]}"
            cv2.putText(frame, cls_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLASS_COLORS[cls_id], 2)
            y_offset += 30
        
        out.write(frame)
        
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{len(frame_files)} frames...")
    
    out.release()
    
    # Final statistics
    print("\n" + "=" * 60)
    print("✅ TRACKING COMPLETED!")
    print("=" * 60)
    print(f"📹 Output: {OUTPUT_VIDEO}")
    print(f"📊 Detection Statistics:")
    print(f"   Total detections: {total_detections}")
    for cls_id in [1, 2, 3, 4]:  # Include referee
        percentage = (class_counts[cls_id] / total_detections * 100) if total_detections > 0 else 0
        print(f"   {CLASS_NAMES[cls_id]}: {class_counts[cls_id]} ({percentage:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
