"""
Test BallRefinerNet with Ground Truth annotations
Shows model predictions vs actual ball positions
"""
import cv2
import numpy as np
import torch
import glob
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.model_tiny import BallRefinerNet
from tqdm import tqdm

# --- CẤU HÌNH ---
MODEL_PATH = "model/tiny_ball_refiner.pth"
TEST_SEQUENCE = "input/SoccerNet/tracking-2023/test/SNMOT-116"
OUTPUT_VIDEO = "output/videos/test_with_gt.mp4"
CROP_SIZE = 64
CONF_THRESHOLD = 0.7

def load_ground_truth(gt_file, seqinfo_file):
    """Load ground truth annotations from MOT format"""
    # Read image dimensions
    img_width, img_height = 1920, 1080
    if os.path.exists(seqinfo_file):
        with open(seqinfo_file, 'r') as f:
            for line in f:
                if 'imWidth' in line:
                    img_width = int(line.split('=')[1].strip())
                elif 'imHeight' in line:
                    img_height = int(line.split('=')[1].strip())
    
    # Read annotations
    annotations = {}
    if os.path.exists(gt_file):
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                x, y, w, h = map(float, parts[2:6])
                
                # Convert to center coordinates
                cx = x + w / 2
                cy = y + h / 2
                
                if frame_id not in annotations:
                    annotations[frame_id] = []
                annotations[frame_id].append({
                    'cx': cx,
                    'cy': cy,
                    'w': w,
                    'h': h
                })
    
    return annotations, img_width, img_height

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = BallRefinerNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"✓ Loaded model from {MODEL_PATH}")
    else:
        print(f"ERROR: Model file {MODEL_PATH} not found!")
        return
    
    # Load ground truth
    gt_file = os.path.join(TEST_SEQUENCE, "gt", "gt.txt")
    seqinfo_file = os.path.join(TEST_SEQUENCE, "seqinfo.ini")
    annotations, img_width, img_height = load_ground_truth(gt_file, seqinfo_file)
    print(f"✓ Loaded {len(annotations)} frames with annotations")
    
    # Get all images
    img_dir = os.path.join(TEST_SEQUENCE, "img1")
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:150]
    print(f"✓ Found {len(img_files)} images\n")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (img_width, img_height))
    
    stats = {'tp': 0, 'fp': 0, 'fn': 0, 'total_frames': 0}
    
    print("Processing frames...")
    for img_path in tqdm(img_files):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        frame_num = int(os.path.basename(img_path).split('.')[0])
        h_img, w_img = frame.shape[:2]
        
        # Get ground truth for this frame
        gt_objects = annotations.get(frame_num, [])
        
        # Draw ground truth (blue circles)
        for obj in gt_objects:
            cv2.circle(frame, (int(obj['cx']), int(obj['cy'])), 15, (255, 0, 0), 2)
            cv2.putText(frame, "GT", (int(obj['cx']) + 20, int(obj['cy'])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Test model around ground truth positions
        detections = []
        if len(gt_objects) > 0:
            for obj in gt_objects:
                # Test in area around GT
                for offset_x in range(-30, 31, 10):
                    for offset_y in range(-30, 31, 10):
                        test_cx = int(obj['cx'] + offset_x)
                        test_cy = int(obj['cy'] + offset_y)
                        
                        if test_cx < CROP_SIZE//2 or test_cx > w_img - CROP_SIZE//2:
                            continue
                        if test_cy < CROP_SIZE//2 or test_cy > h_img - CROP_SIZE//2:
                            continue
                        
                        # Extract crop
                        x1 = test_cx - CROP_SIZE // 2
                        y1 = test_cy - CROP_SIZE // 2
                        crop = frame[y1:y1+CROP_SIZE, x1:x1+CROP_SIZE]
                        
                        # Predict
                        crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
                        crop_tensor = crop_tensor.unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            conf, offset = model(crop_tensor)
                        
                        conf_score = conf.item()
                        
                        if conf_score > CONF_THRESHOLD:
                            dx = offset[0][0].item() * (CROP_SIZE / 2)
                            dy = offset[0][1].item() * (CROP_SIZE / 2)
                            refined_cx = test_cx + dx
                            refined_cy = test_cy + dy
                            detections.append({
                                'cx': refined_cx,
                                'cy': refined_cy,
                                'conf': conf_score
                            })
        
        # Draw detections (green circles)
        for det in detections:
            cv2.circle(frame, (int(det['cx']), int(det['cy'])), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"{det['conf']:.2f}", 
                       (int(det['cx']) + 20, int(det['cy'])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Calculate stats
        stats['total_frames'] += 1
        if len(gt_objects) > 0:
            if len(detections) > 0:
                stats['tp'] += 1  # Detected when ball exists
            else:
                stats['fn'] += 1  # Missed when ball exists
        
        # Add legend
        cv2.rectangle(frame, (10, 10), (350, 130), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame: {frame_num}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Model: Epoch 7", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Blue=GT | Green=Pred", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Detections: {len(detections)}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        out.write(frame)
    
    out.release()
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"✓ Video saved to {OUTPUT_VIDEO}")
    print(f"{'='*50}")
    print(f"Statistics:")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Frames with GT: {stats['tp'] + stats['fn']}")
    print(f"  True Positives: {stats['tp']}")
    print(f"  False Negatives: {stats['fn']}")
    if stats['tp'] + stats['fn'] > 0:
        recall = stats['tp'] / (stats['tp'] + stats['fn'])
        print(f"  Recall: {recall:.2%}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
