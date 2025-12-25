"""
Enhanced Ball Tracking với visualization tốt hơn
Hiển thị cả ground truth để so sánh
"""
import cv2
import numpy as np
import os
import torch
import glob
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ultralytics import YOLO
from model.model_tiny import BallRefinerNet
from tqdm import tqdm

# --- CẤU HÌNH ---
TEST_SEQUENCE = "input/SoccerNet/tracking-2023/test/SNMOT-116"
OUTPUT_VIDEO = "output/videos/tracking_enhanced.mp4"
TINY_MODEL = "model/tiny_ball_refiner.pth"
CROP_SIZE = 64
CONF_THRESH = 0.6  # Giảm threshold để dễ detect hơn
MAX_FRAMES = 300
SEARCH_GRID = 20  # Grid search khi không có tracker

def load_ground_truth(gt_file, seqinfo_file):
    """Load ground truth từ MOT format"""
    img_width, img_height = 1920, 1080
    if os.path.exists(seqinfo_file):
        with open(seqinfo_file, 'r') as f:
            for line in f:
                if 'imWidth' in line:
                    img_width = int(line.split('=')[1].strip())
                elif 'imHeight' in line:
                    img_height = int(line.split('=')[1].strip())
    
    annotations = {}
    if os.path.exists(gt_file):
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                x, y, w, h = map(float, parts[2:6])
                cx, cy = x + w/2, y + h/2
                
                if frame_id not in annotations:
                    annotations[frame_id] = []
                annotations[frame_id].append({'cx': cx, 'cy': cy, 'w': w, 'h': h})
    
    return annotations

def test_crop(model, frame, cx, cy, crop_size, device):
    """Test một crop position"""
    h, w = frame.shape[:2]
    x1 = int(cx - crop_size/2)
    y1 = int(cy - crop_size/2)
    
    if x1 < 0 or y1 < 0 or x1+crop_size > w or y1+crop_size > h:
        return None, None
    
    crop = frame[y1:y1+crop_size, x1:x1+crop_size]
    crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
    crop_tensor = crop_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        conf, offset = model(crop_tensor)
    
    conf_score = conf.item()
    dx = offset[0][0].item() * (crop_size / 2)
    dy = offset[0][1].item() * (crop_size / 2)
    
    refined_cx = cx + dx
    refined_cy = cy + dy
    
    return conf_score, (refined_cx, refined_cy)

def main():
    print("=" * 70)
    print("ENHANCED BALL TRACKING WITH GROUND TRUTH")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[1] Loading Tiny-Net model on {device.upper()}...")
    
    model_tiny = BallRefinerNet().to(device)
    if os.path.exists(TINY_MODEL):
        model_tiny.load_state_dict(torch.load(TINY_MODEL, map_location=device))
        model_tiny.eval()
        print(f"    ✓ Model loaded: {TINY_MODEL}")
    else:
        print(f"    ✗ Model not found!")
        return
    
    # Load data
    img_dir = os.path.join(TEST_SEQUENCE, "img1")
    gt_file = os.path.join(TEST_SEQUENCE, "gt", "gt.txt")
    seqinfo_file = os.path.join(TEST_SEQUENCE, "seqinfo.ini")
    
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:MAX_FRAMES]
    annotations = load_ground_truth(gt_file, seqinfo_file)
    
    print(f"\n[2] Test sequence: {TEST_SEQUENCE}")
    print(f"    Frames: {len(img_files)}")
    print(f"    Ground truth frames: {len(annotations)}")
    
    # Video writer
    first_frame = cv2.imread(img_files[0])
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (w, h))
    
    print(f"\n[3] Processing frames...")
    
    stats = {'detected': 0, 'missed': 0, 'correct': 0}
    
    for img_path in tqdm(img_files):
        frame = cv2.imread(img_path)
        frame_num = int(os.path.basename(img_path).split('.')[0])
        h_img, w_img = frame.shape[:2]
        
        # Get ground truth
        gt_objects = annotations.get(frame_num, [])
        
        # Draw ground truth (blue)
        for obj in gt_objects:
            cv2.circle(frame, (int(obj['cx']), int(obj['cy'])), 15, (255, 100, 0), 2)
            cv2.putText(frame, "GT", (int(obj['cx'])+20, int(obj['cy'])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
        
        # Search for ball using Tiny-Net
        best_conf = 0
        best_pos = None
        
        if len(gt_objects) > 0:
            # Search around ground truth (để test model)
            for obj in gt_objects:
                for dx in range(-40, 41, 20):
                    for dy in range(-40, 41, 20):
                        test_x = obj['cx'] + dx
                        test_y = obj['cy'] + dy
                        
                        conf, pos = test_crop(model_tiny, frame, test_x, test_y, CROP_SIZE, device)
                        if conf is not None and conf > best_conf:
                            best_conf = conf
                            best_pos = pos
        else:
            # Grid search trên toàn ảnh (chậm hơn)
            step = 100
            for x in range(CROP_SIZE//2, w_img-CROP_SIZE//2, step):
                for y in range(CROP_SIZE//2, h_img-CROP_SIZE//2, step):
                    conf, pos = test_crop(model_tiny, frame, x, y, CROP_SIZE, device)
                    if conf is not None and conf > best_conf and conf > CONF_THRESH:
                        best_conf = conf
                        best_pos = pos
        
        # Draw prediction
        if best_conf > CONF_THRESH and best_pos is not None:
            color = (0, 255, 0) if best_conf > 0.8 else (0, 200, 255)
            cv2.circle(frame, (int(best_pos[0]), int(best_pos[1])), 12, color, -1)
            cv2.putText(frame, f"Pred {best_conf:.2f}", 
                       (int(best_pos[0])+20, int(best_pos[1])+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            stats['detected'] += 1
            
            # Check if correct (within 50 pixels of GT)
            if len(gt_objects) > 0:
                for obj in gt_objects:
                    dist = np.sqrt((best_pos[0]-obj['cx'])**2 + (best_pos[1]-obj['cy'])**2)
                    if dist < 50:
                        stats['correct'] += 1
                        break
        elif len(gt_objects) > 0:
            stats['missed'] += 1
        
        # Info overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f"Frame: {frame_num}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Best Confidence: {best_conf:.3f}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if best_conf > CONF_THRESH else (100, 100, 100), 2)
        cv2.putText(frame, f"GT Objects: {len(gt_objects)}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Detected: {stats['detected']}", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Correct: {stats['correct']}", (20, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Missed: {stats['missed']}", (20, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Legend
        cv2.putText(frame, "Blue=Ground Truth | Green=Prediction", (20, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    
    print("\n" + "=" * 70)
    print("TRACKING COMPLETED!")
    print("=" * 70)
    print(f"Output: {OUTPUT_VIDEO}")
    print(f"Total frames: {len(img_files)}")
    print(f"Detections: {stats['detected']}")
    print(f"Correct detections: {stats['correct']}")
    print(f"Missed: {stats['missed']}")
    if stats['detected'] > 0:
        accuracy = stats['correct'] / stats['detected'] * 100
        print(f"Accuracy: {accuracy:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
