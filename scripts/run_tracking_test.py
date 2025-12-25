"""
Smart Ball Tracking trên Test Dataset
Sử dụng: YOLO Global + Tiny Refiner Local + Kalman Tracker
"""
import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import glob
from ultralytics import YOLO
from model.model_tiny import BallRefinerNet
from tqdm import tqdm

# --- CẤU HÌNH ---
TEST_SEQUENCE = "input/SoccerNet/tracking-2023/test/SNMOT-116"  # Chọn sequence test
OUTPUT_VIDEO = "output/videos/tracking_output.mp4"
YOLO_MODEL = "yolov8n.pt"  # Model nhỏ để chạy nhanh trên CPU
TINY_MODEL = "model/tiny_ball_refiner.pth"
CROP_SIZE = 64
CONF_THRESH = 0.7
MAX_FRAMES = 300  # Giới hạn số frame để test nhanh

class KalmanBoxTracker:
    def __init__(self, bbox):
        self.state = np.array(bbox[:2])  # [cx, cy]
        self.velocity = np.array([0.0, 0.0])  # [vx, vy]

    def predict(self):
        self.state += self.velocity
        return self.state

    def update(self, new_pos):
        new_vel = new_pos - (self.state - self.velocity)
        alpha = 0.6
        self.velocity = self.velocity * (1 - alpha) + new_vel * alpha
        self.state = new_pos

def main():
    print("=" * 70)
    print("FOOTBALL BALL TRACKING - TEST OUTPUT")
    print("=" * 70)
    
    # 1. LOAD MODELS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[1] Loading models on {device.upper()}...")
    
    # YOLO Global
    model_global = YOLO(YOLO_MODEL)
    print(f"    ✓ YOLO model: {YOLO_MODEL}")
    
    # Tiny Refiner
    model_tiny = BallRefinerNet().to(device)
    if os.path.exists(TINY_MODEL):
        model_tiny.load_state_dict(torch.load(TINY_MODEL, map_location=device))
        model_tiny.eval()
        print(f"    ✓ Tiny model: {TINY_MODEL}")
    else:
        print(f"    ✗ Model {TINY_MODEL} not found!")
        return
    
    # 2. LOAD TEST IMAGES
    img_dir = os.path.join(TEST_SEQUENCE, "img1")
    if not os.path.exists(img_dir):
        print(f"\n✗ Test sequence not found: {img_dir}")
        return
    
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:MAX_FRAMES]
    print(f"\n[2] Loading test sequence: {TEST_SEQUENCE}")
    print(f"    Total frames: {len(img_files)}")
    
    if len(img_files) == 0:
        print("    ✗ No images found!")
        return
    
    # Get video dimensions
    first_frame = cv2.imread(img_files[0])
    h, w = first_frame.shape[:2]
    
    # 3. CREATE VIDEO WRITER
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (w, h))
    print(f"\n[3] Output video: {OUTPUT_VIDEO} ({w}x{h} @ 20fps)")
    
    # 4. TRACKING LOOP
    print(f"\n[4] Running tracking...")
    tracker = None
    is_lost = True
    stats = {
        'global_detections': 0,
        'local_detections': 0,
        'lost_frames': 0
    }
    
    for idx, img_path in enumerate(tqdm(img_files, desc="Processing")):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        h_img, w_img = frame.shape[:2]
        current_ball_pos = None
        detection_type = "None"
        
        # === LOCAL SEARCH (Kalman + Tiny-Net) ===
        if not is_lost and tracker is not None:
            pred_cx, pred_cy = tracker.predict()
            
            # Crop around prediction
            x1 = int(pred_cx - CROP_SIZE / 2)
            y1 = int(pred_cy - CROP_SIZE / 2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x1 + CROP_SIZE), min(h_img, y1 + CROP_SIZE)
            
            if (x2 - x1) == CROP_SIZE and (y2 - y1) == CROP_SIZE:
                crop = frame[y1:y2, x1:x2]
                
                # Tiny-Net prediction
                crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
                crop_tensor = crop_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    conf, offset = model_tiny(crop_tensor)
                
                conf_score = conf.item()
                
                if conf_score > CONF_THRESH:
                    # Ball found locally!
                    dx = offset[0][0].item() * (CROP_SIZE / 2)
                    dy = offset[0][1].item() * (CROP_SIZE / 2)
                    
                    real_cx = x1 + (CROP_SIZE / 2) + dx
                    real_cy = y1 + (CROP_SIZE / 2) + dy
                    current_ball_pos = np.array([real_cx, real_cy])
                    detection_type = "Local"
                    stats['local_detections'] += 1
                    
                    # Draw green circle (Local tracking)
                    cv2.circle(frame, (int(real_cx), int(real_cy)), 12, (0, 255, 0), 3)
                    cv2.putText(frame, f"Local {conf_score:.2f}", 
                               (int(real_cx) + 15, int(real_cy)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    is_lost = True
            else:
                is_lost = True
        
        # === GLOBAL SEARCH (YOLO) ===
        if is_lost or current_ball_pos is None:
            results = model_global(frame, verbose=False)
            
            # Find best ball detection (class 32 = sports ball in COCO)
            best_conf = 0
            found_box = None
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    # Class 32 = sports ball in COCO
                    if cls_id == 32 and conf > 0.3:
                        if conf > best_conf:
                            best_conf = conf
                            found_box = box.xywh[0].cpu().numpy()
            
            if found_box is not None:
                # Ball found globally!
                current_ball_pos = found_box[:2]
                is_lost = False
                tracker = KalmanBoxTracker([current_ball_pos[0], current_ball_pos[1], 0, 0])
                detection_type = "Global"
                stats['global_detections'] += 1
                
                # Draw red circle (Global detection)
                cv2.circle(frame, (int(current_ball_pos[0]), int(current_ball_pos[1])), 
                          12, (0, 0, 255), 3)
                cv2.putText(frame, f"Global {best_conf:.2f}", 
                           (int(current_ball_pos[0]) + 15, int(current_ball_pos[1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                stats['lost_frames'] += 1
        
        # Update tracker
        if current_ball_pos is not None and tracker is not None:
            tracker.update(current_ball_pos)
            
            # Draw trajectory (last 10 positions)
            if not hasattr(tracker, 'trail'):
                tracker.trail = []
            tracker.trail.append(current_ball_pos.copy())
            if len(tracker.trail) > 10:
                tracker.trail.pop(0)
            
            # Draw trail
            for i in range(len(tracker.trail) - 1):
                pt1 = tuple(tracker.trail[i].astype(int))
                pt2 = tuple(tracker.trail[i + 1].astype(int))
                cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
        
        # Add info overlay
        cv2.rectangle(frame, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame: {idx+1}/{len(img_files)}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {detection_type}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if detection_type != "None" else (0, 0, 255), 2)
        cv2.putText(frame, f"Global: {stats['global_detections']}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Local: {stats['local_detections']}", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Lost: {stats['lost_frames']}", (20, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        out.write(frame)
    
    out.release()
    
    # 5. SUMMARY
    print("\n" + "=" * 70)
    print("TRACKING COMPLETED!")
    print("=" * 70)
    print(f"Output video: {OUTPUT_VIDEO}")
    print(f"Total frames: {len(img_files)}")
    print(f"Global detections: {stats['global_detections']}")
    print(f"Local detections: {stats['local_detections']}")
    print(f"Lost frames: {stats['lost_frames']}")
    total_detections = stats['global_detections'] + stats['local_detections']
    if len(img_files) > 0:
        print(f"Detection rate: {total_detections/len(img_files)*100:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
