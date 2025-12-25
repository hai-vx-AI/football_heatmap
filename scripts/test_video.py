"""
Test script for BallRefinerNet - Creates a demo video showing predictions
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
# Chọn một sequence để test
TEST_SEQUENCE = "input/SoccerNet/tracking-2023/test/SNMOT-116"
OUTPUT_VIDEO = "output/videos/test_output.mp4"
CROP_SIZE = 64
SAMPLE_POINTS = 15  # Số điểm ngẫu nhiên để test mỗi frame

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
    
    # Get all images
    img_dir = os.path.join(TEST_SEQUENCE, "img1")
    if not os.path.exists(img_dir):
        print(f"ERROR: Image directory {img_dir} not found!")
        return
    
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:100]  # Test first 100 frames
    print(f"Found {len(img_files)} images")
    
    if len(img_files) == 0:
        print("No images found!")
        return
    
    # Read first image to get dimensions
    first_img = cv2.imread(img_files[0])
    h, w = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (w, h))
    
    print(f"\nProcessing {len(img_files)} frames...")
    
    for img_path in tqdm(img_files):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        
        h_img, w_img = frame.shape[:2]
        
        # Test với nhiều điểm ngẫu nhiên trên frame
        for _ in range(SAMPLE_POINTS):
            # Random position
            cx = np.random.randint(CROP_SIZE//2, w_img - CROP_SIZE//2)
            cy = np.random.randint(CROP_SIZE//2, h_img - CROP_SIZE//2)
            
            # Extract crop
            x1 = int(cx - CROP_SIZE / 2)
            y1 = int(cy - CROP_SIZE / 2)
            x2 = x1 + CROP_SIZE
            y2 = y1 + CROP_SIZE
            
            if x2 > w_img or y2 > h_img or x1 < 0 or y1 < 0:
                continue
            
            crop = frame[y1:y2, x1:x2]
            
            # Predict
            crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
            crop_tensor = crop_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                conf, offset = model(crop_tensor)
            
            conf_score = conf.item()
            dx = offset[0][0].item() * (CROP_SIZE / 2)
            dy = offset[0][1].item() * (CROP_SIZE / 2)
            
            # Visualize
            if conf_score > 0.8:  # High confidence - likely a ball
                # Calculate refined position
                refined_cx = cx + dx
                refined_cy = cy + dy
                
                # Draw green circle for high confidence detections
                cv2.circle(frame, (int(refined_cx), int(refined_cy)), 8, (0, 255, 0), 2)
                cv2.putText(frame, f"{conf_score:.2f}", 
                           (int(refined_cx) + 10, int(refined_cy)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            elif conf_score > 0.5:  # Medium confidence
                cv2.circle(frame, (int(cx), int(cy)), 3, (0, 165, 255), 1)
        
        # Add frame info
        frame_num = os.path.basename(img_path).split('.')[0]
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Model: BallRefinerNet (Epoch 7)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Write frame
        out.write(frame)
    
    out.release()
    print(f"\n✓ Video saved to {OUTPUT_VIDEO}")
    print(f"  - Frames: {len(img_files)}")
    print(f"  - Resolution: {w}x{h}")
    print(f"  - FPS: 20")

if __name__ == "__main__":
    main()
