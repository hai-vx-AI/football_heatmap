import cv2
import numpy as np
import torch
from ultralytics import YOLO
from model_tiny import BallRefinerNet  # Import mạng đã định nghĩa
import torchvision.transforms as transforms

# --- CẤU HÌNH ---
VIDEO_PATH = "video_bong_da.mp4"
YOLO_MODEL = "yolov8x.pt"  # Hoặc model bạn đã train
TINY_MODEL = "tiny_ball_refiner.pth"  # Model tiny đã train xong
CROP_SIZE = 64
CONF_THRESH = 0.8  # Độ tin cậy của Tiny-Net


class KalmanBoxTracker:
    # (Đơn giản hóa: Dự đoán vận tốc đều)
    def __init__(self, bbox):
        # bbox: [cx, cy, w, h]
        self.state = np.array(bbox[:2])  # [cx, cy]
        self.velocity = np.array([0.0, 0.0])  # [vx, vy]

    def predict(self):
        self.state += self.velocity
        return self.state

    def update(self, new_pos):
        # new_pos: [cx, cy]
        new_vel = new_pos - (self.state - self.velocity)  # Tính vận tốc mới
        # Cập nhật có trọng số (Làm mượt chuyển động)
        alpha = 0.6
        self.velocity = self.velocity * (1 - alpha) + new_vel * alpha
        self.state = new_pos


def main():
    # 1. LOAD MODELS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model Global (Cứu viện)
    model_global = YOLO(YOLO_MODEL)

    # Model Local (Thường trực)
    model_tiny = BallRefinerNet().to(device)
    if os.path.exists(TINY_MODEL):
        model_tiny.load_state_dict(torch.load(TINY_MODEL))
        model_tiny.eval()
    else:
        print("CẢNH BÁO: Chưa có file weights cho Tiny-Net. Chạy demo sẽ lỗi.")

    # Preprocess cho Tiny-Net
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # 0-255 -> 0-1
    ])

    cap = cv2.VideoCapture(VIDEO_PATH)
    tracker = None
    is_lost = True  # Trạng thái ban đầu: Mất dấu

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h_img, w_img = frame.shape[:2]
        current_ball_pos = None  # Kết quả cuối cùng của frame này

        # === LỚP PHÒNG THỦ 1 & 2: DỰ ĐOÁN & LOCAL SEARCH ===
        if not is_lost and tracker is not None:
            # 1. Kalman dự đoán
            pred_cx, pred_cy = tracker.predict()

            # 2. Crop ảnh tại vị trí dự đoán
            x1 = int(pred_cx - CROP_SIZE / 2)
            y1 = int(pred_cy - CROP_SIZE / 2)

            # Check biên (padding nếu cần, ở đây mình clip cho nhanh)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x1 + CROP_SIZE), min(h_img, y1 + CROP_SIZE)

            if (x2 - x1) == CROP_SIZE and (y2 - y1) == CROP_SIZE:
                crop = frame[y1:y2, x1:x2]

                # 3. Tiny-Net kiểm tra
                input_tensor = transform(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    conf, offset = model_tiny(input_tensor)

                conf_score = conf.item()

                # === QUYẾT ĐỊNH ===
                if conf_score > CONF_THRESH:
                    # -> TÌM THẤY! (Không cần YOLO to)
                    dx = offset[0][0].item() * (CROP_SIZE / 2)  # Scale lại
                    dy = offset[0][1].item() * (CROP_SIZE / 2)

                    real_cx = x1 + (CROP_SIZE / 2) + dx
                    real_cy = y1 + (CROP_SIZE / 2) + dy
                    current_ball_pos = np.array([real_cx, real_cy])

                    # Vẽ màu xanh lá (Local Tracking)
                    cv2.circle(frame, (int(real_cx), int(real_cy)), 10, (0, 255, 0), -1)
                    cv2.putText(frame, "Local", (int(real_cx) + 10, int(real_cy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # -> MẤT DẤU! (Tiny-Net bảo chỗ này không có bóng)
                    print(f"Lost track! Conf: {conf_score:.2f}")
                    is_lost = True
            else:
                is_lost = True  # Ra mép ảnh -> coi như mất

        # === LỚP PHÒNG THỦ 3: GLOBAL SEARCH (CỨU VIỆN) ===
        if is_lost or current_ball_pos is None:
            # Chạy YOLOv8 quét toàn ảnh
            results = model_global(frame, verbose=False)

            # Tìm class 'ball' (giả sử ID=1)
            best_conf = 0
            found_box = None

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id == 1 and conf > 0.25:  # ID 1 là bóng
                        if conf > best_conf:
                            best_conf = conf
                            found_box = box.xywh[0].cpu().numpy()  # cx, cy, w, h

            if found_box is not None:
                # TÌM LẠI ĐƯỢC RỒI!
                current_ball_pos = found_box[:2]  # cx, cy
                is_lost = False
                tracker = KalmanBoxTracker([current_ball_pos[0], current_ball_pos[1], 0, 0])

                # Vẽ màu đỏ (Global Recovery)
                cv2.circle(frame, (int(current_ball_pos[0]), int(current_ball_pos[1])), 10, (0, 0, 255), -1)
                cv2.putText(frame, "Global Refind", (int(current_ball_pos[0]) + 10, int(current_ball_pos[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Update tracker nếu có vị trí mới
        if current_ball_pos is not None and tracker is not None:
            tracker.update(current_ball_pos)

        cv2.imshow("Smart Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()