import cv2
import os
import random
import glob
from tqdm import tqdm

# --- CẤU HÌNH ---
DATA_ROOT = r"D:\football_obtracking\path\to\SoccerNet\tracking-2023"  # Folder chứa images và labels
SAVE_DIR = "tiny_dataset"
CROP_SIZE = 64
BALL_CLASS_ID = 1  # ID của bóng trong file label mới


def crop_data():
    os.makedirs(os.path.join(SAVE_DIR, "pos"), exist_ok=True)  # Chứa bóng
    os.makedirs(os.path.join(SAVE_DIR, "neg"), exist_ok=True)  # Không chứa bóng (nền)

    # Lấy danh sách ảnh train
    img_paths = glob.glob(os.path.join(DATA_ROOT, "images", "train", "**", "*.jpg"), recursive=True)

    print(f"Tìm thấy {len(img_paths)} ảnh. Bắt đầu cắt crop...")

    count_pos = 0
    count_neg = 0

    for img_path in tqdm(img_paths):
        # Đường dẫn label tương ứng (đổi images -> labels, .jpg -> .txt)
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

        if not os.path.exists(label_path): continue

        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]

        balls = []

        # 1. Đọc label để tìm bóng
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                cls_id = int(parts[0])
                if cls_id == BALL_CLASS_ID:
                    # Chuyển YOLO -> Pixel center
                    cx, cy = parts[1] * w_img, parts[2] * h_img
                    balls.append((cx, cy))

        # 2. CẮT POSITIVE (Có bóng)
        for (bx, by) in balls:
            # Random dịch chuyển tâm một chút để model học cách sửa sai
            offset_x = random.randint(-10, 10)
            offset_y = random.randint(-10, 10)

            x1 = int(bx + offset_x - CROP_SIZE / 2)
            y1 = int(by + offset_y - CROP_SIZE / 2)

            # Kiểm tra biên
            if x1 < 0 or y1 < 0 or x1 + CROP_SIZE > w_img or y1 + CROP_SIZE > h_img: continue

            crop = img[y1:y1 + CROP_SIZE, x1:x1 + CROP_SIZE]
            cv2.imwrite(f"{SAVE_DIR}/pos/ball_{count_pos}.jpg", crop)
            count_pos += 1

        # 3. CẮT NEGATIVE (Nền đất, không có bóng)
        # Random crop đại một chỗ, nếu xa bóng thì lấy
        for _ in range(2):  # Lấy 2 mẫu nền mỗi ảnh
            rx = random.randint(0, w_img - CROP_SIZE)
            ry = random.randint(0, h_img - CROP_SIZE)
            rcx, rcy = rx + CROP_SIZE / 2, ry + CROP_SIZE / 2

            # Kiểm tra xem có trúng quả bóng nào không
            is_overlap = False
            for (bx, by) in balls:
                if abs(bx - rcx) < CROP_SIZE / 2 and abs(by - rcy) < CROP_SIZE / 2:
                    is_overlap = True;
                    break

            if not is_overlap:
                crop = img[ry:ry + CROP_SIZE, rx:rx + CROP_SIZE]
                cv2.imwrite(f"{SAVE_DIR}/neg/bg_{count_neg}.jpg", crop)
                count_neg += 1

    print(f"Xong! Pos: {count_pos}, Neg: {count_neg}")


if __name__ == "__main__":
    crop_data()
    # Sau khi chạy xong file này, bạn viết một file train đơn giản (PyTorch standard)
    # để train model_tiny.py với data trong folder 'tiny_dataset' nhé.
    # Save model thành 'tiny_ball_refiner.pth'