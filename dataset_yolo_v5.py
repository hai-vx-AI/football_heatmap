import os
import configparser
from tqdm import tqdm

# --- CẤU HÌNH ---
DATA_ROOT = r"D:\football_obtracking\path\to\SoccerNet\tracking-2023"  # Sửa đường dẫn của bạn
PHASES = ["train", "test"]

# NGƯỠNG KÍCH THƯỚC (QUAN TRỌNG)
# Nếu vật thể nhỏ hơn 2% chiều ngang VÀ nhỏ hơn 3% chiều dọc ảnh -> Là Bóng
THRES_W = 0.02
THRES_H = 0.03


def convert_to_yolo_format(img_width, img_height, box):
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    x_center = (box[0] + box[2] / 2.0) * dw
    y_center = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return x_center, y_center, w, h


def main():
    # Mình đổi tên folder output thành labels_fixed để bạn dễ phân biệt
    labels_root = os.path.join(DATA_ROOT, "labels_fixed")

    for phase in PHASES:
        phase_dir = os.path.join(DATA_ROOT, phase)
        if not os.path.exists(phase_dir): continue

        print(f"--- Đang xử lý: {phase} ---")
        snmot_folders = [f for f in os.listdir(phase_dir) if os.path.isdir(os.path.join(phase_dir, f))]

        for snmot_name in tqdm(snmot_folders):
            snmot_dir = os.path.join(phase_dir, snmot_name)

            # Lấy size ảnh
            ini_path = os.path.join(snmot_dir, "seqinfo.ini")
            width, height = 1920, 1080
            if os.path.exists(ini_path):
                config = configparser.ConfigParser()
                config.read(ini_path)
                try:
                    width = int(config['Sequence']['imWidth'])
                    height = int(config['Sequence']['imHeight'])
                except:
                    pass

            out_dir = os.path.join(labels_root, phase, snmot_name, "img1")
            os.makedirs(out_dir, exist_ok=True)

            gt_file = os.path.join(snmot_dir, "gt", "gt.txt")
            if not os.path.exists(gt_file): continue

            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    frame_id = int(parts[0])
                    x, y, w_px, h_px = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])

                    # Tính toán tỷ lệ YOLO trước
                    bbox = convert_to_yolo_format(width, height, [x, y, w_px, h_px])
                    w_norm, h_norm = bbox[2], bbox[3]

                    # --- LOGIC TÁCH BÓNG Ở ĐÂY ---
                    # Nếu chiều rộng < 2% VÀ chiều cao < 3% -> Class 1 (Ball)
                    # Ngược lại -> Class 0 (Player)
                    if w_norm < THRES_W and h_norm < THRES_H:
                        cls_id = 1
                    else:
                        cls_id = 0

                    label_filename = f"{frame_id:06d}.txt"
                    with open(os.path.join(out_dir, label_filename), 'a') as outfile:
                        outfile.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    print(f"\nXong! Folder mới là: {labels_root}")
    print("Bạn nhớ sửa file data.yaml trỏ về folder này và khai báo 2 class nhé!")


if __name__ == "__main__":
    main()