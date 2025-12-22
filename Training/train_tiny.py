import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import cv2
import os
import numpy as np
from model_tiny import BallRefinerNet  # Import mạng của bạn
from tqdm import tqdm

# --- CẤU HÌNH ---
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
DATA_DIR = "tiny_dataset"
SAVE_PATH = "tiny_ball_refiner.pth"


# 1. Tạo Dataset Loader
class TinyDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        # Load Positive (Có bóng) -> Label Class = 1
        pos_files = glob.glob(os.path.join(root_dir, "pos", "*.jpg"))
        for p in pos_files:
            # Giả lập offset (Vì lúc crop mình đã random shift rồi)
            # Ở bài toán thực tế, bạn cần lưu offset chính xác vào tên file hoặc file txt
            # Để đơn giản demo: Coi như bóng luôn ở tâm (offset 0) hoặc để model tự học feature
            self.samples.append((p, 1.0, 0.0, 0.0))  # Path, Conf, dx, dy

        # Load Negative (Nền) -> Label Class = 0
        neg_files = glob.glob(os.path.join(root_dir, "neg", "*.jpg"))
        for p in neg_files:
            self.samples.append((p, 0.0, 0.0, 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, conf, dx, dy = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64))  # Đảm bảo đúng size

        # Chuẩn hóa về [0, 1] và đổi channel (H, W, C) -> (C, H, W)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        return torch.tensor(img), torch.tensor([conf]), torch.tensor([dx, dy])


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    # Load Data
    dataset = TinyDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load Model
    model = BallRefinerNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Loss functions
    criterion_cls = nn.BCELoss()  # Binary Cross Entropy cho phân loại (Có bóng/Ko bóng)
    criterion_reg = nn.MSELoss()  # Mean Squared Error cho vị trí (dx, dy)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for imgs, confs, offsets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            imgs = imgs.to(device)
            confs = confs.to(device).float()
            offsets = offsets.to(device).float()

            optimizer.zero_grad()

            pred_conf, pred_offset = model(imgs)

            # Tính Loss
            loss_cls = criterion_cls(pred_conf, confs)

            # Chỉ tính loss vị trí cho những mẫu CÓ BÓNG (conf=1)
            # Tạo mask để lọc
            mask = (confs > 0.5).squeeze()
            if mask.sum() > 0:
                loss_reg = criterion_reg(pred_offset[mask], offsets[mask])
            else:
                loss_reg = 0.0

            # Tổng loss (cân bằng trọng số nếu cần)
            loss = loss_cls + 0.1 * loss_reg

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(dataloader):.4f}")

    # Lưu model
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Đã lưu model tại: {SAVE_PATH}")


if __name__ == "__main__":
    train()