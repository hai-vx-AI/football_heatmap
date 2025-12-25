import torch
import torch.nn as nn
import torch.nn.functional as F


class BallRefinerNet(nn.Module):
    def __init__(self):
        super(BallRefinerNet, self).__init__()
        # Input: 3 x 64 x 64
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Flatten size: 64 channels * 8 * 8 (sau 3 lần pool) = 4096
        self.fc = nn.Linear(4096, 256)

        # Nhánh 1: Phân loại (Có bóng hay là nền đất?)
        self.classifier = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output 0.0 -> 1.0
        )

        # Nhánh 2: Hồi quy vị trí (Tinh chỉnh tâm bóng)
        self.regressor = nn.Sequential(
            nn.Linear(256, 2),  # dx, dy
            nn.Tanh()  # Output -1.0 -> 1.0 (tương ứng dịch chuyển trong khoảng crop)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x32
        x = self.pool(F.relu(self.conv2(x)))  # 16x16
        x = self.pool(F.relu(self.conv3(x)))  # 8x8
        x = x.view(x.size(0), -1)  # Flatten
        feat = F.relu(self.fc(x))

        conf = self.classifier(feat)
        offset = self.regressor(feat)
        return conf, offset