"""
Enhanced Multi-Class Object Detection Model
Classes: 0=Background, 1=Ball, 2=Team A, 3=Team B, 4=Referee
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassObjectDetector(nn.Module):
    """
    Enhanced model với multi-class classification
    - Input: 64x64x3 RGB crop
    - Output: 
      1. Class probabilities (5 classes: bg, ball, team_a, team_b, referee)
      2. Bounding box offset (dx, dy, dw, dh)
    """
    def __init__(self, num_classes=5):
        super(MultiClassObjectDetector, self).__init__()
        self.num_classes = num_classes
        
        # Backbone CNN
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # After 4 poolings: 64x64 -> 4x4
        # 256 channels * 4 * 4 = 4096
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Multi-class classifier
        self.classifier = nn.Linear(256, num_classes)
        
        # Bounding box regressor (for fine-tuning position)
        self.bbox_regressor = nn.Linear(256, 4)  # dx, dy, dw, dh
        
    def forward(self, x):
        # Backbone
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 4x4
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Outputs
        class_logits = self.classifier(x)  # Raw logits
        bbox_deltas = torch.tanh(self.bbox_regressor(x))  # Normalized to [-1, 1]
        
        return class_logits, bbox_deltas
    
    def predict(self, x):
        """Inference with softmax"""
        class_logits, bbox_deltas = self.forward(x)
        class_probs = F.softmax(class_logits, dim=1)
        return class_probs, bbox_deltas


class BallRefinerNet(nn.Module):
    """
    Legacy binary classification model (kept for compatibility)
    """
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
