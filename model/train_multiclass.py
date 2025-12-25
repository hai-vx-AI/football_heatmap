"""
Train Multi-Class Object Detector
Classes: 0=Background, 1=Ball, 2=Team_A, 3=Team_B, 4=Referee
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import cv2
import os
import numpy as np
from model_multiclass import MultiClassObjectDetector
from tqdm import tqdm

# --- CẤU HÌNH ---
BATCH_SIZE = 32
EPOCHS = 20
START_EPOCH = 0  # Set to 0 to train from scratch, or N to resume from epoch N
LR = 0.001
DATA_DIR = "../input/multiclass_dataset"
SAVE_PATH = "multiclass_detector_5class.pth"
RESUME_TRAINING = False  # Set to True to resume from checkpoint

# Class mapping
CLASS_MAP = {
    'background': 0,
    'ball': 1,
    'team_a': 2,
    'team_b': 3,
    'referee': 4
}

class MultiClassDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        
        for class_name, class_id in CLASS_MAP.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            files = glob.glob(os.path.join(class_dir, "*.jpg"))
            for f in files:
                self.samples.append((f, class_id))
        
        print(f"Loaded {len(self.samples)} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, class_id = self.samples[idx]
        
        # Load image
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load: {path}")
        
        img = cv2.resize(img, (64, 64))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        
        # Dummy bbox offset (for simplicity, can be enhanced later)
        bbox_offset = np.zeros(4, dtype=np.float32)
        
        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(class_id, dtype=torch.long),
            torch.tensor(bbox_offset, dtype=torch.float32)
        )

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}\n")
    
    # Load data
    dataset = MultiClassDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Model
    model = MultiClassObjectDetector(num_classes=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Loss functions
    criterion_cls = nn.CrossEntropyLoss()  # Multi-class classification
    criterion_bbox = nn.MSELoss()  # Bounding box regression
    
    # Resume from checkpoint if exists
    start_epoch = START_EPOCH
    best_acc = 0.0
    
    if RESUME_TRAINING and os.path.exists(SAVE_PATH):
        print(f"📦 Loading checkpoint: {SAVE_PATH}")
        checkpoint = torch.load(SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        print("✓ Checkpoint loaded successfully")
        
        # Resume from START_EPOCH
        start_epoch = START_EPOCH
        best_acc = 0.0  # Reset for new 5-class model
        print(f"Resuming from epoch {start_epoch + 1}")
        print(f"Current best accuracy: {best_acc:.2f}%\n")
    
    # Training loop
    # Training loop
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, class_ids, bbox_offsets in pbar:
            imgs = imgs.to(device)
            class_ids = class_ids.to(device)
            bbox_offsets = bbox_offsets.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            class_logits, pred_bbox = model(imgs)
            
            # Classification loss
            loss_cls = criterion_cls(class_logits, class_ids)
            
            # Bbox loss (only for non-background)
            mask = (class_ids > 0)
            if mask.sum() > 0:
                loss_bbox = criterion_bbox(pred_bbox[mask], bbox_offsets[mask])
            else:
                loss_bbox = torch.tensor(0.0).to(device)
            
            # Total loss
            loss = loss_cls + 0.1 * loss_bbox
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(class_logits, 1)
            correct += (predicted == class_ids).sum().item()
            total += class_ids.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
        
        epoch_acc = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {epoch_acc:.2f}%")
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Saved best model (acc={best_acc:.2f}%)")
        
        print()
    
    print("=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Model saved: {SAVE_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    train()
