"""
Train CNN classifier for team identification from jersey crops.
"""

import argparse
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import cv2
import numpy as np
from tqdm import tqdm
import json


class JerseyDataset(Dataset):
    """Dataset for jersey color classification."""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load annotations
        ann_file = self.data_dir / f"{split}_annotations.json"
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.samples = []
        for item in self.annotations:
            self.samples.append({
                'image_path': self.data_dir / item['crop_path'],
                'label': item['team_id']  # 0 or 1
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['image_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = sample['label']
        
        return image, label


class TeamClassifier(nn.Module):
    """CNN classifier for team identification."""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(TeamClassifier, self).__init__()
        
        # Use ResNet18 as backbone
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_team_classifier(config_path: str = "training/team_classifier_config.yaml"):
    """
    Train team classifier on jersey crops.
    
    Args:
        config_path: Path to training configuration
    """
    print("="*80)
    print("Team Classifier Training")
    print("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_config = config['team_classifier']
    
    # Device
    device = torch.device(train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\nUsing device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((train_config.get('image_size', 224), train_config.get('image_size', 224))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((train_config.get('image_size', 224), train_config.get('image_size', 224))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    data_dir = train_config['data_dir']
    train_dataset = JerseyDataset(data_dir, split='train', transform=train_transform)
    val_dataset = JerseyDataset(data_dir, split='val', transform=val_transform)
    
    print(f"\nDataset:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # DataLoaders
    batch_size = train_config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=train_config.get('num_workers', 4))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=train_config.get('num_workers', 4))
    
    # Model
    model = TeamClassifier(num_classes=2, pretrained=train_config.get('pretrained', True))
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config.get('learning_rate', 0.001),
                          weight_decay=train_config.get('weight_decay', 1e-4))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                     patience=5)
    
    # Training loop
    epochs = train_config.get('epochs', 50)
    best_acc = 0.0
    output_dir = Path(train_config.get('output_dir', 'runs/team_classifier'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, output_dir / 'best_model.pth')
            print(f"✓ Best model saved! (acc: {val_acc:.2f}%)")
        
        # Save checkpoint
        if (epoch + 1) % train_config.get('save_freq', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    print("\n" + "="*80)
    print(f"Training Complete! Best Val Acc: {best_acc:.2f}%")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train team classifier")
    parser.add_argument('--config', type=str, default='training/team_classifier_config.yaml',
                       help='Path to training configuration file')
    
    args = parser.parse_args()
    
    train_team_classifier(args.config)
