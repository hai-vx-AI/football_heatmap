"""
Train LSTM/Transformer model for ball trajectory prediction.
"""

import argparse
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json


class BallTrajectoryDataset(Dataset):
    """Dataset for ball trajectory prediction."""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load trajectory data in new format
        data_file = self.data_dir / f"{split}_sequences.json"
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        self.sequences = data['sequences']
        self.mean = np.array(data['normalization']['mean'], dtype=np.float32)
        self.std = np.array(data['normalization']['std'], dtype=np.float32)
        
        print(f"Loaded {len(self.sequences)} sequences from {data_file}")
        print(f"Normalization: mean={self.mean.tolist()}, std={self.std.tolist()}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_seq = np.array(seq['input'], dtype=np.float32)
        target_seq = np.array(seq['target'], dtype=np.float32)
        
        # Normalize
        input_seq = (input_seq - self.mean) / self.std
        target_seq = (target_seq - self.mean) / self.std
        
        return torch.from_numpy(input_seq), torch.from_numpy(target_seq)


class BallPredictor(nn.Module):
    """LSTM-based ball trajectory predictor."""
    
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, prediction_horizon=5, dropout=0.2):
        super(BallPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # LSTM encoder
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Decoder
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, prediction_horizon * input_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, 2)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Decode to predictions
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Reshape to (batch, prediction_horizon, 2)
        out = out.view(-1, self.prediction_horizon, 2)
        
        return out


class TransformerBallPredictor(nn.Module):
    """Transformer-based ball trajectory predictor."""
    
    def __init__(self, input_dim=2, d_model=128, nhead=8, num_layers=4, 
                 prediction_horizon=5, dropout=0.1):
        super(TransformerBallPredictor, self).__init__()
        
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))  # Max seq length 100
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, prediction_horizon * input_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, 2)
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Use last position
        x = x[:, -1, :]  # (batch, d_model)
        
        # Predict future positions
        out = self.output_proj(x)  # (batch, prediction_horizon * 2)
        out = out.view(batch_size, self.prediction_horizon, 2)
        
        return out


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """Validate model."""
    # Check if validation set is empty
    if len(dataloader) == 0:
        return float('nan')  # Return NaN if no validation data
    
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def train_ball_predictor(config_path: str = "training/ball_predictor_config.yaml"):
    """
    Train ball trajectory predictor.
    
    Args:
        config_path: Path to training configuration
    """
    print("="*80)
    print("Ball Trajectory Predictor Training")
    print("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_config = config['ball_predictor']
    
    # Device
    device = torch.device(train_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\nUsing device: {device}")
    
    # Datasets
    data_dir = train_config['data_dir']
    
    train_dataset = BallTrajectoryDataset(data_dir, split='train')
    val_dataset = BallTrajectoryDataset(data_dir, split='val')
    
    print(f"\nDataset:")
    print(f"  Train sequences: {len(train_dataset)}")
    print(f"  Val sequences: {len(val_dataset)}")
    
    # Get prediction horizon from dataset metadata
    with open(Path(data_dir) / 'train_sequences.json', 'r') as f:
        metadata = json.load(f)['metadata']
        prediction_horizon = metadata['pred_horizon']
    
    # DataLoaders
    batch_size = train_config.get('batch_size', 128)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=train_config.get('num_workers', 4))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=train_config.get('num_workers', 4))
    
    # Model
    model_type = train_config.get('model_type', 'lstm')
    if model_type == 'lstm':
        model = BallPredictor(
            input_dim=2,
            hidden_dim=train_config.get('hidden_dim', 128),
            num_layers=train_config.get('num_layers', 2),
            prediction_horizon=prediction_horizon,
            dropout=train_config.get('dropout', 0.2)
        )
    elif model_type == 'transformer':
        model = TransformerBallPredictor(
            input_dim=2,
            d_model=train_config.get('d_model', 128),
            nhead=train_config.get('nhead', 8),
            num_layers=train_config.get('num_layers', 4),
            prediction_horizon=prediction_horizon,
            dropout=train_config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    print(f"\nModel: {model_type}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config.get('learning_rate', 0.001),
                          weight_decay=train_config.get('weight_decay', 1e-5))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                     patience=10)
    
    # Training loop
    epochs = train_config.get('epochs', 100)
    best_loss = float('inf')
    output_dir = Path(train_config.get('output_dir', 'runs/ball_predictor'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 80)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.6f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        if not torch.isnan(torch.tensor(val_loss)):
            print(f"Val Loss: {val_loss:.6f}")
            # Update learning rate
            scheduler.step(val_loss)
        else:
            print(f"Val Loss: N/A (no validation data)")
            # Use train loss for scheduler if no validation
            scheduler.step(train_loss)
        
        # Save best model (use train loss if no validation)
        loss_to_compare = val_loss if not torch.isnan(torch.tensor(val_loss)) else train_loss
        if loss_to_compare < best_loss:
            best_loss = loss_to_compare
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': train_config
            }, output_dir / 'best_model.pth')
            print(f"✓ Best model saved! (loss: {val_loss:.6f})")
        
        # Save checkpoint
        if (epoch + 1) % train_config.get('save_freq', 20) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    print("\n" + "="*80)
    print(f"Training Complete! Best Val Loss: {best_loss:.6f}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ball trajectory predictor")
    parser.add_argument('--config', type=str, default='training/ball_predictor_config.yaml',
                       help='Path to training configuration file')
    
    args = parser.parse_args()
    
    train_ball_predictor(args.config)
