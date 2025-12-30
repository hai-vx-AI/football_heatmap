"""Monitor training progress for both models."""
import time
import os
from pathlib import Path

def check_team_classifier():
    """Check team classifier training status."""
    best_model = Path("runs/team_classifier/best_model.pth")
    if best_model.exists():
        mtime = os.path.getmtime(best_model)
        age = time.time() - mtime
        return f"✓ Running (last update: {age/60:.1f}min ago)"
    return "⏳ Not started yet"

def check_ball_predictor():
    """Check ball predictor training status."""
    best_model = Path("runs/ball_predictor/best_model.pth")
    if best_model.exists():
        mtime = os.path.getmtime(best_model)
        age = time.time() - mtime
        return f"✓ Completed (Val Loss: check logs)"
    return "⏳ Not started yet"

def check_ball_data():
    """Check ball trajectory data preparation."""
    train_file = Path("data/ball_trajectories/train_sequences.json")
    if train_file.exists():
        mtime = os.path.getmtime(train_file)
        age = time.time() - mtime
        return f"✓ Ready (last update: {age/60:.1f}min ago)"
    return "⏳ Preparing..."

print("="*80)
print("TRAINING STATUS")
print("="*80)
print(f"\n1. Ball Trajectory Data: {check_ball_data()}")
print(f"2. Team Classifier:      {check_team_classifier()}")
print(f"3. Ball Predictor:       {check_ball_predictor()}")
print("\n" + "="*80)
