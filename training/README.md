# Training Pipeline for Football Analysis Models

Complete training system for fine-tuning detection, team classification, and ball tracking.

## Overview

This training pipeline consists of 3 independent models:

1. **YOLO Detector** - Fine-tuned for football-specific detection (players, ball, referees)
2. **Team Classifier** - CNN for identifying team from jersey colors
3. **Ball Predictor** - LSTM/Transformer for predicting ball trajectory

## Quick Start

### 1. Train All Models (Recommended)

```bash
# Prepare data and train all models
python training/train_all.py --all

# Quick test with limited data
python training/train_all.py --all --max-sequences 5
```

### 2. Train Specific Models

```bash
# Only train YOLO
python training/train_all.py --train-yolo

# Train team classifier and ball predictor
python training/train_all.py --train-team-classifier --train-ball-predictor
```

### 3. Data Preparation Only

```bash
# Prepare data for all models
python training/train_all.py --prepare-data-only --prepare-all
```

## Detailed Training

### YOLO Fine-tuning

Fine-tune YOLO on SoccerNet for better detection:

```bash
# 1. Ensure YOLO format data exists
ls input/SoccerNet/yolo_format_v3_smart/

# 2. Train
python training/train_yolo.py --config training/training_config.yaml

# 3. Results in: runs/train/yolo_football/weights/best.pt
```

**Configuration** (in `training/training_config.yaml`):
- Base model: `yolo11n.pt` (change to yolo11s/m/x for larger models)
- Epochs: 100
- Image size: 1280 (higher = better for small ball detection)
- Batch size: 16
- Augmentation: Enabled

### Team Classifier Training

Train CNN to classify team from jersey crops:

```bash
# 1. Prepare jersey crop dataset
python training/prepare_jersey_data.py \
    --soccernet-root input/SoccerNet/tracking-2023 \
    --output-dir data/jersey_crops \
    --max-sequences 20

# 2. Train classifier
python training/train_team_classifier.py --config training/training_config.yaml

# 3. Results in: runs/team_classifier/best_model.pth
```

**Dataset Structure:**
```
data/jersey_crops/
├── train/
│   ├── team0/
│   │   └── *.jpg
│   └── team1/
│       └── *.jpg
├── val/
│   ├── team0/
│   └── team1/
├── train_annotations.json
└── val_annotations.json
```

**Model:** ResNet18 pretrained on ImageNet

### Ball Trajectory Predictor Training

Train LSTM/Transformer to predict ball movement:

```bash
# 1. Prepare trajectory dataset
python training/prepare_ball_trajectory_data.py \
    --soccernet-root input/SoccerNet/tracking-2023 \
    --output-dir data/ball_trajectories \
    --max-sequences 20

# 2. Train predictor
python training/train_ball_predictor.py --config training/training_config.yaml

# 3. Results in: runs/ball_predictor/best_model.pth
```

**Dataset Structure:**
```
data/ball_trajectories/
├── train_trajectories.json
└── val_trajectories.json
```

Each trajectory contains:
- `positions`: List of [x, y, frame_idx, visible] 
- `sequence`: Sequence name
- `total_frames`, `visible_frames`: Statistics

**Models:**
- **LSTM**: Lightweight, fast inference (default)
- **Transformer**: Better accuracy, slower

Change in config: `model_type: "transformer"`

## Configuration

Edit `training/training_config.yaml` to customize:

### YOLO Settings
```yaml
yolo_training:
  base_model: "yolo11n.pt"  # Model size
  epochs: 100
  batch: 16
  imgsz: 1280  # Input resolution
  lr0: 0.01  # Learning rate
  augment: true  # Data augmentation
```

### Team Classifier Settings
```yaml
team_classifier:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  pretrained: true  # ImageNet weights
  image_size: 224
```

### Ball Predictor Settings
```yaml
ball_predictor:
  model_type: "lstm"  # or "transformer"
  sequence_length: 10  # Input frames
  prediction_horizon: 5  # Predict N frames ahead
  epochs: 100
  batch_size: 128
```

## Hardware Requirements

### Minimum (CPU only)
- 16 GB RAM
- 50 GB disk space
- Training time: ~10-20 hours per model

### Recommended (GPU)
- NVIDIA GPU with 8+ GB VRAM
- 32 GB RAM
- Training time: ~2-5 hours per model

## Monitoring Training

All models save:
- Checkpoints every N epochs
- Best model based on validation metric
- Training logs and plots

### TensorBoard (for YOLO)
```bash
tensorboard --logdir runs/train
```

### View Results
```bash
# YOLO
ls runs/train/yolo_football/
# - weights/best.pt
# - results.png
# - confusion_matrix.png

# Team Classifier
ls runs/team_classifier/
# - best_model.pth
# - checkpoint_epoch_*.pth

# Ball Predictor
ls runs/ball_predictor/
# - best_model.pth
```

## Using Trained Models

### Update config.yaml to use trained models:

```yaml
detector:
  model_path: "runs/train/yolo_football/weights/best.pt"

team_color:
  use_trained_classifier: true
  classifier_path: "runs/team_classifier/best_model.pth"

ball_tracker:
  use_predictor: true
  predictor_path: "runs/ball_predictor/best_model.pth"
```

Then run inference as usual:
```bash
python main.py input/video.mp4
```

## Troubleshooting

### Out of Memory
- Reduce batch size in config
- Use smaller YOLO model (yolo11n instead of yolo11x)
- Reduce image size

### Poor Accuracy
- Increase epochs
- Try different learning rates
- Check data quality
- Use data augmentation

### Slow Training
- Enable GPU: `device: "cuda"`
- Increase batch size if memory allows
- Use mixed precision: `amp: true`
- Reduce number of workers if CPU bottleneck

## Advanced Options

### Resume Training
```yaml
# In training_config.yaml
yolo_training:
  resume: true  # Resume from last checkpoint
```

### Multi-GPU Training
```yaml
device: "0,1,2,3"  # Use GPUs 0,1,2,3
```

### Custom Data Split
Edit prepare scripts to change train/val split ratio (default 80/20).

### Export Models
```yaml
yolo_training:
  export_onnx: true  # Export to ONNX format
```

## Citation

If using SoccerNet dataset:
```
@inproceedings{Deliege2021SoccerNetv2,
  title={SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos},
  author={Deliege, Adrien and others},
  booktitle={CVPR Workshops},
  year={2021}
}
```
