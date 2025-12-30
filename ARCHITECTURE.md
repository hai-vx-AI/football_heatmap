# Football Video Analysis System - Architecture

This document provides a comprehensive overview of the four core modules that power the football video analysis system.

---

## Table of Contents

1. [Module 1: Player Detection (YOLO)](#module-1-player-detection-yolo)
2. [Module 2: Ball Detection & Prediction](#module-2-ball-detection--prediction)
3. [Module 3: Team Classification](#module-3-team-classification)
4. [Module 4: Real-Time Possession Analysis](#module-4-real-time-possession-analysis)

---

## Module 1: Player Detection (YOLO)

### Overview
The player detection module uses YOLO (You Only Look Once) deep learning models to identify and localize players, goalkeepers, referees, and balls in football video frames.

### How It Works

#### 1.1 Model Architecture
- **Model Type**: YOLO11/YOLOv8 based object detection
- **Input**: Video frames (RGB images)
- **Output**: Bounding boxes with class labels and confidence scores
- **Supported Classes**:
  - `player` (ID: 0)
  - `ball` (ID: 1)
  - `referee` (ID: 2)
  - `goalkeeper` (ID: 3)

#### 1.2 Detection Pipeline

```
Video Frame (BGR)
    ↓
Preprocessing (Letterbox resize to 1280x1280)
    ↓
YOLO Model Inference
    ↓
Non-Maximum Suppression (NMS)
    ↓
Per-Class Confidence Filtering
    ↓
Geometry Validation (Ball filters)
    ↓
Detections [bbox, confidence, class_id]
```

#### 1.3 Data Requirements

**Training Data Structure:**
```
input/SoccerNet/yolo_dataset/
├── train/
│   ├── images/
│   │   ├── frame_001.jpg
│   │   └── ...
│   └── labels/
│       ├── frame_001.txt  # YOLO format: class_id x_center y_center width height
│       └── ...
└── val/
    ├── images/
    └── labels/
```

**YOLO Label Format (normalized 0-1):**
```
0 0.512 0.345 0.089 0.156  # player at center (0.512, 0.345), size (0.089, 0.156)
1 0.678 0.234 0.012 0.015  # ball
2 0.123 0.456 0.067 0.134  # referee
```

#### 1.4 Model Training

**Quick Training:**
```bash
cd training
python train_yolo.py --data ../input/SoccerNet/yolo_dataset/dataset.yaml \
                     --model yolo11n.pt \
                     --epochs 100 \
                     --imgsz 1280
```

**Training Configuration (`training/training_config.yaml`):**
- **Epochs**: 100-300 (adjust based on dataset size)
- **Image Size**: 1280x1280 (balance between accuracy and speed)
- **Batch Size**: 16-32 (GPU dependent)
- **Augmentation**: Enabled (flip, rotation, mosaic, mixup)
- **Optimizer**: AdamW with cosine learning rate schedule

#### 1.5 Detection Parameters

| Parameter | Purpose | Default | Range |
|-----------|---------|---------|-------|
| `conf_global` | Global confidence threshold | 0.25 | 0.1-0.9 |
| `conf_player` | Player-specific threshold | 0.4 | 0.2-0.8 |
| `conf_ball` | Ball-specific threshold | 0.15 | 0.05-0.5 |
| `conf_referee` | Referee-specific threshold | 0.4 | 0.2-0.8 |
| `conf_goalkeeper` | Goalkeeper-specific threshold | 0.4 | 0.2-0.8 |
| `nms_iou` | NMS IoU threshold | 0.5 | 0.3-0.7 |
| `max_det` | Max detections per frame | 100 | 50-300 |

#### 1.6 Ball-Specific Filters

To reduce false positives, ball detections are filtered by:
- **Area**: `min_area_px=15`, `max_area_px=1500`
- **Aspect Ratio**: `min_aspect=0.6`, `max_aspect=1.8` (mostly circular)

#### 1.7 Model Files

- **Pretrained**: `yolo11n.pt`, `yolo11x.pt`, `yolov8x.pt`
- **Custom Trained**: `data/trained_models/yolo_football_best.pt`

**Implementation**: [`src/detector.py`](src/detector.py)

---

## Module 2: Ball Detection & Prediction

### Overview
The ball tracking module combines YOLO detections with Kalman Filter-based motion prediction to maintain stable ball trajectories even during occlusions or detection failures.

### How It Works

#### 2.1 Two-Phase Tracking System

**Phase 1: Detection-Based Tracking**
- When YOLO detects the ball → use detection directly
- Status: `'detected'`

**Phase 2: Prediction-Based Tracking**
- When YOLO fails to detect → predict ball position using Kalman Filter
- Status: `'predicted'`

**Phase 3: Lost Tracking**
- After 15 consecutive misses → trigger reacquisition or mark as lost
- Status: `'lost'`

#### 2.2 Kalman Filter Motion Model

**State Vector (4D):**
```
x = [cx, cy, vx, vy]
    ↑    ↑   ↑   ↑
    |    |   |   └─ velocity_y
    |    |   └───── velocity_x
    |    └─────── center_y
    └────────── center_x
```

**Prediction Equation:**
```
x(t+1) = F × x(t)

where F = [1  0  dt  0]  # dt = 1/fps (e.g., 0.04s for 25fps)
          [0  1  0  dt]
          [0  0  1   0]
          [0  0  0   1]
```

**Measurement Update:**
```
z(t) = [cx, cy]  # Observed ball center from YOLO
x(t) = x(t|t-1) + K × (z(t) - H × x(t|t-1))
```

#### 2.3 Tracking Pipeline

```
Frame N
    ↓
YOLO Detection → Ball Detections (0-N bboxes)
    ↓
Motion Gating (filter by max displacement)
    ↓
Association (nearest to prediction)
    ↓
┌─────────┬─────────┐
│ MATCHED │ NO MATCH│
└────┬────┴────┬────┘
     ↓         ↓
  Update KF  Predict Only
  Status='detected'  Status='predicted'
     ↓         ↓
  Miss=0    Miss++
     ↓         ↓
     └────┬────┘
          ↓
    miss_counter > 15?
          ↓
        YES → Reacquisition
        NO  → Continue
```

#### 2.4 Reacquisition Strategy

When the ball is lost for 5+ frames, the tracker attempts to reacquire it:

1. **Create Search ROI**: Expand 4x around last known position
2. **Run YOLO in ROI**: Focused detection in smaller region
3. **Validate Candidates**: Check geometry and confidence
4. **Reinitialize Tracker**: Reset Kalman filter with new detection

**Reacquisition Frequency**: Every 3 frames during loss

#### 2.5 Key Parameters

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `track_buffer` | Max frames to keep predicting | 15 |
| `max_displacement_px_per_frame` | Motion gating radius | 80 px |
| `gate_radius_scale` | Search radius multiplier | 1.5x |
| `process_noise_pos` | Position uncertainty | 1.0 |
| `process_noise_vel` | Velocity uncertainty | 0.1 |
| `measurement_noise` | Detection noise | 5.0 |

#### 2.6 Output State

Each frame, the ball tracker returns:
```python
{
    'status': 'detected' | 'predicted' | 'lost',
    'bbox': [x1, y1, x2, y2],           # Current bounding box
    'center': (cx, cy),                  # Current center
    'velocity': (vx, vy),                # Estimated velocity (px/frame)
    'confidence': float,                 # Detection confidence (0 if predicted)
    'miss_counter': int,                 # Consecutive missed detections
    'trajectory': [(cx, cy), ...],       # Recent trajectory (last 30 frames)
}
```

**Implementation**: [`src/ball_tracker.py`](src/ball_tracker.py)

---

## Module 3: Team Classification

### Overview
The team classification module assigns each player to one of two teams based on jersey color analysis using clustering and temporal smoothing.

### How It Works

#### 3.1 Color-Based Classification Pipeline

```
Player Detection (bbox)
    ↓
1. Crop Jersey ROI
   (x: 20%-80%, y: 15%-55% of bbox)
    ↓
2. Preprocess
   - Convert BGR → Lab color space
   - Apply grass mask (remove green pixels)
   - Filter dark pixels (shadow removal)
    ↓
3. Extract Dominant Color
   - Compute median/mean of remaining pixels
    ↓
4. Assign to Team
   - Distance to team centroids (Euclidean in Lab)
   - Assign to nearest cluster
    ↓
5. Temporal Smoothing
   - Per-track EMA (Exponential Moving Average)
   - Voting over last 30 frames
    ↓
Team ID (0 or 1)
```

#### 3.2 Jersey ROI Extraction

**Why only jersey region?**
- Avoids legs/shorts (often same color for both teams or grass)
- Focuses on most discriminative area
- Reduces noise from background

**Crop Parameters:**
```
ROI = bbox[
    x_start : x_end,  # 0.2 to 0.8 (central 60% horizontally)
    y_start : y_end   # 0.15 to 0.55 (chest area vertically)
]
```

#### 3.3 Color Space: Lab vs RGB

**Lab Color Space Advantages:**
- **L**: Lightness (0-100) - separates brightness
- **a**: Green-Red axis (-128 to 127)
- **b**: Blue-Yellow axis (-128 to 127)
- **Perceptually uniform**: Euclidean distance matches human perception
- **Illumination invariant**: Separates color from brightness

**Distance Metric:**
```
distance = √[(L1-L2)² + (a1-a2)² + (b1-b2)²]
```

#### 3.4 Two-Phase Team Learning

**Phase 1: Warmup (First 200 frames)**
- Collect jersey color samples from all players
- Build global sample set (100+ samples minimum)
- Run K-Means clustering (K=2) to find team centroids

**Phase 2: Online Classification (After warmup)**
- Assign players by distance to centroids
- Update centroids incrementally (EMA with β=0.01)
- Apply per-track temporal smoothing

#### 3.5 Temporal Smoothing

**Per-Track EMA:**
```python
color_ema[track_id] = α × color_new + (1 - α) × color_ema[track_id]
```
- **α = 0.2**: Balance between stability and responsiveness

**Voting Window:**
- Keep last 30 team assignments per track
- Final team = majority vote (threshold 60%)
- Prevents flickering between teams

#### 3.6 Goalkeeper Handling

Goalkeepers often wear different colors than their team. Solution:

**Neighbor Voting Method:**
1. Find nearby players within 300px radius
2. Count team assignments of neighbors
3. Assign goalkeeper to majority team (threshold 50%)

**Alternative**: Manual assignment via config

#### 3.7 Grass Mask & Filters

**Grass Removal (HSV):**
- Hue: 35-85 (green range)
- Saturation: > 60
- Value: > 40
- Purpose: Remove grass pixels from jersey ROI

**Dark Pixel Filter:**
- L < 20 in Lab space
- Purpose: Remove shadows and very dark regions

#### 3.8 Key Parameters

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `warmup_frames` | Frames to collect samples | 200 |
| `min_init_samples` | Min samples for clustering | 100 |
| `n_teams` | Number of teams | 2 |
| `ema_alpha` | Per-track color smoothing | 0.2 |
| `vote_window_frames` | Voting window size | 30 |
| `vote_threshold` | Majority vote threshold | 0.6 |
| `centroid_update_beta` | Centroid EMA weight | 0.01 |

#### 3.9 Output

Each player track is augmented with:
```python
{
    'team_id': 0 or 1,           # Assigned team
    'team_color_lab': [L, a, b], # Smoothed jersey color
    'team_confidence': float,    # Voting confidence (0-1)
}
```

**Special Cases:**
- Referees: `team_id = None` (not assigned)
- Goalkeepers: Assigned via neighbor voting or manual config

**Implementation**: [`src/team_assigner.py`](src/team_assigner.py)

---

## Module 4: Real-Time Possession Analysis

### Overview
The possession analysis module determines which player/team controls the ball and tracks possession statistics in real-time.

### How It Works

#### 4.1 Possession Detection Logic

**Basic Rule:**
```
Player has possession IF:
    distance(player_center, ball_center) ≤ possession_radius
```

**Default Radius**: 150 pixels (adjustable based on video resolution)

#### 4.2 Possession Pipeline

```
Frame N
    ↓
Ball State (detected/predicted/lost)
    ↓
Ball Lost? → No Possession
    ↓ NO
Find Nearest Player to Ball
    ↓
Distance ≤ 150px?
    ↓ YES
┌──────────────────┐
│ POSSESSION FOUND │
└────────┬─────────┘
         ↓
    Same Player as Last Frame?
         ↓
    YES → Continue Possession
    NO  → Check Pass Criteria
         ↓
    New Possession Event
         ↓
    Update Statistics
```

#### 4.3 Possession Smoothing

To avoid flickering:

**Minimum Possession Duration:**
- Player must be nearest for 5+ consecutive frames to count as possessor
- Prevents random noise from triggering possession changes

**Smoothing Window:**
- Keep possession history for last 15 frames
- Smooth possession team using majority vote

#### 4.4 Pass Detection

**Pass Criteria:**
```
Pass detected IF:
    1. Possession changed from Player A to Player B
    2. Player A and B are on different teams (optional)
    3. Distance between A and B > min_pass_distance (200px)
    4. Time between possessions < max_pass_duration (50 frames / 2 seconds)
```

**Pass Event Structure:**
```python
{
    'frame_idx': int,
    'from_player_id': int,
    'to_player_id': int,
    'from_team': 0 or 1,
    'to_team': 0 or 1,
    'distance': float,           # Distance between players (px)
    'duration_frames': int,      # Frames between possessions
    'successful': bool,          # True if same team
}
```

#### 4.5 Real-Time Statistics

**Per Frame Output:**
```python
{
    'possessor_id': int or None,         # Current player with ball
    'possessor_team': 0 or 1 or None,    # Current possessing team
    'possession_duration_sec': float,     # How long current player has ball
    'team_possession_pct': {              # Overall possession percentages
        'team_0': 45.2,
        'team_1': 54.8
    },
    'recent_pass': {...} or None,        # Pass info if just detected
}
```

#### 4.6 Statistics Tracking

**Team-Level:**
- Total possession frames per team
- Possession percentage (frames with ball / total frames)
- Possession time in seconds
- Number of passes (successful + unsuccessful)

**Player-Level:**
- Individual possession frames per player
- Individual possession time
- Passes made by each player

#### 4.7 Computation Flow

```python
# Pseudo-code
for each frame:
    if ball_lost:
        possessor = None
    else:
        # Find nearest player
        nearest = min(players, key=lambda p: distance(p, ball))
        
        if distance(nearest, ball) <= radius:
            if nearest == current_possessor:
                # Continue possession
                possession_duration += 1
            else:
                # Check for pass
                if meets_pass_criteria(current, nearest):
                    record_pass(current, nearest)
                
                # New possession
                current_possessor = nearest
                possession_duration = 0
        else:
            possessor = None
    
    # Update statistics
    if possessor:
        team_frames[possessor.team] += 1
        player_frames[possessor.id] += 1
    
    # Compute percentages
    total_frames = sum(team_frames.values())
    for team in teams:
        team_pct[team] = 100 * team_frames[team] / total_frames
```

#### 4.8 Key Parameters

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `possession_radius_px` | Max distance for possession | 150 px |
| `min_frames_for_possession` | Min frames to count | 5 |
| `smoothing_window_frames` | Possession smoothing window | 15 |
| `min_pass_distance_px` | Min distance to count as pass | 200 px |
| `max_pass_duration_frames` | Max time between passes | 50 |

#### 4.9 Rendering

**Visual Indicators:**
- **Possession Highlight**: Circle around player with ball (team color)
- **Possession Bar**: Top overlay showing team percentages
- **Pass Arrow**: Animated arrow showing recent pass direction

**Text Overlays:**
- Current possessor name/ID
- Possession duration
- Team possession percentages

**Implementation**: [`src/possession_analyzer.py`](src/possession_analyzer.py)

---

## Integration Flow

### Complete Pipeline

```
Video Frame
    ↓
┌─────────────────────────────────────┐
│ MODULE 1: PLAYER DETECTION (YOLO)  │
│ Input:  Frame (BGR image)           │
│ Output: Detections [bbox, class]   │
└──────────┬──────────────────────────┘
           ↓
    ┌──────┴──────┐
    │   Split     │
    ├──────┬──────┤
    ↓      ↓      ↓
 Players Ball  Refs
    │      │      │
    ↓      ↓      ↓
┌───────────────────────────────────┐
│ People Tracker (ByteTrack)        │ 
│ Output: Tracks [id, bbox, class]  │
└──────────┬────────────────────────┘
           │
           ↓
┌─────────────────────────────────────┐
│ MODULE 2: BALL DETECTION/PREDICT   │
│ Input:  Ball detections + history  │
│ Output: Ball state [center, vel]   │
└──────────┬──────────────────────────┘
           │
           ↓
┌─────────────────────────────────────┐
│ MODULE 3: TEAM CLASSIFICATION      │
│ Input:  Player tracks + frame      │
│ Output: Tracks with team_id        │
└──────────┬──────────────────────────┘
           │
           ↓
┌─────────────────────────────────────┐
│ MODULE 4: POSSESSION ANALYSIS      │
│ Input:  Tracks + ball state        │
│ Output: Possession stats           │
└──────────┬──────────────────────────┘
           ↓
        Renderer
           ↓
     Output Video
```

---

## Configuration

All modules are configurable via `config.yaml`:

```yaml
# Module 1: Detection
detector:
  model_path: 'data/trained_models/yolo_football_best.pt'
  conf_player: 0.4
  conf_ball: 0.15
  # ...

# Module 2: Ball Tracking
ball_tracker:
  track_buffer: 15
  max_displacement_px_per_frame: 80
  reacquire:
    enabled: true
    roi_scale: 4.0
  # ...

# Module 3: Team Classification
team_color:
  enabled: true
  warmup_frames: 200
  ema_alpha: 0.2
  # ...

# Module 4: Possession
possession:
  enabled: true
  possession_radius_px: 150
  pass_detection_enabled: true
  # ...
```

---

## Performance Optimization

### Speed vs Accuracy Trade-offs

| Component | Fast | Balanced | Accurate |
|-----------|------|----------|----------|
| YOLO Model | yolo11n | yolo11m | yolo11x |
| Input Size | 640 | 1280 | 1280 |
| Conf Threshold | 0.5 | 0.4 | 0.3 |
| Track Buffer | 10 | 15 | 20 |
| Warmup Frames | 100 | 200 | 300 |

### Hardware Requirements

**Minimum (CPU):**
- 4-core CPU
- 8GB RAM
- ~0.5-1 FPS processing speed

**Recommended (GPU):**
- NVIDIA GPU with 6GB+ VRAM
- 16GB RAM
- ~25-30 FPS processing speed (real-time)

---

## Troubleshooting

### Common Issues

**1. Ball Detection Failures:**
- Lower `conf_ball` threshold (0.15 → 0.10)
- Increase `track_buffer` (15 → 25)
- Enable reacquisition

**2. Team Misclassification:**
- Increase `warmup_frames` (200 → 500)
- Adjust grass mask HSV ranges for different field colors
- Check if jersey colors are too similar

**3. Possession Flickering:**
- Increase `possession_radius_px` (150 → 200)
- Increase `smoothing_window_frames` (15 → 30)
- Increase `min_frames_for_possession` (5 → 10)

---

## References

- **YOLO**: [Ultralytics YOLOv8/YOLO11](https://github.com/ultralytics/ultralytics)
- **Kalman Filter**: [FilterPy Library](https://github.com/rlabbe/filterpy)
- **ByteTrack**: [Multi-Object Tracking](https://github.com/ifzhang/ByteTrack)
- **Color Spaces**: [OpenCV Color Conversions](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html)

---

**Last Updated**: December 30, 2025
