# Football Video Analysis System

A comprehensive system for analyzing broadcast football videos with multi-class detection and tracking of players, ball, referees, and goalkeepers, featuring team color-based assignment and temporal stability.

## Features

- **Multi-class Detection**: YOLO-based detection of 4 classes (player, ball, referee, goalkeeper)
- **Robust Tracking**: 
  - ByteTrack for people tracking with occlusion handling
  - Kalman Filter-based ball tracker with reacquisition
- **Team Color Assignment**: Color-based team identification with temporal smoothing
- **Layer-based Rendering**: Toggle visualization of different object types
- **Comprehensive Logging**: Export to JSON, JSONL, and CSV formats

## System Architecture

```
┌─────────────┐
│ Video Input │
└──────┬──────┘
       │
       v
┌─────────────┐
│  Detector   │ (YOLO multi-class)
│  4 classes  │
└──────┬──────┘
       │
       ├──────────────┬────────────────┐
       v              v                v
┌──────────┐   ┌────────────┐  ┌──────────┐
│  People  │   │    Ball    │  │  Team    │
│ Tracker  │   │  Tracker   │  │ Assigner │
│(ByteTrack)   │ (Kalman+   │  │ (Color   │
│          │   │ Reacquire) │  │  Based)  │
└──────┬───┘   └──────┬─────┘  └────┬─────┘
       │              │             │
       └──────────────┴─────────────┘
                      │
                      v
              ┌───────────────┐
              │   Renderer    │
              │ (4 layers +   │
              │  team colors) │
              └───────┬───────┘
                      │
              ┌───────┴────────┐
              v                v
       ┌─────────────┐  ┌──────────┐
       │Output Video │  │   Logs   │
       │   (MP4)     │  │(JSON/CSV)│
       └─────────────┘  └──────────┘
```

## 📁 Cấu trúc thư mục (Updated)

```
football_heatmap/
├── config.yaml                    # Main configuration file
├── main.py                        # Main inference script
├── requirements.txt               # Python dependencies
├── README.md                      # Documentation
│
├── src/                          # Source code modules
│   ├── video_io.py               # Video reading/writing
│   ├── detector.py               # YOLO detector
│   ├── people_tracker.py         # ByteTrack for people
│   ├── ball_tracker.py           # Kalman filter for ball
│   ├── team_assigner.py          # Color-based team assignment
│   ├── renderer.py               # Visualization
│   ├── logger.py                 # Data export
│   └── utils.py                  # Utility functions
│
├── model/                        # Model weights
│   ├── detector/
│   │   └── people_ball_4cls.pt  # YOLO weights (you provide)
│   ├── tracker/
│   ├── model_tiny.py            # (Legacy) BallRefinerNet
│   └── ball_detector/           # (Legacy) Trained weights
│
├── input/                       # Input videos and datasets
│   ├── videos/
│   └── SoccerNet/              # Dataset for training
│
└── output/                     # Generated outputs
    ├── videos/                # Rendered videos
    ├── logs/                  # Tracking logs (JSON/CSV)
    └── debug/                 # Debug outputs (optional)
```

## 🚀 Quick Start (New Pipeline)

### 1. Cài đặt
```bash
# Install PyTorch (choose appropriate version for your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 2. Chuẩn bị model YOLO
```bash
# Place your trained YOLO model (4 classes) at:
# model/detector/people_ball_4cls.pt
# Or train a new one (see Training section below)
```

### 3. Chạy inference

**Regular Video:**
```bash
# Basic usage
python main.py input/videos/match.mp4

# With fast preset (lower quality, faster)
python main.py input/videos/match.mp4 --preset fast

# With accurate preset (higher quality, slower)
python main.py input/videos/match.mp4 --preset accurate
```

**SoccerNet Dataset:**
```bash
# Process a specific sequence
python process_soccernet.py --sequence SNMOT-116

# Process all test sequences
python process_soccernet.py --split test

# Process first 5 sequences with fast preset
python process_soccernet.py --split test --max-sequences 5 --preset fast

# Process train split
python process_soccernet.py --split train
```

### 4. Kết quả
```
output/
├── videos/
│   └── match_overlay.mp4          # Video with overlays
└── logs/
    ├── match_meta.json            # Metadata
    ├── match_frames.jsonl         # Per-frame tracking data
    ├── match_frames_summary.csv   # Frame summary statistics
    └── match_tracks.csv           # Track-level statistics
```

## 📖 Usage Guide

### Basic Commands

**Regular Videos:**
```bash
# Process a video with default config
python main.py input/videos/match.mp4

# Use fast preset (lower quality, faster)
python main.py input/videos/match.mp4 --preset fast

# Custom output name
python main.py input/videos/match.mp4 --output-name match_001
```

**SoccerNet Dataset:**
```bash
# Process a specific sequence
python process_soccernet.py --sequence SNMOT-116

# Process all test sequences
python process_soccernet.py --split test

# Process with fast preset
python process_soccernet.py --split test --preset fast --max-sequences 5
```

### Layer Control

```bash
# Disable specific layers
python main.py video.mp4 --no-referee --no-goalkeeper

# Only show players and ball
python main.py video.mp4 --no-referee --no-goalkeeper
```

## ⚙️ Configuration

Edit `config.yaml` to customize all parameters:

### Key Configuration Sections

#### Detector Settings
```yaml
detector:
  model_path: "model/detector/people_ball_4cls.pt"
  imgsz: 1280  # Higher = better for small ball detection
  class_conf:
    player: 0.25
    ball: 0.08      # Lower threshold for better ball recall
    referee: 0.30
    goalkeeper: 0.25
```

#### Ball Tracker
```yaml
ball_tracker:
  track_buffer: 15  # Frames to keep predicting
  max_displacement_px_per_frame: 80
  reacquire:
    enabled: true
    roi_scale: 4.0
```

#### Team Color Assignment
```yaml
team_color:
  enabled: true
  clustering:
    warmup_frames: 200  # Frames to collect color samples
    n_teams: 2
  per_track:
    vote_window_frames: 30  # Temporal smoothing
```

## 🎯 Training Your Own Model

### Dataset Preparation

1. **Format**: YOLO format with 4 classes
2. **Classes**: `0: player, 1: ball, 2: referee, 3: goalkeeper`

```yaml
# data.yaml
names:
  0: player
  1: ball
  2: referee
  3: goalkeeper

nc: 4
train: images/train
val: images/val
```

### Training Script

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # or yolo8x.pt
model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=1280,
    batch=16,
    name='football_4cls'
)
```

### Update Config

```yaml
detector:
  model_path: "runs/detect/football_4cls/weights/best.pt"
```

## 💡 Tips for Best Results

### Ball Detection
- Use **imgsz ≥ 1280** for small ball detection
- Lower ball confidence threshold (0.05-0.12)
- Enable **reacquisition** for handling missed detections
- Fine-tune motion gating based on your video resolution

### Team Color Assignment
- Ensure good lighting in videos
- **Warmup period** (200+ frames) collects stable color samples
- Adjust **jersey crop ROI** if players' shirts aren't captured well
- Tune **grass mask** parameters for different field colors

### Performance Optimization
- Use **fast preset** for initial testing
- Reduce `imgsz` if GPU memory is limited
- Disable team color assignment if not needed (significant speedup)
- Use GPU for inference (20-40 FPS typical on RTX 3080)

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Ball not detected | Lower `ball_conf`, increase `imgsz`, check training data |
| ID switches | Increase `track_buffer`, lower `match_thresh` |
| Wrong team colors | Adjust `jersey_crop` ROI, tune `grass_mask` HSV |
| Slow processing | Use `--preset fast`, reduce `imgsz`, disable team assignment |

## 📊 Output Format

### Video Output
- MP4 format with overlays
- Color-coded bounding boxes by team
- Ball trail visualization
- Layer toggles for different objects

### Log Files

**meta.json**: Video and processing metadata
```json
{
  "video": {"fps": 25, "width": 1920, "height": 1080},
  "total_frames": 1500,
  "total_tracks": 45
}
```

**frames.jsonl**: Per-frame tracking data (one JSON object per line)
```json
{"frame_idx": 0, "timestamp": 0.0, "tracks": [...], "ball": {...}}
```

**tracks.csv**: Track-level statistics
```csv
track_id,class,team_id,first_frame,last_frame,num_frames,duration_sec
1,player,team0,0,1200,1200,48.0
2,player,team1,5,1180,1175,47.0
```

## 📚 Old Pipeline (Legacy)

The previous ball-only tracking system is still available in `scripts/` directory:
- `run_tracking_enhanced.py`: Original tracking script
- Uses BallRefinerNet for local refinement

## 🔬 Advanced Features

### Debug Mode
```yaml
logger:
  debug:
    enabled: true
    save_jersey_crops: true
    save_ball_roi: true
```

### Custom Colors
```yaml
render:
  colors:
    team0: [0, 255, 0]      # BGR: Green
    team1: [255, 0, 0]      # BGR: Blue
```

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Ultralytics YOLO** for object detection
- **ByteTrack** for multi-object tracking
- **FilterPy** for Kalman filtering
- **SoccerNet** dataset for training data

---

**Made with ❤️ for football analytics**


### 5. Chạy tracking
```bash
cd ../scripts && python run_tracking_enhanced.py
```

## 🎯 Kết quả

**Performance:**
- ✅ Accuracy: **99%**
- ✅ Detection rate: 100%
- ✅ Speed: ~15-20 FPS (CPU)

**Output:** `output/videos/tracking_enhanced.mp4`

## 📝 Scripts chính

| Script | Mô tả |
|--------|-------|
| `scripts/dow.py` | Download SoccerNet dataset |
| `scripts/convert_mot_to_yolo.py` | Convert MOT → YOLO format |
| `model/prepare_crop_data.py` | Tạo training crops |
| `model/train_tiny.py` | Train BallRefinerNet |
| `scripts/run_tracking_enhanced.py` | Tracking test (best) |
| `scripts/run_tracking_test.py` | Full pipeline test |

## 🔧 Configuration

Chỉnh sửa paths trong các file:
- `model/train_tiny.py`: Training settings
- `scripts/run_tracking_enhanced.py`: Tracking settings

## 📄 License

MIT License
