# 🎯 Training Workflow - Giải thích chi tiết

## 📊 Tổng quan hệ thống

Hệ thống training gồm **3 models** hoạt động độc lập nhưng bổ trợ cho nhau:

```
SoccerNet Dataset (videos)
         ↓
    [Data Preparation]
         ↓
    ┌────┴─────┬──────────┐
    ↓          ↓          ↓
YOLO       Team      Ball
Detector   Classifier Predictor
```

---

## 🔄 Workflow Chi Tiết

### **Bước 1: Data Preparation** ⏱️ ~5-10 phút/sequence

#### 1.1 Jersey Crops (cho Team Classifier)

**Input:** Video sequences từ SoccerNet
- `input/SoccerNet/tracking-2023/train/SNMOT-060/` (750 frames)
- `input/SoccerNet/tracking-2023/train/SNMOT-061/` (750 frames)
- ... (4 sequences train, 1 sequence val)

**Process:**
```
Sequence → Frame-by-frame processing:
  │
  ├─ 1. Load frame (1080x1920 BGR)
  │    
  ├─ 2. YOLO Detection
  │    ├─ YOLO11n pretrained (COCO classes)
  │    ├─ Map classes: "person" → "player", "sports ball" → "ball"
  │    └─ Threshold: player conf > 0.25
  │    
  ├─ 3. ByteTrack Tracking
  │    ├─ Track players across frames
  │    └─ Assign unique track_id cho mỗi player
  │    
  ├─ 4. Team Assignment (KMeans clustering)
  │    ├─ Crop jersey area (x: 30-70%, y: 10-55% của bbox)
  │    ├─ Extract color (LAB color space)
  │    ├─ Cluster thành 2 teams
  │    └─ Assign team_id (0 hoặc 1)
  │    
  └─ 5. Save Jersey Crop
       ├─ Resize to 224x224
       ├─ Save as JPG: data/jersey_crops/train/team0/SNMOT060_frame123_track5.jpg
       └─ Metadata: train_annotations.json
```

**Output:**
```
data/jersey_crops/
├── train/
│   ├── team0/           # 3281 images (áo team 0)
│   └── team1/           # 185 images (áo team 1)
├── val/
│   ├── team0/           # 590 images
│   └── team1/           # 44 images
├── train_annotations.json
└── val_annotations.json
```

**Example annotation:**
```json
{
  "image_path": "train/team0/SNMOT060_f100_t12.jpg",
  "team_label": 0,
  "sequence": "SNMOT-060",
  "frame_idx": 100,
  "track_id": 12
}
```

---

#### 1.2 Ball Trajectories (cho Ball Predictor)

**Process:**
```
Sequence → Full sequence processing:
  │
  ├─ 1. Detect ball mọi frame
  │    ├─ YOLO detect "sports ball" (class 32)
  │    ├─ Filter: conf > 0.08, geometry checks
  │    └─ Output: bbox [x1,y1,x2,y2]
  │    
  ├─ 2. Kalman Filter Tracking
  │    ├─ Track ball position (x,y) qua frames
  │    ├─ Handle occlusion (predict khi mất)
  │    └─ Status: 'detected' or 'predicted'
  │    
  └─ 3. Extract Trajectory
       ├─ Save positions: [[x,y,frame,visible], ...]
       ├─ Filter: cần ít nhất 10 frames visible
       └─ Save to JSON
```

**Output:**
```
data/ball_trajectories/
├── train_trajectories.json
└── val_trajectories.json
```

**Example trajectory:**
```json
{
  "sequence": "SNMOT-060",
  "positions": [
    [960.5, 540.2, 0, true],      # [x, y, frame_idx, visible]
    [962.1, 538.7, 1, true],
    [963.8, 537.1, 2, true],
    ...
  ],
  "total_frames": 488,
  "visible_frames": 208
}
```

---

### **Bước 2: Model Training** ⏱️ 2-4 giờ total

#### 2.1 YOLO Fine-tuning (Optional)

**Mục đích:** Improve detection cho football-specific scenarios

**Data:**
- Input: `input/SoccerNet/yolo_format_v3_smart/` (YOLO format)
- Classes: player, ball, referee, goalkeeper

**Training:**
```python
# Base: yolo11n.pt (COCO pretrained)
# Fine-tune 100 epochs
# Image size: 1280x1280
# Batch: 16
# Learning rate: 0.01 → 0.01 (cosine)
```

**Output:**
- `runs/train/yolo_football/weights/best.pt` (improved model)
- Metrics: mAP50, precision, recall

**Hiện tại:** SKIP vì COCO pretrained đã đủ tốt, chỉ train khi cần accuracy cao hơn

---

#### 2.2 Team Classifier Training

**Architecture:** ResNet18 (pretrained ImageNet)

**Data:**
- Train: 3466 jersey crops (80/20 split)
- Val: 634 crops
- Classes: 2 (team 0, team 1)

**Training process:**
```
Input: Jersey crop 224x224 RGB
   ↓
ResNet18 backbone (frozen early layers)
   ↓
Fully Connected Layer (512 → 2)
   ↓
Softmax
   ↓
Output: [P(team0), P(team1)]
```

**Hyperparameters:**
- Epochs: 50
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss: CrossEntropyLoss

**Augmentation:**
- Random crop
- Horizontal flip
- Color jitter (brightness, contrast)

**Output:**
- `runs/team_classifier/best_model.pth`
- Expected accuracy: >90%

---

#### 2.3 Ball Predictor Training

**Architecture:** LSTM (có thể switch sang Transformer)

**Data:**
- Train: Ball trajectories từ 4 sequences
- Val: 1 sequence
- Input: 10 frames (x,y positions)
- Output: Predict 5 frames ahead

**Model:**
```
Input: [x_t-9, y_t-9, ..., x_t, y_t]  (10 timesteps)
   ↓
LSTM Layer 1 (hidden_dim=128)
   ↓
LSTM Layer 2 (hidden_dim=128)
   ↓
Dropout (0.2)
   ↓
Fully Connected (128 → 10)
   ↓
Output: [x_t+1, y_t+1, ..., x_t+5, y_t+5]  (5 predictions)
```

**Training:**
- Epochs: 100
- Batch size: 128
- Learning rate: 0.001
- Loss: MSE (Mean Squared Error)

**Output:**
- `runs/ball_predictor/best_model.pth`
- Expected MSE: 50-100 pixels

---

## 📈 Training Status Hiện Tại

### Data đã chuẩn bị:

✅ **Jersey Crops:**
```
Train: 3466 samples (Team 0: 3281, Team 1: 185)
Val:    634 samples (Team 0: 590,  Team 1: 44)

Từ 5 sequences:
- SNMOT-060: 750 frames, ~800 crops
- SNMOT-061: 750 frames, ~900 crops  
- SNMOT-062: 750 frames, ~950 crops
- SNMOT-063: 750 frames, ~800 crops
- SNMOT-064: 750 frames (val), ~640 crops
```

✅ **Ball Trajectories:**
```
Val: 1 trajectory
- SNMOT-064: 488 frames, 208 visible (42.6%)

Train: Đang prepare thêm từ 4 sequences khác
```

### Models đang train:

1. **Team Classifier** → Ready to train
   - Data: ✅ 3466 train samples
   - Config: ✅ device=cpu, batch=32
   
2. **Ball Predictor** → Cần thêm data
   - Data: ⚠️ Chỉ có 1 trajectory (cần ít nhất 10-20)
   - Fix: Giảm threshold hoặc train với ít sequences hơn

3. **YOLO Fine-tuning** → Optional
   - Skip vì COCO pretrained đã tốt

---

## 🔧 Technical Details

### **Class Mapping (Quan trọng!)**

YOLO11n pretrained dùng **COCO classes** (80 classes):
```python
COCO → Custom mapping:
  0: "person"       → "player"
  32: "sports ball" → "ball"
```

Code tự động detect COCO model và map classes:
```python
if len(model.names) == 80:  # COCO model
    cls_name = COCO_TO_CUSTOM.get(cls_id)
```

### **Team Assignment Algorithm**

1. **Phase 1:** Collect jersey colors từ 100-200 frames đầu
2. **KMeans Clustering:** Group colors thành 2 clusters (LAB space)
3. **Phase 2:** Assign team_id cho mỗi detection
4. **Persistence:** Track duy trì team_id across frames

### **Ball Tracking**

- **Kalman Filter:** Predict position khi ball bị occluded
- **Reacquisition:** Tìm lại ball trong radius khi lost
- **Status:**
  - `detected`: Ball được YOLO detect
  - `predicted`: Kalman predict (không có detection)
  - `lost`: Quá lâu không thấy (>15 frames)

---

## 🚀 Chạy Training

### Quick Start:
```bash
# Data preparation only (test)
python training/train_all.py --prepare-data-only --max-sequences 2

# Full training với 5 sequences
chcp 65001
python training/train_all.py --all --max-sequences 5

# Train từng model riêng
python training/train_team_classifier.py --config training/training_config.yaml
python training/train_ball_predictor.py --config training/training_config.yaml
python training/train_yolo.py --config training/training_config.yaml
```

### Monitor Progress:
```bash
# Check data
python check_data.py

# View logs
ls runs/team_classifier/
ls runs/ball_predictor/
ls runs/train/yolo_football/

# TensorBoard (YOLO only)
tensorboard --logdir runs/train
```

---

## 💡 Tips & Troubleshooting

### **Vấn đề thường gặp:**

1. **No data extracted**
   - ✅ Fixed: Iterator issue, team clustering
   - ✅ Fixed: COCO class mapping

2. **Ball not detected**
   - ✅ Fixed: Map "sports ball" (class 32) → "ball"
   - Threshold: conf > 0.08 (khá thấp)

3. **Training slow**
   - CPU only (PyTorch 2.9.0+cpu)
   - Giảm batch size nếu out of memory
   - Reduce epochs cho test nhanh

4. **Imbalanced team data**
   - Team 0: 3281 samples
   - Team 1: 185 samples (15:1 ratio)
   - Fix: Weighted loss hoặc data augmentation

### **Optimization:**

- **Speed up data prep:** Sample fewer frames (hiện tại: 100/sequence)
- **Improve team classifier:** More augmentation, balanced sampling
- **Better ball prediction:** Collect more trajectories, use Transformer

---

## 📚 References

- **SoccerNet Dataset:** https://www.soccer-net.org/
- **YOLO11:** https://docs.ultralytics.com/
- **ByteTrack:** https://github.com/ifzhang/ByteTrack
- **ResNet:** https://pytorch.org/vision/main/models/resnet.html

---

**Tóm tắt:**
- ✅ Data preparation hoạt động tốt (3466 jersey crops, 1 ball trajectory)
- ✅ Team classifier ready to train
- ⚠️ Ball predictor cần thêm data (process thêm sequences)
- 🔧 Tất cả bugs đã fix: iterator, COCO mapping, device config
- ⏱️ Training time: ~2-4 giờ với CPU, 5 sequences
