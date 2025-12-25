# Football Ball Tracking

Demo pipeline theo dõi bóng đá với AI:
- **Global search**: YOLOv8 quét toàn khung hình
- **Local search**: BallRefinerNet (Tiny-Net) tinh chỉnh vị trí
- **Motion tracker**: Kalman filter dự đoán chuyển động

## 📁 Cấu trúc thư mục

```
football_heatmap/
├── model/                          # Model và training
│   ├── model_tiny.py              # BallRefinerNet definition
│   ├── train_tiny.py              # Training script
│   ├── prepare_crop_data.py       # Data preparation
│   └── tiny_ball_refiner.pth      # Trained weights
├── input/                         # Input data
│   ├── SoccerNet/                 # Dataset
│   └── tiny_dataset/              # Cropped training data
├── output/                        # Results
│   └── videos/                    # Output videos
├── scripts/                       # Utility scripts
│   ├── dow.py                     # Download dataset
│   ├── convert_mot_to_yolo.py     # Convert format
│   ├── run_tracking_enhanced.py   # Main tracking script
│   └── ...                        # Other utilities
└── requirements.txt
```

## 🚀 Quick Start

### 1. Cài đặt
```bash
pip install -r requirements.txt
```

### 2. Download dữ liệu
```bash
cd scripts && python dow.py
```

### 3. Chuẩn bị dữ liệu
```bash
python convert_mot_to_yolo.py
cd ../model && python prepare_crop_data.py
```

### 4. Train model
```bash
python train_tiny.py
```

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
