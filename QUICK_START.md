# 🚀 HƯỚNG DẪN SỬ DỤNG NHANH

## Cấu trúc thư mục mới (đã tổ chức lại)

```
football_heatmap/
├── 📁 model/              # Model và training
│   ├── model_tiny.py
│   ├── train_tiny.py
│   ├── prepare_crop_data.py
│   └── tiny_ball_refiner.pth (4MB)
│
├── 📁 input/              # Dữ liệu đầu vào
│   ├── SoccerNet/         # Dataset gốc
│   └── tiny_dataset/      # Data đã crop
│
├── 📁 output/             # Kết quả
│   └── videos/            # Video outputs
│
└── 📁 scripts/            # Scripts tiện ích
    ├── dow.py                    # Download dataset
    ├── convert_mot_to_yolo.py    # Convert format
    ├── run_tracking_enhanced.py  # ⭐ Tracking chính
    └── ...
```

## ⚡ Chạy ngay (đã train xong)

```bash
# Chạy tracking trên test data
cd scripts
python run_tracking_enhanced.py
```

Output: `output/videos/tracking_enhanced.mp4` (99% accuracy)

## 🔄 Quy trình đầy đủ (từ đầu)

### 1. Download dataset
```bash
cd scripts
python dow.py
```

### 2. Convert format
```bash
python convert_mot_to_yolo.py
```

### 3. Chuẩn bị training data
```bash
cd ../model
python prepare_crop_data.py
```

### 4. Train model
```bash
python train_tiny.py
```
→ Output: `model/tiny_ball_refiner.pth`

### 5. Test tracking
```bash
cd ../scripts
python run_tracking_enhanced.py
```
→ Output: `output/videos/tracking_enhanced.mp4`

## 🎯 Các script test khác

```bash
cd scripts

# Test đơn giản
python test_video.py

# Test với ground truth
python test_with_gt.py

# Full tracking pipeline
python run_tracking_test.py

# Smart tracking với YOLO
python smart_tracking.py
```

## 📊 Kết quả hiện tại

✅ Model đã train: `model/tiny_ball_refiner.pth`
✅ Videos đã tạo:
  - tracking_enhanced.mp4 (99% accuracy) ⭐
  - tracking_output.mp4
  - test_with_gt.mp4
  - test_output.mp4

## 🔧 Troubleshooting

### Import error?
```bash
# Chắc chắn chạy từ thư mục scripts
cd scripts
python run_tracking_enhanced.py
```

### Path error?
Tất cả paths đã được cập nhật:
- `input/SoccerNet/...`
- `output/videos/...`
- `model/...`

### Model not found?
Model weights ở: `model/tiny_ball_refiner.pth` (4MB)

## 📝 Ghi chú

- Tất cả scripts trong `scripts/` đã được cập nhật paths
- Model definition trong `model/model_tiny.py`
- Chạy scripts từ thư mục `scripts/` để đảm bảo paths đúng
- Output videos luôn ở `output/videos/`
