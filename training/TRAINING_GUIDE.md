# Training System - Complete Guide

Hệ thống training hoàn chỉnh với 3 models cho phân tích bóng đá.

## ✅ Hệ thống đã sẵn sàng!

Chạy test để verify:
```bash
python training/test_setup.py
```

## 🚀 Các bước thực hiện

### Bước 1: Chuẩn bị dữ liệu (Test nhỏ)
```bash
# Test với 2 sequences để kiểm tra
python training/train_all.py --prepare-data-only --max-sequences 2
```

Quá trình này sẽ:
- Extract jersey crops từ SoccerNet sequences
- Extract ball trajectories 
- Tạo train/val split (80/20)
- Lưu vào `data/jersey_crops` và `data/ball_trajectories`

⏱️ **Thời gian**: ~5-10 phút cho 2 sequences

### Bước 2: Train từng model riêng

#### 2a. Train YOLO Detector
```bash
python training/train_yolo.py --config training/training_config.yaml
```
- Fine-tune YOLO11n trên SoccerNet format data
- Sử dụng data đã có: `input/SoccerNet/yolo_format_v3_smart/`
- Output: `runs/train/yolo_football/weights/best.pt`

⏱️ **Thời gian**: ~2-3 giờ (100 epochs, GPU)

#### 2b. Train Team Classifier
```bash
# Cần data từ Bước 1
python training/train_team_classifier.py --config training/training_config.yaml
```
- Train ResNet18 phân loại team từ jersey
- Sử dụng data: `data/jersey_crops/`
- Output: `runs/team_classifier/best_model.pth`

⏱️ **Thời gian**: ~30-60 phút (50 epochs, GPU)

#### 2c. Train Ball Predictor
```bash
# Cần data từ Bước 1
python training/train_ball_predictor.py --config training/training_config.yaml
```
- Train LSTM dự đoán trajectory
- Sử dụng data: `data/ball_trajectories/`
- Output: `runs/ball_predictor/best_model.pth`

⏱️ **Thời gian**: ~1-2 giờ (100 epochs, GPU)

### Bước 3: Train tất cả (Full Pipeline)
```bash
# Prepare data + train tất cả models
python training/train_all.py --all --max-sequences 20
```

⏱️ **Thời gian**: ~5-7 giờ (full pipeline, GPU)

## 📊 Monitoring Training

### YOLO
```bash
# TensorBoard
tensorboard --logdir runs/train

# Xem results
ls runs/train/yolo_football/
# - weights/best.pt, last.pt
# - results.png (metrics plot)
# - confusion_matrix.png
```

### Team Classifier & Ball Predictor
```bash
# Checkpoints saved every N epochs
ls runs/team_classifier/
# - best_model.pth (best validation accuracy)
# - checkpoint_epoch_10.pth, checkpoint_epoch_20.pth...

ls runs/ball_predictor/
# - best_model.pth (best validation loss)
# - checkpoint_epoch_20.pth, checkpoint_epoch_40.pth...
```

## ⚙️ Cấu hình

Edit `training/training_config.yaml`:

```yaml
# YOLO
yolo_training:
  base_model: "yolo11n.pt"  # n/s/m/x (size)
  epochs: 100
  batch: 16
  imgsz: 1280
  lr0: 0.01

# Team Classifier
team_classifier:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  pretrained: true

# Ball Predictor
ball_predictor:
  model_type: "lstm"  # or "transformer"
  epochs: 100
  batch_size: 128
  sequence_length: 10
  prediction_horizon: 5
```

## 🔧 Sử dụng Trained Models

Sau khi train xong, update `config.yaml`:

```yaml
detector:
  model_path: "runs/train/yolo_football/weights/best.pt"

team_color:
  # TODO: Add classifier integration code
  use_trained_classifier: true
  classifier_path: "runs/team_classifier/best_model.pth"

ball_tracker:
  # TODO: Add predictor integration code  
  use_predictor: true
  predictor_path: "runs/ball_predictor/best_model.pth"
```

Sau đó chạy inference như bình thường:
```bash
python main.py input/video.mp4
```

## 🐛 Troubleshooting

### Out of Memory
- Giảm batch size
- Dùng model nhỏ hơn (yolo11n thay vì yolo11x)
- Giảm image size

### Không có data sau prepare
- Check: `ls data/jersey_crops/train/`
- Check: `cat data/ball_trajectories/train_trajectories.json`
- Tăng `--max-sequences`
- Giảm threshold trong code (đã set min_visible=20)

### Training chậm
- Enable GPU: `device: "cuda"` trong config
- Tăng batch size nếu có RAM
- Giảm epochs cho test nhanh

### Data preparation chậm
Đây là bình thường vì:
- Phải chạy detection trên mỗi frame
- Phải track players/ball
- Với 2 sequences (~1500 frames) cần ~5-10 phút

Có thể:
- Giảm `samples_per_sequence` trong code
- Skip frames (sample every N frames)

## 📈 Expected Results

### YOLO Fine-tuning
- **Baseline** (yolo11n pretrained): mAP50 ~0.45-0.55
- **After fine-tuning**: mAP50 ~0.60-0.70 (better ball detection)

### Team Classifier
- **Target accuracy**: >90% on validation set
- ResNet18 với jersey crops rất hiệu quả

### Ball Predictor
- **LSTM**: MSE loss ~50-100 pixels
- **Transformer**: MSE loss ~30-80 pixels (better)
- Useful khi ball bị occluded

## 💡 Tips

1. **Start small**: Test với 2-5 sequences trước
2. **Monitor GPU usage**: `nvidia-smi -l 1`
3. **Save checkpoints**: Đã config auto-save
4. **Validate early**: Check sau 10-20 epochs xem có learn không
5. **Data quality**: Jersey crops và trajectories quan trọng hơn quantity

## 📚 Chi tiết

Xem thêm:
- [training/README.md](training/README.md) - Hướng dẫn chi tiết từng model
- [README.md](README.md) - Hướng dẫn chính của project
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide

## ✨ Summary

Training pipeline đã hoàn chỉnh với:
- ✅ 3 models (YOLO, Team Classifier, Ball Predictor)
- ✅ Data preparation scripts
- ✅ Training scripts với checkpointing
- ✅ Configuration files
- ✅ Error handling
- ✅ Master script để chạy tất cả

Bắt đầu với:
```bash
python training/test_setup.py  # Verify setup
python training/train_all.py --prepare-data-only --max-sequences 2  # Test
python training/train_all.py --all --max-sequences 20  # Full training
```
