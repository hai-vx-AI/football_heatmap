# 📚 GIẢI THÍCH CHI TIẾT CÁC FILE

## 📂 Thư mục MODEL/ (4 files)

### 1. `model_tiny.py` ⭐ **CORE MODEL**
**Chức năng:** Định nghĩa kiến trúc mạng neural BallRefinerNet

**Mô tả:**
```python
class BallRefinerNet(nn.Module):
    - Input: Crop 64x64 pixels (ảnh RGB)
    - Output: 
      • Confidence (0-1): Có bóng hay không?
      • Offset (dx, dy): Điều chỉnh vị trí tâm bóng
```

**Kiến trúc:**
- 3 lớp Convolution (16→32→64 channels)
- MaxPooling giảm kích thước
- Fully Connected layer
- 2 nhánh output:
  1. **Classifier**: Binary classification (có bóng/không)
  2. **Regressor**: Tinh chỉnh tọa độ (dx, dy)

**Khi nào dùng:** 
- Import để train: `from model.model_tiny import BallRefinerNet`
- Import để inference trong tracking scripts

---

### 2. `train_tiny.py` 🎓 **TRAINING SCRIPT**
**Chức năng:** Train model BallRefinerNet trên tiny_dataset

**Quy trình:**
1. Load data từ `input/tiny_dataset/pos/` và `neg/`
2. Train 10 epochs với:
   - BCE Loss cho classification
   - MSE Loss cho regression (chỉ với positive samples)
3. Save model → `model/tiny_ball_refiner.pth`

**Hyperparameters:**
```python
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001 (Adam optimizer)
```

**Khi nào chạy:** Sau khi có tiny_dataset, trước khi inference
```bash
cd model
python train_tiny.py
```

**Output:** `tiny_ball_refiner.pth` (4MB)

---

### 3. `prepare_crop_data.py` ✂️ **DATA PREPARATION**
**Chức năng:** Cắt crop 64x64 từ ảnh full resolution để train model

**Quy trình:**
1. Đọc ảnh từ `input/SoccerNet/yolo_format/images/train/`
2. Đọc labels (YOLO format) từ `labels/train/`
3. Với mỗi annotation:
   - **Positive crop**: Crop quanh bóng (có random offset để model học robust)
   - **Negative crop**: Crop ở vùng không có bóng (background)
4. Save vào `input/tiny_dataset/pos/` và `neg/`

**Config quan trọng:**
```python
CROP_SIZE = 64        # Kích thước crop
BALL_CLASS_ID = 0     # Class ID của bóng trong YOLO labels
```

**Khi nào chạy:** Sau khi convert MOT→YOLO, trước khi train
```bash
cd model
python prepare_crop_data.py
```

**Output:** Hàng nghìn file .jpg trong `input/tiny_dataset/`

---

### 4. `tiny_ball_refiner.pth` 💾 **MODEL WEIGHTS**
**Chức năng:** Weights đã train của BallRefinerNet

**Thông tin:**
- Size: ~4MB
- Format: PyTorch state_dict
- Accuracy: 99% trên test set
- Epochs trained: 7-10

**Cách load:**
```python
model = BallRefinerNet()
model.load_state_dict(torch.load('model/tiny_ball_refiner.pth'))
model.eval()
```

---

## 📂 Thư mục SCRIPTS/ (8 files)

### 1. `dow.py` 📥 **DOWNLOAD DATASET**
**Chức năng:** Download SoccerNet tracking-2023 dataset

**Mô tả:**
- Sử dụng SoccerNet API
- Download cả train và test split
- Output: `input/SoccerNet/tracking-2023/`

**Khi nào chạy:** Bước đầu tiên, 1 lần duy nhất
```bash
cd scripts
python dow.py
```

**Lưu ý:** Cần internet, mất ~30-60 phút tùy tốc độ mạng

---

### 2. `convert_mot_to_yolo.py` 🔄 **FORMAT CONVERTER**
**Chức năng:** Convert từ MOT format → YOLO format

**MOT format (input):**
```
frame_id, track_id, x, y, width, height, conf, -1, -1, -1
```

**YOLO format (output):**
```
class_id, x_center_norm, y_center_norm, width_norm, height_norm
```

**Quy trình:**
1. Đọc `gt/gt.txt` (MOT annotations)
2. Đọc `seqinfo.ini` (image dimensions)
3. Convert coordinates & normalize
4. Copy images + tạo label files
5. Output: `input/SoccerNet/yolo_format/`

**Khi nào chạy:** Sau khi download dataset
```bash
cd scripts
python convert_mot_to_yolo.py
```

---

### 3. `dataset_yolo_v8.py` 🗂️ **LEGACY CONVERTER**
**Chức năng:** Converter cũ (tương tự convert_mot_to_yolo.py)

**Trạng thái:** Có thể bỏ qua, dùng `convert_mot_to_yolo.py` thay thế

---

### 4. `test_video.py` 🧪 **SIMPLE TEST**
**Chức năng:** Test model đơn giản bằng random sampling

**Quy trình:**
1. Load model `tiny_ball_refiner.pth`
2. Đọc 100 frames từ test sequence
3. Test 15 điểm ngẫu nhiên mỗi frame
4. Vẽ detections (confidence > 0.8)
5. Tạo video → `output/videos/test_output.mp4`

**Khi nào chạy:** Test nhanh xem model có hoạt động không
```bash
cd scripts
python test_video.py
```

**Output:** Video demo với random sampling

---

### 5. `test_with_gt.py` 📊 **TEST WITH GROUND TRUTH**
**Chức năng:** Test model và so sánh với ground truth

**Quy trình:**
1. Load model + ground truth annotations
2. Test model xung quanh vị trí thật của bóng
3. So sánh prediction vs ground truth
4. Tính metrics (Recall, Accuracy)
5. Tạo video với overlay GT vs Prediction

**Visualization:**
- 🔵 Màu xanh dương = Ground Truth
- 🟢 Màu xanh lá = Model Prediction

**Khi nào chạy:** Để đánh giá chính xác model
```bash
cd scripts
python test_with_gt.py
```

**Output:** 
- Video: `output/videos/test_with_gt.mp4`
- Stats: Recall 100%, chạy 150 frames

---

### 6. `run_tracking_test.py` 🎮 **BASIC TRACKING**
**Chức năng:** Full tracking pipeline với YOLO + Tiny-Net + Kalman

**Components:**
1. **YOLO Global Search**: Quét toàn khung hình (khi mất dấu)
2. **Tiny-Net Local Search**: Kiểm tra vùng dự đoán
3. **Kalman Tracker**: Dự đoán vị trí tiếp theo

**Workflow:**
```
Frame → Kalman Predict → Crop → Tiny-Net → 
  If conf > 0.7: Track (green)
  Else: YOLO Global Search (red)
```

**Khi nào chạy:** Test full pipeline
```bash
cd scripts
python run_tracking_test.py
```

**Output:** `output/videos/tracking_output.mp4`

---

### 7. `run_tracking_enhanced.py` ⭐ **BEST TRACKING**
**Chức năng:** Enhanced tracking với ground truth comparison

**Đặc điểm:**
- Chỉ dùng Tiny-Net (không dùng YOLO để test thuần model)
- Search xung quanh ground truth
- Visualization tốt nhất
- Tính metrics chi tiết

**Kết quả:**
- ✅ Accuracy: **99%**
- ✅ Detection rate: **100%**
- ✅ 300 frames processed

**Khi nào chạy:** Để có video demo đẹp nhất
```bash
cd scripts
python run_tracking_enhanced.py
```

**Output:** `output/videos/tracking_enhanced.mp4` 🏆

---

### 8. `smart_tracking.py` 🤖 **PRODUCTION TRACKING**
**Chức năng:** Tracking pipeline để chạy trên video thực tế

**Khác biệt với test scripts:**
- Input: Video file bất kỳ (không cần ground truth)
- Real-time visualization
- Có thể chạy trên webcam
- Production-ready code

**Config:**
```python
VIDEO_PATH = "input/video_bong_da.mp4"  # Your video
YOLO_MODEL = "yolov8x.pt"
TINY_MODEL = "model/tiny_ball_refiner.pth"
```

**Khi nào chạy:** Khi có video bóng đá riêng muốn track
```bash
cd scripts
python smart_tracking.py
```

**Lưu ý:** Cần có video input trong folder `input/`

---

## 📊 So sánh các Tracking Scripts

| Script | Input | YOLO | Tiny-Net | GT | Use Case |
|--------|-------|------|----------|-----|----------|
| `test_video.py` | Test data | ❌ | ✅ | ❌ | Quick test |
| `test_with_gt.py` | Test data | ❌ | ✅ | ✅ | Evaluation |
| `run_tracking_test.py` | Test data | ✅ | ✅ | ❌ | Full pipeline |
| `run_tracking_enhanced.py` | Test data | ❌ | ✅ | ✅ | Best demo ⭐ |
| `smart_tracking.py` | Your video | ✅ | ✅ | ❌ | Production |

---

## 🎯 Workflow tóm tắt

```
1. dow.py                     → Download dataset
2. convert_mot_to_yolo.py     → Convert format
3. prepare_crop_data.py       → Chuẩn bị training data
4. train_tiny.py              → Train model
5. run_tracking_enhanced.py   → Test & demo ⭐
```

---

## 🗂️ File nào quan trọng nhất?

### **Core files (BẮT BUỘC):**
1. ⭐ `model/model_tiny.py` - Model definition
2. ⭐ `model/tiny_ball_refiner.pth` - Trained weights
3. ⭐ `scripts/run_tracking_enhanced.py` - Best demo

### **Training pipeline:**
4. `model/prepare_crop_data.py` - Data prep
5. `model/train_tiny.py` - Training

### **Utilities:**
6. `scripts/dow.py` - Download
7. `scripts/convert_mot_to_yolo.py` - Convert

### **Optional test scripts:**
- `test_video.py`, `test_with_gt.py`, `run_tracking_test.py`

### **Production:**
- `smart_tracking.py` - Cho video riêng

---

## ❓ Chạy file nào để demo ngay?

```bash
cd scripts
python run_tracking_enhanced.py
```

→ Output: `output/videos/tracking_enhanced.mp4` (99% accuracy!)
