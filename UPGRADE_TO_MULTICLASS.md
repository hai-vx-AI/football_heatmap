# 🎯 NÂNG CẤP MODEL: MULTI-CLASS CLASSIFICATION

## ❌ Vấn đề hiện tại

Model **BallRefinerNet** chỉ làm **binary classification**:
- ✅ Có object / ❌ Không có object
- **KHÔNG phân biệt** được:
  - Ball vs Player
  - Team A vs Team B

### Ví dụ vấn đề:
```
Input: Crop 64x64 chứa cầu thủ
Output: conf=0.95 (có object) ✓
        Nhưng không biết là ball hay player? ❌
```

---

## ✅ Giải pháp: Multi-Class Model

### **Class definition:**
```
0 = Background  (nền sân)
1 = Ball        (quả bóng)
2 = Team A      (đội A - áo màu 1)
3 = Team B      (đội B - áo màu 2)
```

### **Model output:**
```python
class_probs = [0.05, 0.90, 0.03, 0.02]  # Softmax probabilities
# → Predicted: Ball (class 1) với 90% confidence
```

---

## 📊 So sánh Model Cũ vs Mới

| Tiêu chí     | BallRefinerNet (Cũ) | MultiClassObjectDetector (Mới) |
|----------    |---------------------|--------------------------------|
| **Classes**  | 2 (bg/object)       | 4 (bg/ball/team_a/team_b)      |  
| **Output**   | Binary conf (0-1)   | Class probabilities (softmax)  |
| **Loss**     | BCE Loss            | CrossEntropy Loss              |
| **Use case** | Ball detection only | Ball + Player classification   |
| **Accuracy** | 99% (binary)        | TBD (multi-class)              | 

---

## 🔧 Các file mới đã tạo

### 1. `model/model_multiclass.py`
**Model architecture mới:**
```python
class MultiClassObjectDetector(nn.Module):
    - 4 Conv layers (32→64→128→256)
    - BatchNorm + Dropout
    - Multi-class classifier (4 classes)
    - Bbox regressor (dx, dy, dw, dh)
```

### 2. `model/prepare_multiclass_data.py`
**Data preparation với class labeling:**
- Phân loại ball vs player dựa vào kích thước bbox
- Phân chia team A vs B dựa vào vị trí (left/right)
- Output: `input/multiclass_dataset/`
  - `background/` (negative samples)
  - `ball/` (ball crops)
  - `team_a/` (team A players)
  - `team_b/` (team B players)

### 3. `model/train_multiclass.py`
**Training script:**
- CrossEntropy loss cho classification
- MSE loss cho bbox regression
- Save best model → `multiclass_detector.pth`

---

## 🚀 Quy trình Nâng cấp

### **Bước 1: Chuẩn bị multi-class data**
```bash
cd model
python prepare_multiclass_data.py
```

**Output:**
```
input/multiclass_dataset/
├── background/  (~10,000 crops)
├── ball/        (~5,000 crops)
├── team_a/      (~15,000 crops)
└── team_b/      (~15,000 crops)
```

### **Bước 2: Train multi-class model**
```bash
python train_multiclass.py
```

**Hyperparameters:**
- Epochs: 15
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam

**Output:** `model/multiclass_detector.pth`

### **Bước 3: Update tracking scripts**
Cập nhật các tracking scripts để sử dụng model mới:
```python
from model.model_multiclass import MultiClassObjectDetector

model = MultiClassObjectDetector(num_classes=4)
model.load_state_dict(torch.load('multiclass_detector.pth'))

# Inference
class_probs, bbox_deltas = model.predict(crop_tensor)
predicted_class = torch.argmax(class_probs, dim=1)

# 0=bg, 1=ball, 2=team_a, 3=team_b
```

---

## 🎨 Visualization Enhancement

### **Với model mới, có thể:**

1. **Vẽ màu theo class:**
```python
if predicted_class == 1:  # Ball
    color = (0, 255, 255)  # Yellow
elif predicted_class == 2:  # Team A
    color = (0, 0, 255)    # Red
elif predicted_class == 3:  # Team B
    color = (255, 0, 0)    # Blue
```

2. **Tạo heatmap theo đội:**
```python
team_a_positions = []  # Lưu vị trí Team A
team_b_positions = []  # Lưu vị trí Team B
ball_positions = []    # Lưu vị trí Ball

# Generate heatmap for each team
```

3. **Statistics:**
```
Frame 100:
  Team A: 11 players detected
  Team B: 10 players detected
  Ball: 1 detected
  Ball possession: Team A (closer)
```

---

## 📈 Expected Improvements

### **Metrics:**
```
Old model (Binary):
  ✓ Ball detection: 99% accuracy
  ✗ Player detection: Mixed with ball
  ✗ Team classification: Not available

New model (Multi-class):
  ✓ Ball detection: ~95-97% accuracy
  ✓ Player detection: ~92-95% accuracy
  ✓ Team classification: ~85-90% accuracy
  ✓ Overall mAP: ~90%
```

### **Use cases mới:**
- ✅ Tactical analysis (phân tích chiến thuật)
- ✅ Player heatmap by team
- ✅ Ball possession statistics
- ✅ Formation detection
- ✅ Player tracking by jersey color

---

## 🔄 Phân biệt Team thông minh hơn

### **Cách hiện tại (Simple):**
```python
# Dựa vào vị trí trên sân
if cx < w_img/2:
    team = 'team_a'  # Bên trái
else:
    team = 'team_b'  # Bên phải
```

### **Cách nâng cao (Advanced):**

**1. Color-based classification:**
```python
def get_dominant_color(crop):
    # Extract jersey color
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Detect red vs blue jerseys
    return 'team_a' or 'team_b'
```

**2. Deep learning:**
- Train CNN classifier trên jersey colors
- Input: Player crop
- Output: Team A/B probability

**3. Tracking + temporal consistency:**
- Một player không đổi team giữa các frames
- Dùng Kalman filter để maintain team ID

---

## 🎯 Next Steps

### **Để hoàn thiện hệ thống:**

1. ✅ **Đã làm:** Binary ball detection (99% acc)
2. 🔄 **Đang làm:** Multi-class classification
3. ⏭️ **Tiếp theo:**
   - Color-based team classification
   - Player tracking with ID persistence
   - Tactical heatmap generation
   - Ball possession calculation
   - Formation detection

---

## 💡 Tóm tắt

### **Model cũ (BallRefinerNet):**
```
Input: Crop 64x64
Output: [conf] (binary)
Use: Ball detection only
```

### **Model mới (MultiClassObjectDetector):**
```
Input: Crop 64x64
Output: [bg, ball, team_a, team_b] (4 classes)
Use: Full object classification + tracking
```

**→ Giải quyết vấn đề phân biệt ball/player và 2 đội!** ✅
