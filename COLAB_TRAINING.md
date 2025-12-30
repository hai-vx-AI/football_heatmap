# Hướng dẫn Train YOLO trên Google Colab

## Tại sao dùng Google Colab?
- **GPU miễn phí** (Tesla T4/V100)
- Training nhanh hơn **50-100x** so với CPU
- 50 epochs: **~2-3 giờ** trên GPU vs **10-20 giờ** trên CPU

## Bước 1: Chuẩn bị Dataset

### 1.1. Nén dataset (đã làm)
```bash
# Dataset đã được nén tại: yolo_dataset.zip
```

### 1.2. Upload lên Google Drive
1. Truy cập [Google Drive](https://drive.google.com)
2. Tạo folder `football_heatmap`
3. Upload file `yolo_dataset.zip` vào folder này

## Bước 2: Mở Google Colab

### 2.1. Tạo Notebook mới
1. Truy cập [Google Colab](https://colab.research.google.com)
2. Click **"New notebook"**
3. Đổi tên: `Football_YOLO_Training.ipynb`

### 2.2. Bật GPU
1. Menu: **Runtime** → **Change runtime type**
2. **Hardware accelerator**: chọn **GPU**
3. Click **Save**

## Bước 3: Chạy Training trên Colab

### Cell 1: Mount Google Drive và Setup
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install ultralytics pyyaml -q

# Navigate to working directory
import os
os.chdir('/content')

# Verify GPU
!nvidia-smi
```

### Cell 2: Extract Dataset
```python
import zipfile
import shutil
from pathlib import Path

# Extract dataset from Drive
zip_path = '/content/drive/MyDrive/football_heatmap/yolo_dataset.zip'
extract_path = '/content/yolo_dataset'

print("📦 Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content')

# Check for Windows-style paths (backslashes in filenames)
print("\n🔍 Checking for Windows-style paths...")
has_backslash_files = False
for root, dirs, files in os.walk('/content'):
    for file in files:
        if '\\' in file:
            has_backslash_files = True
            print(f"⚠️  Found Windows-style path: {file[:80]}...")
            break
    if has_backslash_files:
        break

# Fix Windows paths if needed
if has_backslash_files:
    print("\n🔧 Fixing Windows-style paths...")
    
    # Find all files with backslashes
    misplaced_files = []
    for root, dirs, files in os.walk('/content'):
        for file in files:
            if '\\' in file:
                misplaced_files.append(os.path.join(root, file))
    
    print(f"Found {len(misplaced_files)} files to fix")
    
    # Create proper directory structure
    os.makedirs(extract_path, exist_ok=True)
    
    moved_count = 0
    for old_path in misplaced_files:
        filename = os.path.basename(old_path)
        # Replace backslashes with forward slashes
        new_relative_path = filename.replace('\\', '/')
        # Remove 'yolo_dataset/' prefix if present
        if new_relative_path.startswith('yolo_dataset/'):
            new_relative_path = new_relative_path[len('yolo_dataset/'):]
        
        new_path = os.path.join(extract_path, new_relative_path)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.move(old_path, new_path)
        moved_count += 1
        
        if moved_count % 500 == 0:
            print(f"  Moved {moved_count}/{len(misplaced_files)} files...")
    
    print(f"✅ Fixed {moved_count} files!")

# Verify dataset structure
print(f"\n📊 Verifying dataset at: {extract_path}")
train_imgs = len(list(Path(extract_path).glob('train/images/*.jpg')))
val_imgs = len(list(Path(extract_path).glob('val/images/*.jpg')))
print(f"Train: {train_imgs} images, Val: {val_imgs} images")

if train_imgs == 0:
    print("\n❌ ERROR: No training images found!")
    print("Showing /content structure:")
    !ls -la /content/ | head -30
else:
    print("\n✅ Dataset ready for training!")
```

### Cell 3: Verify data.yaml
```python
import yaml

# Check data.yaml
data_yaml_path = f'{extract_path}/data.yaml'
with open(data_yaml_path, 'r') as f:
    config = yaml.safe_load(f)
    print(yaml.dump(config, default_flow_style=False))

# Update path to absolute path in Colab
config['path'] = extract_path
with open(data_yaml_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("\nUpdated data.yaml:")
print(yaml.dump(config, default_flow_style=False))
```

### Cell 4: Train YOLO
```python
from ultralytics import YOLO

# Initialize model
model = YOLO('yolo11n.pt')  # or yolo11s.pt, yolo11m.pt for better accuracy

# Training parameters
results = model.train(
    data=f'{extract_path}/data.yaml',
    epochs=50,                  # Số epochs (có thể tăng lên 100)
    imgsz=640,                  # Image size (có thể tăng lên 1280 cho accuracy cao hơn)
    batch=16,                   # Batch size (GPU có thể handle lớn hơn)
    device=0,                   # GPU device
    workers=8,                  # Number of workers
    
    # Optimizer
    optimizer='SGD',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    
    # Loss weights
    box=7.5,
    cls=0.5,
    dfl=1.5,
    
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    
    # Output
    project='runs/train',
    name='yolo_football',
    exist_ok=True,
    
    # Settings
    pretrained=True,
    verbose=True,
    save=True,
    save_period=10,            # Save checkpoint every 10 epochs
    plots=True,
    val=True
)

print("\n" + "="*80)
print("Training Complete!")
print("="*80)
```

### Cell 5: Evaluate Model
```python
# Validate on test set
metrics = model.val()

print(f"\nValidation Results:")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

### Cell 6: Visualize Results
```python
from IPython.display import Image, display

# Show training curves
display(Image(filename='runs/train/yolo_football/results.png'))

# Show confusion matrix
display(Image(filename='runs/train/yolo_football/confusion_matrix.png'))

# Show validation predictions
display(Image(filename='runs/train/yolo_football/val_batch0_pred.jpg'))
```

### Cell 7: Download Trained Model
```python
import shutil

# Copy best model to Drive
output_dir = '/content/drive/MyDrive/football_heatmap/trained_models'
os.makedirs(output_dir, exist_ok=True)

# Copy weights
shutil.copy('runs/train/yolo_football/weights/best.pt', 
            f'{output_dir}/yolo_football_best.pt')
shutil.copy('runs/train/yolo_football/weights/last.pt', 
            f'{output_dir}/yolo_football_last.pt')

# Copy results
shutil.copy('runs/train/yolo_football/results.png', 
            f'{output_dir}/training_results.png')
shutil.copy('runs/train/yolo_football/confusion_matrix.png', 
            f'{output_dir}/confusion_matrix.png')

# Zip all training outputs
shutil.make_archive(
    f'{output_dir}/yolo_football_complete',
    'zip',
    'runs/train/yolo_football'
)

print("✅ Model saved to Google Drive:")
print(f"  - {output_dir}/yolo_football_best.pt")
print(f"  - {output_dir}/yolo_football_last.pt")
print(f"  - {output_dir}/yolo_football_complete.zip")
```

### Cell 8: Test Inference (Optional)
```python
from IPython.display import Image, display
from pathlib import Path

# Test on a sample validation image
test_img = list(Path(extract_path).glob('val/images/*.jpg'))[0]
print(f"Testing on: {test_img.name}")

# Run inference
results = model.predict(test_img, conf=0.25, save=True)
print(f"✅ Test prediction saved!")

# Display result
result_path = Path(results[0].save_dir) / Path(results[0].path).name
display(Image(filename=str(result_path)))

# Print detections
for r in results:
    print(f"\nDetected {len(r.boxes)} objects:")
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  - {model.names[cls]}: {conf:.2f}")
```

## Bước 4: Download Model về Local

1. Từ Google Drive, download file `yolo_football_best.pt`
2. Copy vào folder project:
   ```bash
   # Windows PowerShell
   Move-Item ~\Downloads\yolo_football_best.pt runs\train\yolo_football\weights\best.pt
   ```

## Bước 5: Cập nhật Config để dùng trained model

### 5.1. Sửa config.yaml
```yaml
detector:
  model_path: "runs/train/yolo_football/weights/best.pt"  # Dùng trained model
  imgsz: 640
  # ... rest of config
```

### 5.2. Hoặc dùng khi chạy inference
```bash
# Test với trained YOLO model
python main.py input/SoccerNet/tracking-2023/test/SNMOT-116 \
    --output-name test_116_trained \
    --config config_cpu.yaml \
    --max-frames 500
```

## Tips & Tricks

### Tăng Accuracy
```python
# Trong Cell 4, thay đổi:
imgsz=1280,        # Tăng resolution
epochs=100,        # Train lâu hơn
batch=8,           # Giảm nếu bị OOM
model = YOLO('yolo11s.pt')  # Dùng model lớn hơn
```

### Giảm Training Time
```python
# Trong Cell 4:
imgsz=416,         # Giảm resolution
epochs=30,         # Ít epochs hơn
```

### Theo dõi Training
- Colab sẽ hiển thị progress bar real-time
- Loss giảm dần = model đang học tốt
- mAP tăng dần = accuracy tốt hơn

### Troubleshooting

**1. Out of Memory (OOM)**
```python
# Giảm batch size
batch=8,  # hoặc 4
```

**2. Colab bị disconnect**
```python
# Thêm vào Cell 1 để tự động reconnect
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

**3. Dataset không extract được**
```bash
# Kiểm tra file zip
!unzip -t /content/drive/MyDrive/football_heatmap/yolo_dataset.zip | head -20
```

## Video hướng dẫn chi tiết

Xem [video tutorial](https://www.youtube.com/watch?v=dQw4w9WgXcQ) để hiểu rõ hơn cách setup và train trên Colab.

---

**Thời gian ước tính:**
- Setup: 5-10 phút
- Training (50 epochs, GPU T4): 2-3 giờ
- Download model: 1-2 phút

**Chi phí:** MIỄN PHÍ (Colab free tier)
