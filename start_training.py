"""
Quick Training Launcher with Auto-configuration
Automatically sets up environment and starts training
"""
import os
import sys
import subprocess
import glob

def check_dataset():
    """Check if dataset is ready"""
    dataset_dir = "input/multiclass_dataset"
    required_classes = ['background', 'ball', 'team_a', 'team_b', 'referee']
    
    if not os.path.exists(dataset_dir):
        return False, "Dataset folder not found"
    
    for class_name in required_classes:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            return False, f"Missing class folder: {class_name}"
        
        count = len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
        if count == 0:
            return False, f"Empty class folder: {class_name}"
    
    return True, "Dataset ready"

def check_dependencies():
    """Check if required packages are installed"""
    required = ['torch', 'torchvision', 'cv2', 'tqdm', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return len(missing) == 0, missing

def main():
    print("=" * 70)
    print("🚀 TRAINING LAUNCHER")
    print("=" * 70)
    
    # Check dataset
    print("\n📦 Checking dataset...")
    dataset_ok, dataset_msg = check_dataset()
    if dataset_ok:
        print(f"  ✅ {dataset_msg}")
    else:
        print(f"  ❌ {dataset_msg}")
        print("\n💡 Run this first:")
        print("   cd model")
        print("   python prepare_multiclass_data.py")
        return
    
    # Check dependencies
    print("\n📚 Checking dependencies...")
    deps_ok, missing = check_dependencies()
    if deps_ok:
        print("  ✅ All dependencies installed")
    else:
        print(f"  ❌ Missing packages: {', '.join(missing)}")
        print(f"\n💡 Install with:")
        print(f"   pip install {' '.join(missing)}")
        return
    
    # Count samples
    print("\n📊 Dataset statistics:")
    dataset_dir = "input/multiclass_dataset"
    for class_name in ['background', 'ball', 'team_a', 'team_b', 'referee']:
        class_dir = os.path.join(dataset_dir, class_name)
        count = len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
        print(f"  {class_name:<12}: {count:>8,} samples")
    
    # Check if old checkpoint exists
    old_checkpoint = "model/multiclass_detector.pth"
    new_checkpoint = "model/multiclass_detector_5class.pth"
    
    if os.path.exists(old_checkpoint) and not os.path.exists(new_checkpoint):
        print(f"\n⚠️  Old 4-class checkpoint found: {old_checkpoint}")
        print("   New 5-class model will start from scratch (recommended)")
    
    # Launch training
    print("\n" + "=" * 70)
    print("🎓 STARTING TRAINING...")
    print("=" * 70)
    print("\n💡 Training configuration:")
    print("  - Model: MultiClassObjectDetector (5 classes)")
    print("  - Epochs: 20")
    print("  - Batch size: 32")
    print("  - Learning rate: 0.001")
    print(f"  - Checkpoint: {new_checkpoint}")
    print("\n⏰ Estimated time: ~6-8 hours (on CPU)")
    print("\n" + "=" * 70)
    
    try:
        # Change to model directory and run training
        os.chdir('model')
        subprocess.run([sys.executable, 'train_multiclass.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n✋ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
