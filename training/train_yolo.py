"""
Fine-tune YOLO model on SoccerNet dataset for better football detection.
"""

import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO


def train_yolo(config_path: str = "training/yolo_config.yaml"):
    """
    Train YOLO model on custom football dataset.
    
    Args:
        config_path: Path to training configuration
    """
    print("="*80)
    print("YOLO Fine-tuning for Football Detection")
    print("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_config = config['yolo_training']
    
    # Load pretrained model
    base_model = train_config.get('base_model', 'yolo11n.pt')
    print(f"\nLoading base model: {base_model}")
    model = YOLO(base_model)
    
    # Training parameters
    data_yaml = train_config['data_yaml']
    epochs = train_config.get('epochs', 100)
    imgsz = train_config.get('imgsz', 1280)
    batch = train_config.get('batch', 16)
    device = train_config.get('device', 'cuda')
    workers = train_config.get('workers', 8)
    
    # Hyperparameters
    lr0 = train_config.get('lr0', 0.01)
    lrf = train_config.get('lrf', 0.01)
    momentum = train_config.get('momentum', 0.937)
    weight_decay = train_config.get('weight_decay', 0.0005)
    warmup_epochs = train_config.get('warmup_epochs', 3)
    
    # Data augmentation
    augment = train_config.get('augment', True)
    hsv_h = train_config.get('hsv_h', 0.015)
    hsv_s = train_config.get('hsv_s', 0.7)
    hsv_v = train_config.get('hsv_v', 0.4)
    degrees = train_config.get('degrees', 0.0)
    translate = train_config.get('translate', 0.1)
    scale = train_config.get('scale', 0.5)
    
    print(f"\nTraining Configuration:")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    print(f"  Learning rate: {lr0} -> {lrf}")
    
    # Train
    print("\nStarting training...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=train_config.get('project', 'runs/train'),
        name=train_config.get('name', 'yolo_football'),
        exist_ok=train_config.get('exist_ok', True),
        pretrained=True,
        optimizer=train_config.get('optimizer', 'SGD'),
        verbose=True,
        seed=train_config.get('seed', 42),
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=train_config.get('cos_lr', False),
        close_mosaic=train_config.get('close_mosaic', 10),
        resume=train_config.get('resume', False),
        amp=train_config.get('amp', True),
        fraction=train_config.get('fraction', 1.0),
        profile=False,
        # Hyperparameters
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_momentum=train_config.get('warmup_momentum', 0.8),
        warmup_bias_lr=train_config.get('warmup_bias_lr', 0.1),
        box=train_config.get('box', 7.5),
        cls=train_config.get('cls', 0.5),
        dfl=train_config.get('dfl', 1.5),
        pose=train_config.get('pose', 12.0),
        kobj=train_config.get('kobj', 1.0),
        label_smoothing=train_config.get('label_smoothing', 0.0),
        nbs=train_config.get('nbs', 64),
        # Augmentation
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=train_config.get('shear', 0.0),
        perspective=train_config.get('perspective', 0.0),
        flipud=train_config.get('flipud', 0.0),
        fliplr=train_config.get('fliplr', 0.5),
        mosaic=train_config.get('mosaic', 1.0),
        mixup=train_config.get('mixup', 0.0),
        copy_paste=train_config.get('copy_paste', 0.0),
        # Val settings
        val=True,
        save=True,
        save_period=train_config.get('save_period', -1),
        cache=train_config.get('cache', False),
        plots=True,
    )
    
    # Validation
    print("\n" + "="*80)
    print("Validation Results")
    print("="*80)
    metrics = model.val()
    
    print(f"\nmAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")
    
    # Save best model
    best_model_path = Path(train_config.get('project', 'runs/train')) / train_config.get('name', 'yolo_football') / 'weights' / 'best.pt'
    print(f"\nBest model saved to: {best_model_path}")
    
    # Export to ONNX (optional)
    if train_config.get('export_onnx', False):
        print("\nExporting to ONNX format...")
        model.export(format='onnx')
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO on football dataset")
    parser.add_argument('--config', type=str, default='training/yolo_config.yaml',
                       help='Path to training configuration file')
    
    args = parser.parse_args()
    
    train_yolo(args.config)
