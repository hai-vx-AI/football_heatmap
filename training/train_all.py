"""
Master training script - Train all models sequentially or selectively.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run shell command with error handling."""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} failed with error code {e.returncode}")
        return False


def prepare_data(args):
    """Prepare training data for all models."""
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80)
    
    success = True
    warnings = []
    
    # Prepare jersey crops for team classifier
    if args.train_team_classifier or args.prepare_all:
        cmd = [
            sys.executable, 
            "training/prepare_jersey_data.py",
            "--soccernet-root", args.soccernet_root,
            "--output-dir", "data/jersey_crops",
            "--config", args.config
        ]
        if args.max_sequences:
            cmd.extend(["--max-sequences", str(args.max_sequences)])
        
        if not run_command(cmd, "Jersey Data Preparation"):
            warnings.append("Jersey data preparation had issues")
    
    # Prepare ball trajectories for predictor
    if args.train_ball_predictor or args.prepare_all:
        cmd = [
            sys.executable,
            "training/prepare_ball_trajectory_data.py",
            "--soccernet-root", args.soccernet_root,
            "--output-dir", "data/ball_trajectories",
            "--config", args.config
        ]
        if args.max_sequences:
            cmd.extend(["--max-sequences", str(args.max_sequences)])
        
        if not run_command(cmd, "Ball Trajectory Data Preparation"):
            warnings.append("Ball trajectory data preparation had issues")
    
    if warnings:
        print("\n⚠️  Warnings during data preparation:")
        for w in warnings:
            print(f"  - {w}")
        print("\nYou may want to:")
        print("  - Increase --max-sequences for more data")
        print("  - Check if ball detection is enabled in config")
        print("  - Verify dataset structure")
    
    return success


def train_models(args):
    """Train selected models."""
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    success = True
    
    # Train YOLO
    if args.train_yolo:
        cmd = [
            sys.executable,
            "training/train_yolo.py",
            "--config", args.training_config
        ]
        if not run_command(cmd, "YOLO Fine-tuning"):
            success = False
    
    # Train Team Classifier
    if args.train_team_classifier:
        cmd = [
            sys.executable,
            "training/train_team_classifier.py",
            "--config", args.training_config
        ]
        if not run_command(cmd, "Team Classifier Training"):
            success = False
    
    # Train Ball Predictor
    if args.train_ball_predictor:
        cmd = [
            sys.executable,
            "training/train_ball_predictor.py",
            "--config", args.training_config
        ]
        if not run_command(cmd, "Ball Predictor Training"):
            success = False
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Master training script for football analysis models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data only
  python training/train_all.py --prepare-data-only
  
  # Train all models
  python training/train_all.py --all
  
  # Train specific models
  python training/train_all.py --train-yolo --train-team-classifier
  
  # Quick test with limited data
  python training/train_all.py --all --max-sequences 5
        """
    )
    
    # Data preparation
    parser.add_argument('--prepare-data-only', action='store_true',
                       help='Only prepare data, do not train')
    parser.add_argument('--skip-data-prep', action='store_true',
                       help='Skip data preparation (use existing data)')
    parser.add_argument('--prepare-all', action='store_true',
                       help='Prepare data for all models')
    
    # Model selection
    parser.add_argument('--all', action='store_true',
                       help='Train all models')
    parser.add_argument('--train-yolo', action='store_true',
                       help='Fine-tune YOLO detector')
    parser.add_argument('--train-team-classifier', action='store_true',
                       help='Train team classifier')
    parser.add_argument('--train-ball-predictor', action='store_true',
                       help='Train ball trajectory predictor')
    
    # Paths
    parser.add_argument('--soccernet-root', type=str,
                       default='input/SoccerNet/tracking-2023',
                       help='Root directory of SoccerNet dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to main config file')
    parser.add_argument('--training-config', type=str,
                       default='training/training_config.yaml',
                       help='Path to training config file')
    
    # Data options
    parser.add_argument('--max-sequences', type=int, default=None,
                       help='Maximum sequences to process (for testing)')
    
    args = parser.parse_args()
    
    # If --all, enable all training
    if args.all:
        args.train_yolo = True
        args.train_team_classifier = True
        args.train_ball_predictor = True
        args.prepare_all = True
    
    # If --prepare-data-only, prepare for all models
    if args.prepare_data_only:
        args.prepare_all = True
    
    # Check if any training selected
    if not (args.train_yolo or args.train_team_classifier or 
            args.train_ball_predictor or args.prepare_data_only):
        parser.print_help()
        print("\n⚠️  Please specify at least one training option or use --all")
        sys.exit(1)
    
    # Verify paths exist
    if not Path(args.soccernet_root).exists():
        print(f"✗ Error: SoccerNet root not found: {args.soccernet_root}")
        sys.exit(1)
    
    if not Path(args.config).exists():
        print(f"✗ Error: Config file not found: {args.config}")
        sys.exit(1)
    
    if not Path(args.training_config).exists():
        print(f"✗ Error: Training config not found: {args.training_config}")
        sys.exit(1)
    
    # Create training directory
    Path("training").mkdir(exist_ok=True)
    
    print("="*80)
    print("FOOTBALL ANALYSIS - TRAINING PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  SoccerNet root: {args.soccernet_root}")
    print(f"  Config: {args.config}")
    print(f"  Training config: {args.training_config}")
    if args.max_sequences:
        print(f"  Max sequences: {args.max_sequences}")
    
    print(f"\nSelected tasks:")
    if args.prepare_all or not args.skip_data_prep:
        print(f"  [x] Data preparation")
    if args.train_yolo:
        print(f"  [x] YOLO fine-tuning")
    if args.train_team_classifier:
        print(f"  [x] Team classifier training")
    if args.train_ball_predictor:
        print(f"  [x] Ball predictor training")
    
    # Prepare data
    if not args.skip_data_prep:
        success = prepare_data(args)
        if not success:
            print("\n✗ Data preparation failed!")
            sys.exit(1)
        
        if args.prepare_data_only:
            print("\n" + "="*80)
            print("✓ Data preparation complete!")
            print("="*80)
            return
    
    # Train models
    success = train_models(args)
    
    # Final summary
    print("\n" + "="*80)
    if success:
        print("✓ ALL TRAINING TASKS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nTrained models saved in:")
        if args.train_yolo:
            print("  - runs/train/yolo_football/weights/best.pt")
        if args.train_team_classifier:
            print("  - runs/team_classifier/best_model.pth")
        if args.train_ball_predictor:
            print("  - runs/ball_predictor/best_model.pth")
    else:
        print("✗ SOME TRAINING TASKS FAILED")
        print("="*80)
        sys.exit(1)


if __name__ == "__main__":
    main()
