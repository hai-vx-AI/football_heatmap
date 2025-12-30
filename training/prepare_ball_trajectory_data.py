"""
Prepare training data for ball trajectory predictor.
Extracts ball trajectories and creates sliding window sequences.
"""

import argparse
from pathlib import Path
import json
import yaml
from tqdm import tqdm
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.video_io import SoccerNetSequenceReader
from src.detector import Detector
from src.ball_tracker import BallTracker


def create_sliding_windows(positions, seq_length=10, pred_horizon=5, min_continuous=15):
    """
    Create sliding window sequences from ball positions.
    
    Args:
        positions: List of [x, y, frame_idx, visible]
        seq_length: Input sequence length
        pred_horizon: Prediction horizon (how many frames to predict)
        min_continuous: Minimum continuous visible frames needed
        
    Returns:
        List of (input_seq, target_seq) tuples
    """
    sequences = []
    total_needed = seq_length + pred_horizon
    
    # Find continuous visible segments
    i = 0
    while i < len(positions):
        # Find start of visible segment
        while i < len(positions) and not positions[i][3]:
            i += 1
        
        if i >= len(positions):
            break
        
        # Find end of visible segment
        start = i
        while i < len(positions) and positions[i][3]:
            i += 1
        end = i
        
        segment_length = end - start
        
        # Create sliding windows from this segment
        if segment_length >= min_continuous:
            for j in range(start, end - total_needed + 1):
                window = positions[j:j + total_needed]
                
                # Verify all positions are visible
                if all(p[3] for p in window):
                    input_seq = [[p[0], p[1]] for p in window[:seq_length]]
                    target_seq = [[p[0], p[1]] for p in window[seq_length:]]
                    sequences.append((input_seq, target_seq))
    
    return sequences


def prepare_trajectory_data(soccernet_root: str, output_dir: str, config_path: str = "config.yaml",
                            max_sequences: int = None, seq_length: int = 10, pred_horizon: int = 5,
                            max_frames: int = None):
    """
    Prepare ball trajectory dataset from SoccerNet sequences.
    
    Args:
        soccernet_root: Root directory of SoccerNet dataset
        output_dir: Output directory for trajectory data
        config_path: Path to config.yaml
        max_sequences: Maximum number of sequences to process
        seq_length: Input sequence length for LSTM
        pred_horizon: Prediction horizon (frames to predict)
        max_frames: Maximum frames per sequence (for faster testing)
    """
    soccernet_root = Path(soccernet_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize modules
    detector = Detector(config['detector'])
    
    # Find sequences
    train_dir = soccernet_root / 'train'
    sequences = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    
    if max_sequences:
        sequences = sequences[:max_sequences]
    
    print(f"Found {len(sequences)} sequences")
    print(f"Sequence length: {seq_length}, Prediction horizon: {pred_horizon}")
    
    # Split 80/20 train/val
    split_idx = int(len(sequences) * 0.8)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    # Process sequences
    for split_name, split_seqs in [('train', train_sequences), ('val', val_sequences)]:
        print(f"\nProcessing {split_name} split ({len(split_seqs)} sequences)...")
        
        all_sequences = []
        all_positions_for_norm = []
        
        for seq_dir in tqdm(split_seqs):
            # Load sequence
            reader = SoccerNetSequenceReader(str(seq_dir))
            ball_tracker = BallTracker(config['ball_tracker'], fps=reader.fps)
            
            # Track ball through entire sequence (or max_frames)
            ball_positions = []
            frame_count = 0
            
            for frame_idx, frame in iter(reader):
                # Detect
                detections = detector.detect(frame)
                _, ball_dets = detector.split_detections(detections)
                
                # Track ball
                ball_state = ball_tracker.update(frame_idx, ball_dets, frame.shape, detector)
                
                # Record position
                if ball_state['center'] is not None:
                    x, y = ball_state['center']
                    visible = (ball_state['status'] == 'detected')
                    ball_positions.append([x, y, frame_idx, visible])
                
                # Stop if max_frames reached
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
            
            reader.release()
            
            # Create sliding windows from this sequence
            windows = create_sliding_windows(ball_positions, seq_length, pred_horizon)
            
            if windows:
                all_sequences.extend(windows)
                # Collect positions for normalization
                for pos in ball_positions:
                    if pos[3]:  # visible
                        all_positions_for_norm.append([pos[0], pos[1]])
                
                print(f"  ✓ {seq_dir.name}: {len(windows)} windows from {len(ball_positions)} positions")
            else:
                visible_count = sum(1 for p in ball_positions if p[3])
                print(f"  ✗ {seq_dir.name}: Not enough continuous visible frames ({visible_count} total visible)")
        
        # Calculate normalization parameters
        if all_positions_for_norm:
            all_positions_for_norm = np.array(all_positions_for_norm, dtype=np.float32)
            mean = all_positions_for_norm.mean(axis=0).tolist()
            std = all_positions_for_norm.std(axis=0).tolist()
            # Avoid division by zero
            std = [s if s > 1e-6 else 1.0 for s in std]
        else:
            mean = [0.0, 0.0]
            std = [1.0, 1.0]
        
        # Save in new format
        dataset = {
            'sequences': [
                {
                    'input': input_seq,
                    'target': target_seq
                }
                for input_seq, target_seq in all_sequences
            ],
            'normalization': {
                'mean': mean,
                'std': std
            },
            'metadata': {
                'seq_length': seq_length,
                'pred_horizon': pred_horizon,
                'num_sequences': len(all_sequences)
            }
        }
        
        # Save dataset
        output_file = output_dir / f'{split_name}_sequences.json'
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n{split_name.upper()} Summary:")
        print(f"  Training sequences: {len(all_sequences)}")
        print(f"  Normalization: mean={mean}, std={std}")
        print(f"  Saved to: {output_file}")
    
    print(f"\n✓ Dataset prepared in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ball trajectory dataset")
    parser.add_argument('--soccernet-root', type=str,
                       default='input/SoccerNet/tracking-2023',
                       help='Root directory of SoccerNet dataset')
    parser.add_argument('--output-dir', type=str, default='data/ball_trajectories',
                       help='Output directory for trajectory data')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--max-sequences', type=int, default=None,
                       help='Maximum number of sequences to process')
    parser.add_argument('--seq-length', type=int, default=10,
                       help='Input sequence length for LSTM')
    parser.add_argument('--pred-horizon', type=int, default=5,
                       help='Prediction horizon (frames to predict)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames per sequence (for testing)')
    
    args = parser.parse_args()
    
    prepare_trajectory_data(
        args.soccernet_root,
        args.output_dir,
        args.config,
        args.max_sequences,
        args.seq_length,
        args.pred_horizon,
        args.max_frames
    )
