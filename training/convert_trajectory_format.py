"""
Convert old trajectory format to new sliding window format.
"""

import json
import numpy as np
from pathlib import Path
import argparse


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


def convert_format(input_dir, output_dir, seq_length=10, pred_horizon=5):
    """Convert old format to new format."""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val']:
        # Load old format
        old_file = input_dir / f'{split}_trajectories.json'
        if not old_file.exists():
            print(f"Skipping {split}: {old_file} not found")
            continue
            
        with open(old_file, 'r') as f:
            old_data = json.load(f)
        
        print(f"\nConverting {split} split...")
        print(f"  Old format: {len(old_data)} trajectories")
        
        # Convert to sliding windows
        all_sequences = []
        all_positions_for_norm = []
        
        for traj in old_data:
            positions = traj['positions']
            
            # Collect visible positions for normalization
            for pos in positions:
                if pos[3]:  # visible
                    all_positions_for_norm.append([pos[0], pos[1]])
            
            # Create windows
            windows = create_sliding_windows(positions, seq_length, pred_horizon)
            all_sequences.extend(windows)
            
            if windows:
                print(f"    {traj['sequence']}: {len(windows)} windows")
        
        # Calculate normalization
        if all_positions_for_norm:
            all_positions_for_norm = np.array(all_positions_for_norm, dtype=np.float32)
            mean = all_positions_for_norm.mean(axis=0).tolist()
            std = all_positions_for_norm.std(axis=0).tolist()
            std = [s if s > 1e-6 else 1.0 for s in std]
        else:
            mean = [0.0, 0.0]
            std = [1.0, 1.0]
        
        # Save new format
        new_data = {
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
        
        output_file = output_dir / f'{split}_sequences.json'
        with open(output_file, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        print(f"  New format: {len(all_sequences)} sequences")
        print(f"  Normalization: mean={mean}, std={std}")
        print(f"  Saved to: {output_file}")
    
    print(f"\n✓ Conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert trajectory format")
    parser.add_argument('--input-dir', type=str, default='data/ball_trajectories',
                       help='Input directory with old format')
    parser.add_argument('--output-dir', type=str, default='data/ball_trajectories',
                       help='Output directory for new format')
    parser.add_argument('--seq-length', type=int, default=10,
                       help='Input sequence length')
    parser.add_argument('--pred-horizon', type=int, default=5,
                       help='Prediction horizon')
    
    args = parser.parse_args()
    
    convert_format(args.input_dir, args.output_dir, args.seq_length, args.pred_horizon)
