"""
Prepare training data for team classifier.
Extracts jersey crops from SoccerNet sequences with ground truth team labels.
"""

import argparse
from pathlib import Path
import cv2
import json
import yaml
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.video_io import SoccerNetSequenceReader
from src.detector import Detector
from src.people_tracker import PeopleTracker
from src.team_assigner import TeamAssigner
from src.utils import crop_jersey_roi


def prepare_jersey_data(soccernet_root: str, output_dir: str, config_path: str = "config.yaml",
                       max_sequences: int = None, samples_per_sequence: int = 100):
    """
    Prepare jersey crop dataset from SoccerNet sequences.
    
    Args:
        soccernet_root: Root directory of SoccerNet dataset
        output_dir: Output directory for jersey crops
        config_path: Path to config.yaml
        max_sequences: Maximum number of sequences to process
        samples_per_sequence: Target number of crops per sequence
    """
    soccernet_root = Path(soccernet_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create split directories
    (output_dir / 'train' / 'team0').mkdir(parents=True, exist_ok=True)
    (output_dir / 'train' / 'team1').mkdir(parents=True, exist_ok=True)
    (output_dir / 'val' / 'team0').mkdir(parents=True, exist_ok=True)
    (output_dir / 'val' / 'team1').mkdir(parents=True, exist_ok=True)
    
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
    
    # Split 80/20 train/val
    split_idx = int(len(sequences) * 0.8)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    train_annotations = []
    val_annotations = []
    
    # Process sequences
    for split_name, split_seqs, annotations in [('train', train_sequences, train_annotations),
                                                 ('val', val_sequences, val_annotations)]:
        print(f"\nProcessing {split_name} split ({len(split_seqs)} sequences)...")
        
        # Initialize debug counters for this split
        total_tracks_split = 0
        players_detected_split = 0
        players_with_team_split = 0
        
        for seq_dir in tqdm(split_seqs):
            # Load sequence
            reader = SoccerNetSequenceReader(str(seq_dir))
            
            # Initialize team assigner
            people_tracker = PeopleTracker(config['people_tracker'], fps=reader.fps)
            team_assigner = TeamAssigner(config['team_color'])
            
            crops_saved = {'team0': 0, 'team1': 0}
            
            # Create iterator once
            reader_iter = iter(reader)
            
            # PHASE 1: Process all frames to initialize team clustering
            print(f"\n  Phase 1: Processing {seq_dir.name} for team clustering...")
            all_tracks = []
            frame_data = []  # Store (frame_idx, frame, tracks)
            
            for frame_idx, frame in reader_iter:
                # Detect
                detections = detector.detect(frame)
                people_dets, _ = detector.split_detections(detections)
                
                # Track
                people_tracks = people_tracker.update(frame_idx, people_dets)
                
                # Assign teams (will initialize after enough samples)
                people_tracks = team_assigner.assign_teams(people_tracks, frame)
                
                all_tracks.extend(people_tracks)
                frame_data.append((frame_idx, frame, people_tracks))
                
                # Debug: Count tracks and teams
                total_tracks_split += len(people_tracks)
            
            # PHASE 2: Extract crops from frames with assigned teams
            print(f"  Phase 2: Extracting jersey crops...")
            sample_indices = list(range(0, len(frame_data), max(1, len(frame_data) // samples_per_sequence)))
            
            for idx in sample_indices:
                current_frame_idx, current_frame, people_tracks = frame_data[idx]
                
                # Extract crops
                for track in people_tracks:
                    if track['cls'] != 'player':
                        continue
                    
                    players_detected_split += 1
                    
                    team_id = track.get('team_id')
                    if team_id is None:
                        continue
                    
                    players_with_team_split += 1
                    
                    # Crop jersey
                    bbox = track['bbox']
                    jersey_crop_cfg = config['team_color']['jersey_crop']
                    jersey_crop = crop_jersey_roi(
                        current_frame, bbox,
                        (jersey_crop_cfg['x_start'], jersey_crop_cfg['x_end']),
                        (jersey_crop_cfg['y_start'], jersey_crop_cfg['y_end'])
                    )
                    
                    if jersey_crop is None or jersey_crop.size == 0:
                        continue
                    
                    # Check minimum size
                    h, w = jersey_crop.shape[:2]
                    if h < 30 or w < 20:
                        continue
                    
                    # Save crop
                    team_label = 0 if team_id == 'team0' else 1
                    crop_filename = f"{seq_dir.name}_frame{frame_idx}_track{track['track_id']}.jpg"
                    crop_path = output_dir / split_name / team_id / crop_filename
                    
                    cv2.imwrite(str(crop_path), jersey_crop)
                    
                    # Add to annotations
                    annotations.append({
                        'crop_path': str(Path(split_name) / team_id / crop_filename),
                        'team_id': team_label,
                        'sequence': seq_dir.name,
                        'frame_idx': frame_idx,
                        'track_id': track['track_id']
                    })
                    
                    crops_saved[team_id] += 1
            
            reader.release()
        
        # Save annotations
        with open(output_dir / f'{split_name}_annotations.json', 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"\n{split_name.upper()} Summary:")
        print(f"  Total crops: {len(annotations)}")
        team0_count = sum(1 for a in annotations if a['team_id'] == 0)
        team1_count = sum(1 for a in annotations if a['team_id'] == 1)
        print(f"  Team 0: {team0_count}")
        print(f"  Team 1: {team1_count}")
        
        # Debug info
        print(f"  Debug: Total tracks seen: {total_tracks_split}")
        print(f"  Debug: Player tracks: {players_detected_split}")
        print(f"  Debug: Players with team ID: {players_with_team_split}")
    
    print(f"\n✓ Dataset prepared in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare jersey crop dataset")
    parser.add_argument('--soccernet-root', type=str, 
                       default='input/SoccerNet/tracking-2023',
                       help='Root directory of SoccerNet dataset')
    parser.add_argument('--output-dir', type=str, default='data/jersey_crops',
                       help='Output directory for crops')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--max-sequences', type=int, default=None,
                       help='Maximum number of sequences to process')
    parser.add_argument('--samples-per-sequence', type=int, default=100,
                       help='Target number of crops per sequence')
    
    args = parser.parse_args()
    
    prepare_jersey_data(
        args.soccernet_root,
        args.output_dir,
        args.config,
        args.max_sequences,
        args.samples_per_sequence
    )
