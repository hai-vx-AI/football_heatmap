"""
Main inference pipeline for football video analysis.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    import yaml
    import cv2
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Run: pip install pyyaml opencv-python")
    sys.exit(1)

from src.video_io import VideoReader, VideoWriter, SoccerNetSequenceReader
from src.detector import Detector
from src.people_tracker import PeopleTracker
from src.ball_tracker import BallTracker
from src.team_assigner import TeamAssigner
from src.possession_analyzer import PossessionAnalyzer
from src.renderer import Renderer
from src.logger import Logger


def load_config(config_path: str, preset: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        preset: Optional preset name ('fast', 'accurate')
    
    Returns:
        Configuration dict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply preset if specified
    if preset and preset in config.get('presets', {}):
        preset_config = config['presets'][preset]
        # Deep merge preset into main config
        config = deep_merge(config, preset_config)
        print(f"Applied preset: {preset}")
    
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def run_inference(video_path: str, config: dict, output_name: str = None, max_frames: int = None):
    """
    Run inference on a video.
    
    Args:
        video_path: Path to input video
        config: Configuration dict
        output_name: Output file base name (default: video filename)
        max_frames: Maximum number of frames to process (None = process all)
    """
    video_path_obj = Path(video_path)
    
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video not found: {video_path_obj}")
    
    # Determine output name
    if output_name is None:
        output_name = video_path_obj.stem
    
    # Setup output directories
    output_root = Path(config['output']['videos_dir']).parent
    videos_dir = output_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    output_video_path = videos_dir / f"{output_name}_overlay.mp4"
    
    print("="*80)
    print("Football Video Analysis System")
    print("="*80)
    print(f"Input video: {video_path_obj}")
    print(f"Output video: {output_video_path}")
    print(f"Output directory: {output_root}")
    print("="*80)
    
    # Initialize video reader
    print("\n[1/7] Opening video...")
    
    # Check if it's a SoccerNet sequence or regular video
    if video_path_obj.is_dir():
        # SoccerNet sequence
        print(f"  Detected SoccerNet sequence format")
        reader = SoccerNetSequenceReader(str(video_path_obj))
    else:
        # Regular video file
        reader = VideoReader(str(video_path))
    
    print(f"  Resolution: {reader.width}x{reader.height}")
    print(f"  FPS: {reader.fps:.2f}")
    print(f"  Frames: {reader.frame_count}")
    print(f"  Duration: {reader.frame_count / reader.fps:.2f}s")
    
    # Initialize modules
    print("\n[2/7] Initializing detector...")
    detector = Detector(config['detector'])
    
    print("\n[3/7] Initializing trackers...")
    people_tracker = PeopleTracker(config['people_tracker'], fps=reader.fps, class_names=detector.class_names)
    ball_tracker = BallTracker(config['ball_tracker'], fps=reader.fps)
    
    print("\n[4/7] Initializing team assigner...")
    team_assigner = TeamAssigner(config['team_color'])
    
    print("\n[5/7] Initializing possession analyzer...")
    possession_config = config.get('possession', {'enabled': False})
    possession_analyzer = None
    if possession_config.get('enabled', False):
        possession_analyzer = PossessionAnalyzer(possession_config, fps=reader.fps)
    
    print("\n[6/7] Initializing renderer...")
    renderer = Renderer(config['render'])
    
    print("\n[7/7] Initializing logger...")
    logger = Logger(config['logger'], reader.meta, str(output_root))

    
    # Initialize video writer
    writer = VideoWriter(
        str(output_video_path),
        fps=reader.fps,
        size=(reader.width, reader.height)
    )
    
    print("\n[7/7] Processing video...")
    print("-"*80)
    
    try:
        frame_count = 0
        for frame_idx, frame in reader:
            # Check max frames limit
            if max_frames is not None and frame_count >= max_frames:
                print(f"\nReached max frames limit: {max_frames}")
                break
                
            # Detect
            detections = detector.detect(frame)
            people_dets, ball_dets = detector.split_detections(detections)
            
            # Track people
            people_tracks = people_tracker.update(frame_idx, people_dets)
            
            # Track ball
            ball_state = ball_tracker.update(
                frame_idx, ball_dets, frame.shape, detector
            )
            
            # Assign teams
            people_tracks = team_assigner.assign_teams(people_tracks, frame)
            
            # Analyze possession
            possession_state = None
            if possession_analyzer is not None:
                possession_state = possession_analyzer.update(frame_idx, people_tracks, ball_state)
            
            # Render
            rendered_frame = renderer.render(frame, people_tracks, ball_state, possession_state)
            
            # Write output
            writer.write(rendered_frame)
            
            # Log
            logger.log_frame(frame_idx, reader.fps, detections, 
                           people_tracks, ball_state, possession_state)
            
            # Progress
            frame_count += 1
            if frame_count % 100 == 0:
                progress = (frame_count / reader.frame_count) * 100
                print(f"  Progress: {frame_count}/{reader.frame_count} frames ({progress:.1f}%)")
        
        print("-"*80)
        print(f"✓ Processed {frame_count} frames")
        
    finally:
        # Cleanup
        reader.release()
        writer.release()
    
    # Export logs
    print("\n[8/8] Exporting logs and statistics...")
    logger.export(output_name)
    
    # Print statistics
    if possession_analyzer is not None:
        print("\n" + "="*80)
        print("Possession Statistics")
        print("="*80)
        stats = possession_analyzer.get_statistics()
        for team_id, team_stats in stats['team_statistics'].items():
            print(f"\n{team_id.upper()}:")
            print(f"  Possession: {team_stats['possession_pct']:.1f}%")
            print(f"  Time: {team_stats['possession_time_sec']:.1f}s")
            print(f"  Passes: {team_stats['passes']}")
        print(f"\nTotal passes detected: {stats['total_passes']}")

    
    print("\n" + "="*80)
    print("✓ Processing complete!")
    print("="*80)
    print(f"Output video: {output_video_path}")
    print(f"Logs directory: {output_root / 'logs'}")
    print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Football Video Analysis - Multi-class tracking system"
    )
    
    parser.add_argument(
        'video',
        type=str,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--preset',
        type=str,
        choices=['fast', 'accurate'],
        help='Use preset configuration (fast or accurate)'
    )
    
    parser.add_argument(
        '--output-name',
        type=str,
        help='Output file base name (default: video filename)'
    )
    
    parser.add_argument(
        '--no-player',
        action='store_true',
        help='Disable player layer'
    )
    
    parser.add_argument(
        '--no-ball',
        action='store_true',
        help='Disable ball layer'
    )
    
    parser.add_argument(
        '--no-referee',
        action='store_true',
        help='Disable referee layer'
    )
    
    parser.add_argument(
        '--no-goalkeeper',
        action='store_true',
        help='Disable goalkeeper layer'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to process (useful for testing on CPU)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config, args.preset)
    
    # Override layer settings from command line
    if args.no_player:
        config['render']['layers_enabled']['player'] = False
    if args.no_ball:
        config['render']['layers_enabled']['ball'] = False
    if args.no_referee:
        config['render']['layers_enabled']['referee'] = False
    if args.no_goalkeeper:
        config['render']['layers_enabled']['goalkeeper'] = False
    
    # Run inference
    run_inference(args.video, config, args.output_name, args.max_frames)


if __name__ == '__main__':
    main()
