"""
Logger module for exporting tracking results to JSON, JSONL, and CSV formats.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import pandas as pd


class Logger:
    """
    Logs tracking data and exports to multiple formats.
    """
    
    def __init__(self, config: dict, video_meta: dict, output_dir: str):
        """
        Initialize logger.
        
        Args:
            config: Logger configuration
            video_meta: Video metadata dict
            output_dir: Output directory path
        """
        self.config = config
        self.video_meta = video_meta
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        if config.get('debug', {}).get('enabled', False):
            self.debug_dir = self.output_dir / "debug"
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.debug_dir = None
        
        # Export formats
        self.export_formats = config.get('export_formats', ['jsonl', 'csv', 'json'])
        
        # Data storage
        self.frames_data = []  # Per-frame data
        self.track_data = defaultdict(list)  # Per-track data
        
        # Debug options
        self.debug_config = config.get('debug', {})
        self.max_debug_frames = self.debug_config.get('max_debug_frames', 100)
        self.debug_frame_count = 0
    
    def log_frame(self, frame_idx: int, fps: float, detections: List[Dict],
                  tracks: List[Dict], ball_state: Dict, possession_state: Optional[Dict] = None):
        """
        Log data for a single frame.
        
        Args:
            frame_idx: Frame index
            fps: Video FPS
            detections: Raw detections
            tracks: People tracks
            ball_state: Ball tracking state
            possession_state: Optional possession state
        """
        timestamp = frame_idx / fps if fps > 0 else 0.0
        
        # Build frame data
        frame_data = {
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'detections': self._serialize_detections(detections),
            'tracks': self._serialize_tracks(tracks),
            'ball': self._serialize_ball_state(ball_state)
        }
        
        # Add possession data if available
        if possession_state is not None:
            frame_data['possession'] = {
                'possessor_id': possession_state.get('possessor_id'),
                'possessor_team': possession_state.get('possessor_team'),
                'possession_duration_sec': possession_state.get('possession_duration_sec', 0.0),
                'team_possession_pct': possession_state.get('team_possession_pct', {}),
                'recent_pass': possession_state.get('recent_pass')
            }
        
        self.frames_data.append(frame_data)
        
        # Update per-track data
        for track in tracks:
            track_id = track['track_id']
            track_entry = {
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'bbox': track['bbox'],
                'conf': track['conf'],
                'cls': track['cls'],
                'team_id': track.get('team_id')
            }
            
            # Add possession flag
            if possession_state is not None:
                track_entry['has_possession'] = (possession_state.get('possessor_id') == track_id)
            
            self.track_data[track_id].append(track_entry)
    
    def _serialize_detections(self, detections: List[Dict]) -> List[Dict]:
        """Serialize detections to JSON-friendly format."""
        return [
            {
                'bbox': [float(v) for v in det['bbox']],
                'conf': float(det['conf']),
                'cls': det['cls'],
                'cls_id': int(det['cls_id'])
            }
            for det in detections
        ]
    
    def _serialize_tracks(self, tracks: List[Dict]) -> List[Dict]:
        """Serialize tracks to JSON-friendly format."""
        return [
            {
                'track_id': int(track['track_id']),
                'bbox': [float(v) for v in track['bbox']],
                'conf': float(track['conf']),
                'cls': track['cls'],
                'cls_id': int(track['cls_id']),
                'team_id': track.get('team_id')
            }
            for track in tracks
        ]
    
    def _serialize_ball_state(self, ball_state: Dict) -> Dict:
        """Serialize ball state to JSON-friendly format."""
        center = ball_state.get('center')
        bbox = ball_state.get('bbox')
        
        return {
            'status': ball_state['status'],
            'center': [float(center[0]), float(center[1])] if center else None,
            'bbox': [float(v) for v in bbox] if bbox else None,
            'conf': float(ball_state['conf']),
            'is_predicted': bool(ball_state['is_predicted'])
        }
    
    def export(self, video_name: str):
        """
        Export all logged data to files.
        
        Args:
            video_name: Base name for output files
        """
        print(f"Exporting logs to {self.logs_dir}...")
        
        # Export metadata
        if 'json' in self.export_formats:
            self._export_metadata(video_name)
        
        # Export per-frame data (JSONL)
        if 'jsonl' in self.export_formats:
            self._export_frames_jsonl(video_name)
        
        # Export track-level data (CSV)
        if 'csv' in self.export_formats:
            self._export_tracks_csv(video_name)
            self._export_frames_csv(video_name)
        
        print(f"Export complete. Files saved to {self.logs_dir}")
    
    def _export_metadata(self, video_name: str):
        """Export metadata JSON."""
        meta_path = self.logs_dir / f"{video_name}_meta.json"
        
        meta = {
            'video': self.video_meta,
            'total_frames': len(self.frames_data),
            'total_tracks': len(self.track_data),
            'config': self.config
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"  Metadata: {meta_path}")
    
    def _export_frames_jsonl(self, video_name: str):
        """Export per-frame data as JSONL."""
        jsonl_path = self.logs_dir / f"{video_name}_frames.jsonl"
        
        with open(jsonl_path, 'w') as f:
            for frame_data in self.frames_data:
                f.write(json.dumps(frame_data) + '\n')
        
        print(f"  Frames JSONL: {jsonl_path} ({len(self.frames_data)} frames)")
    
    def _export_frames_csv(self, video_name: str):
        """Export simplified per-frame summary as CSV."""
        csv_path = self.logs_dir / f"{video_name}_frames_summary.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame_idx', 'timestamp', 'num_detections', 'num_tracks',
                'num_players', 'num_referees', 'num_goalkeepers',
                'ball_status', 'ball_x', 'ball_y'
            ])
            
            for frame_data in self.frames_data:
                tracks = frame_data['tracks']
                ball = frame_data['ball']
                
                num_players = sum(1 for t in tracks if t['cls'] == 'player')
                num_referees = sum(1 for t in tracks if t['cls'] == 'referee')
                num_goalkeepers = sum(1 for t in tracks if t['cls'] == 'goalkeeper')
                
                ball_center = ball['center']
                ball_x = ball_center[0] if ball_center else None
                ball_y = ball_center[1] if ball_center else None
                
                writer.writerow([
                    frame_data['frame_idx'],
                    f"{frame_data['timestamp']:.3f}",
                    len(frame_data['detections']),
                    len(tracks),
                    num_players,
                    num_referees,
                    num_goalkeepers,
                    ball['status'],
                    f"{ball_x:.1f}" if ball_x else '',
                    f"{ball_y:.1f}" if ball_y else ''
                ])
        
        print(f"  Frames summary CSV: {csv_path}")
    
    def _export_tracks_csv(self, video_name: str):
        """Export track-level statistics as CSV."""
        csv_path = self.logs_dir / f"{video_name}_tracks.csv"
        
        # Build track summaries
        track_summaries = []
        
        for track_id, track_history in self.track_data.items():
            if len(track_history) == 0:
                continue
            
            # Compute statistics
            first_frame = track_history[0]
            last_frame = track_history[-1]
            
            # Most common class and team
            classes = [f['cls'] for f in track_history]
            teams = [f['team_id'] for f in track_history if f['team_id'] is not None]
            
            most_common_cls = max(set(classes), key=classes.count)
            most_common_team = max(set(teams), key=teams.count) if teams else None
            
            # Average confidence
            avg_conf = sum(f['conf'] for f in track_history) / len(track_history)
            
            # Duration
            duration = last_frame['timestamp'] - first_frame['timestamp']
            
            track_summaries.append({
                'track_id': track_id,
                'class': most_common_cls,
                'team_id': most_common_team,
                'first_frame': first_frame['frame_idx'],
                'last_frame': last_frame['frame_idx'],
                'num_frames': len(track_history),
                'duration_sec': duration,
                'avg_confidence': avg_conf,
                'first_timestamp': first_frame['timestamp'],
                'last_timestamp': last_frame['timestamp']
            })
        
        # Write CSV
        if track_summaries:
            df = pd.DataFrame(track_summaries)
            df = df.sort_values('track_id')
            df.to_csv(csv_path, index=False, float_format='%.3f')
            
            print(f"  Tracks CSV: {csv_path} ({len(track_summaries)} tracks)")
        else:
            print("  No tracks to export")
    
    def save_debug_crop(self, frame_idx: int, crop_type: str, 
                       crop_img: Any, identifier: str):
        """
        Save debug crop image.
        
        Args:
            frame_idx: Frame index
            crop_type: Type of crop ('jersey', 'ball_roi', 'field_mask')
            crop_img: Image to save
            identifier: Unique identifier (e.g., track_id)
        """
        if self.debug_dir is None:
            return
        
        if self.debug_frame_count >= self.max_debug_frames:
            return
        
        # Check if this crop type is enabled
        save_key = f'save_{crop_type}'
        if not self.debug_config.get(save_key, False):
            return
        
        # Create subdirectory
        crop_dir = self.debug_dir / crop_type
        crop_dir.mkdir(exist_ok=True)
        
        # Save image
        import cv2
        filename = f"frame_{frame_idx:06d}_{identifier}.jpg"
        filepath = crop_dir / filename
        cv2.imwrite(str(filepath), crop_img)
        
        self.debug_frame_count += 1
