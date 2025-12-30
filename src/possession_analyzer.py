"""
Ball possession and pass detection analyzer.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque, defaultdict

from .utils import bbox_center, euclidean_distance


class PossessionAnalyzer:
    """
    Analyzes ball possession, passes, and team control statistics.
    """
    
    def __init__(self, config: dict, fps: float = 25.0):
        """
        Initialize possession analyzer.
        
        Args:
            config: Possession analysis configuration
            fps: Video frame rate
        """
        self.config = config
        self.fps = fps
        
        # Possession detection parameters
        self.possession_radius_px = config.get('possession_radius_px', 150)
        self.min_frames_for_possession = config.get('min_frames_for_possession', 5)
        self.smoothing_window = config.get('smoothing_window_frames', 15)
        
        # Pass detection parameters
        self.min_pass_distance_px = config.get('min_pass_distance_px', 200)
        self.max_pass_duration_frames = config.get('max_pass_duration_frames', 50)
        self.pass_detection_enabled = config.get('pass_detection_enabled', True)
        
        # State tracking
        self.possession_history = deque(maxlen=self.smoothing_window)
        self.current_possessor_id = None
        self.current_possessor_team = None
        self.possession_start_frame = None
        
        # Statistics
        self.team_possession_frames = defaultdict(int)  # team_id -> frame count
        self.player_possession_frames = defaultdict(int)  # track_id -> frame count
        self.passes = []  # List of detected passes
        self.last_possessor = None  # For pass detection
        
        # Current frame data
        self.frame_idx = 0
    
    def update(self, frame_idx: int, player_tracks: List[Dict], 
               ball_state: Dict) -> Dict:
        """
        Update possession analysis for current frame.
        
        Args:
            frame_idx: Current frame index
            player_tracks: List of player tracks with team assignments
            ball_state: Ball state dict from ball tracker
        
        Returns:
            Possession state dict with keys:
                - possessor_id: track_id of player with ball (or None)
                - possessor_team: team_id of possessing team (or None)
                - possession_duration_sec: how long current possessor has ball
                - team_possession_pct: dict of team_id -> possession percentage
                - recent_pass: dict with pass info if pass just detected
        """
        self.frame_idx = frame_idx
        
        # Check if ball is available
        if ball_state['status'] == 'lost' or ball_state['center'] is None:
            self._update_no_possession()
            return self._make_output()
        
        ball_center = ball_state['center']
        
        # Find nearest player to ball
        nearest_player = self._find_nearest_player(player_tracks, ball_center)
        
        if nearest_player is None:
            self._update_no_possession()
            return self._make_output()
        
        player_id = nearest_player['track_id']
        player_team = nearest_player.get('team_id')
        player_center = bbox_center(nearest_player['bbox'])
        distance = euclidean_distance(player_center, ball_center)
        
        # Check if player is close enough to have possession
        if distance <= self.possession_radius_px:
            self._update_possession(player_id, player_team)
        else:
            self._update_no_possession()
        
        return self._make_output()
    
    def _find_nearest_player(self, player_tracks: List[Dict], 
                            ball_center: Tuple[float, float]) -> Optional[Dict]:
        """
        Find player nearest to ball (only players, not goalkeepers/referees).
        
        Args:
            player_tracks: List of tracks
            ball_center: Ball position (cx, cy)
        
        Returns:
            Nearest player track or None
        """
        players = [t for t in player_tracks if t['cls'] == 'player']
        
        if len(players) == 0:
            return None
        
        # Find player with minimum distance
        min_dist = float('inf')
        nearest = None
        
        for player in players:
            center = bbox_center(player['bbox'])
            dist = euclidean_distance(center, ball_center)
            
            if dist < min_dist:
                min_dist = dist
                nearest = player
        
        return nearest
    
    def _update_possession(self, player_id: int, player_team: Optional[str]):
        """Update state when player has possession."""
        # Check if possession changed
        if player_id != self.current_possessor_id:
            # Detect pass
            if (self.pass_detection_enabled and 
                self.current_possessor_id is not None and
                self.current_possessor_team == player_team and
                player_team is not None):
                self._detect_pass(player_id, player_team)
            
            # Start new possession
            self.current_possessor_id = player_id
            self.current_possessor_team = player_team
            self.possession_start_frame = self.frame_idx
        
        # Record possession
        self.possession_history.append((player_id, player_team))
        
        # Update statistics (if stable possession)
        frames_with_possession = sum(
            1 for pid, _ in self.possession_history if pid == player_id
        )
        
        if frames_with_possession >= self.min_frames_for_possession:
            self.player_possession_frames[player_id] += 1
            if player_team is not None:
                self.team_possession_frames[player_team] += 1
            self.last_possessor = (player_id, player_team)
    
    def _update_no_possession(self):
        """Update state when no player has clear possession."""
        self.possession_history.append((None, None))
        
        # Check if possession should end
        none_count = sum(1 for pid, _ in self.possession_history if pid is None)
        
        if none_count > self.smoothing_window // 2:
            self.current_possessor_id = None
            self.current_possessor_team = None
            self.possession_start_frame = None
    
    def _detect_pass(self, receiver_id: int, team: str):
        """
        Detect and record a pass.
        
        Args:
            receiver_id: Track ID of receiving player
            team: Team ID
        """
        if self.last_possessor is None:
            return
        
        passer_id, passer_team = self.last_possessor
        
        # Check if pass is valid
        if passer_team != team:
            return
        
        duration_frames = self.frame_idx - (self.possession_start_frame or self.frame_idx)
        
        if duration_frames > self.max_pass_duration_frames:
            return  # Too long, probably not a direct pass
        
        # Record pass
        pass_event = {
            'frame': self.frame_idx,
            'time_sec': self.frame_idx / self.fps,
            'passer_id': passer_id,
            'receiver_id': receiver_id,
            'team_id': team,
            'duration_frames': duration_frames,
            'duration_sec': duration_frames / self.fps
        }
        
        self.passes.append(pass_event)
    
    def _make_output(self) -> Dict:
        """Create possession state output."""
        # Calculate possession duration
        possession_duration_sec = 0.0
        if (self.current_possessor_id is not None and 
            self.possession_start_frame is not None):
            frames = self.frame_idx - self.possession_start_frame
            possession_duration_sec = frames / self.fps
        
        # Calculate team possession percentages
        total_frames = sum(self.team_possession_frames.values())
        team_pct = {}
        
        if total_frames > 0:
            for team_id, frames in self.team_possession_frames.items():
                team_pct[team_id] = (frames / total_frames) * 100.0
        
        # Check for recent pass
        recent_pass = None
        if len(self.passes) > 0 and self.passes[-1]['frame'] == self.frame_idx:
            recent_pass = self.passes[-1]
        
        return {
            'possessor_id': self.current_possessor_id,
            'possessor_team': self.current_possessor_team,
            'possession_duration_sec': possession_duration_sec,
            'team_possession_pct': team_pct,
            'recent_pass': recent_pass,
            'total_passes': len(self.passes)
        }
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive possession statistics.
        
        Returns:
            Statistics dict with team and player possession data
        """
        total_frames = sum(self.team_possession_frames.values())
        
        # Team statistics
        team_stats = {}
        for team_id, frames in self.team_possession_frames.items():
            pct = (frames / total_frames * 100.0) if total_frames > 0 else 0.0
            time_sec = frames / self.fps
            
            team_stats[team_id] = {
                'possession_frames': frames,
                'possession_time_sec': time_sec,
                'possession_pct': pct,
                'passes': len([p for p in self.passes if p['team_id'] == team_id])
            }
        
        # Player statistics
        player_stats = {}
        for player_id, frames in self.player_possession_frames.items():
            time_sec = frames / self.fps
            
            player_stats[player_id] = {
                'possession_frames': frames,
                'possession_time_sec': time_sec,
                'passes_made': len([p for p in self.passes if p['passer_id'] == player_id]),
                'passes_received': len([p for p in self.passes if p['receiver_id'] == player_id])
            }
        
        return {
            'team_statistics': team_stats,
            'player_statistics': player_stats,
            'total_passes': len(self.passes),
            'passes': self.passes
        }
    
    def reset(self):
        """Reset analyzer state."""
        self.possession_history.clear()
        self.current_possessor_id = None
        self.current_possessor_team = None
        self.possession_start_frame = None
        self.team_possession_frames.clear()
        self.player_possession_frames.clear()
        self.passes.clear()
        self.last_possessor = None
        self.frame_idx = 0
