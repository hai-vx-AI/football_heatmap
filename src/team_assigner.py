"""
Team color assignment module using jersey colors with temporal smoothing.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
from sklearn.cluster import KMeans

from .utils import (crop_jersey_roi, extract_color_feature, color_distance_lab, 
                    ema_update, euclidean_distance, bbox_center)


class TeamAssigner:
    """
    Assigns team IDs to players based on jersey colors with temporal stability.
    """
    
    def __init__(self, config: dict):
        """
        Initialize team assigner.
        
        Args:
            config: Team color configuration
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
        if not self.enabled:
            return
        
        # Jersey cropping parameters
        crop_config = config.get('jersey_crop', {})
        self.jersey_x_range = (crop_config.get('x_start', 0.2), 
                               crop_config.get('x_end', 0.8))
        self.jersey_y_range = (crop_config.get('y_start', 0.15), 
                               crop_config.get('y_end', 0.55))
        
        # Minimum size filters
        self.min_person_height = config.get('min_person_height_px', 60)
        self.min_roi_width = config.get('min_roi_width_px', 20)
        self.min_roi_height = config.get('min_roi_height_px', 30)
        
        # Color extraction
        self.color_space = config.get('color_space', 'Lab')
        self.use_median = config.get('use_median', True)
        
        # Grass mask parameters
        grass_config = config.get('grass_mask', {})
        self.grass_mask_enabled = grass_config.get('enabled', True)
        self.grass_mask_params = {
            'h_range': tuple(grass_config.get('h_range', [35, 85])),
            's_min': grass_config.get('s_min', 60),
            'v_min': grass_config.get('v_min', 40)
        } if self.grass_mask_enabled else None
        
        # Dark pixel filter
        filter_config = config.get('filter_dark_pixels', {})
        self.filter_dark = filter_config.get('enabled', True)
        self.min_l_value = filter_config.get('l_min', 20)
        
        # Clustering parameters
        cluster_config = config.get('clustering', {})
        self.n_teams = cluster_config.get('n_teams', 2)
        self.warmup_frames = cluster_config.get('warmup_frames', 200)
        self.min_init_samples = cluster_config.get('min_init_samples', 100)
        self.centroid_update_beta = cluster_config.get('centroid_update_beta', 0.01)
        
        # Per-track smoothing
        track_config = config.get('per_track', {})
        self.ema_alpha = track_config.get('ema_alpha', 0.2)
        self.vote_window = track_config.get('vote_window_frames', 30)
        self.vote_threshold = track_config.get('vote_threshold', 0.6)
        
        # Goalkeeper assignment
        gk_config = config.get('goalkeeper', {})
        self.gk_method = gk_config.get('method', 'neighbor_vote')
        self.gk_min_neighbors = gk_config.get('min_neighbors', 3)
        self.gk_neighbor_radius = gk_config.get('neighbor_radius_px', 300)
        self.gk_vote_threshold = gk_config.get('vote_threshold', 0.5)
        
        # State
        self.centroids = None  # (n_teams, 3) array of Lab centroids
        self.global_samples = []  # Samples collected during warmup
        self.track_color_ema = {}  # track_id -> Lab color EMA
        self.track_votes = defaultdict(lambda: deque(maxlen=self.vote_window))
        self.track_team_stable = {}  # track_id -> stable team_id
        self.frame_count = 0
        self.is_warmed_up = False
    
    def assign_teams(self, tracks: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Assign team IDs to player tracks.
        
        Args:
            tracks: List of track dicts (modified in-place)
            frame: BGR frame for color extraction
        
        Returns:
            Tracks with team_id field updated
        """
        if not self.enabled:
            return tracks
        
        self.frame_count += 1
        
        # Separate tracks by class
        players = [t for t in tracks if t['cls'] == 'player']
        referees = [t for t in tracks if t['cls'] == 'referee']
        goalkeepers = [t for t in tracks if t['cls'] == 'goalkeeper']
        
        # Process referees (no team)
        for ref in referees:
            ref['team_id'] = None
        
        # Process players
        for player in players:
            team_id = self._assign_player_team(player, frame)
            player['team_id'] = team_id
        
        # Process goalkeepers (special logic)
        for gk in goalkeepers:
            team_id = self._assign_goalkeeper_team(gk, players, frame)
            gk['team_id'] = team_id
        
        return tracks
    
    def _assign_player_team(self, track: Dict, frame: np.ndarray) -> Optional[str]:
        """
        Assign team to a player track.
        
        Args:
            track: Player track dict
            frame: BGR frame
        
        Returns:
            Team ID string ('team0', 'team1') or None
        """
        track_id = track['track_id']
        bbox = track['bbox']
        
        # Check if bbox is large enough
        h = bbox[3] - bbox[1]
        if h < self.min_person_height:
            return self.track_team_stable.get(track_id, None)
        
        # Extract jersey color
        color_feat = self._extract_jersey_color(frame, bbox)
        
        if color_feat is None:
            # No valid color extracted, use previous stable assignment
            return self.track_team_stable.get(track_id, None)
        
        # Update EMA
        self.track_color_ema[track_id] = ema_update(
            self.track_color_ema.get(track_id),
            color_feat,
            self.ema_alpha
        )
        
        # Warmup phase: collect samples
        if not self.is_warmed_up:
            self.global_samples.append(self.track_color_ema[track_id].copy())
            
            if (len(self.global_samples) >= self.min_init_samples and 
                self.frame_count >= self.warmup_frames):
                self._initialize_centroids()
            
            return None  # No stable assignment during warmup
        
        # Assign team by nearest centroid
        team_raw = self._find_nearest_centroid(self.track_color_ema[track_id])
        
        # Update vote deque
        self.track_votes[track_id].append(team_raw)
        
        # Compute stable team from majority vote
        team_stable = self._majority_vote(self.track_votes[track_id])
        
        if team_stable is not None:
            self.track_team_stable[track_id] = team_stable
        
        return team_stable
    
    def _assign_goalkeeper_team(self, gk_track: Dict, player_tracks: List[Dict], 
                                frame: np.ndarray) -> Optional[str]:
        """
        Assign team to goalkeeper using neighbor voting.
        
        Args:
            gk_track: Goalkeeper track dict
            player_tracks: List of player tracks with team assignments
            frame: BGR frame
        
        Returns:
            Team ID or None
        """
        if self.gk_method == 'neighbor_vote':
            return self._goalkeeper_neighbor_vote(gk_track, player_tracks)
        else:
            # Fallback: treat like player (color-based)
            return self._assign_player_team(gk_track, frame)
    
    def _goalkeeper_neighbor_vote(self, gk_track: Dict, 
                                  player_tracks: List[Dict]) -> Optional[str]:
        """
        Assign goalkeeper team based on nearest players' teams.
        
        Args:
            gk_track: Goalkeeper track
            player_tracks: Player tracks with team_id
        
        Returns:
            Team ID or None
        """
        gk_center = bbox_center(gk_track['bbox'])
        
        # Find nearby players with team assignment
        neighbors = []
        for player in player_tracks:
            if player['team_id'] is None:
                continue
            
            player_center = bbox_center(player['bbox'])
            dist = euclidean_distance(gk_center, player_center)
            
            if dist <= self.gk_neighbor_radius:
                neighbors.append((dist, player['team_id']))
        
        if len(neighbors) < self.gk_min_neighbors:
            return None  # Not enough neighbors
        
        # Sort by distance and take K nearest
        neighbors.sort(key=lambda x: x[0])
        nearest_k = neighbors[:self.gk_min_neighbors * 2]  # Consider more neighbors
        
        # Count votes
        votes = {}
        for _, team_id in nearest_k:
            votes[team_id] = votes.get(team_id, 0) + 1
        
        # Get majority team
        if len(votes) == 0:
            return None
        
        max_votes = max(votes.values())
        total_votes = sum(votes.values())
        
        if max_votes / total_votes >= self.gk_vote_threshold:
            return max(votes, key=votes.get)
        
        return None
    
    def _extract_jersey_color(self, frame: np.ndarray, 
                              bbox: List[float]) -> Optional[np.ndarray]:
        """
        Extract dominant jersey color from bbox.
        
        Args:
            frame: BGR frame
            bbox: Person bounding box
        
        Returns:
            Lab color feature [L, a, b] or None
        """
        # Crop jersey ROI
        jersey_crop = crop_jersey_roi(frame, bbox, 
                                       self.jersey_x_range, 
                                       self.jersey_y_range)
        
        if jersey_crop is None:
            return None
        
        # Check minimum size
        h, w = jersey_crop.shape[:2]
        if w < self.min_roi_width or h < self.min_roi_height:
            return None
        
        # Extract color feature
        color_feat = extract_color_feature(
            jersey_crop,
            use_median=self.use_median,
            grass_mask_params=self.grass_mask_params,
            min_l=self.min_l_value if self.filter_dark else 0
        )
        
        return color_feat
    
    def _initialize_centroids(self):
        """Initialize team centroids using K-means clustering."""
        if len(self.global_samples) < self.n_teams:
            return
        
        samples = np.array(self.global_samples)
        
        # Run K-means
        kmeans = KMeans(n_clusters=self.n_teams, random_state=42, n_init=10)
        kmeans.fit(samples)
        
        self.centroids = kmeans.cluster_centers_
        self.is_warmed_up = True
        
        print(f"Team centroids initialized with {len(samples)} samples")
        print(f"Centroid 0 (Lab): {self.centroids[0]}")
        print(f"Centroid 1 (Lab): {self.centroids[1]}")
    
    def _find_nearest_centroid(self, color: np.ndarray) -> int:
        """
        Find nearest centroid to given color.
        
        Args:
            color: Lab color vector
        
        Returns:
            Centroid index (0 or 1)
        """
        if self.centroids is None:
            return 0
        
        distances = [color_distance_lab(color, c) for c in self.centroids]
        return int(np.argmin(distances))
    
    def _majority_vote(self, votes: deque) -> Optional[str]:
        """
        Get majority team from vote history.
        
        Args:
            votes: Deque of team indices
        
        Returns:
            Stable team ID or None
        """
        if len(votes) == 0:
            return None
        
        # Count votes
        counts = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        
        max_count = max(counts.values())
        total = len(votes)
        
        if max_count / total >= self.vote_threshold:
            team_idx = max(counts, key=counts.get)
            return f'team{team_idx}'
        
        return None
    
    def update_centroids_slowly(self):
        """
        Slowly update centroids with recent assignments (optional, call periodically).
        """
        if not self.is_warmed_up or self.centroids is None:
            return
        
        # Collect recent samples per team
        team_samples = {i: [] for i in range(self.n_teams)}
        
        for track_id, color_ema in self.track_color_ema.items():
            team_id = self.track_team_stable.get(track_id)
            if team_id is not None:
                team_idx = int(team_id.replace('team', ''))
                team_samples[team_idx].append(color_ema)
        
        # Update each centroid with EMA
        for team_idx, samples in team_samples.items():
            if len(samples) > 10:  # Need enough samples
                mean_color = np.mean(samples, axis=0)
                self.centroids[team_idx] = (
                    (1 - self.centroid_update_beta) * self.centroids[team_idx] +
                    self.centroid_update_beta * mean_color
                )
