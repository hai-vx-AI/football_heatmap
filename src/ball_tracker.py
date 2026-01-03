"""
Ball tracker using Kalman Filter with reacquisition and false positive filtering.

This module implements a robust single-target ball tracking system that combines:
- YOLO-based ball detection
- Kalman Filter for motion prediction (constant velocity model)
- Motion gating for association
- Automatic reacquisition when ball is lost
- Trajectory history for visualization

Author: Football Analysis Team
Last Updated: January 2026
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque
from filterpy.kalman import KalmanFilter

from .utils import bbox_center, euclidean_distance, build_roi_around_point, expand_bbox


class BallTracker:
    """
    Single-target ball tracker with Kalman Filter, motion gating, and reacquisition.
    
    Features:
    - Constant velocity Kalman Filter (4D state: [cx, cy, vx, vy])
    - Motion gating to reject false positives
    - Automatic reacquisition after temporary occlusion
    - Trajectory history tracking
    - Configurable prediction buffer
    
    States:
    - 'detected': Ball detected by YOLO and matched with prediction
    - 'predicted': Ball not detected, using Kalman prediction
    - 'lost': Ball lost for too long (exceeds track_buffer)
    """
    
    def __init__(self, config: dict, fps: float = 25.0):
        """
        Initialize ball tracker.
        
        Args:
            config: Ball tracker configuration dict with keys:
                - track_buffer: Max frames to keep predicting (default: 15)
                - max_displacement_px_per_frame: Motion gate radius (default: 80)
                - gate_radius_scale: Gate multiplier (default: 1.5)
                - kalman: Kalman filter parameters
                - reacquire: Reacquisition settings
            fps: Video frame rate for velocity computation
        """
        self.config = config
        self.fps = fps
        self.dt = 1.0 / fps  # Time step in seconds
        
        # Tracking parameters
        self.track_buffer = config.get('track_buffer', 15)
        self.max_displacement_px = config.get('max_displacement_px_per_frame', 80)
        self.gate_radius_scale = config.get('gate_radius_scale', 1.5)
        self.trajectory_length = config.get('trajectory_length', 30)
        
        # Reacquisition parameters
        self.reacquire_config = config.get('reacquire', {})
        self.reacquire_enabled = self.reacquire_config.get('enabled', True)
        self.reacquire_roi_scale = self.reacquire_config.get('roi_scale', 4.0)
        self.reacquire_roi_size = self.reacquire_config.get('roi_size_px', 320)
        self.reacquire_trigger = self.reacquire_config.get('trigger_after_frames', 5)
        self.reacquire_frequency = self.reacquire_config.get('reacquire_every_n_frames', 3)
        
        # Kalman filter setup
        self._init_kalman_filter(config.get('kalman', {}))
        
        # Tracking state
        self.has_track = False
        self.miss_counter = 0
        self.status = 'lost'  # 'detected', 'predicted', 'lost'
        self.last_bbox = None
        self.last_center = None
        self.predicted_center = None
        self.frame_idx = 0
        
        # Trajectory history
        self.trajectory: Deque[Tuple[float, float]] = deque(maxlen=self.trajectory_length)
        self.velocity_history: Deque[Tuple[float, float]] = deque(maxlen=10)
    
    def _init_kalman_filter(self, kalman_config: dict):
        """
        Initialize Kalman Filter for ball tracking.
        
        State Vector (4D):
            x = [cx, cy, vx, vy]
            - cx, cy: Center position (pixels)
            - vx, vy: Velocity (pixels/frame)
        
        Measurement Vector (2D):
            z = [cx, cy]
            - Only position is measured, velocity is estimated
        
        Motion Model:
            Constant Velocity Model (assumes ball moves with constant velocity)
            x(t+1) = F × x(t)
            where F is state transition matrix
        
        Args:
            kalman_config: Configuration dict with optional keys:
                - process_noise_pos: Position uncertainty (default: 1.0)
                - process_noise_vel: Velocity uncertainty (default: 0.1)
                - measurement_noise: Detection noise (default: 10.0)
        """
        # Initialize Kalman Filter: 4D state, 2D measurement
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix F (constant velocity model)
        # x(t+1) = x(t) + vx*dt
        # y(t+1) = y(t) + vy*dt
        # vx(t+1) = vx(t)
        # vy(t+1) = vy(t)
        self.kf.F = np.array([
            [1, 0, self.dt, 0],     # cx_new = cx + vx * dt
            [0, 1, 0, self.dt],     # cy_new = cy + vy * dt
            [0, 0, 1, 0],           # vx_new = vx (constant)
            [0, 0, 0, 1]            # vy_new = vy (constant)
        ])
        
        # Measurement matrix H (observe position only)
        # z = H × x
        # Measure: [cx, cy] from state [cx, cy, vx, vy]
        self.kf.H = np.array([
            [1, 0, 0, 0],  # Measure cx
            [0, 1, 0, 0]   # Measure cy
        ])
        
        # Process noise covariance Q
        # Higher values = more uncertainty in motion model
        q_pos = kalman_config.get('process_noise_pos', 1.0)
        q_vel = kalman_config.get('process_noise_vel', 0.1)
        self.kf.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        
        # Measurement noise covariance R
        # Higher values = less trust in measurements
        r = kalman_config.get('measurement_noise', 10.0)
        self.kf.R = np.array([[r, 0], [0, r]])
        
        # Initial covariance (high uncertainty at start)
        self.kf.P = np.eye(4) * 1000
    
    def update(self, frame_idx: int, ball_detections: List[Dict], 
               frame_shape: Tuple[int, int], detector=None) -> Dict:
        """
        Update ball tracker with current frame detections.
        
        Tracking Pipeline:
        1. Predict next state using Kalman Filter
        2. Select best detection candidate using motion gating
        3. Update Kalman Filter if match found, else increment miss counter
        4. Attempt reacquisition if ball lost for too long
        5. Return ball state
        
        Args:
            frame_idx: Current frame index
            ball_detections: List of ball detection dicts from YOLO
            frame_shape: (height, width) of frame for ROI computation
            detector: Detector instance for reacquisition (optional)
        
        Returns:
            Ball state dict with keys:
                - status: 'detected' | 'predicted' | 'lost'
                - center: (cx, cy) or None
                - bbox: [x1, y1, x2, y2] or None
                - conf: Detection confidence (0.0 if predicted)
                - is_predicted: True if using Kalman prediction
                - miss_counter: Consecutive frames without detection
                - velocity: (vx, vy) estimated velocity
                - trajectory: List of recent centers
        """
        self.frame_idx = frame_idx
        
        # Step 1: Predict next state using Kalman Filter
        if self.has_track:
            self.kf.predict()
            self.predicted_center = (self.kf.x[0, 0], self.kf.x[1, 0])
        else:
            self.predicted_center = None
        
        # Step 2: Select best candidate with motion gating
        candidate = self._select_best_candidate(ball_detections)
        
        # Step 3: Update or predict
        if candidate is not None:
            return self._handle_detection(candidate)
        else:
            return self._handle_no_detection(detector, frame_shape, ball_detections)
    
    def _handle_detection(self, candidate: Dict) -> Dict:
        """
        Handle successful ball detection.
        
        Args:
            candidate: Best matched detection dict
        
        Returns:
            Ball state dict
        """
        center = bbox_center(candidate['bbox'])
        
        if not self.has_track:
            # Initialize new track
            self._initialize_track(center, candidate['bbox'])
        else:
            # Update existing track with measurement
            measurement = np.array([[center[0]], [center[1]]])
            self.kf.update(measurement)
        
        # Update state
        self.last_bbox = candidate['bbox']
        self.last_center = center
        self.miss_counter = 0
        self.status = 'detected'
        self.has_track = True
        
        # Update trajectory
        self.trajectory.append(center)
        
        # Extract velocity from Kalman state
        velocity = (self.kf.x[2, 0], self.kf.x[3, 0])
        self.velocity_history.append(velocity)
        
        return self._make_output(candidate['conf'], is_predicted=False)
    
    def _handle_no_detection(self, detector, frame_shape: Tuple[int, int], 
                            full_frame_dets: List[Dict]) -> Dict:
        """
        Handle case when no ball detection matched.
        
        Args:
            detector: Detector instance for reacquisition
            frame_shape: Frame dimensions
            full_frame_dets: All ball detections in frame
        
        Returns:
            Ball state dict
        """
        # No track established yet
        if not self.has_track:
            return self._make_lost_output()
        
        # Increment miss counter
        self.miss_counter += 1
        
        # Keep predicting if within buffer
        if self.miss_counter <= self.track_buffer:
            self.status = 'predicted'
            
            # Add predicted center to trajectory
            if self.predicted_center is not None:
                self.trajectory.append(self.predicted_center)
            
            return self._make_output(0.0, is_predicted=True)
        
        # Lost - try reacquisition
        if self._should_attempt_reacquisition():
            reacquired = self._try_reacquire(detector, frame_shape, full_frame_dets)
            if reacquired:
                return reacquired
        
        # Ball is completely lost
        self.status = 'lost'
        self.has_track = False
        return self._make_lost_output()
    
    def _should_attempt_reacquisition(self) -> bool:
        """Check if reacquisition should be attempted."""
        if not self.reacquire_enabled:
            return False
        
        if self.miss_counter < self.reacquire_trigger:
            return False
        
        # Attempt every N frames
        return self.miss_counter % self.reacquire_frequency == 0
    
    def _select_best_candidate(self, detections: List[Dict]) -> Optional[Dict]:
        """
        Select best ball candidate using motion gating and scoring.
        
        Strategy:
        1. If no track yet: Choose detection with highest confidence
        2. If track exists: Apply motion gating and score candidates
        
        Motion Gating:
        - Reject detections too far from predicted position
        - Gate radius = max_displacement_px * gate_radius_scale
        
        Scoring:
        - score = confidence - distance_penalty
        - Prefer high confidence and proximity to prediction
        
        Args:
            detections: List of ball detection dicts
        
        Returns:
            Best detection or None if no valid candidate
        """
        if len(detections) == 0:
            return None
        
        # No track yet: choose highest confidence
        if not self.has_track or self.predicted_center is None:
            return max(detections, key=lambda d: d['conf'])
        
        # Apply motion gating
        gate_radius = self.max_displacement_px * self.gate_radius_scale
        
        valid_candidates = []
        for det in detections:
            center = bbox_center(det['bbox'])
            dist = euclidean_distance(center, self.predicted_center)
            
            # Check if within gate
            if dist <= gate_radius:
                # Compute combined score (confidence - distance penalty)
                # Distance penalty: 0.01 per pixel
                score = det['conf'] - 0.01 * dist
                valid_candidates.append((score, det))
        
        if len(valid_candidates) == 0:
            return None
        
        # Return candidate with highest score
        best_candidate = max(valid_candidates, key=lambda x: x[0])
        return best_candidate[1]
    
    def _initialize_track(self, center: Tuple[float, float], bbox: List[float]):
        """
        Initialize Kalman filter state with first detection.
        
        Args:
            center: Initial ball center (cx, cy)
            bbox: Initial bounding box [x1, y1, x2, y2]
        """
        # Initialize state: [cx, cy, vx=0, vy=0]
        self.kf.x = np.array([
            [center[0]],  # cx
            [center[1]],  # cy
            [0.0],        # vx (initially zero)
            [0.0]         # vy (initially zero)
        ])
        
        # High initial uncertainty
        self.kf.P = np.eye(4) * 1000
        
        self.has_track = True
        self.last_center = center
        self.last_bbox = bbox
        self.trajectory.clear()
        self.trajectory.append(center)
        self.velocity_history.clear()
    
    def _try_reacquire(self, detector, frame_shape: Tuple[int, int], 
                      full_frame_dets: List[Dict]) -> Optional[Dict]:
        """
        Attempt to reacquire ball using ROI-based search.
        
        Strategy:
        1. Build ROI around last known/predicted position
        2. Search for ball detections within ROI
        3. Apply lenient motion gate (2x normal radius)
        4. If found, reinitialize tracker
        
        Args:
            detector: Detector instance (for future ROI detection)
            frame_shape: (height, width) of frame
            full_frame_dets: All ball detections in current frame
        
        Returns:
            Ball state dict if reacquired, else None
        """
        if detector is None or self.predicted_center is None:
            return None
        
        # Build ROI around predicted position
        roi = self._build_reacquisition_roi(frame_shape)
        
        # Search for candidates in ROI
        lenient_gate_radius = self.max_displacement_px * 2.0  # More lenient
        
        for det in full_frame_dets:
            center = bbox_center(det['bbox'])
            
            # Check if in ROI
            if not self._point_in_roi(center, roi):
                continue
            
            # Check lenient motion gate
            dist = euclidean_distance(center, self.predicted_center)
            if dist <= lenient_gate_radius:
                # Reacquire successful!
                measurement = np.array([[center[0]], [center[1]]])
                self.kf.update(measurement)
                
                self.last_bbox = det['bbox']
                self.last_center = center
                self.miss_counter = 0
                self.status = 'detected'
                self.has_track = True
                
                # Add to trajectory
                self.trajectory.append(center)
                
                return self._make_output(det['conf'], is_predicted=False)
        
        return None
    
    def _build_reacquisition_roi(self, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Build ROI for reacquisition search.
        
        Args:
            frame_shape: (height, width)
        
        Returns:
            ROI as (x1, y1, x2, y2)
        """
        if self.last_bbox is not None:
            # Expand around last bbox
            roi_bbox = expand_bbox(self.last_bbox, self.reacquire_roi_scale, frame_shape)
            return (int(roi_bbox[0]), int(roi_bbox[1]), 
                   int(roi_bbox[2]), int(roi_bbox[3]))
        else:
            # Fixed size ROI around predicted center
            return build_roi_around_point(
                self.predicted_center, 
                frame_shape, 
                self.reacquire_roi_size
            )
    
    @staticmethod
    def _point_in_roi(point: Tuple[float, float], roi: Tuple[int, int, int, int]) -> bool:
        """Check if point is inside ROI."""
        x, y = point
        x1, y1, x2, y2 = roi
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def _make_output(self, conf: float, is_predicted: bool) -> Dict:
        """
        Create ball state output dict.
        
        Args:
            conf: Detection confidence
            is_predicted: Whether ball is predicted (not detected)
        
        Returns:
            Ball state dict with all tracking information
        """
        # Get current center
        if self.predicted_center is not None:
            center = self.predicted_center
        elif self.last_center is not None:
            center = self.last_center
        else:
            center = None
        
        # Estimate bbox from center if predicted
        bbox = self.last_bbox
        if is_predicted and center is not None and bbox is not None:
            # Maintain last bbox size
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            bbox = [
                center[0] - w/2, 
                center[1] - h/2,
                center[0] + w/2, 
                center[1] + h/2
            ]
        
        # Extract velocity from Kalman state
        velocity = None
        if self.has_track:
            velocity = (self.kf.x[2, 0], self.kf.x[3, 0])
        
        return {
            'status': self.status,
            'center': center,
            'bbox': bbox,
            'conf': conf,
            'is_predicted': is_predicted,
            'miss_counter': self.miss_counter,
            'velocity': velocity,
            'trajectory': list(self.trajectory)
        }
    
    def _make_lost_output(self) -> Dict:
        """
        Create lost ball state.
        
        Returns:
            Ball state dict indicating ball is lost
        """
        return {
            'status': 'lost',
            'center': None,
            'bbox': None,
            'conf': 0.0,
            'is_predicted': False,
            'miss_counter': self.miss_counter,
            'velocity': None,
            'trajectory': []
        }
    
    def reset(self):
        """
        Reset tracker state to initial conditions.
        
        Use this when starting a new video or after complete ball loss.
        """
        self.has_track = False
        self.miss_counter = 0
        self.status = 'lost'
        self.last_bbox = None
        self.last_center = None
        self.predicted_center = None
        self.trajectory.clear()
        self.velocity_history.clear()
        
        # Reset Kalman filter
        self.kf.x = np.zeros((4, 1))
        self.kf.P = np.eye(4) * 1000
    
    def get_statistics(self) -> Dict:
        """
        Get tracker statistics.
        
        Returns:
            Dict with tracking stats:
                - has_track: Whether tracker is active
                - status: Current status
                - miss_counter: Consecutive misses
                - avg_velocity: Average velocity magnitude
                - trajectory_length: Number of points in trajectory
        """
        avg_velocity = 0.0
        if len(self.velocity_history) > 0:
            velocities = np.array(list(self.velocity_history))
            avg_velocity = float(np.mean(np.linalg.norm(velocities, axis=1)))
        
        return {
            'has_track': self.has_track,
            'status': self.status,
            'miss_counter': self.miss_counter,
            'avg_velocity_px_per_frame': avg_velocity,
            'trajectory_length': len(self.trajectory),
            'frame_idx': self.frame_idx
        }
