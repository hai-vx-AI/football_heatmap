"""
Ball tracker using Kalman Filter with reacquisition and false positive filtering.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from filterpy.kalman import KalmanFilter

from .utils import bbox_center, euclidean_distance, build_roi_around_point, expand_bbox


class BallTracker:
    """
    Single-target ball tracker with Kalman Filter, motion gating, and reacquisition.
    """
    
    def __init__(self, config: dict, fps: float = 25.0):
        """
        Initialize ball tracker.
        
        Args:
            config: Ball tracker configuration
            fps: Video frame rate
        """
        self.config = config
        self.fps = fps
        self.dt = 1.0 / fps
        
        # Tracking parameters
        self.track_buffer = config.get('track_buffer', 15)
        self.max_displacement_px = config.get('max_displacement_px_per_frame', 80)
        self.gate_radius_scale = config.get('gate_radius_scale', 1.5)
        
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
    
    def _init_kalman_filter(self, kalman_config: dict):
        """
        Initialize Kalman Filter for ball tracking.
        State: [cx, cy, vx, vy] (center position + velocity)
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe position only)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise
        q_pos = kalman_config.get('process_noise_pos', 1.0)
        q_vel = kalman_config.get('process_noise_vel', 0.1)
        self.kf.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        
        # Measurement noise
        r = kalman_config.get('measurement_noise', 10.0)
        self.kf.R = np.array([[r, 0], [0, r]])
        
        # Initial covariance
        self.kf.P *= 1000
    
    def update(self, frame_idx: int, ball_detections: List[Dict], 
               frame_shape: Tuple[int, int], detector=None) -> Dict:
        """
        Update ball tracker with detections.
        
        Args:
            frame_idx: Current frame index
            ball_detections: List of ball detections
            frame_shape: (height, width) of frame
            detector: Detector instance for reacquisition (optional)
        
        Returns:
            Ball state dict with keys:
                - status: 'detected', 'predicted', 'lost'
                - center: (cx, cy) or None
                - bbox: [x1, y1, x2, y2] or None
                - conf: confidence score
                - is_predicted: bool
        """
        self.frame_idx = frame_idx
        
        # Predict next state
        if self.has_track:
            self.kf.predict()
            self.predicted_center = (self.kf.x[0, 0], self.kf.x[1, 0])
        else:
            self.predicted_center = None
        
        # Select best candidate with motion gating
        candidate = self._select_best_candidate(ball_detections)
        
        if candidate is not None:
            # Update Kalman filter with measurement
            center = bbox_center(candidate['bbox'])
            
            if not self.has_track:
                # Initialize track
                self._initialize_track(center, candidate['bbox'])
            else:
                # Update track
                self.kf.update(np.array([[center[0]], [center[1]]]))
            
            self.last_bbox = candidate['bbox']
            self.last_center = center
            self.miss_counter = 0
            self.status = 'detected'
            self.has_track = True
            
            return self._make_output(candidate['conf'], is_predicted=False)
        
        # No candidate found
        if not self.has_track:
            return self._make_lost_output()
        
        # Increment miss counter
        self.miss_counter += 1
        
        # Keep predicting if within buffer
        if self.miss_counter <= self.track_buffer:
            self.status = 'predicted'
            return self._make_output(0.0, is_predicted=True)
        
        # Lost - try reacquisition
        if self.reacquire_enabled and self.miss_counter % self.reacquire_frequency == 0:
            reacquired = self._try_reacquire(detector, frame_shape, ball_detections)
            if reacquired:
                return reacquired
        
        # Ball is lost
        self.status = 'lost'
        self.has_track = False
        return self._make_lost_output()
    
    def _select_best_candidate(self, detections: List[Dict]) -> Optional[Dict]:
        """
        Select best ball candidate using motion gating.
        
        Args:
            detections: List of ball detections
        
        Returns:
            Best detection or None
        """
        if len(detections) == 0:
            return None
        
        # If no track yet, take highest confidence
        if not self.has_track or self.predicted_center is None:
            return max(detections, key=lambda d: d['conf'])
        
        # Apply motion gating
        gate_radius = self.max_displacement_px * self.gate_radius_scale
        
        valid_candidates = []
        for det in detections:
            center = bbox_center(det['bbox'])
            dist = euclidean_distance(center, self.predicted_center)
            
            if dist <= gate_radius:
                # Compute score (prefer high confidence and small distance)
                score = det['conf'] - 0.01 * dist
                valid_candidates.append((score, det))
        
        if len(valid_candidates) == 0:
            return None
        
        # Return candidate with highest score
        return max(valid_candidates, key=lambda x: x[0])[1]
    
    def _initialize_track(self, center: Tuple[float, float], bbox: List[float]):
        """Initialize Kalman filter state with first detection."""
        self.kf.x = np.array([[center[0]], [center[1]], [0], [0]])
        self.kf.P *= 1000  # High initial uncertainty
        self.has_track = True
        self.last_center = center
        self.last_bbox = bbox
    
    def _try_reacquire(self, detector, frame_shape: Tuple[int, int], 
                      full_frame_dets: List[Dict]) -> Optional[Dict]:
        """
        Try to reacquire ball using ROI search.
        
        Args:
            detector: Detector instance
            frame_shape: (height, width)
            full_frame_dets: Already detected balls in full frame
        
        Returns:
            Ball state dict if reacquired, else None
        """
        if detector is None or self.predicted_center is None:
            return None
        
        # Build ROI around predicted position
        if self.last_bbox is not None:
            # Use last bbox to determine ROI size
            roi_bbox = expand_bbox(self.last_bbox, self.reacquire_roi_scale, frame_shape)
            roi = (int(roi_bbox[0]), int(roi_bbox[1]), 
                   int(roi_bbox[2]), int(roi_bbox[3]))
        else:
            # Use fixed ROI size
            roi = build_roi_around_point(self.predicted_center, frame_shape, 
                                         self.reacquire_roi_size)
        
        # Note: ROI detection would be done externally by main loop
        # For now, check if any detection from full frame is in ROI
        for det in full_frame_dets:
            center = bbox_center(det['bbox'])
            if (roi[0] <= center[0] <= roi[2] and 
                roi[1] <= center[1] <= roi[3]):
                # Found candidate in ROI
                dist = euclidean_distance(center, self.predicted_center)
                if dist <= self.max_displacement_px * 2:  # More lenient gate
                    # Reacquire
                    self.kf.update(np.array([[center[0]], [center[1]]]))
                    self.last_bbox = det['bbox']
                    self.last_center = center
                    self.miss_counter = 0
                    self.status = 'detected'
                    self.has_track = True
                    return self._make_output(det['conf'], is_predicted=False)
        
        return None
    
    def _make_output(self, conf: float, is_predicted: bool) -> Dict:
        """Create ball state output dict."""
        if self.predicted_center is not None:
            center = self.predicted_center
        elif self.last_center is not None:
            center = self.last_center
        else:
            center = None
        
        # Estimate bbox from center if predicted
        bbox = self.last_bbox
        if is_predicted and center is not None and bbox is not None:
            # Use last bbox size
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            bbox = [center[0] - w/2, center[1] - h/2,
                    center[0] + w/2, center[1] + h/2]
        
        return {
            'status': self.status,
            'center': center,
            'bbox': bbox,
            'conf': conf,
            'is_predicted': is_predicted,
            'miss_counter': self.miss_counter
        }
    
    def _make_lost_output(self) -> Dict:
        """Create lost ball state."""
        return {
            'status': 'lost',
            'center': None,
            'bbox': None,
            'conf': 0.0,
            'is_predicted': False,
            'miss_counter': self.miss_counter
        }
    
    def reset(self):
        """Reset tracker state."""
        self.has_track = False
        self.miss_counter = 0
        self.status = 'lost'
        self.last_bbox = None
        self.last_center = None
        self.predicted_center = None
        self.kf.x = np.zeros((4, 1))
        self.kf.P = np.eye(4) * 1000
