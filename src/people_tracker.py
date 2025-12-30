"""
People tracker using ByteTrack for tracking player, referee, and goalkeeper.
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict, deque


try:
    from boxmot import ByteTrack
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    print("Warning: boxmot not available. Install with: pip install boxmot")


class PeopleTracker:
    """
    Tracker for people (player, referee, goalkeeper) using ByteTrack.
    """
    
    def __init__(self, config: dict, fps: float = 25.0, class_names: dict = None):
        """
        Initialize people tracker.
        
        Args:
            config: People tracker configuration
            fps: Video frame rate
            class_names: Class ID to name mapping (from detector model)
        """
        self.config = config
        self.fps = fps
        
        # Store class names mapping from detector
        self.class_names = class_names or {0: 'player', 1: 'ball', 2: 'referee', 3: 'goalkeeper'}
        
        # Tracker parameters
        self.track_high_thresh = config.get('track_high_thresh', 0.25)
        self.track_low_thresh = config.get('track_low_thresh', 0.10)
        self.new_track_thresh = config.get('new_track_thresh', 0.25)
        self.match_thresh = config.get('match_thresh', 0.8)
        self.track_buffer = config.get('track_buffer', 30)
        self.min_box_area = config.get('min_box_area', 300)
        self.min_track_frames = config.get('min_track_frames', 10)
        
        # Initialize tracker
        if BYTETRACK_AVAILABLE:
            self.tracker = ByteTrack(
                track_thresh=self.track_high_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
                frame_rate=int(fps)
            )
        else:
            # Fallback to simple tracker
            self.tracker = SimpleTracker(config)
        
        # Track history for filtering short tracks
        self.track_history = defaultdict(lambda: {'frames': 0, 'cls_votes': deque(maxlen=10)})
        self.frame_idx = 0
    
    def update(self, frame_idx: int, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            frame_idx: Current frame index
            detections: List of people detections
        
        Returns:
            List of tracked objects with track_id
        """
        self.frame_idx = frame_idx
        
        if len(detections) == 0:
            # Update with empty detections
            if BYTETRACK_AVAILABLE:
                tracks_output = self.tracker.update(np.empty((0, 6)), None)
            else:
                tracks_output = self.tracker.update(detections)
            
            return self._parse_tracks(tracks_output, detections)
        
        # Convert detections to tracker format: [x1, y1, x2, y2, conf, cls_id]
        det_array = self._detections_to_array(detections)
        
        # Update tracker
        if BYTETRACK_AVAILABLE:
            tracks_output = self.tracker.update(det_array, None)
        else:
            tracks_output = self.tracker.update(detections)
        
        # Parse tracker output
        tracks = self._parse_tracks(tracks_output, detections)
        
        # Update track history
        for track in tracks:
            track_id = track['track_id']
            self.track_history[track_id]['frames'] += 1
            self.track_history[track_id]['cls_votes'].append(track['cls'])
        
        # Filter tracks
        filtered_tracks = self._filter_tracks(tracks)
        
        return filtered_tracks
    
    def _detections_to_array(self, detections: List[Dict]) -> np.ndarray:
        """
        Convert detection list to numpy array for tracker.
        
        Args:
            detections: List of detection dicts
        
        Returns:
            Array of shape (N, 6) with [x1, y1, x2, y2, conf, cls_id]
        """
        if len(detections) == 0:
            return np.empty((0, 6))
        
        det_list = []
        for det in detections:
            bbox = det['bbox']
            conf = det['conf']
            cls_id = det['cls_id']
            det_list.append([bbox[0], bbox[1], bbox[2], bbox[3], conf, cls_id])
        
        return np.array(det_list)
    
    def _parse_tracks(self, tracks_output, detections: List[Dict]) -> List[Dict]:
        """
        Parse tracker output into track dicts.
        
        Args:
            tracks_output: Output from tracker
            detections: Original detections for class info
        
        Returns:
            List of track dicts
        """
        tracks = []
        
        if tracks_output is None or len(tracks_output) == 0:
            return tracks
        
        # ByteTrack output format: [x1, y1, x2, y2, track_id, conf, cls_id, ...]
        for track_data in tracks_output:
            if len(track_data) < 7:
                continue
            
            x1, y1, x2, y2 = track_data[:4]
            track_id = int(track_data[4])
            conf = float(track_data[5])
            cls_id = int(track_data[6])
            
            # Map cls_id to class name
            cls_name = self._get_class_name(cls_id)
            
            track = {
                'track_id': track_id,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'conf': conf,
                'cls': cls_name,
                'cls_id': cls_id,
                'team_id': None  # Will be assigned by team assigner
            }
            
            tracks.append(track)
        
        return tracks
    
    def _get_class_name(self, cls_id: int) -> str:
        """Map class ID to name using detector's class names."""
        return self.class_names.get(cls_id, 'player')
    
    def _filter_tracks(self, tracks: List[Dict]) -> List[Dict]:
        """
        Filter tracks based on quality criteria.
        
        Args:
            tracks: List of tracks
        
        Returns:
            Filtered tracks
        """
        filtered = []
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            
            # Filter by minimum box area
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < self.min_box_area:
                continue
            
            # Optional: smooth class assignment using majority vote
            if track_id in self.track_history:
                history = self.track_history[track_id]
                if len(history['cls_votes']) > 0:
                    # Use most common class in recent history
                    cls_counts = {}
                    for cls in history['cls_votes']:
                        cls_counts[cls] = cls_counts.get(cls, 0) + 1
                    track['cls'] = max(cls_counts, key=cls_counts.get)
            
            filtered.append(track)
        
        return filtered
    
    def get_confirmed_tracks(self) -> List[int]:
        """
        Get list of track IDs that have enough frames to be considered confirmed.
        
        Returns:
            List of confirmed track IDs
        """
        confirmed = []
        for track_id, history in self.track_history.items():
            if history['frames'] >= self.min_track_frames:
                confirmed.append(track_id)
        return confirmed


class SimpleTracker:
    """
    Fallback simple tracker using IOU matching (when ByteTrack not available).
    """
    
    def __init__(self, config: dict):
        self.next_track_id = 1
        self.active_tracks = {}
        self.max_age = config.get('track_buffer', 30)
        self.min_iou = 0.3
    
    def update(self, detections: List[Dict]) -> List:
        """Update tracks with simple IOU matching."""
        # Age existing tracks
        for track_id in list(self.active_tracks.keys()):
            self.active_tracks[track_id]['age'] += 1
            if self.active_tracks[track_id]['age'] > self.max_age:
                del self.active_tracks[track_id]
        
        # Match detections to tracks
        matched_tracks = []
        unmatched_dets = list(detections)
        
        for track_id, track in list(self.active_tracks.items()):
            best_match = None
            best_iou = self.min_iou
            
            for det in unmatched_dets:
                iou = self._compute_iou(track['bbox'], det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = det
            
            if best_match:
                # Update track
                track['bbox'] = best_match['bbox']
                track['conf'] = best_match['conf']
                track['cls'] = best_match['cls']
                track['cls_id'] = best_match['cls_id']
                track['age'] = 0
                
                matched_tracks.append([
                    track['bbox'][0], track['bbox'][1],
                    track['bbox'][2], track['bbox'][3],
                    track_id, track['conf'], track['cls_id']
                ])
                
                unmatched_dets.remove(best_match)
        
        # Create new tracks for unmatched detections
        for det in unmatched_dets:
            track_id = self.next_track_id
            self.next_track_id += 1
            
            self.active_tracks[track_id] = {
                'bbox': det['bbox'],
                'conf': det['conf'],
                'cls': det['cls'],
                'cls_id': det['cls_id'],
                'age': 0
            }
            
            matched_tracks.append([
                det['bbox'][0], det['bbox'][1],
                det['bbox'][2], det['bbox'][3],
                track_id, det['conf'], det['cls_id']
            ])
        
        return np.array(matched_tracks) if matched_tracks else np.empty((0, 7))
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IOU between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
