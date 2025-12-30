"""
Renderer module for drawing overlays on video frames.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque


class Renderer:
    """
    Renders detection and tracking overlays on video frames.
    """
    
    def __init__(self, config: dict):
        """
        Initialize renderer.
        
        Args:
            config: Render configuration
        """
        self.config = config
        
        # Layer toggles
        self.layers_enabled = config.get('layers_enabled', {
            'player': True,
            'ball': True,
            'referee': True,
            'goalkeeper': True
        })
        
        # Colors (BGR format)
        colors = config.get('colors', {})
        self.color_team0 = tuple(colors.get('team0', [0, 255, 0]))
        self.color_team1 = tuple(colors.get('team1', [255, 0, 0]))
        self.color_referee = tuple(colors.get('referee', [0, 165, 255]))
        self.color_goalkeeper = tuple(colors.get('goalkeeper', [255, 255, 0]))
        self.color_ball_detected = tuple(colors.get('ball_detected', [0, 255, 255]))
        self.color_ball_predicted = tuple(colors.get('ball_predicted', [128, 128, 128]))
        self.color_ball_lost = tuple(colors.get('ball_lost', [0, 0, 255]))
        self.color_unknown = tuple(colors.get('unknown', [255, 255, 255]))
        
        # Drawing options
        self.draw_track_id = config.get('draw_track_id', True)
        self.draw_team_id = config.get('draw_team_id', True)
        self.draw_confidence = config.get('draw_confidence', False)
        self.bbox_thickness = config.get('bbox_thickness', 2)
        self.text_scale = config.get('text_scale', 0.6)
        self.text_thickness = config.get('text_thickness', 2)
        
        # Ball rendering
        self.ball_radius = config.get('ball_radius', 8)
        self.ball_trail_length = config.get('ball_trail_length', 10)
        
        # Ball trail history
        self.ball_trail = deque(maxlen=self.ball_trail_length)
    
    def render(self, frame: np.ndarray, tracks: List[Dict], 
               ball_state: Dict, possession_state: Optional[Dict] = None) -> np.ndarray:
        """
        Render all overlays on frame.
        
        Args:
            frame: Input BGR frame
            tracks: List of people tracks
            ball_state: Ball tracking state
            possession_state: Optional possession state dict
        
        Returns:
            Frame with overlays
        """
        output = frame.copy()
        
        # Draw people tracks
        for track in tracks:
            cls = track['cls']
            
            # Check if this player has possession
            has_possession = False
            if possession_state is not None and possession_state.get('possessor_id') == track['track_id']:
                has_possession = True
            
            if cls == 'player' and self.layers_enabled.get('player', True):
                self._draw_player(output, track, has_possession)
            elif cls == 'referee' and self.layers_enabled.get('referee', True):
                self._draw_referee(output, track)
            elif cls == 'goalkeeper' and self.layers_enabled.get('goalkeeper', True):
                self._draw_goalkeeper(output, track, has_possession)
        
        # Draw ball
        if self.layers_enabled.get('ball', True):
            self._draw_ball(output, ball_state)
        
        # Draw possession info
        if possession_state is not None:
            self._draw_possession_info(output, possession_state)
        
        # Draw legend
        self._draw_legend(output)
        
        return output
    
    def _draw_player(self, frame: np.ndarray, track: Dict, has_possession: bool = False):
        """Draw player with team color."""
        bbox = track['bbox']
        track_id = track['track_id']
        team_id = track.get('team_id')
        conf = track['conf']
        
        # Determine color
        if team_id == 'team0':
            color = self.color_team0
        elif team_id == 'team1':
            color = self.color_team1
        else:
            color = self.color_unknown
        
        # Draw bbox (thicker if has possession)
        thickness = self.bbox_thickness + 2 if has_possession else self.bbox_thickness
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw possession indicator
        if has_possession:
            # Draw circle around player
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = max((x2 - x1), (y2 - y1)) // 2 + 10
            cv2.circle(frame, (center_x, center_y), radius, color, 2)
        
        # Draw label
        label_parts = []
        if self.draw_track_id:
            label_parts.append(f"ID:{track_id}")
        if self.draw_team_id and team_id:
            label_parts.append(team_id)
        if self.draw_confidence:
            label_parts.append(f"{conf:.2f}")
        if has_possession:
            label_parts.append("⚽")
        
        if label_parts:
            label = " ".join(label_parts)
            self._draw_label(frame, label, (x1, y1 - 5), color)
    
    def _draw_referee(self, frame: np.ndarray, track: Dict):
        """Draw referee."""
        bbox = track['bbox']
        track_id = track['track_id']
        conf = track['conf']
        
        # Draw bbox
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color_referee, self.bbox_thickness)
        
        # Draw label
        label_parts = ["REF"]
        if self.draw_track_id:
            label_parts.append(f"ID:{track_id}")
        if self.draw_confidence:
            label_parts.append(f"{conf:.2f}")
        
        label = " ".join(label_parts)
        self._draw_label(frame, label, (x1, y1 - 5), self.color_referee)
    
    def _draw_goalkeeper(self, frame: np.ndarray, track: Dict, has_possession: bool = False):
        """Draw goalkeeper."""
        bbox = track['bbox']
        track_id = track['track_id']
        team_id = track.get('team_id')
        conf = track['conf']
        
        # Determine color (same as team or special GK color)
        if team_id == 'team0':
            color = self.color_team0
        elif team_id == 'team1':
            color = self.color_team1
        else:
            color = self.color_goalkeeper
        
        # Draw bbox (thicker if has possession)
        thickness = self.bbox_thickness + 2 if has_possession else self.bbox_thickness
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw possession indicator
        if has_possession:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = max((x2 - x1), (y2 - y1)) // 2 + 10
            cv2.circle(frame, (center_x, center_y), radius, color, 2)
        
        # Draw label
        label_parts = ["GK"]
        if self.draw_track_id:
            label_parts.append(f"ID:{track_id}")
        if self.draw_team_id and team_id:
            label_parts.append(team_id)
        if self.draw_confidence:
            label_parts.append(f"{conf:.2f}")
        if has_possession:
            label_parts.append("⚽")
        
        label = " ".join(label_parts)
        self._draw_label(frame, label, (x1, y1 - 5), color)
    
    def _draw_ball(self, frame: np.ndarray, ball_state: Dict):
        """Draw ball with trail."""
        status = ball_state['status']
        center = ball_state.get('center')
        bbox = ball_state.get('bbox')
        
        if center is None:
            return
        
        # Add to trail
        self.ball_trail.append(center)
        
        # Determine color based on status
        if status == 'detected':
            color = self.color_ball_detected
        elif status == 'predicted':
            color = self.color_ball_predicted
        else:
            color = self.color_ball_lost
        
        # Draw trail
        if len(self.ball_trail) > 1:
            points = np.array([(int(x), int(y)) for x, y in self.ball_trail], dtype=np.int32)
            for i in range(len(points) - 1):
                alpha = (i + 1) / len(points)  # Fade effect
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, tuple(points[i]), tuple(points[i + 1]), 
                        trail_color, 2, cv2.LINE_AA)
        
        # Draw ball circle
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(frame, (cx, cy), self.ball_radius, color, -1)
        cv2.circle(frame, (cx, cy), self.ball_radius + 2, (255, 255, 255), 1)
        
        # Draw status label
        label = f"Ball: {status}"
        self._draw_label(frame, label, (cx + 12, cy), color, bg_color=(0, 0, 0))
    
    def _draw_label(self, frame: np.ndarray, text: str, 
                    position: Tuple[int, int], color: Tuple[int, int, int],
                    bg_color: Optional[Tuple[int, int, int]] = None):
        """Draw text label with optional background."""
        x, y = position
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 
            self.text_scale, self.text_thickness
        )
        
        # Draw background rectangle if specified
        if bg_color is not None:
            cv2.rectangle(frame, 
                         (x, y - text_h - baseline - 2), 
                         (x + text_w, y + 2),
                         bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y - baseline), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, 
                   color, self.text_thickness, cv2.LINE_AA)
    
    def _draw_possession_info(self, frame: np.ndarray, possession_state: Dict):
        """Draw possession information overlay."""
        # Draw possession bar at top
        h, w = frame.shape[:2]
        bar_height = 40
        
        # Get possession percentages
        team_pct = possession_state.get('team_possession_pct', {})
        team0_pct = team_pct.get('team0', 0.0)
        team1_pct = team_pct.get('team1', 0.0)
        
        # Draw background
        cv2.rectangle(frame, (0, 0), (w, bar_height), (0, 0, 0), -1)
        
        # Draw team0 portion
        if team0_pct > 0:
            team0_width = int(w * team0_pct / 100.0)
            cv2.rectangle(frame, (0, 0), (team0_width, bar_height), self.color_team0, -1)
        
        # Draw team1 portion
        if team1_pct > 0:
            team1_width = int(w * team1_pct / 100.0)
            cv2.rectangle(frame, (w - team1_width, 0), (w, bar_height), self.color_team1, -1)
        
        # Draw text
        text0 = f"TEAM0: {team0_pct:.1f}%"
        text1 = f"TEAM1: {team1_pct:.1f}%"
        
        cv2.putText(frame, text0, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)
        cv2.putText(frame, text1, (w - 150, 25), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2)
        
        # Draw current possessor
        possessor_team = possession_state.get('possessor_team')
        if possessor_team is not None:
            duration = possession_state.get('possession_duration_sec', 0.0)
            text_poss = f"{possessor_team.upper()}: {duration:.1f}s"
            text_size = cv2.getTextSize(text_poss, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, text_poss, (text_x, 25), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)
        
        # Draw recent pass indicator
        recent_pass = possession_state.get('recent_pass')
        if recent_pass is not None:
            pass_text = f"PASS! {recent_pass['passer_id']} -> {recent_pass['receiver_id']}"
            cv2.putText(frame, pass_text, (w//2 - 100, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
    
    def _draw_legend(self, frame: np.ndarray):
        """Draw legend showing team colors and controls."""
        h, w = frame.shape[:2]
        
        # Legend background
        legend_h = 120
        legend_w = 200
        x_start = w - legend_w - 10
        y_start = 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), 
                     (x_start + legend_w, y_start + legend_h),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw legend items
        y_offset = y_start + 20
        line_height = 25
        
        legend_items = [
            ("Team 0", self.color_team0),
            ("Team 1", self.color_team1),
            ("Referee", self.color_referee),
            ("Goalkeeper", self.color_goalkeeper)
        ]
        
        for text, color in legend_items:
            # Color box
            cv2.rectangle(frame, 
                         (x_start + 10, y_offset - 10), 
                         (x_start + 30, y_offset + 5),
                         color, -1)
            
            # Text
            cv2.putText(frame, text, (x_start + 40, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 
                       1, cv2.LINE_AA)
            
            y_offset += line_height
    
    def reset(self):
        """Reset renderer state (e.g., ball trail)."""
        self.ball_trail.clear()
