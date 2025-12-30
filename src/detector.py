"""
Detector module using YOLO for multi-class detection (player, ball, referee, goalkeeper).
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import torch

from .utils import letterbox, map_bbox_to_original, bbox_area, bbox_aspect_ratio


class Detector:
    """
    YOLO-based detector with per-class confidence filtering and geometry checks.
    Supports both COCO pretrained models and custom football models.
    """
    
    # Custom model class mapping (for fine-tuned models)
    CLASS_NAMES = {
        0: 'player',
        1: 'ball',
        2: 'referee',
        3: 'goalkeeper'
    }
    
    CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}
    
    # COCO class mapping (for pretrained models like yolo11n.pt)
    COCO_TO_CUSTOM = {
        0: 'player',      # person -> player
        32: 'ball',       # sports ball -> ball
    }
    
    def __init__(self, config: dict):
        """
        Initialize detector.
        
        Args:
            config: Detector configuration dict
        """
        self.config = config
        self.model_path = config['model_path']
        self.imgsz = config['imgsz']
        self.conf_global = config['conf_global']
        
        # Per-class thresholds
        self.class_conf = config['class_conf']
        self.nms_iou = config['nms_iou']
        self.max_det = config['max_det']
        
        # Ball filters
        self.ball_filters = config.get('ball_filters', {})
        
        # Load model
        print(f"Loading YOLO model from {self.model_path}...")
        self.model = YOLO(self.model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Detect if using COCO pretrained model or custom trained model
        self.is_coco_model = len(self.model.names) == 80  # COCO has 80 classes
        
        if self.is_coco_model:
            print(f"Detected COCO pretrained model - will map classes automatically")
            self.class_names = self.CLASS_NAMES  # Use hardcoded mapping
        else:
            # Custom trained model - use model's class names directly
            print(f"Detected custom trained model with classes: {self.model.names}")
            self.class_names = self.model.names  # Use model's class mapping
            
        self.class_ids = {v: k for k, v in self.class_names.items()}
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection on a single frame.
        
        Args:
            frame: Input BGR image
        
        Returns:
            List of detections with bbox, conf, cls, cls_id
        """
        # Preprocess: letterbox resize
        img_input, meta = letterbox(frame, new_shape=(self.imgsz, self.imgsz))
        
        # Run inference
        results = self.model.predict(
            img_input,
            conf=self.conf_global,
            iou=0.45,
            max_det=self.max_det['people'] + self.max_det['ball'],
            verbose=False,
            device=self.device
        )[0]  # Get first result directly
        
        # Parse results
        if results.boxes is None or len(results.boxes) == 0:
            return []
        
        detections = []
        boxes = results.boxes
        
        # Batch process boxes for efficiency
        xyxy_batch = boxes.xyxy.cpu().numpy()
        conf_batch = boxes.conf.cpu().numpy()
        cls_batch = boxes.cls.cpu().numpy().astype(int)
        
        for i in range(len(boxes)):
            bbox_orig = map_bbox_to_original(xyxy_batch[i].tolist(), meta)
            conf = float(conf_batch[i])
            cls_id = int(cls_batch[i])
            
            # Get class name - handle both COCO and custom models
            if self.is_coco_model:
                cls_name = self.COCO_TO_CUSTOM.get(cls_id, 'unknown')
            else:
                cls_name = self.class_names.get(cls_id, 'unknown')
            
            if cls_name == 'unknown':
                continue
            
            detections.append({
                'bbox': bbox_orig,
                'conf': conf,
                'cls': cls_name,
                'cls_id': cls_id
            })
        
        # Apply per-class filtering
        return self._filter_detections(detections, frame.shape)
    
    def _filter_detections(self, detections: List[Dict], img_shape: Tuple[int, int]) -> List[Dict]:
        """Apply per-class confidence and geometry filters."""
        filtered = []
        
        for det in detections:
            cls_name = det['cls']
            
            # Check confidence threshold
            if det['conf'] < self.class_conf.get(cls_name, 0.25):
                continue
            
            # Apply geometry filters
            bbox = det['bbox']
            if cls_name == 'ball':
                if not self._is_valid_ball_detection(bbox):
                    continue
            elif bbox_area(bbox) < 100:  # People minimum area check
                continue
            
            filtered.append(det)
        
        return filtered
    
    def _is_valid_ball_detection(self, bbox: List[float]) -> bool:
        """Check if ball detection passes geometry filters."""
        area = bbox_area(bbox)
        aspect_ratio = bbox_aspect_ratio(bbox)
        
        # Area filter
        min_area = self.ball_filters.get('min_area_px', 0)
        max_area = self.ball_filters.get('max_area_px', float('inf'))
        if not (min_area <= area <= max_area):
            return False
        
        # Aspect ratio filter (ball should be roughly circular)
        min_ar = self.ball_filters.get('aspect_ratio_min', 0.5)
        max_ar = self.ball_filters.get('aspect_ratio_max', 2.0)
        if not (min_ar <= aspect_ratio <= max_ar):
            return False
        
        return True
    
    def split_detections(self, detections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Split detections into people and ball groups.
        
        Args:
            detections: All detections
        
        Returns:
            Tuple of (people_detections, ball_detections)
        """
        people_dets = []
        ball_dets = []
        
        for det in detections:
            if det['cls'] == 'ball':
                ball_dets.append(det)
            else:  # player, referee, goalkeeper
                people_dets.append(det)
        
        return people_dets, ball_dets
    
    def detect_in_roi(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Run ball detection in a specific ROI (for reacquisition).
        
        Args:
            frame: Full frame image
            roi: (x1, y1, x2, y2) region of interest
        
        Returns:
            List of ball detections in original frame coordinates
        """
        x1, y1, x2, y2 = roi
        
        # Crop ROI
        roi_img = frame[y1:y2, x1:x2]
        
        if roi_img.size == 0:
            return []
        
        # Detect in ROI
        detections = self.detect(roi_img)
        
        # Map detections back to full frame coordinates
        for det in detections:
            bbox = det['bbox']
            det['bbox'] = [
                bbox[0] + x1,
                bbox[1] + y1,
                bbox[2] + x1,
                bbox[3] + y1
            ]
        
        # Filter only ball detections
        ball_dets = [det for det in detections if det['cls'] == 'ball']
        
        return ball_dets
