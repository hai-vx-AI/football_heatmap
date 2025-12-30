"""
Utility functions for geometry, color conversion, and image processing.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List


def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), 
              color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, dict]:
    """
    Resize and pad image to new_shape while maintaining aspect ratio.
    
    Args:
        img: Input image (H, W, C)
        new_shape: Target size (height, width)
        color: Padding color
    
    Returns:
        Padded image and metadata dict with scale and padding info
    """
    shape = img.shape[:2]  # current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    meta = {
        'scale': r,
        'pad': (left, top),
        'original_shape': shape,
        'new_shape': new_shape
    }
    
    return img, meta


def map_bbox_to_original(bbox: List[float], meta: dict) -> List[float]:
    """
    Map bounding box from letterboxed coordinates back to original image coordinates.
    
    Args:
        bbox: [x1, y1, x2, y2] in letterboxed coordinates
        meta: Metadata from letterbox function
    
    Returns:
        [x1, y1, x2, y2] in original coordinates
    """
    x1, y1, x2, y2 = bbox
    scale = meta['scale']
    pad_left, pad_top = meta['pad']
    
    # Remove padding and scale
    x1 = (x1 - pad_left) / scale
    y1 = (y1 - pad_top) / scale
    x2 = (x2 - pad_left) / scale
    y2 = (y2 - pad_top) / scale
    
    return [x1, y1, x2, y2]


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Get center point of bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_area(bbox: List[float]) -> float:
    """Calculate area of bounding box."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_aspect_ratio(bbox: List[float]) -> float:
    """Calculate aspect ratio (width/height) of bounding box."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return w / h if h > 0 else 0


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def crop_bbox(img: np.ndarray, bbox: List[float], margin: float = 0.0) -> Optional[np.ndarray]:
    """
    Crop image region defined by bbox with optional margin.
    
    Args:
        img: Input image
        bbox: [x1, y1, x2, y2]
        margin: Margin ratio to expand bbox (e.g., 0.1 = 10% expansion)
    
    Returns:
        Cropped image or None if invalid
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Add margin
    if margin > 0:
        w_bbox = x2 - x1
        h_bbox = y2 - y1
        x1 -= w_bbox * margin
        y1 -= h_bbox * margin
        x2 += w_bbox * margin
        y2 += h_bbox * margin
    
    # Clamp to image boundaries
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    
    # Check validity
    if x2 <= x1 or y2 <= y1:
        return None
    
    return img[y1:y2, x1:x2]


def crop_jersey_roi(img: np.ndarray, bbox: List[float], 
                    x_range: Tuple[float, float] = (0.2, 0.8),
                    y_range: Tuple[float, float] = (0.15, 0.55)) -> Optional[np.ndarray]:
    """
    Crop jersey region from person bbox (upper torso, avoiding legs and ground).
    
    Args:
        img: Input image
        bbox: Person bounding box [x1, y1, x2, y2]
        x_range: (start_ratio, end_ratio) for horizontal crop
        y_range: (start_ratio, end_ratio) for vertical crop
    
    Returns:
        Cropped jersey region or None if invalid
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    # Calculate jersey ROI
    jersey_x1 = x1 + w * x_range[0]
    jersey_x2 = x1 + w * x_range[1]
    jersey_y1 = y1 + h * y_range[0]
    jersey_y2 = y1 + h * y_range[1]
    
    return crop_bbox(img, [jersey_x1, jersey_y1, jersey_x2, jersey_y2])


def bgr_to_lab(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to Lab color space."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


def bgr_to_hsv(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to HSV color space."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def create_grass_mask(img_hsv: np.ndarray, 
                      h_range: Tuple[int, int] = (35, 85),
                      s_min: int = 60, 
                      v_min: int = 40) -> np.ndarray:
    """
    Create binary mask for grass/field pixels in HSV space.
    
    Args:
        img_hsv: Image in HSV color space
        h_range: Hue range for green (35-85 typical for grass)
        s_min: Minimum saturation
        v_min: Minimum value
    
    Returns:
        Binary mask (1 = grass, 0 = not grass)
    """
    h, s, v = cv2.split(img_hsv)
    
    mask = np.zeros(h.shape, dtype=np.uint8)
    mask[(h >= h_range[0]) & (h <= h_range[1]) & 
         (s >= s_min) & (v >= v_min)] = 1
    
    return mask


def extract_color_feature(img_bgr: np.ndarray, 
                          use_median: bool = True,
                          grass_mask_params: Optional[dict] = None,
                          min_l: int = 20) -> Optional[np.ndarray]:
    """
    Extract dominant color feature from image (typically jersey crop).
    
    Args:
        img_bgr: Input BGR image
        use_median: If True, use median color; else use k-means
        grass_mask_params: Parameters for grass masking (h_range, s_min, v_min)
        min_l: Minimum L value in Lab to filter dark pixels
    
    Returns:
        Color feature vector [L, a, b] or None if insufficient pixels
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    
    # Convert to Lab and HSV
    img_lab = bgr_to_lab(img_bgr)
    img_hsv = bgr_to_hsv(img_bgr)
    
    # Create mask to filter unwanted pixels
    mask = np.ones(img_bgr.shape[:2], dtype=bool)
    
    # Filter grass pixels
    if grass_mask_params:
        grass_mask = create_grass_mask(
            img_hsv,
            h_range=grass_mask_params.get('h_range', (35, 85)),
            s_min=grass_mask_params.get('s_min', 60),
            v_min=grass_mask_params.get('v_min', 40)
        )
        mask &= (grass_mask == 0)
    
    # Filter dark pixels
    if min_l > 0:
        mask &= (img_lab[:, :, 0] >= min_l)
    
    # Extract valid pixels
    valid_pixels = img_lab[mask]
    
    if len(valid_pixels) < 10:  # Need minimum pixels
        return None
    
    if use_median:
        # Use median color (robust to outliers)
        color_feat = np.median(valid_pixels, axis=0)
    else:
        # Use k-means to find dominant color
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(2, len(valid_pixels)), random_state=42, n_init=3)
        kmeans.fit(valid_pixels)
        
        # Find largest cluster
        labels = kmeans.labels_
        cluster_sizes = np.bincount(labels)
        dominant_cluster = np.argmax(cluster_sizes)
        color_feat = kmeans.cluster_centers_[dominant_cluster]
    
    return color_feat


def color_distance_lab(color1: np.ndarray, color2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two Lab colors.
    
    Args:
        color1: [L, a, b]
        color2: [L, a, b]
    
    Returns:
        Distance value
    """
    return np.linalg.norm(color1 - color2)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def build_roi_around_point(center: Tuple[float, float], 
                           img_shape: Tuple[int, int],
                           roi_size: int = 320) -> Tuple[int, int, int, int]:
    """
    Build ROI (region of interest) centered around a point.
    
    Args:
        center: (cx, cy) center point
        img_shape: (height, width) of image
        roi_size: Size of square ROI
    
    Returns:
        (x1, y1, x2, y2) ROI coordinates, clamped to image bounds
    """
    cx, cy = center
    h, w = img_shape[:2]
    
    half_size = roi_size // 2
    x1 = int(clamp(cx - half_size, 0, w))
    y1 = int(clamp(cy - half_size, 0, h))
    x2 = int(clamp(cx + half_size, 0, w))
    y2 = int(clamp(cy + half_size, 0, h))
    
    return (x1, y1, x2, y2)


def expand_bbox(bbox: List[float], scale: float, 
                img_shape: Tuple[int, int]) -> List[float]:
    """
    Expand bounding box by scale factor, clamped to image bounds.
    
    Args:
        bbox: [x1, y1, x2, y2]
        scale: Scale factor (e.g., 2.0 = double size)
        img_shape: (height, width) of image
    
    Returns:
        Expanded bbox [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    h, w = img_shape[:2]
    
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1
    
    new_w = bw * scale
    new_h = bh * scale
    
    x1 = clamp(cx - new_w / 2, 0, w)
    y1 = clamp(cy - new_h / 2, 0, h)
    x2 = clamp(cx + new_w / 2, 0, w)
    y2 = clamp(cy + new_h / 2, 0, h)
    
    return [x1, y1, x2, y2]


def ema_update(prev_value: Optional[np.ndarray], 
               new_value: np.ndarray, 
               alpha: float = 0.2) -> np.ndarray:
    """
    Exponential moving average update.
    
    Args:
        prev_value: Previous EMA value (None if first update)
        new_value: New observation
        alpha: Smoothing factor (0 = no change, 1 = full update)
    
    Returns:
        Updated EMA value
    """
    if prev_value is None:
        return new_value
    return (1 - alpha) * prev_value + alpha * new_value
