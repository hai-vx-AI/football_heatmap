"""
Video I/O module for reading and writing video files.
"""

import cv2
import numpy as np
from typing import Iterator, Tuple, Optional
from pathlib import Path
import configparser


class VideoReader:
    """
    Video reader that provides frame-by-frame iteration with metadata.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to input video file
        """
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        # Read metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (self.width, self.height)
        
        self.current_frame_idx = 0
    
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Iterate over frames.
        
        Yields:
            Tuple of (frame_index, frame_bgr)
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            yield self.current_frame_idx, frame
            self.current_frame_idx += 1
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame.
        
        Returns:
            Tuple of (success, frame_bgr)
        """
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx += 1
        return ret, frame
    
    def get_timestamp(self, frame_idx: Optional[int] = None) -> float:
        """
        Get timestamp in seconds for a frame.
        
        Args:
            frame_idx: Frame index (uses current if None)
        
        Returns:
            Timestamp in seconds
        """
        if frame_idx is None:
            frame_idx = self.current_frame_idx
        return frame_idx / self.fps if self.fps > 0 else 0.0
    
    def seek(self, frame_idx: int) -> bool:
        """
        Seek to specific frame.
        
        Args:
            frame_idx: Target frame index
        
        Returns:
            True if successful
        """
        ret = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        if ret:
            self.current_frame_idx = frame_idx
        return ret
    
    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    @property
    def meta(self) -> dict:
        """Get video metadata."""
        return {
            'path': self.video_path,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'duration_sec': self.frame_count / self.fps if self.fps > 0 else 0
        }


class VideoWriter:
    """
    Video writer for saving processed frames.
    """
    
    def __init__(self, output_path: str, fps: float, size: Tuple[int, int], 
                 fourcc: str = 'mp4v'):
        """
        Initialize video writer.
        
        Args:
            output_path: Output video file path
            fps: Frames per second
            size: (width, height) of frames
            fourcc: Video codec fourcc code
        """
        self.output_path = str(output_path)
        self.fps = fps
        self.size = size
        self.fourcc = fourcc
        
        # Create output directory if needed
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize writer
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc_code,
            fps,
            size
        )
        
        if not self.writer.isOpened():
            raise ValueError(f"Could not create video writer: {self.output_path}")
        
        self.frame_count = 0
    
    def write(self, frame: np.ndarray):
        """
        Write a frame to video.
        
        Args:
            frame: BGR frame to write
        """
        if frame.shape[:2][::-1] != self.size:
            # Resize if needed
            frame = cv2.resize(frame, self.size)
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def release(self):
        """Release video writer resources."""
        if self.writer is not None:
            self.writer.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    @property
    def meta(self) -> dict:
        """Get writer metadata."""
        return {
            'path': self.output_path,
            'fps': self.fps,
            'size': self.size,
            'frame_count': self.frame_count
        }


class SoccerNetSequenceReader:
    """
    Reader for SoccerNet MOT format sequences (image sequences).
    Compatible with SoccerNet tracking-2023 dataset.
    """
    
    def __init__(self, sequence_path: str):
        """
        Initialize SoccerNet sequence reader.
        
        Args:
            sequence_path: Path to sequence directory (e.g., SNMOT-116)
        """
        self.sequence_path = Path(sequence_path)
        
        if not self.sequence_path.exists():
            raise ValueError(f"Sequence not found: {self.sequence_path}")
        
        # Read seqinfo.ini
        seqinfo_path = self.sequence_path / "seqinfo.ini"
        if not seqinfo_path.exists():
            raise ValueError(f"seqinfo.ini not found in {self.sequence_path}")
        
        config = configparser.ConfigParser()
        config.read(seqinfo_path)
        
        # Parse metadata
        self.name = config.get('Sequence', 'name')
        self.img_dir = config.get('Sequence', 'imDir')
        self.fps = float(config.get('Sequence', 'frameRate'))
        self.frame_count = int(config.get('Sequence', 'seqLength'))
        self.width = int(config.get('Sequence', 'imWidth'))
        self.height = int(config.get('Sequence', 'imHeight'))
        self.img_ext = config.get('Sequence', 'imExt')
        self.size = (self.width, self.height)
        
        # Build image paths
        self.img_path = self.sequence_path / self.img_dir
        if not self.img_path.exists():
            raise ValueError(f"Image directory not found: {self.img_path}")
        
        self.current_frame_idx = 0
    
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Iterate over frames.
        
        Yields:
            Tuple of (frame_index, frame_bgr)
        """
        for frame_idx in range(self.frame_count):
            # SoccerNet uses 1-based indexing for images
            img_name = f"{frame_idx + 1:06d}{self.img_ext}"
            img_file = self.img_path / img_name
            
            if not img_file.exists():
                print(f"Warning: Image not found: {img_file}")
                continue
            
            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"Warning: Could not read image: {img_file}")
                continue
            
            self.current_frame_idx = frame_idx
            yield frame_idx, frame
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame.
        
        Returns:
            Tuple of (success, frame_bgr)
        """
        if self.current_frame_idx >= self.frame_count:
            return False, None
        
        # SoccerNet uses 1-based indexing
        img_name = f"{self.current_frame_idx + 1:06d}{self.img_ext}"
        img_file = self.img_path / img_name
        
        if not img_file.exists():
            return False, None
        
        frame = cv2.imread(str(img_file))
        if frame is None:
            return False, None
        
        self.current_frame_idx += 1
        return True, frame
    
    def get_timestamp(self, frame_idx: Optional[int] = None) -> float:
        """
        Get timestamp in seconds for a frame.
        
        Args:
            frame_idx: Frame index (uses current if None)
        
        Returns:
            Timestamp in seconds
        """
        if frame_idx is None:
            frame_idx = self.current_frame_idx
        return frame_idx / self.fps if self.fps > 0 else 0.0
    
    def seek(self, frame_idx: int) -> bool:
        """
        Seek to specific frame.
        
        Args:
            frame_idx: Target frame index
        
        Returns:
            True if successful
        """
        if 0 <= frame_idx < self.frame_count:
            self.current_frame_idx = frame_idx
            return True
        return False
    
    def release(self):
        """Release resources (no-op for image sequences)."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    @property
    def meta(self) -> dict:
        """Get sequence metadata."""
        return {
            'path': str(self.sequence_path),
            'name': self.name,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'duration_sec': self.frame_count / self.fps if self.fps > 0 else 0
        }
