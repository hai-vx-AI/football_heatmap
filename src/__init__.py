"""
Football Video Analysis System - Source Package
"""

__version__ = "1.0.0"
__author__ = "hai-vx-AI"

from .video_io import VideoReader, VideoWriter, SoccerNetSequenceReader
from .detector import Detector
from .people_tracker import PeopleTracker
from .ball_tracker import BallTracker
from .team_assigner import TeamAssigner
from .possession_analyzer import PossessionAnalyzer
from .renderer import Renderer
from .logger import Logger

__all__ = [
    'VideoReader',
    'VideoWriter',
    'SoccerNetSequenceReader',
    'Detector',
    'PeopleTracker',
    'BallTracker',
    'TeamAssigner',
    'PossessionAnalyzer',
    'Renderer',
    'Logger',
]
