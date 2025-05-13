"""Coordinate transformation module."""

import numpy as np
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration


class CoordinateTransformer:
    def __init__(self, config):
        self.config = config
        self.pitch_config = SoccerPitchConfiguration()
        self.keypoint_threshold = config['detection']['keypoint_confidence_threshold']
    
    def update(self, key_points):
        """Update transformation based on detected keypoints."""
        if key_points is None:
            return None
        
        # Filter keypoints by confidence
        filter_mask = key_points.confidence[0] > self.keypoint_threshold
        frame_reference_points = key_points.xy[0][filter_mask]
        pitch_reference_points = np.array(self.pitch_config.vertices)[filter_mask]
        
        if len(frame_reference_points) >= 4:
            transformer = ViewTransformer(
                source=frame_reference_points,
                target=pitch_reference_points
            )
            return transformer
        
        return None
    
    def transform_points(self, points, transformer):
        """Transform points using the provided transformer."""
        if transformer is None:
            return np.array([])
        
        return transformer.transform_points(points=points)
