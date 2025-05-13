"""Object tracking module."""

import numpy as np
import supervision as sv
from collections import deque


class ObjectTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.tracker.reset()
        self.ball_trail = []
        self.transformation_matrices = deque(maxlen=10)
    
    def update(self, detections):
        """Update tracking for detections."""
        return self.tracker.update_with_detections(detections=detections)
    
    def update_ball_trail(self, position):
        """Update ball trail with new position."""
        if position is not None:
            self.ball_trail.append(position)
            if len(self.ball_trail) > 30:
                self.ball_trail.pop(0)
    
    def update_transformation(self, matrix):
        """Update transformation matrix history."""
        if matrix is not None:
            self.transformation_matrices.append(matrix)
    
    def get_averaged_transformation(self):
        """Get averaged transformation matrix."""
        if len(self.transformation_matrices) == 0:
            return None
        return np.mean(np.array(self.transformation_matrices), axis=0)
    
    def reset(self):
        """Reset tracker state."""
        self.tracker.reset()
        self.ball_trail.clear()
        self.transformation_matrices.clear()
