"""Object detection modules."""

import cv2
import numpy as np
import supervision as sv
from inference import get_model
import time


class ObjectDetector:
    def __init__(self, model_id, api_key, confidence_threshold=0.5):
        self.model = get_model(model_id=model_id, api_key=api_key)
        self.confidence_threshold = confidence_threshold
        
        # Class IDs
        self.BALL_ID = 0
        self.GOALKEEPER_ID = 1
        self.PLAYER_ID = 2
        self.REFEREE_ID = 3
    
    def detect(self, frame):
        """Detect all objects in frame."""
        result = self.model.infer(
            frame,
            confidence=self.confidence_threshold
        )[0]
        return sv.Detections.from_inference(result)
    
    def detect_players_only(self, frame):
        """Detect only players."""
        detections = self.detect(frame)
        return detections[detections.class_id == self.PLAYER_ID]
    
    def detect_ball(self, frame):
        """Detect ball with padding."""
        detections = self.detect(frame)
        ball_detections = detections[detections.class_id == self.BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
        return ball_detections
    
    def detect_categories(self, frame):
        """Detect and separate all categories."""
        detections = self.detect(frame)
        
        return {
            'all': detections,
            'ball': detections[detections.class_id == self.BALL_ID],
            'goalkeepers': detections[detections.class_id == self.GOALKEEPER_ID],
            'players': detections[detections.class_id == self.PLAYER_ID],
            'referees': detections[detections.class_id == self.REFEREE_ID]
        }


class FieldDetector:
    def __init__(self, model_id, api_key, confidence_threshold=0.5):
        self.model = get_model(model_id=model_id, api_key=api_key)
        self.confidence_threshold = confidence_threshold
    
    def detect_keypoints(self, frame):
        """Detect field keypoints."""
        result = self.model.infer(
            frame,
            confidence=self.confidence_threshold
        )[0]
        return sv.KeyPoints.from_inference(result)
