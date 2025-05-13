"""Frame processing module."""

import cv2
import numpy as np
import supervision as sv
from .team_resolver import TeamResolver
from .coordinate_transformer import CoordinateTransformer


class FrameProcessor:
    def __init__(self, player_detector, field_detector, team_classifier, tracker, config):
        self.player_detector = player_detector
        self.field_detector = field_detector
        self.team_classifier = team_classifier
        self.tracker = tracker
        self.config = config
        
        self.team_resolver = TeamResolver()
        self.coordinate_transformer = CoordinateTransformer(config)
    
    def process_frame(self, frame):
        """Process a single frame."""
        # Player detection
        result = self.player_detector.detect_categories(frame)
        all_detections = result['all']
        
        # Separate ball detections
        ball_detections = result['ball']
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
        
        # Process other detections
        non_ball_detections = all_detections[all_detections.class_id != self.player_detector.BALL_ID]
        non_ball_detections = non_ball_detections.with_nms(
            threshold=self.config['detection']['nms_threshold'],
            class_agnostic=True
        )
        non_ball_detections = self.tracker.update(non_ball_detections)
        
        # Separate by type
        goalkeepers = non_ball_detections[non_ball_detections.class_id == self.player_detector.GOALKEEPER_ID]
        players = non_ball_detections[non_ball_detections.class_id == self.player_detector.PLAYER_ID]
        referees = non_ball_detections[non_ball_detections.class_id == self.player_detector.REFEREE_ID]
        
        # Team assignment
        if len(players) > 0:
            players = self._assign_teams(frame, players)
            
            if len(goalkeepers) > 0:
                goalkeepers = self.team_resolver.resolve_goalkeeper_teams(players, goalkeepers)
        
        # Create detections dict
        detections = {
            'ball': ball_detections,
            'goalkeepers': goalkeepers,
            'players': players,
            'referees': referees
        }
        
        # Field detection and transformation
        transformer = self._update_field_transformation(frame)
        
        # Update ball trail
        if len(ball_detections) > 0 and transformer is not None:
            ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = transformer.transform_points(points=ball_xy)
            if len(pitch_ball_xy) > 0:
                self.tracker.update_ball_trail(pitch_ball_xy[0])
        
        return detections, transformer
    
    def _assign_teams(self, frame, players):
        """Assign teams to players."""
        try:
            player_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
            team_ids = self.team_classifier.predict(player_crops)
            players.class_id = team_ids
        except Exception as e:
            print(f"Team assignment error: {e}")
            players.class_id = np.zeros(len(players), dtype=int)
        
        return players
    
    def _update_field_transformation(self, frame):
        """Update field transformation."""
        key_points = self.field_detector.detect_keypoints(frame)
        transformer = self.coordinate_transformer.update(key_points)
        
        if transformer is not None:
            self.tracker.update_transformation(transformer.m)
            # Use averaged transformation
            averaged_matrix = self.tracker.get_averaged_transformation()
            if averaged_matrix is not None:
                transformer.m = averaged_matrix
        
        return transformer
