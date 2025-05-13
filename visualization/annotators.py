"""Visualization annotators module."""

import cv2
import numpy as np
import supervision as sv


class FootballAnnotator:
    def __init__(self, config):
        self.config = config
        self._init_annotators()
    
    def _init_annotators(self):
        """Initialize supervision annotators."""
        colors = [
            self.config['display']['team_colors']['team_1'],
            self.config['display']['team_colors']['team_2'],
            self.config['display']['referee_color']
        ]
        
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(colors),
            thickness=2
        )
        
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(colors),
            text_color=sv.Color.from_hex('#000000'),
            text_position=sv.Position.BOTTOM_CENTER
        )
        
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex(self.config['display']['ball_color']),
            base=25,
            height=21,
            outline_thickness=1
        )
    
    def annotate_frame(self, frame, detections):
        """Annotate frame with detections."""
        annotated_frame = frame.copy()
        
        # Merge all player-type detections
        all_detections = []
        for category in ['players', 'goalkeepers', 'referees']:
            if len(detections[category]) > 0:
                all_detections.append(detections[category])
        
        if all_detections:
            merged = sv.Detections.merge(all_detections)
            
            annotated_frame = self.ellipse_annotator.annotate(
                scene=annotated_frame,
                detections=merged
            )
            
            if self.config['display']['show_tracking_ids'] and merged.tracker_id is not None:
                labels = [f"#{tracker_id}" for tracker_id in merged.tracker_id]
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame,
                    detections=merged,
                    labels=labels
                )
        
        # Annotate ball
        if self.config['display']['show_ball'] and len(detections['ball']) > 0:
            annotated_frame = self.triangle_annotator.annotate(
                scene=annotated_frame,
                detections=detections['ball']
            )
        
        return annotated_frame
