"""Visualization annotators module with pose support - Fixed version."""

import cv2
import numpy as np
import supervision as sv


class FootballAnnotator:
    def __init__(self, config):
        self.config = config
        self._init_annotators()
        self.enable_pose = config.get('display', {}).get('show_pose', True)
        
        # Team colors for pose visualization
        self.team1_color = self._hex_to_rgb(config['display']['team_colors']['team_1'])
        self.team2_color = self._hex_to_rgb(config['display']['team_colors']['team_2'])
        self.default_pose_color = (0, 255, 0)  # Green
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
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
    
    def annotate_frame_with_pose(self, frame, detections, poses=None):
        """Annotate frame with detections and pose estimations."""
        # First apply standard annotations
        annotated = self.annotate_frame(frame, detections)
        
        # Then add pose annotations if available
        if poses and self.enable_pose:
            try:
                annotated = self._annotate_poses(annotated, detections, poses)
            except Exception as e:
                print(f"Error annotating poses: {e}")
        
        return annotated
    
    def annotate_frame(self, frame, detections):
        """Standard frame annotation without poses."""
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
    
    def _annotate_poses(self, frame, detections, poses):
        """Add pose annotations to frame."""
        from models.detector import PoseDetector
        pose_drawer = PoseDetector()  # Just for drawing method
        
        # Draw poses for players
        if 'players' in poses and poses['players']:
            players_detections = detections['players']
            for i, pose in enumerate(poses['players']):
                if pose is not None and i < len(players_detections):
                    try:
                        # Get color based on team
                        if hasattr(players_detections, 'class_id') and players_detections.class_id is not None:
                            team_id = players_detections.class_id[i] if i < len(players_detections.class_id) else 0
                        else:
                            team_id = 0
                        color = self.team1_color if team_id == 0 else self.team2_color
                        
                        # Draw pose
                        frame = pose_drawer.draw_pose(frame, pose, color=color, thickness=2)
                    except Exception as e:
                        print(f"Error drawing player pose {i}: {e}")
        
        # Draw poses for goalkeepers
        if 'goalkeepers' in poses and poses['goalkeepers']:
            goalkeepers_detections = detections['goalkeepers']
            for i, pose in enumerate(poses['goalkeepers']):
                if pose is not None and i < len(goalkeepers_detections):
                    try:
                        # Get color based on team
                        if hasattr(goalkeepers_detections, 'class_id') and goalkeepers_detections.class_id is not None:
                            team_id = goalkeepers_detections.class_id[i] if i < len(goalkeepers_detections.class_id) else 0
                        else:
                            team_id = 0
                        color = self.team1_color if team_id == 0 else self.team2_color
                        
                        # Draw pose with thicker lines for goalkeepers
                        frame = pose_drawer.draw_pose(frame, pose, color=color, thickness=3)
                    except Exception as e:
                        print(f"Error drawing goalkeeper pose {i}: {e}")
        
        return frame
    
    def draw_stats_with_pose(self, frame, stats, pose_stats=None):
        """Draw statistics including pose information."""
        annotated = frame.copy()
        
        y_offset = 30
        for key, value in stats.items():
            text = f"{key}: {value}"
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # Add pose statistics if available
        if pose_stats:
            y_offset += 10
            cv2.putText(annotated, "Pose Stats:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
            
            for key, value in pose_stats.items():
                text = f"  {key}: {value}"
                cv2.putText(annotated, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
        
        return annotated