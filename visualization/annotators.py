"""Visualization annotators module with pose and segmentation support."""

import cv2
import numpy as np
import supervision as sv


class FootballAnnotator:
    def __init__(self, config):
        self.config = config
        self._init_annotators()
        self.enable_pose = config.get('display', {}).get('show_pose', True)
        self.enable_segmentation = config.get('display', {}).get('show_segmentation', True)
        
        # Team colors for pose visualization
        self.team1_color = self._hex_to_rgb(config['display']['team_colors']['team_1'])
        self.team2_color = self._hex_to_rgb(config['display']['team_colors']['team_2'])
        self.default_pose_color = (0, 255, 0)  # Green
        
        # Segmentation transparency
        self.segmentation_alpha = config.get('display', {}).get('segmentation_alpha', 0.6)
    
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
    
    def annotate_frame_with_pose_and_segmentation(self, frame, detections, poses=None, segmentations=None):
        """Annotate frame with detections, pose estimations, and segmentations."""
        # First apply segmentations if available
        if segmentations and self.enable_segmentation:
            frame = self._annotate_segmentations(frame, detections, segmentations)
        
        # Then apply standard annotations
        annotated = self.annotate_frame(frame, detections)
        
        # Finally add pose annotations if available
        if poses and self.enable_pose:
            try:
                annotated = self._annotate_poses(annotated, detections, poses)
            except Exception as e:
                print(f"Error annotating poses: {e}")
        
        return annotated
    
    def annotate_frame_with_pose(self, frame, detections, poses=None):
        """Backward compatible method for pose-only annotation."""
        return self.annotate_frame_with_pose_and_segmentation(frame, detections, poses, None)
    
    def _annotate_segmentations(self, frame, detections, segmentations):
        """Add segmentation masks to frame."""
        overlay = frame.copy()
        
        # Draw segmentation masks for players
        if 'players' in segmentations and segmentations['players']:
            players_detections = detections['players']
            for i, mask in enumerate(segmentations['players']):
                if mask is not None and i < len(players_detections):
                    try:
                        # Get color based on team
                        if hasattr(players_detections, 'class_id') and players_detections.class_id is not None:
                            team_id = players_detections.class_id[i] if i < len(players_detections.class_id) else 0
                        else:
                            team_id = 0
                        color = self.team1_color if team_id == 0 else self.team2_color
                        
                        # Apply mask with color
                        mask_bool = mask > 0.5 if isinstance(mask, np.ndarray) else mask
                        overlay[mask_bool] = color
                    except Exception as e:
                        print(f"Error drawing player segmentation {i}: {e}")
        
        # Draw segmentation masks for goalkeepers
        if 'goalkeepers' in segmentations and segmentations['goalkeepers']:
            goalkeepers_detections = detections['goalkeepers']
            for i, mask in enumerate(segmentations['goalkeepers']):
                if mask is not None and i < len(goalkeepers_detections):
                    try:
                        # Get color based on team
                        if hasattr(goalkeepers_detections, 'class_id') and goalkeepers_detections.class_id is not None:
                            team_id = goalkeepers_detections.class_id[i] if i < len(goalkeepers_detections.class_id) else 0
                        else:
                            team_id = 0
                        color = self.team1_color if team_id == 0 else self.team2_color
                        
                        # Apply mask with color (brighter for goalkeepers)
                        brighter_color = tuple(min(255, int(c * 1.3)) for c in color)
                        mask_bool = mask > 0.5 if isinstance(mask, np.ndarray) else mask
                        overlay[mask_bool] = brighter_color
                    except Exception as e:
                        print(f"Error drawing goalkeeper segmentation {i}: {e}")
        
        # Blend overlay with original frame
        frame = cv2.addWeighted(frame, 1 - self.segmentation_alpha, overlay, self.segmentation_alpha, 0)
        return frame
    
    def annotate_frame(self, frame, detections):
        """Standard frame annotation without poses or segmentations."""
        annotated_frame = frame.copy()
        
        # Merge all player-type detections
        all_detections = []
        for category in ['players', 'goalkeepers', 'referees']:
            if len(detections[category]) > 0:
                all_detections.append(detections[category])
        
        if all_detections:
            merged = sv.Detections.merge(all_detections)
            
            # Only draw bounding boxes if segmentation is not enabled
            if not self.enable_segmentation:
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
    
    def draw_stats_with_pose_and_segmentation(self, frame, stats, pose_stats=None, seg_stats=None):
        """Draw statistics including pose and segmentation information."""
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
        
        # Add segmentation statistics if available
        if seg_stats:
            y_offset += 10
            cv2.putText(annotated, "Segmentation Stats:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
            y_offset += 25
            
            for key, value in seg_stats.items():
                text = f"  {key}: {value}"
                cv2.putText(annotated, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
                y_offset += 25
        
        return annotated
    
    def draw_stats_with_pose(self, frame, stats, pose_stats=None):
        """Backward compatible method."""
        return self.draw_stats_with_pose_and_segmentation(frame, stats, pose_stats, None)