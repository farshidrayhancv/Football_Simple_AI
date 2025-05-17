"""Visualization annotators module with player possession highlighting."""

import cv2
import numpy as np
import supervision as sv


class FootballAnnotator:
    def __init__(self, config, possession_detector=None):
        self.config = config
        self.possession_detector = possession_detector
        self._init_annotators()
        self.enable_pose = config.get('display', {}).get('show_pose', True)
        self.enable_segmentation = config.get('display', {}).get('show_segmentation', True)
        self.enable_possession = config.get('possession_detection', {}).get('enable', True)
        
        print(f"FootballAnnotator initialized with possession_detection={self.enable_possession}")
        
        # Load team colors for pose visualization (in BGR format)
        self.team1_color = self._hex_to_bgr(config['display']['team_colors']['team_1'])
        self.team2_color = self._hex_to_bgr(config['display']['team_colors']['team_2'])
        self.referee_color = self._hex_to_bgr(config['display']['referee_color'])
        self.ball_color = self._hex_to_bgr(config['display']['ball_color'])
        self.default_pose_color = (0, 255, 0)  # Green in BGR
        
        # Segmentation transparency
        self.segmentation_alpha = config.get('display', {}).get('segmentation_alpha', 0.6)
    
    def _hex_to_bgr(self, hex_color):
        """Convert hex color to BGR tuple for OpenCV."""
        hex_color = hex_color.lstrip('#')
        # Get RGB values
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        # Return BGR for OpenCV
        return (b, g, r)
    
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
    
    def annotate_frame_with_all_features(self, frame, detections, poses=None, segmentations=None, possession_result=None):
        """Annotate frame with all available features."""
        # First apply segmentations if available
        if segmentations and self.enable_segmentation:
            frame = self._annotate_segmentations(frame, detections, segmentations)
        
        # Then apply standard annotations
        annotated = self.annotate_frame(frame, detections)
        
        # Add pose annotations if available
        if poses and self.enable_pose:
            try:
                annotated = self._annotate_poses(annotated, detections, poses)
            except Exception as e:
                print(f"Error annotating poses: {e}")
        
        # Add possession visualization if enabled
        if self.enable_possession and possession_result and possession_result.get('player_id') is not None:
            try:
                print(f"Highlighting player with possession: ID={possession_result.get('player_id')}, Team={possession_result.get('team_id')}")
                
                # Highlight player with possession
                if self.possession_detector:
                    # Use our instance
                    self.possession_detector.current_possession = possession_result.get('player_id')
                    self.possession_detector.current_team = possession_result.get('team_id')
                    annotated = self.possession_detector.highlight_possession(annotated, detections)
                else:
                    # Import and create one if we don't have it
                    from models.player_possession_detector import PlayerPossessionDetector
                    temp_detector = PlayerPossessionDetector()
                    temp_detector.current_possession = possession_result.get('player_id')
                    temp_detector.current_team = possession_result.get('team_id')
                    annotated = temp_detector.highlight_possession(annotated, detections)
            except Exception as e:
                print(f"Error highlighting possession: {e}")
                import traceback
                traceback.print_exc()
        
        return annotated
    
    # Backward compatibility methods
    def annotate_frame_with_pose_and_segmentation(self, frame, detections, poses=None, segmentations=None):
        """Backward compatible method for pose and segmentation annotation."""
        return self.annotate_frame_with_all_features(frame, detections, poses, segmentations, None)
    
    def annotate_frame_with_pose(self, frame, detections, poses=None):
        """Backward compatible method for pose-only annotation."""
        return self.annotate_frame_with_all_features(frame, detections, poses, None, None)
    
    def _annotate_segmentations(self, frame, detections, segmentations):
        """Add segmentation masks to frame."""
        overlay = frame.copy()
        
        # Draw segmentation masks for players
        if 'players' in segmentations and segmentations['players']:
            players_detections = detections['players']
            for i, mask in enumerate(segmentations['players']):
                if mask is not None and i < len(players_detections):
                    try:
                        # Get color based on team from config
                        if hasattr(players_detections, 'class_id') and players_detections.class_id is not None:
                            team_id = int(players_detections.class_id[i]) if i < len(players_detections.class_id) else 0
                            team_key = f"team_{team_id + 1}"  # team_1 or team_2
                            hex_color = self.config['display']['team_colors'][team_key]
                            color = self._hex_to_bgr(hex_color)
                        else:
                            # Default to team 1 color if team can't be determined
                            hex_color = self.config['display']['team_colors']['team_1']
                            color = self._hex_to_bgr(hex_color)
                        
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
                        # Get color based on team from config
                        if hasattr(goalkeepers_detections, 'class_id') and goalkeepers_detections.class_id is not None:
                            team_id = int(goalkeepers_detections.class_id[i]) if i < len(goalkeepers_detections.class_id) else 0
                            team_key = f"team_{team_id + 1}"  # team_1 or team_2
                            hex_color = self.config['display']['team_colors'][team_key]
                            color = self._hex_to_bgr(hex_color)
                        else:
                            # Default to team 1 color if team can't be determined
                            hex_color = self.config['display']['team_colors']['team_1']
                            color = self._hex_to_bgr(hex_color)
                        
                        # Make goalkeepers slightly brighter
                        brighter_color = tuple(min(255, int(c * 1.3)) for c in color)
                        
                        # Apply mask with color
                        mask_bool = mask > 0.5 if isinstance(mask, np.ndarray) else mask
                        overlay[mask_bool] = brighter_color
                    except Exception as e:
                        print(f"Error drawing goalkeeper segmentation {i}: {e}")
        
        # Draw segmentation masks for referees
        if 'referees' in segmentations and segmentations['referees']:
            referees_detections = detections['referees']
            for i, mask in enumerate(segmentations['referees']):
                if mask is not None and i < len(referees_detections):
                    try:
                        # Get referee color directly from config
                        hex_color = self.config['display']['referee_color']
                        color = self._hex_to_bgr(hex_color)
                        
                        # Apply mask with color
                        mask_bool = mask > 0.5 if isinstance(mask, np.ndarray) else mask
                        overlay[mask_bool] = color
                    except Exception as e:
                        print(f"Error drawing referee segmentation {i}: {e}")
        
        # Draw segmentation for ball if available
        if 'ball' in segmentations and segmentations['ball']:
            ball_detections = detections['ball']
            for i, mask in enumerate(segmentations['ball']):
                if mask is not None and i < len(ball_detections):
                    try:
                        # Get ball color directly from config
                        hex_color = self.config['display']['ball_color']
                        color = self._hex_to_bgr(hex_color)
                        
                        # Apply mask with color
                        mask_bool = mask > 0.5 if isinstance(mask, np.ndarray) else mask
                        overlay[mask_bool] = color
                    except Exception as e:
                        print(f"Error drawing ball segmentation {i}: {e}")
    
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
        
        # Draw poses for referees
        if 'referees' in poses and poses['referees']:
            referees_detections = detections['referees']
            for i, pose in enumerate(poses['referees']):
                if pose is not None and i < len(referees_detections):
                    try:
                        # Use referee color from config
                        frame = pose_drawer.draw_pose(frame, pose, color=self.referee_color, thickness=2)
                    except Exception as e:
                        print(f"Error drawing referee pose {i}: {e}")
        
        return frame
    
    def draw_stats_with_all_features(self, frame, stats, pose_stats=None, seg_stats=None, possession_result=None):
        """Draw statistics including pose, segmentation, and possession information."""
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
        
        # Add possession information if available
        if possession_result and possession_result.get('player_id') is not None:
            y_offset += 10
            cv2.putText(annotated, "Possession:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
            
            player_id = possession_result.get('player_id')
            team_id = possession_result.get('team_id')
            
            team_text = f"Team {team_id + 1}" if team_id in [0, 1] else "Referee/Other"
            text = f"  Player #{player_id} ({team_text})"
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        return annotated
    
    # Backward compatibility methods
    def draw_stats_with_pose_and_segmentation(self, frame, stats, pose_stats=None, seg_stats=None):
        return self.draw_stats_with_all_features(frame, stats, pose_stats, seg_stats, None)
    
    def draw_stats_with_pose(self, frame, stats, pose_stats=None):
        return self.draw_stats_with_all_features(frame, stats, pose_stats, None, None)