"""Frame processing module with pose, segmentation, and SAHI support."""

import cv2
import numpy as np
import supervision as sv
from .team_resolver import TeamResolver
from .coordinate_transformer import CoordinateTransformer
from .sahi_processor import SAHIProcessor


class FrameProcessor:
    def __init__(self, player_detector, field_detector, team_classifier, tracker, config):
        self.player_detector = player_detector
        self.field_detector = field_detector
        self.team_classifier = team_classifier
        self.tracker = tracker
        self.config = config
        
        self.team_resolver = TeamResolver()
        self.coordinate_transformer = CoordinateTransformer(config)
        self.sahi_processor = SAHIProcessor(config)
        
        # Check if features are enabled
        self.enable_pose = config.get('display', {}).get('show_pose', True)
        self.enable_segmentation = config.get('display', {}).get('show_segmentation', True)
        self.enable_sahi = config.get('sahi', {}).get('enable', False)
    
    def process_frame(self, frame):
        """Process a single frame with SAHI, pose estimation and segmentation."""
        # Process with SAHI if enabled
        if self.enable_sahi:
            detections_dict, poses, segmentations = self.sahi_processor.process_with_sahi(
                frame, 
                self.player_detector,
                self.enable_pose,
                self.enable_segmentation
            )
        else:
            # Use standard processing
            if hasattr(self.player_detector, 'detect_with_pose_and_segmentation'):
                detections_dict, poses, segmentations = self.player_detector.detect_with_pose_and_segmentation(frame)
            else:
                # Fallback to basic detection
                result = self.player_detector.detect_categories(frame)
                detections_dict = result
                poses = None
                segmentations = None
        
        # Get all detections
        all_detections = self._merge_all_detections(detections_dict)
        
        # Separate ball detections
        ball_detections = detections_dict['ball']
        if len(ball_detections) > 0:
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
        
        # Calculate statistics
        pose_stats = None
        seg_stats = None
        sahi_stats = None
        
        if poses:
            pose_stats = self._calculate_pose_stats(poses)
        
        if segmentations:
            seg_stats = self._calculate_segmentation_stats(segmentations)
        
        if self.enable_sahi:
            sahi_stats = self._calculate_sahi_stats(detections)
        
        return detections, transformer, poses, pose_stats, segmentations, seg_stats, sahi_stats
    
    def _merge_all_detections(self, detections_dict):
        """Merge all category detections."""
        all_detections = []
        
        for category in ['ball', 'goalkeepers', 'players', 'referees']:
            if category in detections_dict and len(detections_dict[category]) > 0:
                all_detections.append(detections_dict[category])
        
        if all_detections:
            return sv.Detections.merge(all_detections)
        else:
            return sv.Detections.empty()
    
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
    
    def _calculate_pose_stats(self, poses):
        """Calculate statistics from pose data."""
        stats = {}
        
        # Count detected poses
        player_poses_detected = sum(1 for p in poses.get('players', []) if p is not None)
        goalkeeper_poses_detected = sum(1 for p in poses.get('goalkeepers', []) if p is not None)
        
        stats['player_poses'] = player_poses_detected
        stats['goalkeeper_poses'] = goalkeeper_poses_detected
        stats['total_poses'] = player_poses_detected + goalkeeper_poses_detected
        
        # Calculate average pose confidence if available
        all_confidences = []
        for category in ['players', 'goalkeepers']:
            if category in poses:
                for pose in poses[category]:
                    if pose and 'confidence' in pose:
                        all_confidences.extend(pose['confidence'])
        
        if all_confidences:
            stats['avg_confidence'] = f"{np.mean(all_confidences):.2f}"
        
        return stats
    
    def _calculate_segmentation_stats(self, segmentations):
        """Calculate statistics from segmentation data."""
        stats = {}
        
        # Count detected segmentations
        player_segs_detected = sum(1 for s in segmentations.get('players', []) if s is not None)
        goalkeeper_segs_detected = sum(1 for s in segmentations.get('goalkeepers', []) if s is not None)
        
        stats['player_segments'] = player_segs_detected
        stats['goalkeeper_segments'] = goalkeeper_segs_detected
        stats['total_segments'] = player_segs_detected + goalkeeper_segs_detected
        
        # Calculate average mask size if available
        mask_sizes = []
        for category in ['players', 'goalkeepers']:
            if category in segmentations:
                for mask in segmentations[category]:
                    if mask is not None and isinstance(mask, np.ndarray):
                        mask_sizes.append(np.sum(mask > 0.5))
        
        if mask_sizes:
            stats['avg_mask_size'] = f"{np.mean(mask_sizes):.0f} pixels"
        
        return stats
    
    def _calculate_sahi_stats(self, detections):
        """Calculate SAHI statistics."""
        stats = {}
        
        total_detections = 0
        for category in ['players', 'goalkeepers', 'referees', 'ball']:
            total_detections += len(detections[category])
        
        stats['total_detections'] = total_detections
        stats['slices'] = f"{self.config['sahi']['slice_rows']} x {self.config['sahi']['slice_cols']}"
        
        return stats