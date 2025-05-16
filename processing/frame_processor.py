"""Frame processing module with player possession detection."""

import cv2
import numpy as np
import supervision as sv
from .team_resolver import TeamResolver
from .coordinate_transformer import CoordinateTransformer
from .sahi_processor import SAHIProcessor


class FrameProcessor:
    def __init__(self, player_detector, field_detector, team_classifier, tracker, config, possession_detector=None):
        self.player_detector = player_detector
        self.field_detector = field_detector
        self.team_classifier = team_classifier
        self.tracker = tracker
        self.config = config
        self.possession_detector = possession_detector
        
        self.team_resolver = TeamResolver()
        self.coordinate_transformer = CoordinateTransformer(config)
        self.sahi_processor = SAHIProcessor(config)
        
        # Check if features are enabled
        self.enable_pose = config.get('display', {}).get('show_pose', True)
        self.enable_segmentation = config.get('display', {}).get('show_segmentation', True)
        self.enable_sahi = config.get('sahi', {}).get('enable', False)
        self.enable_possession = config.get('possession_detection', {}).get('enable', True)
        
        # Processing resolution
        self.processing_resolution = config.get('processing', {}).get('resolution', None)
        if self.processing_resolution:
            self.processing_width = self.processing_resolution[0]
            self.processing_height = self.processing_resolution[1]
            print(f"Processing resolution set to: {self.processing_width}x{self.processing_height}")
    
    def process_frame(self, frame):
        """Process a single frame at standardized resolution."""
        original_height, original_width = frame.shape[:2]
        
        # Resize frame to processing resolution if specified
        scale_factor = 1.0
        if self.processing_resolution:
            processing_frame = cv2.resize(frame, (self.processing_width, self.processing_height))
            scale_x = original_width / self.processing_width
            scale_y = original_height / self.processing_height
            scale_factor = (scale_x, scale_y)
        else:
            processing_frame = frame
            scale_factor = (1.0, 1.0)
        
        # Process with SAHI if enabled
        if self.enable_sahi:
            detections_dict, poses, segmentations = self.sahi_processor.process_with_sahi(
                processing_frame, 
                self.player_detector,
                self.enable_pose,
                self.enable_segmentation
            )
        else:
            # Use standard processing
            if hasattr(self.player_detector, 'detect_with_pose_and_segmentation'):
                detections_dict, poses, segmentations = self.player_detector.detect_with_pose_and_segmentation(processing_frame)
            else:
                # Fallback to basic detection
                result = self.player_detector.detect_categories(processing_frame)
                detections_dict = result
                poses = None
                segmentations = None
        
        # Scale detections back to original resolution
        if scale_factor != (1.0, 1.0):
            detections_dict = self._scale_detections_dict(detections_dict, scale_factor)
            poses = self._scale_poses(poses, scale_factor)
            segmentations = self._scale_segmentations(segmentations, scale_factor, 
                                                    (original_height, original_width))
        
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
        
        # Team assignment (use original frame for crops)
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
        
        # Field detection and transformation (use original frame)
        transformer = self._update_field_transformation(frame)
        
        # Update ball trail and get ball position
        ball_position = None
        if len(ball_detections) > 0:
            ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            if transformer is not None:
                pitch_ball_xy = transformer.transform_points(points=ball_xy)
                if len(pitch_ball_xy) > 0:
                    ball_position = pitch_ball_xy[0]
                    self.tracker.update_ball_trail(ball_position)
            else:
                # Use frame coordinates if no transformer
                if len(ball_xy) > 0:
                    ball_position = ball_xy[0]
        
        # Process possession detection if enabled
        possession_result = None
        if self.enable_possession and self.possession_detector is not None and ball_position is not None:
            possession_result = self.possession_detector.update(detections, ball_position)
        
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
        
        return detections, transformer, poses, pose_stats, segmentations, seg_stats, sahi_stats, possession_result
    
    # Rest of your existing methods remain unchanged
    def _scale_detections_dict(self, detections_dict, scale_factor):
        """Scale all detections back to original resolution."""
        scale_x, scale_y = scale_factor
        scaled_dict = {}
        
        for category, detections in detections_dict.items():
            if isinstance(detections, sv.Detections) and len(detections) > 0:
                # Create a new Detections object with copied arrays instead of detections.copy()
                scaled_dict[category] = sv.Detections(
                    xyxy=detections.xyxy.copy(),
                    confidence=detections.confidence.copy() if detections.confidence is not None else None,
                    class_id=detections.class_id.copy() if detections.class_id is not None else None,
                    tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None
                )
                
                # Scale bounding boxes
                scaled_dict[category].xyxy[:, [0, 2]] *= scale_x
                scaled_dict[category].xyxy[:, [1, 3]] *= scale_y
            else:
                scaled_dict[category] = detections
        
        return scaled_dict
    
    def _scale_poses(self, poses, scale_factor):
        """Scale pose keypoints back to original resolution."""
        if poses is None:
            return None
        
        scale_x, scale_y = scale_factor
        scaled_poses = {}
        
        for category in poses.keys():
            if category in poses:
                scaled_category = []
                for pose in poses[category]:
                    if pose is not None:
                        scaled_pose = pose.copy()
                        # Scale keypoints
                        scaled_pose['keypoints'][:, 0] *= scale_x
                        scaled_pose['keypoints'][:, 1] *= scale_y
                        # Scale bbox if present
                        if 'bbox' in scaled_pose and scaled_pose['bbox'] is not None:
                            scaled_pose['bbox'][[0, 2]] *= scale_x
                            scaled_pose['bbox'][[1, 3]] *= scale_y
                        scaled_category.append(scaled_pose)
                    else:
                        scaled_category.append(None)
                scaled_poses[category] = scaled_category
        
        return scaled_poses
    
    def _scale_segmentations(self, segmentations, scale_factor, original_size):
        """Scale segmentation masks back to original resolution."""
        if segmentations is None:
            return None
        
        scaled_segmentations = {}
        original_height, original_width = original_size
        
        for category in segmentations.keys():
            if category in segmentations:
                scaled_category = []
                for mask in segmentations[category]:
                    if mask is not None and isinstance(mask, np.ndarray):
                        # Resize mask to original dimensions
                        scaled_mask = cv2.resize(mask.astype(np.uint8), 
                                               (original_width, original_height),
                                               interpolation=cv2.INTER_NEAREST)
                        scaled_category.append(scaled_mask.astype(bool))
                    else:
                        scaled_category.append(None)
                scaled_segmentations[category] = scaled_category
        
        return scaled_segmentations
    
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
        referee_poses_detected = sum(1 for p in poses.get('referees', []) if p is not None)
        
        stats['player_poses'] = player_poses_detected
        stats['goalkeeper_poses'] = goalkeeper_poses_detected
        stats['referee_poses'] = referee_poses_detected
        stats['total_poses'] = player_poses_detected + goalkeeper_poses_detected + referee_poses_detected
        
        # Calculate average pose confidence if available
        all_confidences = []
        for category in poses.keys():
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
        referee_segs_detected = sum(1 for s in segmentations.get('referees', []) if s is not None)
        
        stats['player_segments'] = player_segs_detected
        stats['goalkeeper_segments'] = goalkeeper_segs_detected
        stats['referee_segments'] = referee_segs_detected
        stats['total_segments'] = player_segs_detected + goalkeeper_segs_detected + referee_segs_detected
        
        # Calculate average mask size if available
        mask_sizes = []
        for category in segmentations.keys():
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