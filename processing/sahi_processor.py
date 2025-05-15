"""SAHI (Slicing Adaptive Inference) implementation with support for adaptive padding."""

import numpy as np
import cv2
import supervision as sv
from typing import List, Tuple, Dict, Optional


class SAHISlicer:
    def __init__(self, slice_rows=2, slice_cols=2, overlap_ratio=0.2):
        """Initialize SAHI slicer."""
        self.slice_rows = slice_rows
        self.slice_cols = slice_cols
        self.overlap_ratio = overlap_ratio
    
    def slice_image(self, image: np.ndarray) -> List[Dict]:
        """Slice image into grid with overlap."""
        height, width = image.shape[:2]
        
        # Calculate slice dimensions
        slice_height = height // self.slice_rows
        slice_width = width // self.slice_cols
        
        # Calculate overlap
        overlap_h = int(slice_height * self.overlap_ratio)
        overlap_w = int(slice_width * self.overlap_ratio)
        
        slices = []
        
        for row in range(self.slice_rows):
            for col in range(self.slice_cols):
                # Calculate slice boundaries with overlap
                y1 = max(0, row * slice_height - overlap_h)
                y2 = min(height, (row + 1) * slice_height + overlap_h)
                x1 = max(0, col * slice_width - overlap_w)
                x2 = min(width, (col + 1) * slice_width + overlap_w)
                
                # Extract slice
                slice_img = image[y1:y2, x1:x2]
                
                # Store slice info
                slice_info = {
                    'image': slice_img,
                    'offset': (x1, y1),
                    'original_size': (width, height),
                    'slice_id': (row, col),
                    'bbox': (x1, y1, x2, y2)
                }
                
                slices.append(slice_info)
        
        return slices
    
    def merge_detections(self, slice_results: List[Dict], nms_threshold: float = 0.5) -> Dict:
        """Merge detections from all slices."""
        all_xyxy = []
        all_confidence = []
        all_class_id = []
        all_tracker_id = []
        all_poses = []
        all_masks = []
        
        for result in slice_results:
            if result['detections'] is None or len(result['detections']) == 0:
                continue
            
            detections = result['detections']
            offset_x, offset_y = result['offset']
            
            # Adjust coordinates to full image
            adjusted_xyxy = detections.xyxy.copy()
            adjusted_xyxy[:, [0, 2]] += offset_x
            adjusted_xyxy[:, [1, 3]] += offset_y
            
            all_xyxy.append(adjusted_xyxy)
            all_confidence.append(detections.confidence)
            all_class_id.append(detections.class_id)
            
            if detections.tracker_id is not None:
                all_tracker_id.append(detections.tracker_id)
            
            # Adjust poses if available
            if 'poses' in result and result['poses'] is not None:
                adjusted_poses = []
                for pose in result['poses']:
                    if pose is not None:
                        adjusted_pose = pose.copy()
                        adjusted_pose['keypoints'][:, 0] += offset_x
                        adjusted_pose['keypoints'][:, 1] += offset_y
                        adjusted_poses.append(adjusted_pose)
                    else:
                        adjusted_poses.append(None)
                all_poses.extend(adjusted_poses)
            
            # Adjust masks if available
            if 'masks' in result and result['masks'] is not None:
                for mask in result['masks']:
                    if mask is not None:
                        # Create full-size mask
                        full_mask = np.zeros(result['original_size'][::-1], dtype=bool)
                        y1, x1 = offset_y, offset_x
                        y2, x2 = y1 + mask.shape[0], x1 + mask.shape[1]
                        full_mask[y1:y2, x1:x2] = mask
                        all_masks.append(full_mask)
                    else:
                        all_masks.append(None)
        
        if not all_xyxy:
            return {
                'detections': sv.Detections.empty(),
                'poses': [],
                'masks': []
            }
        
        # Concatenate all detections
        xyxy = np.vstack(all_xyxy)
        confidence = np.concatenate(all_confidence)
        class_id = np.concatenate(all_class_id)
        
        # Create merged detections
        merged_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        
        # Apply NMS to remove duplicates
        merged_detections = merged_detections.with_nms(
            threshold=nms_threshold,
            class_agnostic=False
        )
        
        # Filter poses and masks to match NMS results
        # This is a simplified approach - in practice, you'd need to track which detections were kept
        kept_indices = self._get_kept_indices(xyxy, merged_detections.xyxy)
        
        filtered_poses = []
        filtered_masks = []
        
        if all_poses:
            for idx in kept_indices:
                if idx < len(all_poses):
                    filtered_poses.append(all_poses[idx])
        
        if all_masks:
            for idx in kept_indices:
                if idx < len(all_masks):
                    filtered_masks.append(all_masks[idx])
        
        return {
            'detections': merged_detections,
            'poses': filtered_poses,
            'masks': filtered_masks
        }
    
    def _get_kept_indices(self, original_xyxy: np.ndarray, kept_xyxy: np.ndarray) -> List[int]:
        """Find which original detections were kept after NMS."""
        kept_indices = []
        
        for kept_box in kept_xyxy:
            # Find closest original box
            diffs = np.abs(original_xyxy - kept_box).sum(axis=1)
            idx = np.argmin(diffs)
            kept_indices.append(idx)
        
        return kept_indices


class SAHIProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.sahi_config = config.get('sahi', {})
        self.enabled = self.sahi_config.get('enable', False)
        
        if self.enabled:
            self.slicer = SAHISlicer(
                slice_rows=self.sahi_config.get('slice_rows', 2),
                slice_cols=self.sahi_config.get('slice_cols', 2),
                overlap_ratio=self.sahi_config.get('overlap_ratio', 0.2)
            )
    
    def process_with_sahi(self, frame: np.ndarray, detector, 
                         enable_pose: bool = True, 
                         enable_segmentation: bool = True) -> Dict:
        """Process frame using SAHI slicing."""
        if not self.enabled:
            # Process normally without SAHI
            return self._process_single_frame(frame, detector, enable_pose, enable_segmentation)
        
        # Slice the image
        slices = self.slicer.slice_image(frame)
        
        # Process each slice
        slice_results = []
        for slice_info in slices:
            slice_img = slice_info['image']
            
            # Process slice
            if hasattr(detector, 'detect_with_pose_and_segmentation'):
                detections, poses, segmentations = detector.detect_with_pose_and_segmentation(slice_img)
            else:
                detections = detector.detect_categories(slice_img)
                poses = None
                segmentations = None
            
            # Combine results by category
            combined_detections = self._combine_category_detections(detections)
            combined_poses = self._combine_category_data(poses) if poses else None
            combined_masks = self._combine_category_data(segmentations) if segmentations else None
            
            slice_results.append({
                'detections': combined_detections,
                'poses': combined_poses,
                'masks': combined_masks,
                'offset': slice_info['offset'],
                'original_size': slice_info['original_size']
            })
        
        # Merge results from all slices
        merged_results = self.slicer.merge_detections(
            slice_results, 
            nms_threshold=self.config['detection']['nms_threshold']
        )
        
        # Split back into categories
        final_detections, final_poses, final_segmentations = self._split_into_categories(
            merged_results, detector
        )
        
        return final_detections, final_poses, final_segmentations
    
    def _process_single_frame(self, frame: np.ndarray, detector, 
                            enable_pose: bool, enable_segmentation: bool) -> Dict:
        """Process a single frame without SAHI."""
        if hasattr(detector, 'detect_with_pose_and_segmentation'):
            return detector.detect_with_pose_and_segmentation(frame)
        else:
            detections = detector.detect_categories(frame)
            return detections, None, None
    
    def _combine_category_detections(self, detections: Dict) -> sv.Detections:
        """Combine detections from all categories into one."""
        all_detections = []
        
        for category in ['players', 'goalkeepers', 'referees', 'ball']:
            if category in detections and len(detections[category]) > 0:
                all_detections.append(detections[category])
        
        if all_detections:
            return sv.Detections.merge(all_detections)
        else:
            return sv.Detections.empty()
    
    def _combine_category_data(self, data: Dict) -> List:
        """Combine poses or masks from all categories."""
        if data is None:
            return []
        
        combined = []
        for category in ['players', 'goalkeepers', 'referees']:
            if category in data:
                combined.extend(data[category])
        
        return combined
    
    def _split_into_categories(self, merged_results: Dict, detector) -> Tuple[Dict, Dict, Dict]:
        """Split merged results back into categories."""
        detections = merged_results['detections']
        poses = merged_results.get('poses', [])
        masks = merged_results.get('masks', [])
        
        # Split detections by class ID
        final_detections = {
            'ball': detections[detections.class_id == detector.BALL_ID],
            'goalkeepers': detections[detections.class_id == detector.GOALKEEPER_ID],
            'players': detections[detections.class_id == detector.PLAYER_ID],
            'referees': detections[detections.class_id == detector.REFEREE_ID]
        }
        
        # Split poses and masks - now include referees
        final_poses = {'players': [], 'goalkeepers': [], 'referees': []} if poses else None
        final_masks = {'players': [], 'goalkeepers': [], 'referees': []} if masks else None
        
        if poses or masks:
            pose_idx = 0
            mask_idx = 0
            
            for i in range(len(detections)):
                class_id = detections.class_id[i]
                
                if class_id == detector.PLAYER_ID:
                    if poses and pose_idx < len(poses):
                        final_poses['players'].append(poses[pose_idx])
                        pose_idx += 1
                    if masks and mask_idx < len(masks):
                        final_masks['players'].append(masks[mask_idx])
                        mask_idx += 1
                        
                elif class_id == detector.GOALKEEPER_ID:
                    if poses and pose_idx < len(poses):
                        final_poses['goalkeepers'].append(poses[pose_idx])
                        pose_idx += 1
                    if masks and mask_idx < len(masks):
                        final_masks['goalkeepers'].append(masks[mask_idx])
                        mask_idx += 1
                        
                elif class_id == detector.REFEREE_ID:
                    if poses and pose_idx < len(poses):
                        final_poses['referees'].append(poses[pose_idx])
                        pose_idx += 1
                    if masks and mask_idx < len(masks):
                        final_masks['referees'].append(masks[mask_idx])
                        mask_idx += 1
        
        return final_detections, final_poses, final_masks