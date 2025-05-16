"""Extended detector with pose estimation and SAM segmentation support - with adaptive padding."""

import cv2
import numpy as np
import supervision as sv
from inference import get_model
from ultralytics import YOLO, SAM
import time
import torch


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


class PoseDetector:
    def __init__(self, model_name='yolo11n-pose.pt', device='cpu'):
        """Initialize pose detector with YOLO pose model."""
        try:
            self.model = YOLO(model_name).to(device)
            self.device = device
        except Exception as e:
            print(f"Error loading pose model: {e}")
            self.model = YOLO(model_name)
            self.device = 'cpu'
        
        # COCO pose keypoints
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Skeleton connections for drawing
        self.skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # legs
            [5, 11], [6, 12], [5, 6],  # torso
            [5, 7], [6, 8], [7, 9], [8, 10],  # arms
            [1, 2], [1, 0], [2, 0], [0, 3], [0, 4]  # head
        ]
    
    def detect_poses(self, frame, boxes=None):
        """Detect poses in frame or within specific boxes.
        
        Args:
            frame: The frame to analyze
            boxes: Bounding boxes to analyze. If None, analyze whole frame (not recommended)
        
        Returns:
            List of pose data
        """
        if boxes is None or len(boxes) == 0:
            # We should avoid running on full frame for performance,
            # but keep this as a fallback
            print("Warning: Running pose detection on full frame is inefficient")
            try:
                results = self.model(frame, verbose=False, device=self.device)
                return self._process_results(results, frame.shape[:2])
            except Exception as e:
                print(f"Pose detection error on full frame: {e}")
                return []
        else:
            # Run on cropped regions (recommended approach)
            all_poses = []
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                
                # Ensure valid crop bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    all_poses.append(None)
                    continue
                
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    all_poses.append(None)
                    continue
                
                try:
                    results = self.model(crop, verbose=False, device=self.device)
                    poses = self._process_results(results, crop.shape[:2])
                    
                    # Adjust coordinates back to full frame
                    if poses and len(poses) > 0:
                        pose = poses[0]  # Take first detection in crop
                        if pose['keypoints'] is not None:
                            # Adjust keypoint coordinates
                            pose['keypoints'][:, 0] += x1
                            pose['keypoints'][:, 1] += y1
                        all_poses.append(pose)
                    else:
                        all_poses.append(None)
                except Exception as e:
                    print(f"Pose detection error on crop: {e}")
                    all_poses.append(None)
            
            return all_poses
    
    def _process_results(self, results, img_shape):
        """Process YOLO results into pose data."""
        poses = []
        
        try:
            for r in results:
                if r is None:
                    continue
                    
                # Check if we have keypoints
                if not hasattr(r, 'keypoints') or r.keypoints is None:
                    continue
                    
                # Check keypoints data shape
                if not hasattr(r.keypoints, 'data') or r.keypoints.data.shape[0] == 0:
                    continue
                
                # Process each detected pose
                for i in range(r.keypoints.data.shape[0]):
                    try:
                        keypoints = r.keypoints.data[i].cpu().numpy()
                        
                        # Get bounding box if available
                        bbox = None
                        if hasattr(r, 'boxes') and r.boxes is not None:
                            if hasattr(r.boxes, 'xyxy') and i < len(r.boxes.xyxy):
                                bbox = r.boxes.xyxy[i].cpu().numpy()
                        
                        # Extract pose data
                        pose_data = {
                            'keypoints': keypoints[:, :2],  # x, y coordinates
                            'confidence': keypoints[:, 2],   # confidence scores
                            'bbox': bbox
                        }
                        
                        poses.append(pose_data)
                    except Exception as e:
                        print(f"Error processing pose {i}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error in _process_results: {e}")
        
        return poses
    
    def draw_pose(self, frame, pose_data, color=(0, 255, 0), thickness=2):
        """Draw pose keypoints and skeleton on frame."""
        if pose_data is None or pose_data['keypoints'] is None:
            return frame
        
        keypoints = pose_data['keypoints']
        confidence = pose_data['confidence']
        
        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            if confidence[i] > 0.3:  # Lower threshold for visibility
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 0), 2)  # Black border
        
        # Draw skeleton
        for connection in self.skeleton:
            kpt1, kpt2 = connection
            
            if kpt1 < len(confidence) and kpt2 < len(confidence):
                if confidence[kpt1] > 0.3 and confidence[kpt2] > 0.3:
                    pt1 = tuple(keypoints[kpt1].astype(int))
                    pt2 = tuple(keypoints[kpt2].astype(int))
                    cv2.line(frame, pt1, pt2, color, thickness)
        
        return frame


class SegmentationDetector:
    def __init__(self, model_name='sam2.1_b.pt', device='cpu'):
        """Initialize SAM model for segmentation."""
        try:
            self.model = SAM(model_name).to(device)
            self.device = device
            print(f"Loaded SAM model: {model_name}")
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            self.model = None
    
    def segment_boxes(self, frame, boxes):
        """Segment objects within bounding boxes using simple center-point prompts.
        
        Args:
            frame: The full frame
            boxes: List of bounding boxes to segment
            
        Returns:
            List of segmentation masks
        """
        if self.model is None or len(boxes) == 0:
            return []
        
        try:
            # Convert boxes to list format for SAM
            boxes_list = boxes.tolist() if isinstance(boxes, np.ndarray) else boxes
            
            # Create results list
            masks = []
            
            # Process each box individually with simple foreground/background prompts
            for box in boxes_list:
                x1, y1, x2, y2 = map(int, box)
                
                # Very simple strategy: center point is foreground, corners are background
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Single foreground point (center of box)
                fg_points = [[center_x, center_y]]
                
                # Four background points (outside the box)
                # Make sure they're not too far to avoid segmenting other objects
                margin = min((x2 - x1), (y2 - y1)) // 10  # 10% of box size
                bg_points = [
                    [max(0, x1 - margin), max(0, y1 - margin)],              # Top-left
                    [min(frame.shape[1] - 1, x2 + margin), max(0, y1 - margin)],  # Top-right
                    [max(0, x1 - margin), min(frame.shape[0] - 1, y2 + margin)],  # Bottom-left
                    [min(frame.shape[1] - 1, x2 + margin), min(frame.shape[0] - 1, y2 + margin)]  # Bottom-right
                ]
                
                # Combine points
                all_points = fg_points + bg_points
                all_labels = [1] * len(fg_points) + [0] * len(bg_points)
                
                # Run SAM with box and simple point prompts
                try:
                    result = self.model(
                        frame,
                        bboxes=[box],
                        points=[all_points],
                        labels=[all_labels],
                        verbose=False,
                        device=self.device
                    )
                    
                    # Extract mask
                    if result and result[0].masks is not None:
                        mask_data = result[0].masks.data.cpu().numpy()
                        if mask_data.ndim >= 2 and mask_data.shape[0] > 0:
                            masks.append(mask_data[0])  # Take first mask
                        else:
                            masks.append(None)
                    else:
                        masks.append(None)
                except Exception as e:
                    print(f"Error segmenting box {box}: {e}")
                    masks.append(None)
            
            return masks
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            return [None] * len(boxes)
    
    def debug_segment_box(self, image, box):
        """Debug version with simpler point prompts for player segmentation."""
        if self.model is None:
            return None, image
        
        try:
            # Convert box to list
            box_list = box.tolist() if isinstance(box, np.ndarray) else box
            x1, y1, x2, y2 = map(int, box_list)
            
            # Very simple strategy: center point is foreground, corners are background
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Single foreground point (center of box)
            fg_points = [[center_x, center_y]]
            
            # Four background points (outside the box)
            margin = min((x2 - x1), (y2 - y1)) // 10  # 10% of box size
            bg_points = [
                [max(0, x1 - margin), max(0, y1 - margin)],              # Top-left
                [min(image.shape[1] - 1, x2 + margin), max(0, y1 - margin)],  # Top-right
                [max(0, x1 - margin), min(image.shape[0] - 1, y2 + margin)],  # Bottom-left
                [min(image.shape[1] - 1, x2 + margin), min(image.shape[0] - 1, y2 + margin)]  # Bottom-right
            ]
            
            # Combine points
            all_points = fg_points + bg_points
            all_labels = [1] * len(fg_points) + [0] * len(bg_points)
            
            # Run SAM with box and simple point prompts
            result = self.model(
                image,
                bboxes=[box_list],
                points=[all_points],
                labels=[all_labels],
                verbose=False,
                device=self.device
            )
            
            # Extract mask
            mask = None
            if result and result[0].masks is not None:
                mask_data = result[0].masks.data.cpu().numpy()
                if mask_data.ndim >= 2 and mask_data.shape[0] > 0:
                    mask = mask_data[0]
            
            # Visualize result for debugging
            vis_image = image.copy()
            if mask is not None:
                mask_bool = mask > 0.5
                vis_image[mask_bool] = vis_image[mask_bool] * 0.7 + np.array([0, 0, 255]) * 0.3
                
            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw foreground points
            for px, py in fg_points:
                cv2.circle(vis_image, (px, py), 5, (0, 255, 0), -1)  # Green for foreground
                
            # Draw background points
            for px, py in bg_points:
                cv2.circle(vis_image, (px, py), 5, (0, 0, 255), -1)  # Red for background
                
            return mask, vis_image
                
        except Exception as e:
            print(f"Debug segmentation error: {e}")
            return None, image

class EnhancedObjectDetector(ObjectDetector):
    """Object detector with integrated pose estimation and segmentation."""
    
    def __init__(self, model_id, api_key, confidence_threshold=0.5, 
                 enable_pose=True, pose_model='yolo11m-pose.pt',
                 enable_segmentation=True, sam_model='sam2.1_s.pt', 
                 padding_ratio=0.1, device='cpu'):
        super().__init__(model_id, api_key, confidence_threshold)
        
        self.enable_pose = enable_pose
        self.enable_segmentation = enable_segmentation
        self.padding_ratio = padding_ratio  # Consistent padding ratio for both pose and segmentation
        self.device = device
        
        if self.enable_pose:
            try:
                self.pose_detector = PoseDetector(pose_model, device)
            except Exception as e:
                print(f"Failed to initialize pose detector: {e}")
                self.enable_pose = False
        
        if self.enable_segmentation:
            try:
                self.segmentation_detector = SegmentationDetector(sam_model, device)
            except Exception as e:
                print(f"Failed to initialize segmentation detector: {e}")
                self.enable_segmentation = False
    
    def _apply_padding(self, boxes, frame_shape, padding_ratio=0.1):
        """Apply percentage-based padding to bounding boxes.
        
        Args:
            boxes: Array of bounding boxes in xyxy format
            frame_shape: Shape of the frame (height, width)
            padding_ratio: Ratio of box size to add as padding (e.g., 0.1 = 10%)
            
        Returns:
            Array of padded boxes
        """
        padded_boxes = []
        height, width = frame_shape[:2]
        
        for box in boxes:
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Calculate padding based on percentage of box dimensions
            pad_x = int(box_width * padding_ratio)
            pad_y = int(box_height * padding_ratio)
            
            # Apply padding with bounds checking
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)
            
            padded_boxes.append([x1, y1, x2, y2])
        
        return np.array(padded_boxes)
    
    def detect_with_pose_and_segmentation(self, frame):
        """Detect objects, estimate poses, and segment players with percentage-based padding.
           This method follows a strict pipeline:
           1. First detect all objects using base object detection
           2. Apply percentage-based padding to human boxes
           3. Run pose and segmentation ONLY on the padded boxes
        """
        # STEP 1: Get standard object detections first
        detections = self.detect_categories(frame)
        
        # Initialize outputs
        poses = {'players': [], 'goalkeepers': [], 'referees': []}
        segmentations = {'players': [], 'goalkeepers': [], 'referees': []}
        
        # STEP 2: Process each category of humans separately
        for category in ['players', 'goalkeepers', 'referees']:
            # Skip if no detections in this category
            if len(detections[category]) == 0:
                continue
            
            # Get boxes for this category
            boxes = detections[category].xyxy
            
            # Apply padding to boxes (same padding for both pose and segmentation)
            padded_boxes = self._apply_padding(boxes, frame.shape, self.padding_ratio)
            
            # STEP 3: Run pose estimation if enabled
            if self.enable_pose:
                try:
                    # Detect poses on padded boxes
                    pose_results = self.pose_detector.detect_poses(frame, padded_boxes)
                    poses[category] = pose_results
                except Exception as e:
                    print(f"Error during pose estimation for {category}: {e}")
                    poses[category] = [None] * len(boxes)
            
            # STEP 4: Run segmentation if enabled
            if self.enable_segmentation:
                try:
                    # Run segmentation on padded boxes
                    seg_results = self.segmentation_detector.segment_boxes(frame, padded_boxes)
                    segmentations[category] = seg_results
                except Exception as e:
                    print(f"Error during segmentation for {category}: {e}")
                    segmentations[category] = [None] * len(boxes)
        
        # If pose or segmentation is disabled, initialize with None values
        if not self.enable_pose:
            poses = None
        
        if not self.enable_segmentation:
            segmentations = None
        
        return detections, poses, segmentations
    
    # Keep backward compatibility
    def detect_with_pose(self, frame):
        """Backward compatible method for pose-only detection."""
        detections, poses, _ = self.detect_with_pose_and_segmentation(frame)
        return detections, poses


class FieldDetector:
    def __init__(self, model_id, api_key, confidence_threshold=0.5):
        self.model = get_model(model_id=model_id, api_key=api_key)
        self.confidence_threshold = confidence_threshold
    
    def detect_keypoints(self, frame):
        """Detect field keypoints."""
        result = self.model.infer(
            frame,
            confidence=self.confidence_threshold,

        )[0]
        return sv.KeyPoints.from_inference(result)