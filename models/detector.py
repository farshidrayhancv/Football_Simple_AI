"""Extended detector with pose estimation support - Fixed version."""

import cv2
import numpy as np
import supervision as sv
from inference import get_model
from ultralytics import YOLO
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


class PoseDetector:
    def __init__(self, model_name='yolo11x-pose.pt', device='cuda'):
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
        """Detect poses in frame or within specific boxes."""
        if boxes is None:
            # Run on full frame
            try:
                results = self.model(frame, verbose=False, device=self.device)
                return self._process_results(results, frame.shape[:2])
            except Exception as e:
                print(f"Pose detection error on full frame: {e}")
                return []
        else:
            # Run on cropped regions
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


class EnhancedObjectDetector(ObjectDetector):
    """Object detector with integrated pose estimation."""
    
    def __init__(self, model_id, api_key, confidence_threshold=0.5, 
                 enable_pose=True, pose_model='yolov8m-pose.pt', device='cpu'):
        super().__init__(model_id, api_key, confidence_threshold)
        
        self.enable_pose = enable_pose
        if self.enable_pose:
            try:
                self.pose_detector = PoseDetector(pose_model, device)
            except Exception as e:
                print(f"Failed to initialize pose detector: {e}")
                self.enable_pose = False
    
    def detect_with_pose(self, frame):
        """Detect objects and estimate poses for players."""
        # Get standard detections
        detections = self.detect_categories(frame)
        
        if not self.enable_pose:
            return detections, None
        
        # Combine players and goalkeepers for pose estimation
        all_players = []
        player_indices = []
        
        # Add players
        if len(detections['players']) > 0:
            all_players.extend(detections['players'].xyxy)
            player_indices.extend(['player'] * len(detections['players']))
        
        # Add goalkeepers
        if len(detections['goalkeepers']) > 0:
            all_players.extend(detections['goalkeepers'].xyxy)
            player_indices.extend(['goalkeeper'] * len(detections['goalkeepers']))
        
        # Estimate poses
        poses = {
            'players': [],
            'goalkeepers': []
        }
        
        if all_players:
            try:
                pose_results = self.pose_detector.detect_poses(frame, np.array(all_players))
                
                # Organize poses by player type
                player_idx = 0
                goalkeeper_idx = 0
                
                for i, (pose, player_type) in enumerate(zip(pose_results, player_indices)):
                    if player_type == 'player':
                        poses['players'].append(pose)
                        player_idx += 1
                    else:
                        poses['goalkeepers'].append(pose)
                        goalkeeper_idx += 1
                        
            except Exception as e:
                print(f"Error during pose estimation: {e}")
                # Return empty poses on error
                poses['players'] = [None] * len(detections['players'])
                poses['goalkeepers'] = [None] * len(detections['goalkeepers'])
        
        return detections, poses


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