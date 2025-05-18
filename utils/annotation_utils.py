"""Utility for extracting and storing frame annotations in JSON format."""

import os
import json
import numpy as np
import cv2
import supervision as sv
from typing import Dict, List, Any, Optional, Tuple
import time

class AnnotationStore:
    """Class to store and retrieve annotations for video frames."""
    
    def __init__(self, video_path: str, output_dir: str = "annotations"):
        """
        Initialize annotation store.
        
        Args:
            video_path: Path to the video being processed
            output_dir: Directory to store annotation data
        """
        self.video_path = video_path
        self.video_basename = os.path.splitext(os.path.basename(video_path))[0]
        self.output_dir = output_dir
        self.annotations_dir = os.path.join(output_dir, self.video_basename)
        self.metadata_path = os.path.join(self.annotations_dir, "metadata.json")
        self.is_processing = False
        self.total_frames = 0
        self.processed_frames = 0
        
        # Create output directories
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        # Initialize metadata
        self._init_metadata()
    
    def _init_metadata(self):
        """Initialize or load metadata."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Get video properties
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {self.video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            self.metadata = {
                "video_path": self.video_path,
                "fps": fps,
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "processed_frames": 0,
                "is_processing": False,
                "created_at": time.time(),
                "features": {
                    "detection": True,
                    "pose": False,
                    "segmentation": False,
                    "possession": False
                }
            }
            self._save_metadata()
        
        self.total_frames = self.metadata["total_frames"]
        self.processed_frames = self.metadata["processed_frames"]
        self.is_processing = self.metadata["is_processing"]
    
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def update_metadata(self, **kwargs):
        """Update metadata with provided values."""
        for key, value in kwargs.items():
            if key in self.metadata:
                self.metadata[key] = value
            elif key.startswith("features.") and len(key.split(".")) == 2:
                feature_key = key.split(".")[1]
                if feature_key in self.metadata["features"]:
                    self.metadata["features"][feature_key] = value
        
        self._save_metadata()
    
    def _detections_to_dict(self, detections: Dict[str, sv.Detections]) -> Dict[str, List[Dict[str, Any]]]:
        """Convert supervision Detections to serializable dictionaries."""
        result = {}
        
        for category, detection in detections.items():
            if len(detection) == 0:
                result[category] = []
                continue
            
            category_result = []
            for i in range(len(detection)):
                box = detection.xyxy[i].tolist() if detection.xyxy is not None else None
                confidence = float(detection.confidence[i]) if detection.confidence is not None else None
                class_id = int(detection.class_id[i]) if detection.class_id is not None else None
                tracker_id = int(detection.tracker_id[i]) if detection.tracker_id is not None else None
                
                detection_dict = {
                    "box": box,
                    "confidence": confidence,
                    "class_id": class_id,
                    "tracker_id": tracker_id
                }
                category_result.append(detection_dict)
            
            result[category] = category_result
        
        return result
    
    def _poses_to_dict(self, poses: Optional[Dict[str, List[Dict[str, Any]]]]) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """Convert pose data to serializable dictionaries."""
        if poses is None:
            return None
        
        result = {}
        
        for category, pose_list in poses.items():
            category_result = []
            
            for pose in pose_list:
                if pose is None:
                    category_result.append(None)
                    continue
                
                pose_dict = {
                    "keypoints": pose["keypoints"].tolist() if isinstance(pose["keypoints"], np.ndarray) else None,
                    "confidence": pose["confidence"].tolist() if isinstance(pose["confidence"], np.ndarray) else None,
                    "bbox": pose["bbox"].tolist() if isinstance(pose["bbox"], np.ndarray) else None
                }
                category_result.append(pose_dict)
            
            result[category] = category_result
        
        return result
    
    def _segmentations_to_dict(self, segmentations: Optional[Dict[str, List[np.ndarray]]]) -> Optional[Dict[str, List[str]]]:
        """Convert segmentation masks to encoded strings (we don't store full masks as JSON)."""
        if segmentations is None:
            return None
        
        result = {}
        
        for category, mask_list in segmentations.items():
            category_result = []
            
            for mask in mask_list:
                if mask is None:
                    category_result.append(None)
                    continue
                
                # For JSON, we just store the shape and a count of non-zero elements (to save space)
                mask_info = {
                    "shape": mask.shape,
                    "count": int(np.sum(mask > 0.5))
                }
                category_result.append(mask_info)
            
            result[category] = category_result
        
        return result
    
    def save_frame_annotations(self, frame_index: int, detections: Dict[str, sv.Detections], 
                              poses: Optional[Dict[str, List[Dict[str, Any]]]] = None,
                              segmentations: Optional[Dict[str, List[np.ndarray]]] = None,
                              transformer_matrix: Optional[np.ndarray] = None,
                              possession_result: Optional[Dict[str, Any]] = None):
        """
        Save annotations for a single frame.
        
        Args:
            frame_index: Index of the frame
            detections: Dictionary of detections by category
            poses: Dictionary of pose data by category
            segmentations: Dictionary of segmentation masks by category
            transformer_matrix: Transformation matrix for pitch coordinates
            possession_result: Player possession detection result
        """
        # Convert detections to dictionary
        detections_dict = self._detections_to_dict(detections)
        
        # Convert poses to dictionary
        poses_dict = self._poses_to_dict(poses)
        
        # Convert segmentations to dictionary (storing just metadata, not full masks)
        segmentations_dict = self._segmentations_to_dict(segmentations)
        
        # Create frame annotation structure
        frame_annotation = {
            "frame_index": frame_index,
            "timestamp": frame_index / self.metadata["fps"],
            "detections": detections_dict,
            "poses": poses_dict,
            "segmentations": segmentations_dict,
            "transformer_matrix": transformer_matrix.tolist() if transformer_matrix is not None else None,
            "possession": possession_result
        }
        
        # Save to file
        frame_path = os.path.join(self.annotations_dir, f"frame_{frame_index:06d}.json")
        with open(frame_path, 'w') as f:
            json.dump(frame_annotation, f)
        
        # Update metadata
        self.processed_frames = max(self.processed_frames, frame_index + 1)
        self.update_metadata(
            processed_frames=self.processed_frames,
            features={
                "detection": True,
                "pose": poses is not None,
                "segmentation": segmentations is not None,
                "possession": possession_result is not None
            }
        )
    
    def get_frame_annotation(self, frame_index: int) -> Optional[Dict[str, Any]]:
        """Get annotation for a specific frame."""
        frame_path = os.path.join(self.annotations_dir, f"frame_{frame_index:06d}.json")
        if os.path.exists(frame_path):
            with open(frame_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_latest_frame_index(self) -> int:
        """Get index of latest processed frame."""
        return self.processed_frames - 1
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for this annotation store."""
        return self.metadata
    
    def set_processing_status(self, is_processing: bool):
        """Set processing status."""
        self.is_processing = is_processing
        self.update_metadata(is_processing=is_processing)
    
    def list_available_videos(cls) -> List[str]:
        """List all videos with available annotations."""
        annotations_dir = "annotations"
        if not os.path.exists(annotations_dir):
            return []
        
        return [d for d in os.listdir(annotations_dir) 
                if os.path.isdir(os.path.join(annotations_dir, d))]


def apply_annotations_to_frame(frame: np.ndarray, annotation: Dict[str, Any], 
                              config: Dict[str, Any]) -> np.ndarray:
    """
    Apply annotations to frame for visualization.
    
    Args:
        frame: Original video frame
        annotation: Annotation data for the frame
        config: Visualization configuration
    
    Returns:
        Annotated frame
    """
    if annotation is None:
        return frame
    
    annotated = frame.copy()
    
    # Convert detection dictionaries back to Supervision Detections
    detections = {}
    for category, detection_list in annotation["detections"].items():
        if not detection_list:
            detections[category] = sv.Detections.empty()
            continue
        
        boxes = []
        confidences = []
        class_ids = []
        tracker_ids = []
        
        for det in detection_list:
            boxes.append(det["box"])
            confidences.append(det["confidence"])
            class_ids.append(det["class_id"])
            tracker_ids.append(det["tracker_id"])
        
        detections[category] = sv.Detections(
            xyxy=np.array(boxes) if boxes else np.zeros((0, 4)),
            confidence=np.array(confidences) if confidences else None,
            class_id=np.array(class_ids) if class_ids else None,
            tracker_id=np.array(tracker_ids) if tracker_ids else None
        )
    
    # Ensure display config has all required keys
    if 'display' not in config:
        config['display'] = {}
    
    # Add default values for any missing display keys
    required_keys = {
        'show_tracking_ids': True,
        'show_ball': True,
        'show_voronoi': False,
        'show_pose': config['display'].get('show_pose', True),
        'show_segmentation': config['display'].get('show_segmentation', True),
        'show_possession_detection': config['display'].get('show_possession', True),
        'segmentation_alpha': 0.6
    }
    
    for key, default_value in required_keys.items():
        if key not in config['display']:
            config['display'][key] = default_value
    
    # Apply standard detections (bounding boxes)
    from visualization.annotators import FootballAnnotator
    annotator = FootballAnnotator(config)
    annotated = annotator.annotate_frame(annotated, detections)
    
    # Apply poses if available
    if annotation.get("poses") and config.get("display", {}).get("show_pose", True):
        from models.detector import PoseDetector
        pose_drawer = PoseDetector()
        
        # Team colors (in BGR format)
        team1_color = _hex_to_bgr(config["display"]["team_colors"]["team_1"])
        team2_color = _hex_to_bgr(config["display"]["team_colors"]["team_2"])
        referee_color = _hex_to_bgr(config["display"]["referee_color"])
        
        for category in ["players", "goalkeepers", "referees"]:
            if category not in annotation["poses"]:
                continue
                
            for i, pose_data in enumerate(annotation["poses"][category]):
                if pose_data is None:
                    continue
                
                # Convert pose data
                pose = {
                    "keypoints": np.array(pose_data["keypoints"]),
                    "confidence": np.array(pose_data["confidence"]),
                    "bbox": np.array(pose_data["bbox"]) if pose_data["bbox"] else None
                }
                
                # Determine color
                if category == "referees":
                    color = referee_color
                else:
                    # Get team from detections
                    team_id = detections[category].class_id[i] if i < len(detections[category]) else 0
                    color = team1_color if team_id == 0 else team2_color
                
                annotated = pose_drawer.draw_pose(annotated, pose, color)
    
    # Highlight possession if available
    if annotation.get("possession") and annotation["possession"].get("player_id") is not None and config["display"].get("show_possession", True):
        from models.player_possession_detector import PlayerPossessionDetector
        possession_detector = PlayerPossessionDetector()
        possession_detector.current_possession = annotation["possession"]["player_id"]
        possession_detector.current_team = annotation["possession"]["team_id"]
        annotated = possession_detector.highlight_possession(annotated, detections)
    
    return annotated


def _hex_to_bgr(hex_color):
    """Convert hex color to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    # Get RGB values
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # Return BGR for OpenCV
    return (b, g, r)