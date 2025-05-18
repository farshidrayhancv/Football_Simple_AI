"""Modified frame processor that saves annotations as JSON."""

import cv2
import numpy as np
import supervision as sv
from typing import Dict, Any, Optional, Tuple, List
import os
import json

# Import original frame processor
from processing.frame_processor import FrameProcessor

# Import annotation utilities
from utils.annotation_utils import AnnotationStore

class JSONFrameProcessor(FrameProcessor):
    """Frame processor that saves annotations as JSON during processing."""
    
    def __init__(self, player_detector, field_detector, team_classifier, tracker, config, 
                 possession_detector=None, annotation_store=None):
        """
        Initialize frame processor with annotation store.
        
        Args:
            player_detector: Object detector
            field_detector: Field keypoint detector
            team_classifier: Team classifier
            tracker: Object tracker
            config: Configuration dictionary
            possession_detector: Player possession detector
            annotation_store: Annotation store for saving JSON annotations
        """
        super().__init__(player_detector, field_detector, team_classifier, tracker, config, possession_detector)
        self.annotation_store = annotation_store
    
    def process_frame_and_save(self, frame, frame_index):
        """
        Process a frame and save annotations.
        
        Args:
            frame: Video frame to process
            frame_index: Index of the frame in the video
            
        Returns:
            Tuple of processing results
        """
        # Process frame normally
        results = self.process_frame(frame)
        
        # Save annotations if annotation store is provided
        if self.annotation_store is not None:
            detections, transformer, poses, pose_stats, segmentations, seg_stats, sahi_stats, possession_result = results
            
            transformer_matrix = transformer.m if transformer is not None else None
            
            self.annotation_store.save_frame_annotations(
                frame_index=frame_index,
                detections=detections,
                poses=poses,
                segmentations=segmentations,
                transformer_matrix=transformer_matrix,
                possession_result=possession_result
            )
        
        return results


class VideoProcessorWithJSON:
    """Video processor that saves annotations as JSON during processing."""
    
    def __init__(self, config, annotation_dir="annotations"):
        """
        Initialize video processor with JSON annotation support.
        
        Args:
            config: Configuration dictionary
            annotation_dir: Directory to store annotations
        """
        self.config = config
        self.annotation_dir = annotation_dir
        self.processing_resolution = config.get('processing', {}).get('resolution', None)
        self.enable_possession = config.get('possession_detection', {}).get('enable', True)
        
        # Create annotation directory
        os.makedirs(annotation_dir, exist_ok=True)
    
    def process_video(self, video_path, output_path, player_detector, field_detector, 
                     team_classifier, tracker, possession_detector=None, 
                     start_frame=0, end_frame=None, stride=1):
        """
        Process video and save annotations as JSON.
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)
            player_detector: Object detector
            field_detector: Field keypoint detector
            team_classifier: Team classifier
            tracker: Object tracker
            possession_detector: Player possession detector
            start_frame: First frame to process
            end_frame: Last frame to process (None = process all)
            stride: Frame stride for processing
            
        Returns:
            Path to annotation directory
        """
        # Create annotation store
        annotation_store = AnnotationStore(video_path, self.annotation_dir)
        annotation_store.set_processing_status(True)
        
        # Create frame processor with annotation store
        frame_processor = JSONFrameProcessor(
            player_detector=player_detector,
            field_detector=field_detector,
            team_classifier=team_classifier,
            tracker=tracker,
            config=self.config,
            possession_detector=possession_detector,
            annotation_store=annotation_store
        )
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Adjust end frame if not specified
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
        
        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Create video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None
        
        # Process frames
        frame_count = start_frame
        from tqdm import tqdm
        
        with tqdm(total=(end_frame - start_frame) // stride, desc="Processing frames") as pbar:
            while frame_count < end_frame:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames according to stride
                if (frame_count - start_frame) % stride != 0:
                    frame_count += 1
                    continue
                
                # Process frame and save annotations
                results = frame_processor.process_frame_and_save(frame, frame_count)
                
                # Write to output video if provided
                if out is not None:
                    # Apply annotations to frame for visualization
                    from visualization.annotators import FootballAnnotator
                    annotator = FootballAnnotator(self.config, possession_detector)
                    
                    detections, transformer, poses, pose_stats, segmentations, seg_stats, sahi_stats, possession_result = results
                    
                    if self.config.get('display', {}).get('show_segmentation', True):
                        annotated = annotator.annotate_frame_with_pose_and_segmentation(
                            frame, detections, poses, segmentations
                        )
                    elif self.config.get('display', {}).get('show_pose', True):
                        annotated = annotator.annotate_frame_with_pose(
                            frame, detections, poses
                        )
                    else:
                        annotated = annotator.annotate_frame(frame, detections)
                    
                    # Add possession visualization if enabled
                    if self.enable_possession and possession_detector is not None:
                        annotated = possession_detector.highlight_possession(annotated, detections)
                    
                    out.write(annotated)
                
                frame_count += 1
                pbar.update(1)
        
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        
        # Update processing status
        annotation_store.set_processing_status(False)
        
        return annotation_store.annotations_dir