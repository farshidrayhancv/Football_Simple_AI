#!/usr/bin/env python3
"""
Process video and store complete annotations for later viewing
"""

import os
import sys
import argparse
import cv2
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm

# Import required components
from config.config_loader import ConfigLoader
from models.detector import EnhancedObjectDetector, FieldDetector
from models.classifier import TeamClassifierModule
from models.tracker import ObjectTracker
from models.player_possession_detector import PlayerPossessionDetector
from processing.frame_processor import FrameProcessor
from utils.annotation_utils import AnnotationStore
from utils.video_utils import VideoProcessor

class VideoAnnotationProcessor:
    """Process video and create rich annotations for later viewing."""
    
    def __init__(self, config_path):
        """Initialize with configuration."""
        # Load configuration
        self.config = ConfigLoader(config_path).config
        print(f"Loaded configuration from {config_path}")
        
        # Initialize models
        print("Initializing models...")
        self.player_detector = EnhancedObjectDetector(
            model_id=self.config['models']['player_detection_model_id'],
            api_key=self.config['api_keys']['roboflow_api_key'],
            confidence_threshold=self.config['detection']['confidence_threshold'],
            enable_pose=self.config.get('display', {}).get('show_pose', True),
            enable_segmentation=self.config.get('display', {}).get('show_segmentation', True),
            device=self.config['performance']['device']
        )
        
        self.field_detector = FieldDetector(
            model_id=self.config['models']['field_detection_model_id'],
            api_key=self.config['api_keys']['roboflow_api_key']
        )
        
        self.team_classifier = TeamClassifierModule(
            device=self.config['performance']['device'],
            hf_token=self.config['api_keys']['huggingface_token'],
            model_path=self.config['models']['siglip_model_path']
        )
        
        self.tracker = ObjectTracker()
        
        # Initialize possession detector if enabled
        self.possession_detector = None
        if self.config.get('possession_detection', {}).get('enable', True):
            self.possession_detector = PlayerPossessionDetector(
                proximity_threshold=self.config.get('possession_detection', {}).get('proximity_threshold', 250),
                frame_proximity_threshold=self.config.get('possession_detection', {}).get('frame_proximity_threshold', 30),
                coordinate_system=self.config.get('possession_detection', {}).get('coordinate_system', 'pitch'),
                possession_frames=self.config.get('possession_detection', {}).get('possession_frames', 3),
                possession_duration=self.config.get('possession_detection', {}).get('possession_duration', 3),
                no_possession_frames=self.config.get('possession_detection', {}).get('no_possession_frames', 10)
            )
        
        # Initialize frame processor
        self.frame_processor = FrameProcessor(
            player_detector=self.player_detector,
            field_detector=self.field_detector,
            team_classifier=self.team_classifier,
            tracker=self.tracker,
            config=self.config,
            possession_detector=self.possession_detector
        )
        
        self.video_utils = VideoProcessor(self.config)
    
    def train_team_classifier(self, video_path):
        """Train team classifier on video."""
        print("Training team classifier...")
        crops = self.video_utils.collect_player_crops(
            video_path, 
            self.player_detector,
            self.config['video']['stride']
        )
        
        if crops:
            self.team_classifier.train(crops)
            print(f"Team classifier trained on {len(crops)} player crops")
        else:
            print("Warning: No player crops found for team classification")
    
    def process_video(self, video_path, output_dir="annotations", start_frame=0, end_frame=None, stride=1):
        """Process video and save annotations."""
        # Create annotation store
        annotation_store = AnnotationStore(video_path, output_dir)
        annotation_store.set_processing_status(True)
        
        # Train team classifier
        self.train_team_classifier(video_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
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
        
        # Process frames
        frame_count = start_frame
        
        # Create previews directory
        previews_dir = os.path.join(output_dir, "previews")
        os.makedirs(previews_dir, exist_ok=True)
        
        # Keep track of field keypoints and transformation
        field_keypoints = None  # Will store keypoints
        pitch_transformer = None  # Will store transformer
        
        with tqdm(total=(end_frame-start_frame)//stride, desc="Processing frames") as pbar:
            while frame_count < end_frame:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames according to stride
                if (frame_count - start_frame) % stride != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                try:
                    # If we don't have field keypoints yet, detect them first
                    if field_keypoints is None:
                        # Detect field keypoints
                        field_keypoints = self.field_detector.detect_keypoints(frame)
                    
                    # Process frame
                    results = self.frame_processor.process_frame(frame)
                    
                    # Unpack results
                    detections, transformer, poses, pose_stats, segmentations, seg_stats, sahi_stats, possession_result = results
                    
                    # Store transformer for future frames if needed
                    if transformer is not None:
                        pitch_transformer = transformer
                    
                    # Save complete annotation data
                    transformer_matrix = transformer.m if transformer is not None else None
                    
                    # If transformer is None but we have a previous one, use that
                    if transformer_matrix is None and pitch_transformer is not None:
                        transformer_matrix = pitch_transformer.m
                    
                    # Store field keypoints if detected
                    field_keypoints_data = None
                    if field_keypoints is not None:
                        field_keypoints_data = {
                            'xy': field_keypoints.xy[0].tolist() if len(field_keypoints.xy) > 0 else [],
                            'confidence': field_keypoints.confidence[0].tolist() if len(field_keypoints.confidence) > 0 else []
                        }
                    
                    # Save frame annotations with additional data
                    annotation_store.save_frame_annotations(
                        frame_index=frame_count,
                        detections=detections,
                        poses=poses,
                        segmentations=segmentations,
                        transformer_matrix=transformer_matrix,
                        field_keypoints=field_keypoints_data,
                        possession_result=possession_result
                    )
                    
                    # Save preview image every 100 frames
                    if frame_count % 100 == 0:
                        from visualization.annotators import FootballAnnotator
                        annotator = FootballAnnotator(self.config)
                        
                        if segmentations:
                            annotated = annotator.annotate_frame_with_pose_and_segmentation(
                                frame, detections, poses, segmentations
                            )
                        elif poses:
                            annotated = annotator.annotate_frame_with_pose(
                                frame, detections, poses
                            )
                        else:
                            annotated = annotator.annotate_frame(frame, detections)
                        
                        # Add possession highlighting if available
                        if possession_result and self.possession_detector:
                            self.possession_detector.current_possession = possession_result.get('player_id')
                            self.possession_detector.current_team = possession_result.get('team_id')
                            annotated = self.possession_detector.highlight_possession(annotated, detections)
                        
                        preview_path = os.path.join(previews_dir, f"frame_{frame_count:06d}.jpg")
                        cv2.imwrite(preview_path, annotated)
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
                
                frame_count += 1
                pbar.update(1)
        
        # Clean up
        cap.release()
        
        # Set processing status to complete
        annotation_store.set_processing_status(False)
        
        print(f"Processing complete! Annotations stored in: {os.path.join(output_dir, os.path.basename(video_path).split('.')[0])}")

def main():
    parser = argparse.ArgumentParser(description='Process football video and store detailed annotations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default='annotations', help='Output directory for annotations')
    parser.add_argument('--start', type=int, default=0, help='Start frame')
    parser.add_argument('--end', type=int, default=None, help='End frame (None for all)')
    parser.add_argument('--stride', type=int, default=1, help='Frame stride')
    
    args = parser.parse_args()
    
    processor = VideoAnnotationProcessor(args.config)
    processor.process_video(args.video, args.output, args.start, args.end, args.stride)

if __name__ == "__main__":
    main()