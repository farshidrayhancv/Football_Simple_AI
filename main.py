#!/usr/bin/env python3
"""
Main Football AI Application with Pose Estimation and Segmentation
"""

import argparse
import os
import sys

from config.config_loader import ConfigLoader
from models.detector import EnhancedObjectDetector, FieldDetector
from models.classifier import TeamClassifierModule
from models.tracker import ObjectTracker
from processing.frame_processor import FrameProcessor
from visualization.annotators import FootballAnnotator
from visualization.pitch_renderer import PitchRenderer
from caching.cache_manager import CacheManager
from utils.video_utils import VideoProcessor


class FootballAI:
    def __init__(self, config_path):
        """Initialize Football AI with configuration."""
        # Load configuration
        self.config = ConfigLoader(config_path).config
        
        # Initialize all components
        self._init_models()
        self._init_processors()
        self._init_visualization()
        self.cache_manager = CacheManager(self.config)
        self.video_processor = VideoProcessor(self.config)
    
    def _init_models(self):
        """Initialize all required models."""
        print("Loading models...")
        
        # Use enhanced detector with pose estimation and segmentation
        enable_pose = self.config.get('display', {}).get('show_pose', True)
        enable_segmentation = self.config.get('display', {}).get('show_segmentation', True)
        pose_model = self.config.get('models', {}).get('pose_model', 'yolov8m-pose.pt')
        sam_model = self.config.get('models', {}).get('sam_model', 'sam2.1_b.pt')
        
        self.player_detector = EnhancedObjectDetector(
            model_id=self.config['models']['player_detection_model_id'],
            api_key=self.config['api_keys']['roboflow_api_key'],
            confidence_threshold=self.config['detection']['confidence_threshold'],
            enable_pose=enable_pose,
            pose_model=pose_model,
            enable_segmentation=enable_segmentation,
            sam_model=sam_model,
            device=self.config['performance']['device']
        )
        
        self.field_detector = FieldDetector(
            model_id=self.config['models']['field_detection_model_id'],
            api_key=self.config['api_keys']['roboflow_api_key']
        )
        
        # Pass model path from config
        self.team_classifier = TeamClassifierModule(
            device=self.config['performance']['device'],
            hf_token=self.config['api_keys']['huggingface_token'],
            model_path=self.config['models']['siglip_model_path']
        )
        
        self.tracker = ObjectTracker()
    
    def _init_processors(self):
        """Initialize processors."""
        self.frame_processor = FrameProcessor(
            player_detector=self.player_detector,
            field_detector=self.field_detector,
            team_classifier=self.team_classifier,
            tracker=self.tracker,
            config=self.config
        )
    
    def _init_visualization(self):
        """Initialize visualization components."""
        self.annotator = FootballAnnotator(self.config)
        self.pitch_renderer = PitchRenderer(self.config)
    
    def train_team_classifier(self, video_path):
        """Train the team classifier on the video."""
        print("Training team classifier...")
        
        # Try to load from cache first
        cached_classifier = self.cache_manager.load_team_classifier(video_path)
        if cached_classifier:
            self.team_classifier.classifier = cached_classifier
            return
        
        # Collect training data
        crops = self.video_processor.collect_player_crops(
            video_path, 
            self.player_detector,
            self.config['video']['stride']
        )
        
        # Train classifier
        if crops:
            self.team_classifier.train(crops)
            # Save to cache
            self.cache_manager.save_team_classifier(video_path, self.team_classifier.classifier)
    
    def run(self, output_path=None):
        """Main processing loop."""
        video_path = self.config['video']['input_path']
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"Processing video: {video_path}")
        
        # Train team classifier
        self.train_team_classifier(video_path)
        
        # Process video with pose estimation and segmentation
        self.video_processor.process_video_with_pose_and_segmentation(
            video_path=video_path,
            output_path=output_path,
            frame_processor=self.frame_processor,
            annotator=self.annotator,
            pitch_renderer=self.pitch_renderer,
            tracker=self.tracker
        )


def main():
    parser = argparse.ArgumentParser(description='Football AI Demo with Pose and Segmentation')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                      help='Output video path (optional)')
    args = parser.parse_args()
    
    try:
        football_ai = FootballAI(args.config)
        football_ai.run(args.output)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())