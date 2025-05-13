#!/usr/bin/env python3
"""
Football AI Demo - Final Optimized Headless version
Works around ONNX CUDA issues and uses proper video encoding
"""

import os
import argparse
import yaml
import cv2
import numpy as np
import supervision as sv
from collections import deque
from tqdm import tqdm
import torch
from transformers import AutoProcessor, SiglipVisionModel
from inference import get_model
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram,
    draw_paths_on_pitch
)
import pickle
import hashlib


class FootballAI:
    def __init__(self, config_path):
        """Initialize the Football AI system with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set environment variables
        os.environ["HF_TOKEN"] = self.config['api_keys']['huggingface_token']
        os.environ["ROBOFLOW_API_KEY"] = self.config['api_keys']['roboflow_api_key']
        
        # Disable ONNX CUDA to avoid errors
        os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CPUExecutionProvider]"
        
        # Check GPU availability
        self.use_gpu = torch.cuda.is_available() and self.config['performance']['use_gpu']
        if self.use_gpu:
            self.device = torch.device('cuda:0')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
        # Initialize all components
        self.init_models()
        self.init_annotators()
        self.init_tracking()
        self.pitch_config = SoccerPitchConfiguration()
        self.team_classifier = None
        self.ball_trail = []
        self.transformation_matrices = deque(maxlen=self.config['ball_tracking']['avg_frames'])
        self.frame_width = None
        self.frame_height = None
        
        # Optimization
        self.team_assignment_cache = {}
        self.team_assignment_interval = 30
        self.last_team_update_frame = 0
        self.blank_pitch = draw_pitch(self.pitch_config)
    
    def init_models(self):
        """Initialize all required models."""
        print("Loading models...")
        
        # Roboflow models (will use their own inference backend)
        self.player_detection_model = get_model(
            model_id=self.config['models']['player_detection_model_id'],
            api_key=self.config['api_keys']['roboflow_api_key']
        )
        
        self.field_detection_model = get_model(
            model_id=self.config['models']['field_detection_model_id'],
            api_key=self.config['api_keys']['roboflow_api_key']
        )
        
        # SigLIP model on GPU if available
        hf_token = self.config['api_keys']['huggingface_token']
        self.siglip_model = SiglipVisionModel.from_pretrained(
            self.config['models']['siglip_model_path'],
            token=hf_token,  # Use 'token' instead of deprecated 'use_auth_token'
            torch_dtype=torch.float16 if self.use_gpu else torch.float32
        ).to(self.device)
        
        self.siglip_processor = AutoProcessor.from_pretrained(
            self.config['models']['siglip_model_path'],
            token=hf_token
        )
        
        self.siglip_model.eval()
        if self.use_gpu:
            torch.cuda.empty_cache()
    
    def init_annotators(self):
        """Initialize all annotators for visualization."""
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
    
    def init_tracking(self):
        """Initialize tracking system."""
        self.tracker = sv.ByteTrack()
        self.tracker.reset()
        self.BALL_ID = 0
        self.GOALKEEPER_ID = 1
        self.PLAYER_ID = 2
        self.REFEREE_ID = 3
    
    def get_video_hash(self, video_path):
        """Get a unique hash for the video file."""
        stats = os.stat(video_path)
        unique_string = f"{video_path}_{stats.st_size}_{stats.st_mtime}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def get_cache_path(self, video_path):
        """Get the cache file path for a video."""
        video_hash = self.get_video_hash(video_path)
        cache_dir = self.config['caching'].get('cache_directory', '')
        if not cache_dir:
            cache_dir = os.path.dirname(video_path)
        os.makedirs(cache_dir, exist_ok=True)
        cache_filename = f".team_classifier_cache_{video_hash}.pkl"
        return os.path.join(cache_dir, cache_filename)
    
    def load_cached_classifier(self, video_path):
        """Load a cached team classifier if it exists."""
        cache_path = self.get_cache_path(video_path)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                classifier = cached_data['classifier']
                print(f"Loaded cached team classifier from {cache_path}")
                classifier.device = self.device
                return classifier
            except Exception as e:
                print(f"Failed to load cache: {e}")
                return None
        return None
    
    def save_cached_classifier(self, video_path, classifier):
        """Save the team classifier to cache."""
        cache_path = self.get_cache_path(video_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'classifier': classifier,
                    'video_path': video_path,
                    'timestamp': os.path.getmtime(video_path)
                }, f)
            print(f"Saved team classifier cache to {cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def train_team_classifier(self, video_path):
        """Train the team classifier on the video or load from cache."""
        if self.config['caching'].get('enable_team_classifier_cache', True):
            cached_classifier = self.load_cached_classifier(video_path)
            if cached_classifier is not None:
                self.team_classifier = cached_classifier
                return
        
        print("Training team classifier...")
        frame_generator = sv.get_video_frames_generator(
            source_path=video_path,
            stride=self.config['video']['stride']
        )
        
        crops = []
        for frame in tqdm(frame_generator, desc='Collecting crops'):
            result = self.player_detection_model.infer(
                frame,
                confidence=self.config['detection']['confidence_threshold']
            )[0]
            detections = sv.Detections.from_inference(result)
            players_detections = detections[detections.class_id == self.PLAYER_ID]
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            crops.extend(players_crops)
        
        self.team_classifier = TeamClassifier(device=self.device.type)
        self.team_classifier.fit(crops)
        print("Team classifier training complete!")
        
        if self.config['caching'].get('enable_team_classifier_cache', True):
            self.save_cached_classifier(video_path, self.team_classifier)
    
    def resolve_goalkeepers_team_id(self, players, goalkeepers):
        """Assign goalkeepers to teams based on proximity to team centroids."""
        if len(goalkeepers) == 0:
            return np.array([], dtype=int)
        
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        
        team_0_players = players_xy[players.class_id == 0] if len(players_xy) > 0 else np.array([])
        team_1_players = players_xy[players.class_id == 1] if len(players_xy) > 0 else np.array([])
        
        if len(team_0_players) == 0 or len(team_1_players) == 0:
            return np.zeros(len(goalkeepers), dtype=int)
        
        team_0_centroid = team_0_players.mean(axis=0)
        team_1_centroid = team_1_players.mean(axis=0)
        
        goalkeepers_team_id = []
        for goalkeeper_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
            dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
            goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
        
        return np.array(goalkeepers_team_id, dtype=int)
    
    def batch_process_team_assignments(self, frame, players_detections, tracker_ids):
        """Process team assignments in batch for efficiency."""
        if len(players_detections) == 0:
            return np.array([], dtype=int)
        
        # Batch crop all players
        crops = []
        valid_indices = []
        
        for i, (xyxy, tracker_id) in enumerate(zip(players_detections.xyxy, tracker_ids)):
            cache_key = f"tracker_{tracker_id}"
            
            # Check cache first
            if cache_key in self.team_assignment_cache:
                continue
                
            crop = sv.crop_image(frame, xyxy)
            crops.append(crop)
            valid_indices.append(i)
        
        # Process batch if we have new crops
        if crops:
            try:
                batch_assignments = self.team_classifier.predict(crops)
                
                # Update cache
                for idx, assignment in zip(valid_indices, batch_assignments):
                    tracker_id = tracker_ids[idx]
                    cache_key = f"tracker_{tracker_id}"
                    self.team_assignment_cache[cache_key] = assignment
            except Exception as e:
                print(f"Batch assignment error: {e}")
        
        # Build final assignments from cache
        assignments = np.zeros(len(players_detections), dtype=int)
        for i, tracker_id in enumerate(tracker_ids):
            cache_key = f"tracker_{tracker_id}"
            assignments[i] = self.team_assignment_cache.get(cache_key, 0)
        
        return assignments
    
    def process_frame(self, frame, frame_count):
        """Process a single frame and return annotated frames."""
        if self.frame_width is None:
            self.frame_height, self.frame_width = frame.shape[:2]
        
        # Player detection
        result = self.player_detection_model.infer(
            frame,
            confidence=self.config['detection']['confidence_threshold']
        )[0]
        detections = sv.Detections.from_inference(result)
        
        # Separate ball detections
        ball_detections = detections[detections.class_id == self.BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
        
        # Process other detections
        all_detections = detections[detections.class_id != self.BALL_ID]
        all_detections = all_detections.with_nms(
            threshold=self.config['detection']['nms_threshold'],
            class_agnostic=True
        )
        all_detections = self.tracker.update_with_detections(detections=all_detections)
        
        # Separate by type
        goalkeepers_detections = all_detections[all_detections.class_id == self.GOALKEEPER_ID]
        players_detections = all_detections[all_detections.class_id == self.PLAYER_ID]
        referees_detections = all_detections[all_detections.class_id == self.REFEREE_ID]
        
        # Team assignment (optimized)
        if len(players_detections) > 0 and self.team_classifier is not None:
            # Update teams periodically
            should_update = (frame_count - self.last_team_update_frame) >= self.team_assignment_interval
            
            if should_update:
                self.last_team_update_frame = frame_count
                # Clear old cache entries
                current_trackers = set(players_detections.tracker_id)
                self.team_assignment_cache = {
                    k: v for k, v in self.team_assignment_cache.items()
                    if int(k.split('_')[1]) in current_trackers
                }
            
            # Batch process assignments
            players_detections.class_id = self.batch_process_team_assignments(
                frame, players_detections, players_detections.tracker_id
            )
            
            # Handle goalkeepers
            if len(goalkeepers_detections) > 0:
                goalkeepers_detections.class_id = self.resolve_goalkeepers_team_id(
                    players_detections, goalkeepers_detections
                )
        else:
            if len(players_detections) > 0:
                players_detections.class_id = np.zeros(len(players_detections), dtype=int)
            if len(goalkeepers_detections) > 0:
                goalkeepers_detections.class_id = np.zeros(len(goalkeepers_detections), dtype=int)
        
        if len(referees_detections) > 0:
            referees_detections.class_id = np.full(len(referees_detections), 2, dtype=int)
        
        # Merge detections
        detections_to_merge = []
        if len(players_detections) > 0:
            detections_to_merge.append(players_detections)
        if len(goalkeepers_detections) > 0:
            detections_to_merge.append(goalkeepers_detections)
        if len(referees_detections) > 0:
            detections_to_merge.append(referees_detections)
        
        if detections_to_merge:
            all_detections = sv.Detections.merge(detections_to_merge)
        else:
            all_detections = sv.Detections.empty()
        
        # Field detection
        field_result = self.field_detection_model.infer(
            frame,
            confidence=self.config['detection']['confidence_threshold']
        )[0]
        key_points = sv.KeyPoints.from_inference(field_result)
        
        # Filter keypoints
        filter_mask = key_points.confidence[0] > self.config['detection']['keypoint_confidence_threshold']
        frame_reference_points = key_points.xy[0][filter_mask]
        pitch_reference_points = np.array(self.pitch_config.vertices)[filter_mask]
        
        # Create transformation
        if len(frame_reference_points) >= 4:
            transformer = ViewTransformer(
                source=frame_reference_points,
                target=pitch_reference_points
            )
            self.transformation_matrices.append(transformer.m)
            if len(self.transformation_matrices) > 0:
                transformer.m = np.mean(np.array(self.transformation_matrices), axis=0)
        else:
            transformer = None
        
        # Create annotated frames
        annotated_frame = self.annotate_frame(frame, all_detections, ball_detections)
        pitch_view = self.create_pitch_view(
            players_detections, referees_detections, ball_detections, transformer
        )
        
        return annotated_frame, pitch_view
    
    def annotate_frame(self, frame, all_detections, ball_detections):
        """Annotate the original frame with detections."""
        annotated_frame = frame.copy()
        
        annotated_frame = self.ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections
        )
        
        if self.config['display']['show_tracking_ids']:
            labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=all_detections,
                labels=labels
            )
        
        if self.config['display']['show_ball']:
            annotated_frame = self.triangle_annotator.annotate(
                scene=annotated_frame,
                detections=ball_detections
            )
        
        return annotated_frame
    
    def create_pitch_view(self, players_detections, referees_detections, ball_detections, transformer):
        """Create the top-down pitch view."""
        pitch_view = self.blank_pitch.copy()
        
        if transformer is None:
            return pitch_view
        
        # Transform coordinates
        if len(ball_detections) > 0:
            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)
            
            if self.config['ball_tracking']['enable'] and len(pitch_ball_xy) > 0:
                self.ball_trail.append(pitch_ball_xy.flatten())
                if len(self.ball_trail) > self.config['ball_tracking']['trail_length']:
                    self.ball_trail.pop(0)
        else:
            pitch_ball_xy = np.array([])
        
        # Transform player coordinates
        players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_players_xy = transformer.transform_points(points=players_xy) if len(players_xy) > 0 else np.array([])
        
        referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_referees_xy = transformer.transform_points(points=referees_xy) if len(referees_xy) > 0 else np.array([])
        
        # Draw ball trail
        if self.config['ball_tracking']['enable'] and len(self.ball_trail) > 1:
            try:
                valid_trail = [pos for pos in self.ball_trail if pos.shape[0] >= 2]
                
                if len(valid_trail) > 1:
                    pitch_view = draw_paths_on_pitch(
                        config=self.pitch_config,
                        paths=[valid_trail],
                        color=sv.Color.WHITE,
                        pitch=pitch_view
                    )
            except Exception as e:
                pass
        
        # Draw players
        if len(pitch_players_xy) > 0:
            team_1_mask = players_detections.class_id == 0
            if np.any(team_1_mask):
                pitch_view = draw_points_on_pitch(
                    config=self.pitch_config,
                    xy=pitch_players_xy[team_1_mask],
                    face_color=sv.Color.from_hex(self.config['display']['team_colors']['team_1']),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=pitch_view
                )
            
            team_2_mask = players_detections.class_id == 1
            if np.any(team_2_mask):
                pitch_view = draw_points_on_pitch(
                    config=self.pitch_config,
                    xy=pitch_players_xy[team_2_mask],
                    face_color=sv.Color.from_hex(self.config['display']['team_colors']['team_2']),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=pitch_view
                )
        
        # Draw referees
        if len(pitch_referees_xy) > 0:
            pitch_view = draw_points_on_pitch(
                config=self.pitch_config,
                xy=pitch_referees_xy,
                face_color=sv.Color.from_hex(self.config['display']['referee_color']),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=pitch_view
            )
        
        # Draw ball
        if len(pitch_ball_xy) > 0:
            pitch_view = draw_points_on_pitch(
                config=self.pitch_config,
                xy=pitch_ball_xy,
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK,
                radius=10,
                pitch=pitch_view
            )
        
        return pitch_view
    
    def run(self, output_path=None):
        """Main processing loop - saves output to video file."""
        try:
            video_path = self.config['video']['input_path']
            
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            print(f"Starting with video: {video_path}")
            if self.use_gpu:
                print(f"GPU Memory before processing: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
            
            # Train team classifier
            self.train_team_classifier(video_path)
            
            # Open video
            print(f"Opening video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print("ERROR: Failed to open video!")
                return
            
            # Get video info
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Video info: {total_frames} frames, {fps} FPS, {width}x{height}")
            
            # Set output path
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(
                    os.path.dirname(video_path),
                    f"{base_name}_football_ai_output.mp4"
                )
            
            # Calculate output dimensions
            target_height = 600
            original_width = int(width * target_height / height)
            pitch_width = 800
            out_width = original_width + pitch_width
            out_height = target_height
            
            # Create video writer with compatible codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
            
            if not out.isOpened():
                print("Failed to open video writer, trying alternative codec...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_path = output_path.replace('.mp4', '.avi')
                out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
            
            print(f"Saving output to: {output_path}")
            print("Processing video (press Ctrl+C to stop)...")
            
            # Process video with progress bar
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                frame_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    annotated_frame, pitch_view = self.process_frame(frame, frame_count)
                    
                    # Resize frames
                    annotated_frame_resized = cv2.resize(annotated_frame, (original_width, target_height))
                    pitch_view_resized = cv2.resize(pitch_view, (pitch_width, target_height))
                    
                    # Combine frames
                    combined_frame = np.hstack([annotated_frame_resized, pitch_view_resized])
                    
                    # Write frame
                    out.write(combined_frame)
                    
                    frame_count += 1
                    pbar.update(1)
                    
                    # Periodic updates
                    if frame_count % 100 == 0:
                        if self.use_gpu:
                            gpu_memory = torch.cuda.memory_allocated() / 1024**3
                            pbar.set_postfix({"GPU Memory": f"{gpu_memory:.1f} GB"})
                            torch.cuda.empty_cache()
                        
                        # Save preview
                        if self.config['video'].get('save_previews', True):
                            preview_path = output_path.replace('.mp4', f'_preview_{frame_count}.jpg').replace('.avi', f'_preview_{frame_count}.jpg')
                            cv2.imwrite(preview_path, combined_frame)
            
            # Save final preview
            if frame_count > 0:
                preview_path = output_path.replace('.mp4', '_preview_final.jpg').replace('.avi', '_preview_final.jpg')
                cv2.imwrite(preview_path, combined_frame)
                print(f"Preview image saved to: {preview_path}")
            
            # Cleanup
            print("Finalizing video...")
            cap.release()
            out.release()
            
            if self.use_gpu:
                torch.cuda.empty_cache()
                print(f"Final GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
            
            print(f"Processing complete! Output saved to: {output_path}")
            
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            if self.use_gpu:
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"ERROR during processing: {e}")
            import traceback
            traceback.print_exc()
            
            # Ensure cleanup
            try:
                if 'cap' in locals():
                    cap.release()
                if 'out' in locals():
                    out.release()
                if self.use_gpu:
                    torch.cuda.empty_cache()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description='Football AI Demo (Final Optimized)')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                      help='Output video path (optional)')
    args = parser.parse_args()
    
    football_ai = FootballAI(args.config)
    football_ai.run(args.output)


if __name__ == "__main__":
    main()
