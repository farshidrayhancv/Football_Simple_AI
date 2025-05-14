"""Video processing utilities with pose estimation support."""

import cv2
import numpy as np
import supervision as sv
import os
from tqdm import tqdm


class VideoProcessor:
    def __init__(self, config):
        self.config = config
    
    def collect_player_crops(self, video_path, detector, stride):
        """Collect player crops from video for training."""
        cap = cv2.VideoCapture(video_path)
        
        crops = []
        frame_count = 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames//stride, desc='Collecting crops')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % stride == 0:
                players = detector.detect_players_only(frame)
                if len(players) > 0:
                    player_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
                    crops.extend(player_crops)
                pbar.update(1)
            
            frame_count += 1
        
        cap.release()
        pbar.close()
        
        print(f"Collected {len(crops)} player crops")
        return crops
    
    def process_video_with_pose(self, video_path, output_path, frame_processor, 
                               annotator, pitch_renderer, tracker):
        """Process video with pose estimation support."""
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set output path
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"{base_name}_football_ai_with_pose.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_width = width + 800
        out_height = max(height, 600)
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        print(f"Processing {total_frames} frames with pose estimation...")
        
        # Process frames
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with pose estimation
                detections, transformer, poses, pose_stats = frame_processor.process_frame(frame)
                
                # Create visualizations with pose
                if poses:
                    annotated_frame = annotator.annotate_frame_with_pose(frame, detections, poses)
                else:
                    annotated_frame = annotator.annotate_frame(frame, detections)
                
                # Add pose statistics to frame
                if pose_stats:
                    stats = {
                        'Frame': frame_count,
                        'FPS': 0 ## TODO int(1/frame_processor.player_detector.pose_detector.model.predictor.speed['inference'] * 1000) if hasattr(frame_processor.player_detector, 'pose_detector') else 'N/A'
                    }
                    annotated_frame = annotator.draw_stats_with_pose(annotated_frame, stats, pose_stats)
                
                
                # Render pitch view
                pitch_view = pitch_renderer.render(
                    detections, transformer, tracker.ball_trail
                )
                
                # Combine frames
                combined = self._create_combined_view(
                    annotated_frame, pitch_view, width, height
                )
                
                # Write frame
                out.write(combined)
                
                frame_count += 1
                pbar.update(1)
                
                # Save previews
                if frame_count % 500 == 0:
                    preview_path = output_path.replace('.mp4', f'_preview_{frame_count}.jpg')
                    cv2.imwrite(preview_path, combined)
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Processing complete! Output saved to: {output_path}")
    
    def process_video(self, video_path, output_path, frame_processor, 
                     annotator, pitch_renderer, tracker):
        """Process video (fallback for when pose is not needed)."""
        return self.process_video_with_pose(
            video_path, output_path, frame_processor, 
            annotator, pitch_renderer, tracker
        )
    
    def _create_combined_view(self, annotated_frame, pitch_view, width, height):
        """Create combined view of original and pitch."""
        # Resize frames
        annotated_frame = cv2.resize(annotated_frame, (width, height))
        pitch_view = cv2.resize(pitch_view, (800, 600))
        
        # Create combined frame
        out_height = max(height, 600)
        out_width = width + 800
        combined = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        
        # Place frames
        combined[:height, :width] = annotated_frame
        combined[:600, width:] = pitch_view
        
        # Add labels
        cv2.putText(combined, "Original + Detection + Pose", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Tactical View", (width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined