"""Video processing utilities with pose, segmentation, and SAHI support."""

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
    
    def process_video_with_pose_and_segmentation(self, video_path, output_path, frame_processor, 
                                                annotator, pitch_renderer, tracker):
        """Process video with pose estimation, segmentation, and SAHI support."""
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
            output_suffix = "_football_ai"
            if self.config.get('sahi', {}).get('enable', False):
                output_suffix += "_sahi"
            if self.config.get('display', {}).get('show_pose', False):
                output_suffix += "_pose"
            if self.config.get('display', {}).get('show_segmentation', False):
                output_suffix += "_seg"
            output_path = f"{base_name}{output_suffix}.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_width = width + 800
        out_height = max(height, 600)
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        # Processing description
        features = []
        if self.config.get('sahi', {}).get('enable', False):
            features.append("SAHI")
        if self.config.get('display', {}).get('show_pose', False):
            features.append("pose estimation")
        if self.config.get('display', {}).get('show_segmentation', False):
            features.append("segmentation")
        
        features_str = " with " + ", ".join(features) if features else ""
        print(f"Processing {total_frames} frames{features_str}...")
        
        # Process frames
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with all features
                results = frame_processor.process_frame(frame)
                
                # Unpack results (now includes SAHI stats)
                if len(results) == 7:
                    detections, transformer, poses, pose_stats, segmentations, seg_stats, sahi_stats = results
                else:
                    # Fallback for older versions
                    detections, transformer, poses, pose_stats, segmentations, seg_stats = results[:6]
                    sahi_stats = None
                
                # Create visualizations
                if segmentations:
                    annotated_frame = annotator.annotate_frame_with_pose_and_segmentation(
                        frame, detections, poses, segmentations
                    )
                elif poses:
                    annotated_frame = annotator.annotate_frame_with_pose(frame, detections, poses)
                else:
                    annotated_frame = annotator.annotate_frame(frame, detections)
                
                # Add statistics to frame
                stats = {
                    'Frame': frame_count,
                    'Players': len(detections['players']),
                    'Goalkeepers': len(detections['goalkeepers'])
                }
                
                # Add SAHI info if enabled
                if sahi_stats:
                    annotated_frame = self._draw_sahi_info(annotated_frame, sahi_stats)
                
                # Draw all statistics
                annotated_frame = annotator.draw_stats_with_pose_and_segmentation(
                    annotated_frame, stats, pose_stats, seg_stats
                )
                
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
    
    def _draw_sahi_info(self, frame, sahi_stats):
        """Draw SAHI information on frame."""
        if not sahi_stats:
            return frame
        
        # Draw SAHI info in top right corner
        y_offset = 30
        x_offset = frame.shape[1] - 200
        
        cv2.putText(frame, "SAHI Enabled", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(frame, f"Slices: {sahi_stats['slices']}", (x_offset, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def process_video_with_pose(self, video_path, output_path, frame_processor, 
                               annotator, pitch_renderer, tracker):
        """Backward compatible method."""
        return self.process_video_with_pose_and_segmentation(
            video_path, output_path, frame_processor, 
            annotator, pitch_renderer, tracker
        )
    
    def process_video(self, video_path, output_path, frame_processor, 
                     annotator, pitch_renderer, tracker):
        """Backward compatible method."""
        return self.process_video_with_pose_and_segmentation(
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
        label = "Original"
        if self.config.get('sahi', {}).get('enable', False):
            label += " + SAHI"
        if self.config.get('display', {}).get('show_pose', False):
            label += " + Pose"
        if self.config.get('display', {}).get('show_segmentation', False):
            label += " + Segmentation"
        
        cv2.putText(combined, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Tactical View", (width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined