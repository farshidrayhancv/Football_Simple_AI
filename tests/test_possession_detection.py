"""Test script for possession detection."""

import cv2
import argparse
import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from models.detector import EnhancedObjectDetector
from models.tracker import ObjectTracker
from models.possession_detector import PossessionDetector
from visualization.annotators import FootballAnnotator
import supervision as sv


def test_possession_detection(config_path, video_path):
    """Test possession detection on a video."""
    # Load config
    config = ConfigLoader(config_path).config
    
    # Initialize models
    print("Loading object detector...")
    player_detector = EnhancedObjectDetector(
        model_id=config['models']['player_detection_model_id'],
        api_key=config['api_keys']['roboflow_api_key'],
        confidence_threshold=config['detection']['confidence_threshold'],
        device=config['performance']['device']
    )
    
    # Initialize tracker
    tracker = ObjectTracker()
    
    # Initialize possession detector
    possession_detector = PossessionDetector(
        proximity_threshold=config.get('possession_detection', {}).get('proximity_threshold', 50),
        possession_frames=config.get('possession_detection', {}).get('possession_frames', 5)
    )
    
    # Initialize annotator
    annotator = FootballAnnotator(config)
    
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
    
    print(f"Video: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = video_path.replace('.mp4', '_possession_detection.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video
    frame_count = 0
    
    # Create progress bar
    from tqdm import tqdm
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for faster processing during testing
            if frame_count % 5 != 0:
                frame_count += 1
                pbar.update(1)
                continue
            
            start_time = time.time()
            
            # Run detection
            detections = player_detector.detect_categories(frame)
            
            # Update tracker
            all_detections = []
            for category in ['players', 'goalkeepers', 'referees']:
                if len(detections[category]) > 0:
                    all_detections.append(detections[category])
            
            if all_detections:
                merged = sv.Detections.merge(all_detections)
                merged = tracker.update(merged)
                
                # Extract tracked detections back to categories
                for category in ['players', 'goalkeepers', 'referees']:
                    if category == 'players':
                        detections[category] = merged[merged.class_id == player_detector.PLAYER_ID]
                    elif category == 'goalkeepers': 
                        detections[category] = merged[merged.class_id == player_detector.GOALKEEPER_ID]
                    elif category == 'referees':
                        detections[category] = merged[merged.class_id == player_detector.REFEREE_ID]
            
            # Get ball position
            ball_position = None
            if len(detections['ball']) > 0:
                ball_xy = detections['ball'].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                if len(ball_xy) > 0:
                    ball_position = ball_xy[0]
            
            # Update possession detector
            possession_results = possession_detector.update(detections, ball_position, frame)
            
            # Draw visualizations
            # First annotate standard detections
            vis_frame = annotator.annotate_frame(frame.copy(), detections)
            
            # Highlight player with possession
            vis_frame = possession_detector.highlight_possession(vis_frame, detections)
            
            # Draw possession statistics
            vis_frame = possession_detector.draw_possession_stats(vis_frame)
            
            # Add processing info
            proc_time = time.time() - start_time
            cv2.putText(vis_frame, f"Frame: {frame_count} | Time: {proc_time:.3f}s", 
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            # Resize for display if needed
            display_frame = cv2.resize(vis_frame, (min(1280, width), min(720, height)))
            # cv2.imshow('Possession Detection', display_frame)
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q'):
                # break
            
            # Write to output video
            out.write(vis_frame)
            
            frame_count += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    
    print(f"Processing complete! Output saved to: {output_path}")
    
    # Print possession statistics
    print("\nPossession Statistics:")
    
    total_frames = sum(possession_detector.team_possession.values())
    if total_frames > 0:
        team1_percentage = (possession_detector.team_possession[0] / total_frames) * 100
        team2_percentage = (possession_detector.team_possession[1] / total_frames) * 100
    else:
        team1_percentage = team2_percentage = 0
    
    print(f"Team 1: {team1_percentage:.1f}%")
    print(f"Team 2: {team2_percentage:.1f}%")
    
    # Print top possession holders
    print("\nTop Possession Holders:")
    sorted_players = sorted(possession_detector.possession_stats.items(), 
                           key=lambda x: x[1]['frames'], reverse=True)[:5]
    
    for player_id, stats in sorted_players:
        team_id = stats['team']
        frames = stats['frames']
        percentage = (frames / total_frames) * 100 if total_frames > 0 else 0
        print(f"Player #{player_id} (Team {team_id+1}): {frames} frames ({percentage:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test possession detection')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--video', required=True, help='Path to video file')
    
    args = parser.parse_args()
    test_possession_detection(args.config, args.video)