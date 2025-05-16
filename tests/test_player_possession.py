"""Test script for player possession detection."""

import cv2
import argparse
import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from models.detector import EnhancedObjectDetector, FieldDetector
from models.tracker import ObjectTracker
from models.player_possession_detector import PlayerPossessionDetector
from sports.common.view import ViewTransformer
import supervision as sv


def test_player_possession(config_path, video_path):
    """Test player possession detection on a video."""
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
    
    # Initialize field detector for transformation
    field_detector = FieldDetector(
        model_id=config['models']['field_detection_model_id'],
        api_key=config['api_keys']['roboflow_api_key'],
        confidence_threshold=config['detection']['confidence_threshold']
    )
    
    # Initialize tracker
    tracker = ObjectTracker()
    
    # Initialize possession detector with configurable parameters
    proximity_threshold = config.get('possession_detection', {}).get('proximity_threshold', 50)
    possession_frames = config.get('possession_detection', {}).get('possession_frames', 3)
    
    print(f"Initializing possession detector with threshold={proximity_threshold}, frames={possession_frames}")
    possession_detector = PlayerPossessionDetector(
        proximity_threshold=proximity_threshold,
        possession_frames=possession_frames
    )
    
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
    output_path = video_path.replace('.mp4', '_player_possession.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video
    frame_count = 0
    processing_stride = 1  # Process every frame (adjust for speed if needed)
    
    # Create progress bar
    from tqdm import tqdm
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for faster processing during testing
            if frame_count % processing_stride != 0:
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
            
            # Detect field keypoints and create transformer
            keypoints = field_detector.detect_keypoints(frame)
            transformer = None
            
            if keypoints and len(keypoints.xy) > 0 and len(keypoints.xy[0]) >= 4:
                # Filter keypoints by confidence
                keypoint_threshold = config['detection']['keypoint_confidence_threshold']
                filter_mask = keypoints.confidence[0] > keypoint_threshold
                frame_reference_points = keypoints.xy[0][filter_mask]
                
                from sports.configs.soccer import SoccerPitchConfiguration
                pitch_config = SoccerPitchConfiguration()
                pitch_reference_points = np.array(pitch_config.vertices)[filter_mask]
                
                if len(frame_reference_points) >= 4:
                    transformer = ViewTransformer(
                        source=frame_reference_points,
                        target=pitch_reference_points
                    )
            
            # Get ball position and transform if possible
            ball_position = None
            pitch_ball_position = None
            
            if len(detections['ball']) > 0:
                ball_xy = detections['ball'].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                if len(ball_xy) > 0:
                    ball_position = ball_xy[0]  # Frame coordinates
                    
                    # Transform to pitch coordinates if possible
                    if transformer is not None:
                        pitch_ball_xy = transformer.transform_points(points=ball_xy)
                        if len(pitch_ball_xy) > 0:
                            pitch_ball_position = pitch_ball_xy[0]
            
            # Similarly transform player positions if possible
            transformed_detections = None
            if transformer is not None and pitch_ball_position is not None:
                transformed_detections = {}
                
                for category in ['players', 'goalkeepers', 'referees']:
                    if len(detections[category]) > 0:
                        # Get positions in frame coordinates
                        frame_positions = detections[category].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                        
                        # Transform to pitch coordinates
                        pitch_positions = transformer.transform_points(points=frame_positions)
                        
                        if len(pitch_positions) > 0:
                            # Create a copy of the detections
                            # transformed_detections[category] = detections[category].copy() // TODO: copy Doesnt work
                            transformed_detections[category] = detections[category]
                            
                            # Replace xyxy with transformed positions
                            # We'll create small boxes around the transformed positions
                            # This is just for the possession detector to find closest player
                            transformed_boxes = []
                            for pos in pitch_positions:
                                x, y = pos
                                transformed_boxes.append([x-5, y-5, x+5, y+5])
                            
                            transformed_detections[category].xyxy = np.array(transformed_boxes)
                        else:
                            transformed_detections[category] = detections[category]
                    else:
                        transformed_detections[category] = detections[category]
                
                # Ball category just stays the same
                transformed_detections['ball'] = detections['ball']
            
            # Update possession detector
            possession_result = None
            
            if pitch_ball_position is not None and transformed_detections is not None:
                # Use pitch coordinates for more accurate possession detection
                possession_result = possession_detector.update(transformed_detections, pitch_ball_position)
            elif ball_position is not None:
                # Fallback to frame coordinates if transformation failed
                possession_result = possession_detector.update(detections, ball_position)
            else:
                print("No ball position detected")
            
            # Create visualization
            # First annotate standard detections
            vis_frame = frame.copy()
            
            # Draw bounding boxes for all detections
            for category in ['players', 'goalkeepers', 'referees']:
                if len(detections[category]) > 0:
                    for i, box in enumerate(detections[category].xyxy):
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Draw bounding box
                        if category == 'referees':
                            color = (0, 255, 0)  # Green for referees
                        else:
                            # Get team color
                            team_id = int(detections[category].class_id[i])
                            color = (0, 0, 255) if team_id == 0 else (255, 0, 0)  # Red or Blue for teams
                        
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add tracker ID if available
                        if hasattr(detections[category], 'tracker_id') and detections[category].tracker_id is not None:
                            tracker_id = detections[category].tracker_id[i]
                            cv2.putText(vis_frame, f"#{tracker_id}", (x1, y2 + 15),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw ball
            if len(detections['ball']) > 0:
                for box in detections['ball'].xyxy:
                    x1, y1, x2, y2 = box.astype(int)
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(vis_frame, (center_x, center_y), 10, (0, 255, 255), -1)
            
            # Highlight player with possession
            vis_frame = possession_detector.highlight_possession(vis_frame, detections)
            
            # Add frame info
            proc_time = time.time() - start_time
            cv2.putText(vis_frame, f"Frame: {frame_count} | Processing time: {proc_time:.3f}s", 
                      (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if possession_result and possession_result['player_id'] is not None:
                cv2.putText(vis_frame, f"Current possession: Player #{possession_result['player_id']}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add coordinate type used
                coord_type = "pitch" if pitch_ball_position is not None and transformed_detections is not None else "frame"
                cv2.putText(vis_frame, f"Using {coord_type} coordinates", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(vis_frame, "No player has possession", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write to output video
            out.write(vis_frame)
            
            frame_count += 1
            pbar.update(1)
            
            # Save sample frame every 500 frames
            if frame_count % 500 == 0:
                sample_path = f"frame_{frame_count}_possession.jpg"
                cv2.imwrite(sample_path, vis_frame)
                print(f"Saved sample frame to {sample_path}")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Processing complete! Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test player possession detection')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--video', required=True, help='Path to video file')
    
    args = parser.parse_args()
    test_player_possession(args.config, args.video)