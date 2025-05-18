#!/usr/bin/env python3
"""
Simple Streamlit app to view football videos with annotations
"""

import os
import cv2
import numpy as np
import streamlit as st
import json
import yaml
import time
from models.detector import PoseDetector
from models.player_possession_detector import PlayerPossessionDetector

# Set up the page
st.set_page_config(
    page_title="Football Video Viewer",
    layout="wide"
)

# Title
st.title("Football Video Annotations Viewer")

# Utility functions
def hex_to_bgr(hex_color):
    """Convert hex color to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

def load_config():
    """Load configuration from config.yaml."""
    try:
        with open("config_temp.yaml", 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading config.yaml: {e}")
        return None

def get_video_path_from_config(config):
    """Get video path from config."""
    if not config or 'video' not in config or 'input_path' not in config['video']:
        return None
    return config['video']['input_path']

def check_annotations_exist(video_path):
    """Check if annotations exist for the given video."""
    if not video_path:
        return False
        
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    annotation_dir = os.path.join("annotations", video_basename)
    
    if os.path.exists(annotation_dir):
        metadata_path = os.path.join(annotation_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                if not metadata.get("is_processing", True) and metadata.get("processed_frames", 0) > 0:
                    return True
            except Exception as e:
                st.error(f"Error checking annotations: {e}")
    
    return False

def get_frame(video_path, frame_index):
    """Get a specific frame from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    return frame

def load_annotation(video_path, frame_index):
    """Load annotation for a specific frame."""
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    frame_path = os.path.join("annotations", video_basename, f"frame_{frame_index:06d}.json")
    
    if os.path.exists(frame_path):
        try:
            with open(frame_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading annotation: {e}")
    
    return None

def draw_annotations(frame, annotation):
    """Draw annotations on frame."""
    if annotation is None:
        return frame
    
    annotated = frame.copy()
    
    # Import necessary modules
    import supervision as sv
    
    # Convert detections to Supervision format
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
    
    # Define colors
    colors = {
        "team_1": hex_to_bgr("#00BFFF"),  # Light blue
        "team_2": hex_to_bgr("#FF1493"),  # Deep pink
        "referee": hex_to_bgr("#FFD700"),  # Gold
        "ball": hex_to_bgr("#FFD700")      # Gold
    }
    
    # Draw player bounding boxes
    for category in ['players', 'goalkeepers', 'referees']:
        if len(detections.get(category, sv.Detections.empty())) > 0:
            category_detections = detections[category]
            
            for i in range(len(category_detections)):
                # Get box
                box = category_detections.xyxy[i].astype(int)
                x1, y1, x2, y2 = box
                
                # Determine color
                if category == 'referees':
                    color = colors["referee"]
                else:
                    team_id = category_detections.class_id[i] if category_detections.class_id is not None else 0
                    color = colors["team_1"] if team_id == 0 else colors["team_2"]
                
                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID if available
                if category_detections.tracker_id is not None:
                    tracker_id = category_detections.tracker_id[i]
                    cv2.putText(annotated, f"#{tracker_id}", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw ball
    if len(detections.get('ball', sv.Detections.empty())) > 0:
        for box in detections['ball'].xyxy:
            x1, y1, x2, y2 = box.astype(int)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            radius = max(5, (x2 - x1) // 4)
            cv2.circle(annotated, (center_x, center_y), radius, colors["ball"], -1)
    
    # Draw poses if available
    if annotation.get("poses"):
        pose_drawer = PoseDetector()
        
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
                    color = colors["referee"]
                else:
                    team_id = detections[category].class_id[i] if i < len(detections[category]) else 0
                    color = colors["team_1"] if team_id == 0 else colors["team_2"]
                
                # Draw pose
                annotated = pose_drawer.draw_pose(annotated, pose, color)
    
    # Highlight player with possession if available
    if annotation.get("possession") and annotation["possession"].get("player_id") is not None:
        possession_detector = PlayerPossessionDetector()
        possession_detector.current_possession = annotation["possession"]["player_id"]
        possession_detector.current_team = annotation["possession"]["team_id"]
        annotated = possession_detector.highlight_possession(annotated, detections)
    
    return annotated

# Main app logic
def main():
    # Load configuration
    config = load_config()
    
    if not config:
        st.error("Error: Failed to load config.yaml file.")
        return
    
    # Get video path from config
    video_path = get_video_path_from_config(config)
    
    if not video_path:
        st.error("Error: No video path found in config.yaml.")
        return
    
    if not os.path.exists(video_path):
        st.error(f"Error: Video file not found: {video_path}")
        return
    
    # Display video info
    st.write(f"**Video**: {os.path.basename(video_path)}")
    
    # Check if annotations exist
    annotations_exist = check_annotations_exist(video_path)
    
    if not annotations_exist:
        st.error("No annotations found for this video. Please process it first using:")
        st.code(f"python main.py --config config.yaml --video {video_path}")
        return
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Get video basename for annotation path
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    annotation_path = os.path.join("annotations", video_basename)
    
    # Load metadata
    metadata_path = os.path.join(annotation_path, "metadata.json")
    processed_frames = 0
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        processed_frames = metadata.get("processed_frames", 0)
    
    # Setup UI
    st.sidebar.header("Video Controls")
    
    # Add display options
    st.sidebar.header("Display Options")
    show_detections = st.sidebar.checkbox("Show Detections", value=True)
    show_poses = st.sidebar.checkbox("Show Poses", value=True)
    show_possession = st.sidebar.checkbox("Show Possession", value=True)
    
    # Frame navigation
    st.sidebar.header("Navigation")
    frame_index = st.sidebar.slider("Frame", 0, min(processed_frames-1, total_frames-1), 0)
    
    # Play controls
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button("⏮️", help="First Frame"):
            frame_index = 0
    
    # Create a session state for play mode if it doesn't exist
    if "play_mode" not in st.session_state:
        st.session_state.play_mode = False
    
    with col2:
        play_text = "⏸️" if st.session_state.play_mode else "▶️"
        play_help = "Pause" if st.session_state.play_mode else "Play"
        if st.button(play_text, help=play_help):
            st.session_state.play_mode = not st.session_state.play_mode
    
    with col3:
        if st.button("⏭️", help="Last Frame"):
            frame_index = min(processed_frames-1, total_frames-1)
    
    # Play speed
    play_speed = st.sidebar.slider("Play Speed", 0.1, 2.0, 1.0, 0.1)
    
    # Display frame with annotations
    frame = get_frame(video_path, frame_index)
    
    if frame is None:
        st.error(f"Could not read frame {frame_index} from video")
    else:
        # Get annotation
        annotation = load_annotation(video_path, frame_index)
        
        if annotation is not None:
            # Apply annotations
            annotated_frame = draw_annotations(frame, annotation)
            
            # Display annotated frame
            st.image(annotated_frame, caption=f"Frame {frame_index}", use_container_width=True, channels="BGR")
            
            # Display metadata
            with st.expander("Frame Info", expanded=False):
                frame_info = {
                    "Frame": frame_index,
                    "Timestamp": annotation.get("timestamp"),
                    "Detections": {
                        "Players": len(annotation["detections"]["players"]),
                        "Goalkeepers": len(annotation["detections"]["goalkeepers"]),
                        "Referees": len(annotation["detections"]["referees"]),
                        "Ball": len(annotation["detections"]["ball"])
                    }
                }
                
                if annotation.get("possession"):
                    possession_info = {
                        "Player ID": annotation["possession"].get("player_id"),
                        "Team ID": annotation["possession"].get("team_id")
                    }
                    frame_info["Possession"] = possession_info
                
                st.json(frame_info)
        else:
            # Display original frame
            st.image(frame, caption=f"Frame {frame_index} (No Annotation)", use_container_width=True, channels="BGR")
    
    # Auto-refresh if in play mode
    if st.session_state.play_mode and frame_index < min(processed_frames-1, total_frames-1):
        time.sleep(1.0 / (fps * play_speed))
        st.rerun()

if __name__ == "__main__":
    main()