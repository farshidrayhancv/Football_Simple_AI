# """Streamlit app for real-time visualization of football video annotations."""

# import os
# import sys
# import tempfile
# import subprocess
# import time
# import yaml
# import shutil
# import json
# import threading
# import streamlit as st
# from PIL import Image
# import cv2
# import numpy as np
# from typing import Dict, Any, Optional, List, Tuple
# import glob

# # Add parent directory to path for imports
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.append(current_dir)


# # With these:
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # Direct imports
# from utils.annotation_utils import AnnotationStore, apply_annotations_to_frame
# # Try a different import approach
# try:
#     from processing.json_processor import VideoProcessorWithJSON
# except ImportError:
#     # Fallback to direct import
#     import processing.json_processing
#     VideoProcessorWithJSON = processing.json_processing.VideoProcessorWithJSON

"""
Enhanced Streamlit app for viewing and processing football video annotations.
Includes capability to run models directly from the app when annotations aren't available.
"""

import os
import sys
import tempfile
import time
import yaml
import json
import glob
import threading
import streamlit as st
import cv2
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

# Set page config
st.set_page_config(page_title="Football AI - Viewer & Processor", layout="wide")

# App title
st.title("Football AI - Annotation Viewer & Processor")
st.markdown("View and process football videos with AI annotations and tactical view")

# Initialize session state
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "config_path" not in st.session_state:
    st.session_state.config_path = None
if "frame_index" not in st.session_state:
    st.session_state.frame_index = 0
if "play_mode" not in st.session_state:
    st.session_state.play_mode = False
if "play_speed" not in st.session_state:
    st.session_state.play_speed = 1.0
if "display_options" not in st.session_state:
    st.session_state.display_options = {
        "show_detections": True,
        "show_poses": True,
        "show_segmentations": True,
        "show_possession": True,
        "show_pitch": True
    }
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "processing_thread" not in st.session_state:
    st.session_state.processing_thread = None

# Function to update config with new video path
def update_config_video_path(config_path, video_path):
    """Update the video input_path in the config file."""
    try:
        # Read existing config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update video input path
        if 'video' not in config:
            config['video'] = {}
        config['video']['input_path'] = video_path
        
        # Create a backup
        backup_path = f"{config_path}.bak"
        try:
            import shutil
            shutil.copy2(config_path, backup_path)
        except Exception as e:
            print(f"Warning: Could not create config backup: {e}")
        
        # Write updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return True
    except Exception as e:
        st.error(f"Failed to update config: {e}")
        return False

# Function to initialize models
@st.cache_resource
def initialize_models(config_path):
    """Initialize Football AI models."""
    # Import required components
    from config.config_loader import ConfigLoader
    from models.detector import EnhancedObjectDetector, FieldDetector
    from models.classifier import TeamClassifierModule
    from models.tracker import ObjectTracker
    from models.player_possession_detector import PlayerPossessionDetector
    
    try:
        # Load configuration
        config = ConfigLoader(config_path).config
        
        # Initialize player detector
        player_detector = EnhancedObjectDetector(
            model_id=config['models']['player_detection_model_id'],
            api_key=config['api_keys']['roboflow_api_key'],
            confidence_threshold=config['detection']['confidence_threshold'],
            enable_pose=config.get('display', {}).get('show_pose', True),
            enable_segmentation=config.get('display', {}).get('show_segmentation', True),
            device=config['performance']['device']
        )
        
        # Initialize field detector
        field_detector = FieldDetector(
            model_id=config['models']['field_detection_model_id'],
            api_key=config['api_keys']['roboflow_api_key']
        )
        
        # Initialize team classifier
        team_classifier = TeamClassifierModule(
            device=config['performance']['device'],
            hf_token=config['api_keys']['huggingface_token'],
            model_path=config['models']['siglip_model_path']
        )
        
        # Initialize tracker
        tracker = ObjectTracker()
        
        # Initialize possession detector if enabled
        possession_detector = None
        if config.get('possession_detection', {}).get('enable', True):
            possession_detector = PlayerPossessionDetector(
                proximity_threshold=config.get('possession_detection', {}).get('proximity_threshold', 250),
                frame_proximity_threshold=config.get('possession_detection', {}).get('frame_proximity_threshold', 30),
                coordinate_system=config.get('possession_detection', {}).get('coordinate_system', 'pitch'),
                possession_frames=config.get('possession_detection', {}).get('possession_frames', 3),
                possession_duration=config.get('possession_detection', {}).get('possession_duration', 3),
                no_possession_frames=config.get('possession_detection', {}).get('no_possession_frames', 10)
            )
        
        return config, player_detector, field_detector, team_classifier, tracker, possession_detector
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

# Function to get frame from video
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

# Function to convert hex color to BGR for OpenCV
def hex_to_bgr(hex_color):
    """Convert hex color to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    # Get RGB values
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # Return BGR for OpenCV
    return (b, g, r)

# Function to process video and save annotations
def process_video_thread(video_path, config_path, start_frame=0, end_frame=None, stride=1):
    """Process video in a background thread and save annotations."""
    try:
        st.session_state.is_processing = True
        
        # Create annotation directory structure
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        annotation_dir = os.path.join("annotations")
        os.makedirs(annotation_dir, exist_ok=True)
        video_annotation_dir = os.path.join(annotation_dir, video_basename)
        os.makedirs(video_annotation_dir, exist_ok=True)
        
        # Initialize metadata
        metadata_path = os.path.join(video_annotation_dir, "metadata.json")
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Could not open video: {video_path}")
            st.session_state.is_processing = False
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Adjust end frame if not specified
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
            
        # Create initial metadata
        metadata = {
            "video_path": video_path,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "processed_frames": 0,
            "is_processing": True,
            "created_at": time.time(),
            "features": {
                "detection": True,
                "pose": True,
                "segmentation": True,
                "possession": True
            }
        }
        
        # Save initial metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Initialize models
        config, player_detector, field_detector, team_classifier, tracker, possession_detector = initialize_models(config_path)
        
        # Initialize frame processor
        from processing.frame_processor import FrameProcessor
        frame_processor = FrameProcessor(
            player_detector=player_detector,
            field_detector=field_detector,
            team_classifier=team_classifier,
            tracker=tracker,
            config=config,
            possession_detector=possession_detector
        )
        
        # Train team classifier first
        from utils.video_utils import VideoProcessor
        video_utils = VideoProcessor(config)
        
        # Collect player crops for team classification
        crops = video_utils.collect_player_crops(
            video_path, 
            player_detector,
            config['video']['stride']
        )
        
        # Train classifier
        if crops:
            team_classifier.train(crops)
        
        # Skip to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames
        frame_count = start_frame
        while frame_count < end_frame and st.session_state.is_processing:
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
                results = frame_processor.process_frame(frame)
                
                # Unpack results
                detections, transformer, poses, pose_stats, segmentations, seg_stats, sahi_stats, possession_result = results
                
                # Convert data for JSON storage
                from utils.annotation_utils import AnnotationStore
                annotation_store = AnnotationStore(video_path, "annotations")
                
                # Save frame annotations
                transformer_matrix = transformer.m if transformer is not None else None
                
                annotation_store.save_frame_annotations(
                    frame_index=frame_count,
                    detections=detections,
                    poses=poses,
                    segmentations=segmentations,
                    transformer_matrix=transformer_matrix,
                    possession_result=possession_result
                )
                
                # Update metadata
                metadata["processed_frames"] = frame_count + 1
                metadata["features"] = {
                    "detection": True,
                    "pose": poses is not None,
                    "segmentation": segmentations is not None,
                    "possession": possession_result is not None
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        
        # Update metadata to indicate processing is complete
        metadata["is_processing"] = False
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        st.session_state.is_processing = False
        print(f"Video processing complete. Annotations saved to: {video_annotation_dir}")
    
    except Exception as e:
        st.session_state.is_processing = False
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()

# Function to apply annotations to frame
def apply_annotations_to_frame(frame, annotation, config):
    """Apply annotations to frame using simplified approach."""
    if annotation is None:
        return frame
    
    # Import required classes only when needed
    import supervision as sv
    from models.detector import PoseDetector
    from models.player_possession_detector import PlayerPossessionDetector
    
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
    
    # Draw bounding boxes if detections enabled
    if config["display"]["show_detections"]:
        # Draw all human detections (players, goalkeepers, referees)
        all_humans = []
        for category in ['players', 'goalkeepers', 'referees']:
            if len(detections.get(category, sv.Detections.empty())) > 0:
                all_humans.append(detections[category])
        
        if all_humans:
            merged = sv.Detections.merge(all_humans)
            
            # Define colors based on team ids
            colors = [
                hex_to_bgr(config["display"]["team_colors"]["team_1"]),  # Team 1
                hex_to_bgr(config["display"]["team_colors"]["team_2"]),  # Team 2
                hex_to_bgr(config["display"]["referee_color"])           # Referee
            ]
            
            # Draw bounding boxes
            for i, box in enumerate(merged.xyxy):
                x1, y1, x2, y2 = box.astype(int)
                team_id = merged.class_id[i] if merged.class_id is not None else 0
                color = colors[min(team_id, 2)]  # Default to team 1 if unknown
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw tracker ID if available
                if merged.tracker_id is not None:
                    tracker_id = merged.tracker_id[i]
                    cv2.putText(annotated, f"#{tracker_id}", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw ball
        ball = detections.get('ball', sv.Detections.empty())
        if len(ball) > 0:
            ball_color = hex_to_bgr(config["display"]["ball_color"])
            for box in ball.xyxy:
                x1, y1, x2, y2 = box.astype(int)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                radius = max(5, (x2 - x1) // 4)
                cv2.circle(annotated, (center_x, center_y), radius, ball_color, -1)
    
    # Draw poses if enabled
    if config["display"]["show_poses"] and annotation.get("poses"):
        pose_drawer = PoseDetector()
        
        # Team colors (in BGR format)
        team1_color = hex_to_bgr(config["display"]["team_colors"]["team_1"])
        team2_color = hex_to_bgr(config["display"]["team_colors"]["team_2"])
        referee_color = hex_to_bgr(config["display"]["referee_color"])
        
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
                
                # Draw pose
                annotated = pose_drawer.draw_pose(annotated, pose, color)
    
    # Highlight possession if enabled
    if config["display"]["show_possession"] and annotation.get("possession") and annotation["possession"].get("player_id") is not None:
        # Create temporary possession detector
        possession_detector = PlayerPossessionDetector()
        possession_detector.current_possession = annotation["possession"]["player_id"]
        possession_detector.current_team = annotation["possession"]["team_id"]
        
        # Highlight player with possession
        annotated = possession_detector.highlight_possession(annotated, detections)
    
    return annotated

# Function to render pitch view
def render_pitch_view(annotation, config_path):
    """Render the tactical pitch view using existing PitchRenderer."""
    if not annotation or not annotation.get("transformer_matrix"):
        return None
    
    try:
        # Import required modules
        import supervision as sv
        from config.config_loader import ConfigLoader
        from visualization.pitch_renderer import PitchRenderer
        from sports.common.view import ViewTransformer
        
        # Load configuration
        config = ConfigLoader(config_path).config
        
        # Convert annotations to format needed by pitch renderer
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
        
        # Create transformer correctly with both source and target
        transformer = ViewTransformer(
            source=np.zeros((1, 2)),
            target=np.zeros((1, 2))  # Add target parameter to fix the error
        )
        transformer.m = np.array(annotation["transformer_matrix"])
        
        # Create pitch renderer
        pitch_renderer = PitchRenderer(config)
        
        # Create ball trail (just use current position for simplicity)
        ball_trail = []
        if len(detections.get('ball', sv.Detections.empty())) > 0:
            ball_xy = detections['ball'].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            if len(ball_xy) > 0:
                # Transform to pitch coordinates
                pitch_xy = transformer.transform_points(points=ball_xy)
                if len(pitch_xy) > 0:
                    ball_trail = [pitch_xy[0]]
        
        # Render pitch
        pitch_view = pitch_renderer.render(detections, transformer, ball_trail)
        return pitch_view
    
    except Exception as e:
        st.error(f"Error rendering pitch view: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Function to process a single frame
def process_single_frame(frame, frame_index, config_path):
    """Process a single frame and return the results."""
    try:
        # Initialize models
        config, player_detector, field_detector, team_classifier, tracker, possession_detector = initialize_models(config_path)
        
        # Initialize frame processor
        from processing.frame_processor import FrameProcessor
        frame_processor = FrameProcessor(
            player_detector=player_detector,
            field_detector=field_detector,
            team_classifier=team_classifier,
            tracker=tracker,
            config=config,
            possession_detector=possession_detector
        )
        
        # Process frame
        results = frame_processor.process_frame(frame)
        
        # Return results
        return results
    except Exception as e:
        st.error(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return None

# Video selection section
st.header("1. Select Video")
source_tab1, source_tab2 = st.tabs(["Upload Video", "Use Existing Video"])

# Initialize video path
video_path = None

with source_tab1:
    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])
    if uploaded_file:
        # Save uploaded file to temp location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
        st.session_state.video_path = video_path
        st.success(f"Video uploaded successfully")

with source_tab2:
    video_dir = "videos"
    if os.path.exists(video_dir):
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
        if video_files:
            selected_video = st.selectbox("Select a video from videos directory", video_files)
            video_path = os.path.join(video_dir, selected_video)
            st.session_state.video_path = video_path
            st.success(f"Selected video: {video_path}")
        else:
            st.warning(f"No video files found in {video_dir} directory")
    else:
        st.warning(f"Directory {video_dir} does not exist")

# Load processed videos with annotations
annotation_dir = "annotations"
if os.path.exists(annotation_dir):
    processed_videos = [d for d in os.listdir(annotation_dir) if os.path.isdir(os.path.join(annotation_dir, d))]
    
    if processed_videos:
        st.header("2. Select Previously Processed Video")
        selected_processed = st.selectbox(
            "Select a processed video", 
            ["None"] + processed_videos
        )
        
        if selected_processed != "None":
            # Find the original video
            video_found = False
            for ext in ['.mp4', '.mov', '.avi']:
                potential_path = os.path.join("videos", f"{selected_processed}{ext}")
                if os.path.exists(potential_path):
                    video_path = potential_path
                    st.session_state.video_path = video_path
                    st.success(f"Loaded processed video: {video_path}")
                    video_found = True
                    break
            
            if not video_found:
                st.warning(f"Original video file not found for {selected_processed}. Please make sure it exists in the videos directory.")

# Select config file
st.header("3. Configuration")
config_files = glob.glob("*.yaml")
available_configs = [f for f in config_files if os.path.exists(f)]

if available_configs:
    config_path = st.selectbox("Select config file", available_configs)
    st.session_state.config_path = config_path
    st.success(f"Using config: {config_path}")
else:
    st.error("No config files found. Please create config.yaml or config_temp.yaml")
    config_path = None

# Process video button
st.header("4. Process Video")
col1, col2 = st.columns(2)

start_frame = 0
end_frame = None
stride = 1

with st.expander("Advanced Processing Options"):
    start_frame = st.number_input("Start Frame", min_value=0, value=0)
    end_frame_input = st.number_input("End Frame (0 = all)", min_value=0, value=0)
    end_frame = None if end_frame_input == 0 else end_frame_input
    stride = st.number_input("Frame Stride", min_value=1, value=1)

with col1:
    if st.button("Start Processing", disabled=st.session_state.is_processing or not video_path or not config_path):
        # Update config with video path
        if update_config_video_path(config_path, video_path):
            st.success(f"Updated config to use selected video: {video_path}")
        
        # Start processing thread
        st.session_state.is_processing = True
        processing_thread = threading.Thread(
            target=process_video_thread,
            args=(video_path, config_path, start_frame, end_frame, stride)
        )
        processing_thread.daemon = True
        processing_thread.start()
        st.session_state.processing_thread = processing_thread
        st.rerun()

with col2:
    if st.button("Stop Processing", disabled=not st.session_state.is_processing):
        # This won't actually stop the thread, but will update the UI
        st.session_state.is_processing = False
        st.warning("Processing will stop after current frame completes.")
        st.rerun()

# Display processing status
if st.session_state.is_processing:
    progress_container = st.empty()
    
    # Check progress
    if video_path:
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        metadata_path = os.path.join("annotations", video_basename, "metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            processed = metadata.get("processed_frames", 0)
            total = metadata.get("total_frames", 100)
            
            if processed > 0 and total > 0:
                progress = processed / total
                progress_container.progress(progress, f"Processing: {processed}/{total} frames ({progress:.1%})")

# Main viewer section
st.header("5. Annotation Viewer")

if st.session_state.video_path and os.path.exists(st.session_state.video_path):
    # Load video info
    cap = cv2.VideoCapture(st.session_state.video_path)
    if cap.isOpened():
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Get video basename for annotation path
        video_basename = os.path.splitext(os.path.basename(st.session_state.video_path))[0]
        annotation_path = os.path.join("annotations", video_basename)
        
        # Check if annotations exist
        annotations_exist = os.path.exists(annotation_path)
        metadata = None
        processed_frames = 0
        
        if annotations_exist:
            metadata_path = os.path.join(annotation_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                processed_frames = metadata.get("processed_frames", 0)
        
        # Display video info in sidebar
        st.sidebar.subheader("Video Information")
        st.sidebar.text(f"Dimensions: {width}x{height}")
        st.sidebar.text(f"FPS: {fps}")
        st.sidebar.text(f"Total Frames: {total_frames}")
        if annotations_exist and metadata:
            st.sidebar.text(f"Processed Frames: {processed_frames}")
            
            # Display options
            st.sidebar.subheader("Display Options")
            st.session_state.display_options["show_detections"] = st.sidebar.checkbox(
                "Show Detections", value=st.session_state.display_options["show_detections"]
            )
            
            # Only show toggles for available features
            if metadata["features"].get("pose", False):
                st.session_state.display_options["show_poses"] = st.sidebar.checkbox(
                    "Show Poses", value=st.session_state.display_options["show_poses"]
                )
            
            if metadata["features"].get("segmentation", False):
                st.session_state.display_options["show_segmentations"] = st.sidebar.checkbox(
                    "Show Segmentations", value=st.session_state.display_options["show_segmentations"]
                )
            
            if metadata["features"].get("possession", False):
                st.session_state.display_options["show_possession"] = st.sidebar.checkbox(
                    "Show Possession", value=st.session_state.display_options["show_possession"]
                )
            
            st.session_state.display_options["show_pitch"] = st.sidebar.checkbox(
                "Show Pitch View", value=st.session_state.display_options["show_pitch"]
            )
            
            # Frame navigation
            st.sidebar.subheader("Navigation")
            
            # Slider for frame selection
            max_frame = processed_frames - 1 if processed_frames > 0 else total_frames - 1
            new_frame_index = st.sidebar.slider("Frame", 0, max_frame, st.session_state.frame_index)
            
            # Play controls
            col1, col2, col3 = st.sidebar.columns(3)
            
            with col1:
                if st.button("⏮️", help="First Frame"):
                    new_frame_index = 0
                    st.session_state.play_mode = False
            
            with col2:
                play_text = "⏸️" if st.session_state.play_mode else "▶️"
                play_help = "Pause" if st.session_state.play_mode else "Play"
                if st.button(play_text, help=play_help):
                    st.session_state.play_mode = not st.session_state.play_mode
            
            with col3:
                if st.button("⏭️", help="Last Frame"):
                    new_frame_index = max_frame
                    st.session_state.play_mode = False
            
            # Play speed
            st.session_state.play_speed = st.sidebar.slider("Play Speed", 0.1, 2.0, 1.0, 0.1)
            
            # Update frame index
            if st.session_state.play_mode and new_frame_index < max_frame:
                st.session_state.frame_index = new_frame_index + 1
            else:
                st.session_state.frame_index = new_frame_index
        else:
            # No annotations - offer to run models directly
            st.sidebar.text("No annotations found.")
            
            # Options for viewing frames
            st.sidebar.subheader("Navigation")
            
            # Slider for frame selection
            new_frame_index = st.sidebar.slider("Frame", 0, total_frames - 1, st.session_state.frame_index)
            st.session_state.frame_index = new_frame_index
            
            # Display options when processing on-the-fly
            st.sidebar.subheader("Display Options")
            st.session_state.display_options["show_detections"] = st.sidebar.checkbox(
                "Show Detections", value=st.session_state.display_options["show_detections"]
            )
            st.session_state.display_options["show_poses"] = st.sidebar.checkbox(
                "Show Poses", value=st.session_state.display_options["show_poses"]
            )
            st.session_state.display_options["show_possession"] = st.sidebar.checkbox(
                "Show Possession", value=st.session_state.display_options["show_possession"]
            )
            st.session_state.display_options["show_pitch"] = st.sidebar.checkbox(
                "Show Pitch View", value=st.session_state.display_options["show_pitch"]
            )
        
        # Get original frame
        frame = get_frame(st.session_state.video_path, st.session_state.frame_index)
        if frame is None:
            st.error(f"Could not read frame {st.session_state.frame_index} from video")
        else:
            # Create display columns
            if st.session_state.display_options["show_pitch"]:
                col1, col2 = st.columns(2)
            else:
                col1 = st
            
            # Get annotation if it exists
            annotation = None
            if annotations_exist:
                frame_path = os.path.join(annotation_path, f"frame_{st.session_state.frame_index:06d}.json")
                if os.path.exists(frame_path):
                    with open(frame_path, 'r') as f:
                        annotation = json.load(f)
            
            # If annotations don't exist but config is available, process on-the-fly
            on_the_fly_results = None
            if annotation is None and config_path and st.session_state.config_path:
                # Display a message
                with st.spinner(f"Processing frame {st.session_state.frame_index} on-the-fly..."):
                    # Process the frame
                    on_the_fly_results = process_single_frame(frame, st.session_state.frame_index, config_path)
            
            # Display frame with annotations or processing
            with col1:
                if annotation is not None:
                    # Apply annotations from saved data
                    config = {
                        "display": {
                            "show_detections": st.session_state.display_options["show_detections"],
                            "show_pose": st.session_state.display_options["show_poses"],
                            "show_segmentation": st.session_state.display_options["show_segmentations"],
                            "show_possession": st.session_state.display_options["show_possession"],
                            "show_tracking_ids": True,  # Add this to avoid KeyError
                            "show_ball": True,          # Add this to avoid KeyError
                            "show_voronoi": False,      # Add this to avoid KeyError
                            "segmentation_alpha": 0.6,  # Add this to avoid KeyError
                            "team_colors": {
                                "team_1": "#00BFFF",  # Light blue
                                "team_2": "#FF1493"   # Deep pink
                            },
                            "referee_color": "#FFD700",  # Gold
                            "ball_color": "#FFD700"     # Gold
                        }
                    }
                    
                    annotated_frame = apply_annotations_to_frame(frame, annotation, config)
                    st.image(annotated_frame, channels="BGR", use_container_width=True, caption="Annotated Video")
                elif on_the_fly_results is not None:
                    # Process frame and display results
                    from visualization.annotators import FootballAnnotator
                    detections, transformer, poses, pose_stats, segmentations, seg_stats, sahi_stats, possession_result = on_the_fly_results
                    
                    # Create config for annotation
                    config_loader = ConfigLoader(config_path)
                    config = config_loader.config
                    
                    # Create annotator
                    annotator = FootballAnnotator(config)
                    
                    # Process based on selected options
                    if st.session_state.display_options["show_poses"] and poses:
                        annotated_frame = annotator.annotate_frame_with_pose(frame, detections, poses)
                    else:
                        annotated_frame = annotator.annotate_frame(frame, detections)
                    
                    # Add possession highlighting if enabled
                    if st.session_state.display_options["show_possession"] and possession_result:
                        from models.player_possession_detector import PlayerPossessionDetector
                        possession_detector = PlayerPossessionDetector()
                        possession_detector.current_possession = possession_result.get("player_id")
                        possession_detector.current_team = possession_result.get("team_id")
                        annotated_frame = possession_detector.highlight_possession(annotated_frame, detections)
                    
                    st.image(annotated_frame, channels="BGR", use_container_width=True, caption="Processed Video (On-the-fly)")
                    
                    # Save this processed frame as annotation for future use
                    if st.button("Save this processed frame"):
                        try:
                            # Create annotation directory structure
                            video_basename = os.path.splitext(os.path.basename(st.session_state.video_path))[0]
                            annotation_dir = os.path.join("annotations")
                            os.makedirs(annotation_dir, exist_ok=True)
                            video_annotation_dir = os.path.join(annotation_dir, video_basename)
                            os.makedirs(video_annotation_dir, exist_ok=True)
                            
                            # Create or load metadata
                            metadata_path = os.path.join(video_annotation_dir, "metadata.json")
                            if os.path.exists(metadata_path):
                                with open(metadata_path, 'r') as f:
                                    metadata = json.load(f)
                            else:
                                cap = cv2.VideoCapture(st.session_state.video_path)
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                cap.release()
                                
                                metadata = {
                                    "video_path": st.session_state.video_path,
                                    "fps": fps,
                                    "width": width,
                                    "height": height,
                                    "total_frames": total_frames,
                                    "processed_frames": 0,
                                    "is_processing": False,
                                    "created_at": time.time(),
                                    "features": {
                                        "detection": True,
                                        "pose": poses is not None,
                                        "segmentation": segmentations is not None,
                                        "possession": possession_result is not None
                                    }
                                }
                            
                            # Create annotation store
                            from utils.annotation_utils import AnnotationStore
                            annotation_store = AnnotationStore(st.session_state.video_path, "annotations")
                            
                            # Save frame annotation
                            transformer_matrix = transformer.m if transformer is not None else None
                            
                            annotation_store.save_frame_annotations(
                                frame_index=st.session_state.frame_index,
                                detections=detections,
                                poses=poses,
                                segmentations=segmentations,
                                transformer_matrix=transformer_matrix,
                                possession_result=possession_result
                            )
                            
                            # Update metadata
                            metadata["processed_frames"] = max(metadata["processed_frames"], st.session_state.frame_index + 1)
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=2)
                            
                            st.success(f"Frame {st.session_state.frame_index} saved as annotation!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving annotation: {e}")
                            import traceback
                            traceback.print_exc()
                else:
                    st.image(frame, channels="BGR", use_container_width=True, caption="Original Video (No Annotations)")
            
            # Pitch view
            if st.session_state.display_options["show_pitch"]:
                with col2:
                    if annotation is not None and st.session_state.config_path:
                        # Use saved annotation for pitch view
                        pitch_view = render_pitch_view(annotation, st.session_state.config_path)
                        if pitch_view is not None:
                            st.image(pitch_view, channels="BGR", use_container_width=True, caption="Tactical Pitch View")
                        else:
                            st.warning("Could not render pitch view from saved annotation.")
                    elif on_the_fly_results is not None:
                        # Create pitch view from on-the-fly results
                        detections, transformer, poses, pose_stats, segmentations, seg_stats, sahi_stats, possession_result = on_the_fly_results
                        
                        if transformer is not None:
                            try:
                                from visualization.pitch_renderer import PitchRenderer
                                config_loader = ConfigLoader(config_path)
                                pitch_renderer = PitchRenderer(config_loader.config)
                                
                                # Create ball trail (just current position for simplicity)
                                ball_trail = []
                                if 'ball' in detections and len(detections['ball']) > 0:
                                    ball_xy = detections['ball'].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                                    if len(ball_xy) > 0:
                                        # Transform to pitch coordinates
                                        pitch_xy = transformer.transform_points(points=ball_xy)
                                        if len(pitch_xy) > 0:
                                            ball_trail = [pitch_xy[0]]
                                
                                # Render pitch
                                pitch_view = pitch_renderer.render(detections, transformer, ball_trail)
                                
                                if pitch_view is not None:
                                    st.image(pitch_view, channels="BGR", use_container_width=True, caption="Tactical Pitch View (On-the-fly)")
                                else:
                                    st.warning("Could not render pitch view from on-the-fly processing.")
                            except Exception as e:
                                st.error(f"Error rendering pitch view: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            st.warning("No field transformation available for pitch view.")
                    else:
                        st.warning("No annotation available for pitch view.")
            
            # Show annotation data in expandable
            if annotation is not None:
                with st.expander("Show Annotation Data", expanded=False):
                    # Only show relevant parts
                    display_annotation = {
                        "frame_index": annotation["frame_index"],
                        "timestamp": annotation["timestamp"],
                        "detections": {
                            category: [
                                {"box": d["box"], "tracker_id": d["tracker_id"], "class_id": d["class_id"]}
                                for d in detection_list
                            ]
                            for category, detection_list in annotation["detections"].items()
                            if detection_list
                        }
                    }
                    
                    if annotation.get("possession"):
                        display_annotation["possession"] = annotation["possession"]
                    
                    st.json(display_annotation)
            
            # Auto-refresh if in play mode
            if st.session_state.play_mode:
                time.sleep(1.0 / (fps * st.session_state.play_speed))
                st.rerun()
    else:
        st.error("Could not open video file.")
else:
    st.info("Please select a video first.")

if __name__ == "__main__":
    # This helps when running directly with streamlit run
    pass