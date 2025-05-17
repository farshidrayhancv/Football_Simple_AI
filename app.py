import tempfile
import numpy as np
import os
import sys
import yaml
import torch
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from tqdm import tqdm

# Display version info and setup instructions at startup
st.set_page_config(page_title="Football AI - Advanced Tactical Analysis", layout="wide", initial_sidebar_state="expanded")

# Check NumPy version and warn if needed
import numpy as np
numpy_version = np.__version__
if numpy_version.startswith("2."):
    st.warning(f"""
    ⚠️ NumPy version {numpy_version} detected. This may cause compatibility issues.
    
    Some modules require NumPy 1.x. Please consider running:
    ```
    pip install numpy==1.24.3
    ```
    """)

# Handle module imports with error catching
@st.cache_resource
def import_football_ai_modules():
    try:
        # Import Football AI modules with error handling
        from config.config_loader import ConfigLoader
        from models.detector import EnhancedObjectDetector, FieldDetector
        from models.classifier import TeamClassifierModule
        from models.tracker import ObjectTracker
        from models.player_possession_detector import PlayerPossessionDetector
        from processing.frame_processor import FrameProcessor
        from visualization.annotators import FootballAnnotator
        from visualization.pitch_renderer import PitchRenderer
        from caching.cache_manager import CacheManager
        from utils.video_utils import VideoProcessor
        
        return {
            'ConfigLoader': ConfigLoader,
            'EnhancedObjectDetector': EnhancedObjectDetector,
            'FieldDetector': FieldDetector,
            'TeamClassifierModule': TeamClassifierModule,
            'ObjectTracker': ObjectTracker,
            'PlayerPossessionDetector': PlayerPossessionDetector,
            'FrameProcessor': FrameProcessor,
            'FootballAnnotator': FootballAnnotator,
            'PitchRenderer': PitchRenderer,
            'CacheManager': CacheManager,
            'VideoProcessor': VideoProcessor,
            'import_successful': True
        }
    except Exception as e:
        st.error(f"Error importing Football AI modules: {str(e)}")
        return {'import_successful': False, 'error': str(e)}

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return None

def initialize_football_ai(config, modules):
    """Initialize Football AI components."""
    if not modules.get('import_successful', False):
        st.error("Cannot initialize Football AI: Module import failed")
        return None
    
    try:
        # Set CUDA environment variables if GPU available
        if torch.cuda.is_available():
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Initialize object detectors
        player_detector = modules['EnhancedObjectDetector'](
            model_id=config['models']['player_detection_model_id'],
            api_key=config['api_keys']['roboflow_api_key'],
            confidence_threshold=config['detection']['confidence_threshold'],
            enable_pose=config.get('display', {}).get('show_pose', True),
            enable_segmentation=config.get('display', {}).get('show_segmentation', True),
            pose_model=config.get('models', {}).get('pose_model', 'yolo11m-pose.pt'),
            sam_model=config.get('models', {}).get('sam_model', 'sam2.1_b.pt'),
            padding_ratio=config.get('detection', {}).get('padding_ratio', 0.1),
            device=config['performance']['device']
        )
        
        field_detector = modules['FieldDetector'](
            model_id=config['models']['field_detection_model_id'],
            api_key=config['api_keys']['roboflow_api_key']
        )
        
        # Initialize team classifier
        team_classifier = modules['TeamClassifierModule'](
            device=config['performance']['device'],
            hf_token=config['api_keys']['huggingface_token'],
            model_path=config['models']['siglip_model_path']
        )
        
        # Initialize tracker
        tracker = modules['ObjectTracker']()
        
        # Initialize possession detector if enabled
        possession_detector = None
        if config.get('possession_detection', {}).get('enable', True):
            possession_detector = modules['PlayerPossessionDetector'](
                proximity_threshold=config.get('possession_detection', {}).get('proximity_threshold', 250),
                frame_proximity_threshold=config.get('possession_detection', {}).get('frame_proximity_threshold', 30),
                coordinate_system=config.get('possession_detection', {}).get('coordinate_system', 'pitch'),
                possession_frames=config.get('possession_detection', {}).get('possession_frames', 3),
                possession_duration=config.get('possession_detection', {}).get('possession_duration', 3),
                no_possession_frames=config.get('possession_detection', {}).get('no_possession_frames', 10)
            )
        
        # Initialize frame processor
        frame_processor = modules['FrameProcessor'](
            player_detector=player_detector,
            field_detector=field_detector,
            team_classifier=team_classifier,
            tracker=tracker,
            config=config,
            possession_detector=possession_detector
        )
        
        # Initialize visualization components
        annotator = modules['FootballAnnotator'](
            config=config,
            possession_detector=possession_detector
        )
        
        pitch_renderer = modules['PitchRenderer'](config)
        
        return {
            'player_detector': player_detector,
            'field_detector': field_detector,
            'team_classifier': team_classifier,
            'tracker': tracker,
            'possession_detector': possession_detector,
            'frame_processor': frame_processor,
            'annotator': annotator,
            'pitch_renderer': pitch_renderer
        }
    except Exception as e:
        st.error(f"Error initializing Football AI: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def process_video(video_path, stframe, football_ai, ui_settings, save_output=False, output_file=None):
    """Process video with Football AI."""
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer if requested
    out = None
    if save_output:
        if not output_file:
            output_file = "football_ai_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_width = width + 800  # Original + tactical view
        out_height = max(height, 600)
        out = cv2.VideoWriter(output_file, fourcc, fps, (out_width, out_height))
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    # Train team classifier if needed
    with st.spinner('Training team classifier...'):
        # Collect player crops
        crops = []
        sample_frames = min(30, total_frames)  # Limit number of frames for training
        stride = max(1, total_frames // sample_frames)
        
        for i in range(0, total_frames, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect players
            detections = football_ai['player_detector'].detect_categories(frame)
            try:
                import supervision as sv
                player_crops = [sv.crop_image(frame, xyxy) for xyxy in detections['players'].xyxy]
                crops.extend(player_crops)
            except Exception as e:
                st.error(f"Error creating player crops: {e}")
                continue
        
        # Train classifier
        if crops:
            football_ai['team_classifier'].train(crops)
            st.toast('Team classifier trained successfully!')
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Initialize processing variables
    frame_count = 0
    stop_requested = False
    
    # Main processing loop
    while True:
        # Check if stop button was clicked
        if 'stop_processing' in st.session_state and st.session_state['stop_processing']:
            stop_requested = True
            break
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        try:
            results = football_ai['frame_processor'].process_frame(frame)
            
            # Unpack results
            detections, transformer, poses, pose_stats, segmentations, seg_stats, sahi_stats, possession_result = results
            
            # Create visualizations
            annotated_frame = football_ai['annotator'].annotate_frame_with_all_features(
                frame, detections, poses, segmentations, possession_result
            )
            
            # Render pitch view
            pitch_view = football_ai['pitch_renderer'].render(
                detections, transformer, football_ai['tracker'].ball_trail
            )
            
            # Combine frames
            out_height = max(height, 600)
            out_width = width + 800
            combined = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            
            # Place frames
            combined[:height, :width] = annotated_frame
            resized_pitch = cv2.resize(pitch_view, (800, 600))
            combined[:600, width:] = resized_pitch
            
            # Add labels
            features = []
            if ui_settings.get('show_pose', False):
                features.append("Pose")
            if ui_settings.get('show_segmentation', False):
                features.append("Segmentation")
            if ui_settings.get('enable_possession', False):
                features.append("Player Possession")
            if ui_settings.get('enable_sahi', False):
                features.append("SAHI")
            
            features_str = " with " + ", ".join(features) if features else ""
            cv2.putText(combined, f"Original{features_str}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Tactical View", (width + 10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display frame
            stframe.image(combined, channels="BGR")
            
            # Save frame if requested
            if out is not None:
                out.write(combined)
            
        except Exception as e:
            st.error(f"Error processing frame {frame_count}: {e}")
            import traceback
            st.code(traceback.format_exc())
        
        # Update progress
        frame_count += 1
        progress_percentage = min(100, int((frame_count / total_frames) * 100))
        progress_bar.progress(progress_percentage)
        
        # Check if we need to slow down processing for realtime display
        if ui_settings.get('realtime_processing', False):
            import time
            time.sleep(1/fps)
    
    # Clean up
    cap.release()
    if out is not None:
        out.release()
    
    # Reset progress
    progress_bar.empty()
    
    return not stop_requested


def main():
    st.title("Football AI - Advanced Tactical Analysis")
    st.subheader("Computer Vision-based Football Analysis System")

    # Initialize session state
    if 'stop_processing' not in st.session_state:
        st.session_state['stop_processing'] = False

    # Import Football AI modules
    football_ai_modules = import_football_ai_modules()
    if not football_ai_modules.get('import_successful', False):
        st.error("Failed to import required modules. Please check error messages and installation.")
        st.code(football_ai_modules.get('error', 'Unknown error'))
        
        st.warning("""
        ### Potential fixes:
        
        1. **NumPy Version Issue**: Try downgrading NumPy:
           ```
           pip install numpy==1.24.3
           ```
        
        2. **ONNX Runtime Issue**: Try reinstalling ONNX:
           ```
           pip uninstall -y onnxruntime onnxruntime-gpu
           pip install onnxruntime
           ```
        
        3. **Dependency Issues**: Install missing packages:
           ```
           pip install 'inference[transformers]' 'inference[gaze]' 'inference[grounding-dino]'
           ```
        """)
        return
        
    st.sidebar.title("Settings")
    
    # Load default config path
    default_config_path = "config_temp.yaml"
    if not os.path.exists(default_config_path):
        default_config_path = st.sidebar.text_input("Enter path to config.yaml", value="config_temp.yaml")
    
    # Load configuration
    config = load_config(default_config_path)
    if not config:
        st.error(f"Could not load configuration from {default_config_path}")
        return
    
    # Video selection
    st.sidebar.subheader("Video Input")
    demo_selected = st.sidebar.radio(label="Video Source", options=["Upload Your Own", "Use Test Video"], horizontal=True)
    
    # File uploader
    uploaded_video = None
    video_path = None
    
    if demo_selected == "Upload Your Own":
        uploaded_video = st.sidebar.file_uploader('Upload a video file', type=['mp4','mov', 'avi', 'm4v', 'asf'])
        if uploaded_video:
            # Save uploaded video to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(uploaded_video.read())
            video_path = temp_file.name
            st.sidebar.video(video_path)
    else:
        # Use path to a local test video
        video_path_input = st.sidebar.text_input("Path to test video", value=config['video']['input_path'])
        if os.path.exists(video_path_input):
            video_path = video_path_input
            try:
                st.sidebar.video(video_path)
            except Exception as e:
                st.sidebar.warning(f"Could not display video preview: {str(e)}")
    
    # Team settings
    st.sidebar.subheader("Team Information")
    team1_name = st.sidebar.text_input(label='First Team Name', value="Team 1")
    team2_name = st.sidebar.text_input(label='Second Team Name', value="Team 2")
    
    # Create tabs for settings and visualization
    tab1, tab2, tab3 = st.tabs(["How to Use", "Team Colors & Features", "Detection Settings"])
    
    with tab1:
        st.header(':blue[Welcome to Football AI!]')
        st.subheader('Key Features:', divider='blue')
        st.markdown("""
            1. **Player Detection & Tracking**: Identifies all players, goalkeepers, referees, and the ball
            2. **Team Classification**: Automatically assigns players to their teams using AI
            3. **Pose Estimation**: Detects player poses for advanced movement analysis
            4. **Player Segmentation**: Creates precise player silhouettes for better visualization
            5. **Possession Detection**: Identifies which player has possession of the ball
            6. **Tactical Mapping**: Projects player positions onto a tactical pitch view
            7. **Ball Tracking**: Tracks ball movement and visualizes its path
        """)
        
        st.subheader('How to use:', divider='blue')
        st.markdown("""
            1. Upload your own video or specify the path to a test video
            2. Enter team names for easier identification
            3. Configure team colors in the "Team Colors & Features" tab
            4. Adjust detection settings in the "Detection Settings" tab
            5. Click "Start Processing" to begin analysis
            6. View the results with both the original video and tactical map side by side
            7. Save outputs for later review if needed
        """)
        
        st.write("Version 1.0.0")
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Team color selection
            st.subheader("Team Colors")
            team1_p_color = st.color_picker(f"{team1_name} Players", value="#00BFFF")  # Light blue
            team1_gk_color = st.color_picker(f"{team1_name} Goalkeeper", value="#FFFF00")  # Yellow
            team2_p_color = st.color_picker(f"{team2_name} Players", value="#FF1493")  # Deep pink
            team2_gk_color = st.color_picker(f"{team2_name} Goalkeeper", value="#00FF00")  # Green
            referee_color = st.color_picker("Referee", value="#FFFFFF")  # White
            
            # Store colors in session state
            st.session_state['team1_p_color'] = team1_p_color
            st.session_state['team1_gk_color'] = team1_gk_color
            st.session_state['team2_p_color'] = team2_p_color
            st.session_state['team2_gk_color'] = team2_gk_color
            st.session_state['referee_color'] = referee_color
        
        with col2:
            # Advanced features toggles
            st.subheader("Advanced Features")
            enable_pose = st.toggle("Enable Pose Estimation", value=True)
            enable_segmentation = st.toggle("Enable Player Segmentation", value=True)
            enable_possession = st.toggle("Enable Player Possession Detection", value=True)
            enable_sahi = st.toggle("Enable SAHI (Small Object Detection)", value=False)
            
            # Store feature selections in session state
            st.session_state['enable_pose'] = enable_pose
            st.session_state['enable_segmentation'] = enable_segmentation
            st.session_state['enable_possession'] = enable_possession
            st.session_state['enable_sahi'] = enable_sahi
            
            # Visualization options
            st.subheader("Visualization Options")
            show_pose = st.checkbox("Show Pose Visualization", value=True) if enable_pose else False
            show_segmentation = st.checkbox("Show Segmentation Visualization", value=True) if enable_segmentation else False
            show_possession = st.checkbox("Show Possession Visualization", value=True) if enable_possession else False
            show_ball_tracks = st.checkbox("Show Ball Tracks", value=True)
            
            # Store visualization options in session state
            st.session_state['show_pose'] = show_pose
            st.session_state['show_segmentation'] = show_segmentation
            st.session_state['show_possession'] = show_possession
            st.session_state['show_ball_tracks'] = show_ball_tracks
    
    with tab3:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Detection settings
            st.subheader("Detection Parameters")
            player_model_conf_thresh = st.slider('Player Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
            keypoints_model_conf_thresh = st.slider('Field Keypoints Confidence', min_value=0.0, max_value=1.0, value=0.5)
            
            # SAHI settings if enabled
            if enable_sahi:
                st.subheader("SAHI Settings")
                sahi_slice_rows = st.slider('SAHI Slice Rows', min_value=1, max_value=4, value=2)
                sahi_slice_cols = st.slider('SAHI Slice Columns', min_value=1, max_value=4, value=2)
                sahi_overlap = st.slider('SAHI Overlap Ratio', min_value=0.0, max_value=0.5, value=0.2)
            
        with col2:
            # Possession detection settings
            if enable_possession:
                st.subheader("Possession Detection")
                proximity_threshold = st.slider('Proximity Threshold (pixels)', min_value=10, max_value=100, value=30)
                possession_frames = st.slider('Required Possession Frames', min_value=1, max_value=10, value=3)
                coordinate_system = st.radio("Coordinate System", ["frame", "pitch"], horizontal=True)
            
            # Output settings
            st.subheader("Output Settings")
            save_output = st.checkbox("Save Output Video", value=False)
            output_filename = None
            if save_output:
                output_filename = st.text_input("Output Filename", value="football_ai_output.mp4")
            
            # Real-time processing toggle
            realtime_processing = st.checkbox("Real-time Processing (slower but smoother)", value=False)
    
    # Process control buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    processing_ready = video_path is not None
    
    with col1:
        pass
    
    with col2:
        if processing_ready:
            start_button = st.button("Start Processing", disabled=not processing_ready)
        else:
            st.warning("Please provide a valid video file to process")
            start_button = False
            
        if start_button:
            st.session_state['stop_processing'] = False
            
            # Update config from UI settings
            config['detection']['confidence_threshold'] = player_model_conf_thresh
            config['detection']['keypoint_confidence_threshold'] = keypoints_model_conf_thresh
            config['display']['show_pose'] = show_pose if 'show_pose' in locals() else False
            config['display']['show_segmentation'] = show_segmentation if 'show_segmentation' in locals() else False
            config['display']['show_possession_detection'] = show_possession if 'show_possession' in locals() else False
            config['display']['show_ball'] = show_ball_tracks if 'show_ball_tracks' in locals() else True
            config['display']['team_colors']['team_1'] = team1_p_color
            config['display']['team_colors']['team_2'] = team2_p_color
            
            # Update SAHI settings
            config['sahi']['enable'] = enable_sahi
            if enable_sahi and 'sahi_slice_rows' in locals():
                config['sahi']['slice_rows'] = sahi_slice_rows
                config['sahi']['slice_cols'] = sahi_slice_cols
                config['sahi']['overlap_ratio'] = sahi_overlap
            
            # Update possession settings
            config['possession_detection']['enable'] = enable_possession
            if enable_possession and 'proximity_threshold' in locals():
                if coordinate_system == "frame":
                    config['possession_detection']['frame_proximity_threshold'] = proximity_threshold
                else:
                    config['possession_detection']['proximity_threshold'] = proximity_threshold
                config['possession_detection']['possession_frames'] = possession_frames
                config['possession_detection']['coordinate_system'] = coordinate_system
            
            # Initialize Football AI components
            football_ai = initialize_football_ai(config, football_ai_modules)
            
            if football_ai:
                # Create a placeholder for the video display
                stframe = st.empty()
                
                # Process the video
                ui_settings = {
                    'show_pose': show_pose if 'show_pose' in locals() else False,
                    'show_segmentation': show_segmentation if 'show_segmentation' in locals() else False,
                    'enable_possession': enable_possession,
                    'show_possession': show_possession if 'show_possession' in locals() else False,
                    'enable_sahi': enable_sahi,
                    'show_ball_tracks': show_ball_tracks if 'show_ball_tracks' in locals() else True,
                    'realtime_processing': realtime_processing
                }
                
                completed = process_video(
                    video_path=video_path,
                    stframe=stframe,
                    football_ai=football_ai,
                    ui_settings=ui_settings,
                    save_output=save_output,
                    output_file=output_filename if save_output else None
                )
                
                if completed:
                    st.success("Processing completed!")
                else:
                    st.warning("Processing stopped by user.")
            else:
                st.error("Failed to initialize Football AI components.")
    
    with col3:
        if 'stop_processing' in st.session_state and not st.session_state['stop_processing'] and processing_ready:
            stop_button = st.button("Stop Processing")
            if stop_button:
                st.session_state['stop_processing'] = True
    
    # Display visualization area
    st.markdown("---")
    stframe = st.empty()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass