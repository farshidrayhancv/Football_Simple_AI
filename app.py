import os
import sys
import tempfile
import subprocess
import streamlit as st
import time
import yaml
import shutil
from PIL import Image

# Set page config
st.set_page_config(page_title="Football AI - Live Viewer", layout="wide")

# App title
st.title("Football AI - Live Viewer")
st.markdown("Upload a video and see frames as they're processed")

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
        shutil.copy2(config_path, backup_path)
        
        # Write updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return True
    except Exception as e:
        st.error(f"Failed to update config: {e}")
        return False

# Extract the latest frame from an MP4 video using ffmpeg
def extract_latest_frame(video_path, output_path):
    """Extract the last frame from a video file using ffmpeg."""
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return False
    
    try:
        # Use ffmpeg to extract the latest frame
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", video_path,  # Input file
            "-vf", "select=gte(n\\,0)",  # Select frames from the end
            "-vframes", "1",  # Extract just one frame
            "-y",  # Overwrite output file if exists
            output_path  # Output file
        ]
        
        # Run ffmpeg command
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        st.error(f"Error extracting frame: {e}")
        return False

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
        st.success(f"Video uploaded successfully")

with source_tab2:
    video_dir = "videos"
    if os.path.exists(video_dir):
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
        if video_files:
            selected_video = st.selectbox("Select a video from videos directory", video_files)
            video_path = os.path.join(video_dir, selected_video)
            st.success(f"Selected video: {video_path}")
        else:
            st.warning(f"No video files found in {video_dir} directory")
    else:
        st.warning(f"Directory {video_dir} does not exist")

# Select config file
st.header("2. Processing Configuration")
config_files = ["config.yaml", "config_temp.yaml"]
available_configs = [f for f in config_files if os.path.exists(f)]

if available_configs:
    config_path = st.selectbox("Select config file", available_configs)
    st.success(f"Using config: {config_path}")
else:
    st.error("No config files found. Please create config.yaml or config_temp.yaml")
    config_path = None

# Create output directory if not exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
temp_dir = os.path.join(output_dir, "temp")
os.makedirs(temp_dir, exist_ok=True)

# State management
if "process" not in st.session_state:
    st.session_state.process = None
if "log_file" not in st.session_state:
    st.session_state.log_file = None
if "output_file" not in st.session_state:
    st.session_state.output_file = None
if "original_video" not in st.session_state:
    st.session_state.original_video = None
if "last_frame_time" not in st.session_state:
    st.session_state.last_frame_time = 0

# Create a frame thumbnail
frame_path = os.path.join(temp_dir, "latest_frame.jpg")

# Video display section
st.header("3. Live Frame View")

# Create frame display containers
frame_display = st.empty()
frame_info = st.empty()

# Display original video for reference
if video_path and os.path.exists(video_path):
    st.header("Original Video")
    st.video(video_path)

# Runner section
st.header("4. Run Processing")

col1, col2 = st.columns(2)

# Create placeholders for output and status
log_placeholder = st.empty()
status_placeholder = st.empty()

# Start button
with col1:
    start_disabled = video_path is None or config_path is None or st.session_state.process is not None
    start_button = st.button("Start Processing", disabled=start_disabled)
    if start_button and video_path and config_path:
        # Update config file with new video path
        if update_config_video_path(config_path, video_path):
            st.success(f"Updated config to use selected video: {video_path}")
        else:
            st.error("Failed to update config file. Processing may use incorrect video.")
        
        # Create output filename
        output_basename = os.path.basename(video_path)
        output_filename = f"processed_{int(time.time())}_{output_basename}"
        output_path = os.path.join(output_dir, output_filename)
        st.session_state.output_file = output_path
        st.session_state.original_video = video_path
        
        # Create log file
        log_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.log')
        log_file.close()
        st.session_state.log_file = log_file.name
        
        # Build command
        cmd = [
            sys.executable,  # Current Python interpreter
            "main.py",
            "--config", config_path,
            "--output", output_path
        ]
        
        # Parse environment variables
        custom_env = dict(os.environ)
        custom_env["PYTHONMALLOC"] = "malloc"  # Add this to help prevent segfaults
        
        # Start subprocess
        try:
            with open(st.session_state.log_file, 'w') as f:
                st.session_state.process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=custom_env
                )
            st.success(f"Started processing with PID: {st.session_state.process.pid}")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to start process: {e}")
            st.session_state.process = None

# Stop button
with col2:
    stop_button = st.button("Stop Processing", disabled=st.session_state.process is None)
    if stop_button and st.session_state.process is not None:
        try:
            st.session_state.process.terminate()
            time.sleep(2)  # Give it time to terminate gracefully
            if st.session_state.process.poll() is None:  # If still running
                st.session_state.process.kill()  # Force kill
            st.warning("Process stopped by user")
        except Exception as e:
            st.error(f"Error stopping process: {e}")
        finally:
            st.session_state.process = None
            st.rerun()

# Display process status
if st.session_state.process is not None:
    # Check if process is still running
    if st.session_state.process.poll() is None:
        status_placeholder.info("⏳ Processing is running...")
        
        # Check if processed video exists and extract current frame
        if st.session_state.output_file and os.path.exists(st.session_state.output_file):
            # Only try to extract a frame every 2 seconds to avoid too many ffmpeg calls
            current_time = time.time()
            if current_time - st.session_state.last_frame_time > 2:
                # Extract the latest frame
                if extract_latest_frame(st.session_state.output_file, frame_path):
                    try:
                        # Display the frame
                        img = Image.open(frame_path)
                        frame_display.image(img, caption="Latest processed frame", use_column_width=True)
                        frame_info.success("✅ Showing latest processed frame")
                        # Update last frame time
                        st.session_state.last_frame_time = current_time
                    except Exception as e:
                        frame_info.error(f"Error displaying frame: {e}")
                else:
                    frame_info.warning("Waiting for processed frames...")
        
        # Read and display log
        if st.session_state.log_file and os.path.exists(st.session_state.log_file):
            try:
                with open(st.session_state.log_file, 'r') as f:
                    log_content = f.read()
                
                # Show only the last 30 lines to avoid overwhelming the UI
                lines = log_content.split('\n')
                last_lines = lines[-30:]
                
                log_placeholder.text_area("Log Output (last 30 lines)", 
                                        "\n".join(last_lines), 
                                        height=400)
                
                # Show progress if visible
                for line in reversed(lines):
                    if "Processing frames:" in line and "%" in line:
                        status_placeholder.success(line)
                        break
            except Exception as e:
                log_placeholder.error(f"Error reading log: {e}")
        
        # Refresh every 2 seconds
        time.sleep(2)
        st.rerun()
    else:
        # Process has ended
        exit_code = st.session_state.process.poll()
        if exit_code == 0:
            status_placeholder.success(f"✅ Processing completed successfully!")
        else:
            status_placeholder.error(f"❌ Processing failed with exit code: {exit_code}")
        
        # Display full log
        if st.session_state.log_file and os.path.exists(st.session_state.log_file):
            try:
                with open(st.session_state.log_file, 'r') as f:
                    log_content = f.read()
                log_placeholder.text_area("Complete Log", log_content, height=400)
            except Exception as e:
                log_placeholder.error(f"Error reading log: {e}")
        
        # Show the final frame
        if st.session_state.output_file and os.path.exists(st.session_state.output_file):
            if extract_latest_frame(st.session_state.output_file, frame_path):
                try:
                    img = Image.open(frame_path)
                    frame_display.image(img, caption="Final processed frame", use_column_width=True)
                    frame_info.success("✅ Processing complete - showing final frame")
                except Exception as e:
                    frame_info.error(f"Error displaying final frame: {e}")
        
        # Show the completed video
        if st.session_state.output_file and os.path.exists(st.session_state.output_file):
            st.header("Completed Processed Video")
            st.video(st.session_state.output_file)
        
        # Clear process state
        st.session_state.process = None

# List all processed videos
st.header("5. All Processed Videos")
if os.path.exists(output_dir):
    processed_videos = [f for f in os.listdir(output_dir) 
                       if f.endswith(('.mp4', '.mov', '.avi')) and os.path.isfile(os.path.join(output_dir, f))]
    
    if processed_videos:
        # Sort by modification time (newest first)
        processed_videos.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
        
        selected_video = st.selectbox("Select processed video", processed_videos)
        if selected_video:
            st.video(os.path.join(output_dir, selected_video))
    else:
        st.info("No processed videos found")