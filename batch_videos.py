#!/usr/bin/env python3
"""
Batch process multiple football videos
"""

import os
import glob
import argparse
import yaml


def process_videos(config_path, input_pattern, output_dir):
    """Process multiple videos matching the pattern."""
    # Find all matching videos
    video_files = glob.glob(input_pattern)
    
    if not video_files:
        print(f"No videos found matching pattern: {input_pattern}")
        return
    
    print(f"Found {len(video_files)} videos to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each video
    for i, video_path in enumerate(video_files, 1):
        print(f"\nProcessing video {i}/{len(video_files)}: {video_path}")
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_football_ai.mp4")
        
        # Update config with current video
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['video']['input_path'] = video_path
        
        # Create temporary config
        temp_config_path = f".temp_config_{i}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Run headless processing
        cmd = f"python3 main.py --config {temp_config_path} --output {output_path}"
        print(f"Running: {cmd}")
        os.system(cmd)
        
        # Cleanup temp config
        os.remove(temp_config_path)
    
    print(f"\nBatch processing complete! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Batch process football videos')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--input', type=str, required=False, default='videos/*.mp4',
                      help='Input pattern (e.g., "videos/*.mp4")')
    parser.add_argument('--output', type=str, default='output',
                      help='Output directory')
    args = parser.parse_args()
    
    process_videos(args.config, args.input, args.output)


if __name__ == "__main__":
    main()
