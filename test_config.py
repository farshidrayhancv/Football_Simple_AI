#!/usr/bin/env python3
"""
Test script to verify configuration and API keys
"""

import os
import sys
import yaml
import argparse
from transformers import AutoProcessor


def test_config(config_path):
    """Test the configuration file and API keys."""
    print("Testing configuration...")
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Config file loaded successfully")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False
    
    # Test HuggingFace token
    try:
        hf_token = config['api_keys']['huggingface_token']
        if hf_token == "YOUR_HUGGINGFACE_TOKEN":
            print("✗ HuggingFace token not configured (still has placeholder value)")
            return False
        
        # Try to load a simple model to test the token
        print("Testing HuggingFace token...")
        processor = AutoProcessor.from_pretrained(
            "bert-base-uncased",  # Using a simple model for testing
            use_auth_token=hf_token
        )
        print("✓ HuggingFace token is valid")
    except Exception as e:
        print(f"✗ HuggingFace token error: {e}")
        print("  Please check your token at: https://huggingface.co/settings/tokens")
        return False
    
    # Test Roboflow API key
    try:
        rf_key = config['api_keys']['roboflow_api_key']
        if rf_key == "YOUR_ROBOFLOW_API_KEY":
            print("✗ Roboflow API key not configured (still has placeholder value)")
            return False
        print("✓ Roboflow API key configured")
    except Exception as e:
        print(f"✗ Roboflow API key error: {e}")
        return False
    
    # Test video path
    try:
        video_path = config['video']['input_path']
        if os.path.exists(video_path):
            print(f"✓ Video file found: {video_path}")
        else:
            print(f"✗ Video file not found: {video_path}")
            return False
    except Exception as e:
        print(f"✗ Video path error: {e}")
        return False
    
    print("\nAll tests passed! Your configuration is ready.")
    return True


def main():
    parser = argparse.ArgumentParser(description='Test Football AI Configuration')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    args = parser.parse_args()
    
    success = test_config(args.config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
