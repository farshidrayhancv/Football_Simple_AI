#!/usr/bin/env python3
"""
Test single frame detection for debugging
"""

import cv2
import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from models.detector import ObjectDetector, FieldDetector
from visualization.annotators import FootballAnnotator


def test_single_frame(config_path, image_path):
    """Test detection on a single frame."""
    # Load config
    config = ConfigLoader(config_path).config
    
    # Initialize models
    player_detector = ObjectDetector(
        model_id=config['models']['player_detection_model_id'],
        api_key=config['api_keys']['roboflow_api_key'],
        confidence_threshold=config['detection']['confidence_threshold']
    )
    
    field_detector = FieldDetector(
        model_id=config['models']['field_detection_model_id'],
        api_key=config['api_keys']['roboflow_api_key'],
        confidence_threshold=config['detection']['confidence_threshold']
    )
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    # Run player detection
    print("Running player detection...")
    detections = player_detector.detect_categories(frame)
    
    print(f"Detected:")
    print(f"  Players: {len(detections['players'])}")
    print(f"  Goalkeepers: {len(detections['goalkeepers'])}")
    print(f"  Ball: {len(detections['ball'])}")
    print(f"  Referees: {len(detections['referees'])}")
    
    # Run field detection
    print("\nRunning field detection...")
    keypoints = field_detector.detect_keypoints(frame)
    print(f"Detected {len(keypoints.xy[0])} keypoints")
    
    # Visualize
    annotator = FootballAnnotator(config)
    annotated = annotator.annotate_frame(frame, detections)
    
    # Draw keypoints
    for i, (point, conf) in enumerate(zip(keypoints.xy[0], keypoints.confidence[0])):
        x, y = point.astype(int)
        color = (0, 255, 0) if conf > config['detection']['keypoint_confidence_threshold'] else (0, 0, 255)
        cv2.circle(annotated, (x, y), 5, color, -1)
        cv2.putText(annotated, f"{i}", (x+5, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save and display
    output_path = image_path.replace('.', '_detected.')
    cv2.imwrite(output_path, annotated)
    print(f"\nSaved to: {output_path}")
    
    # Display
    cv2.imshow('Detection Test', cv2.resize(annotated, (1280, 720)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test single frame detection')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--image', required=True, help='Path to test image')
    
    args = parser.parse_args()
    test_single_frame(args.config, args.image)
