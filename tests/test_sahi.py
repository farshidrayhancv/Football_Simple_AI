#!/usr/bin/env python3
"""
Test SAHI functionality
"""

import cv2
import argparse
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from models.detector import EnhancedObjectDetector
from processing.sahi_processor import SAHIProcessor
from visualization.annotators import FootballAnnotator


def test_sahi_on_frame(config_path, image_path):
    """Test SAHI on a single frame."""
    # Load config
    config = ConfigLoader(config_path).config
    
    # Initialize detector
    detector = EnhancedObjectDetector(
        model_id=config['models']['player_detection_model_id'],
        api_key=config['api_keys']['roboflow_api_key'],
        confidence_threshold=config['detection']['confidence_threshold'],
        enable_pose=config.get('display', {}).get('show_pose', True),
        enable_segmentation=config.get('display', {}).get('show_segmentation', True),
        device=config['performance']['device']
    )
    
    # Initialize SAHI processor
    sahi_processor = SAHIProcessor(config)
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Test with SAHI
    if config.get('sahi', {}).get('enable', False):
        print(f"\nTesting with SAHI ({config['sahi']['slice_rows']}x{config['sahi']['slice_cols']} slices)...")
        detections, poses, segmentations = sahi_processor.process_with_sahi(
            frame, detector,
            enable_pose=config.get('display', {}).get('show_pose', True),
            enable_segmentation=config.get('display', {}).get('show_segmentation', True)
        )
    else:
        print("\nSAHI disabled, using standard detection...")
        detections, poses, segmentations = detector.detect_with_pose_and_segmentation(frame)
    
    # Print results
    print(f"\nDetection Results:")
    print(f"  Players: {len(detections['players'])}")
    print(f"  Goalkeepers: {len(detections['goalkeepers'])}")
    print(f"  Ball: {len(detections['ball'])}")
    print(f"  Referees: {len(detections['referees'])}")
    
    # Visualize
    annotator = FootballAnnotator(config)
    
    if segmentations:
        annotated = annotator.annotate_frame_with_pose_and_segmentation(
            frame, detections, poses, segmentations
        )
    elif poses:
        annotated = annotator.annotate_frame_with_pose(frame, detections, poses)
    else:
        annotated = annotator.annotate_frame(frame, detections)
    
    # Add SAHI grid visualization if enabled
    if config.get('sahi', {}).get('enable', False):
        annotated = draw_sahi_grid(annotated, config)
    
    # Save and display
    output_path = image_path.replace('.', '_sahi.')
    cv2.imwrite(output_path, annotated)
    print(f"\nSaved to: {output_path}")
    
    # Display
    cv2.imshow('SAHI Test', cv2.resize(annotated, (1280, 720)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_sahi_grid(frame, config):
    """Draw SAHI grid overlay on frame."""
    sahi_config = config.get('sahi', {})
    rows = sahi_config.get('slice_rows', 2)
    cols = sahi_config.get('slice_cols', 2)
    overlap = sahi_config.get('overlap_ratio', 0.2)
    
    height, width = frame.shape[:2]
    slice_height = height // rows
    slice_width = width // cols
    
    # Draw grid lines
    overlay = frame.copy()
    
    # Vertical lines
    for col in range(1, cols):
        x = col * slice_width
        cv2.line(overlay, (x, 0), (x, height), (255, 255, 0), 2)
    
    # Horizontal lines
    for row in range(1, rows):
        y = row * slice_height
        cv2.line(overlay, (0, y), (width, y), (255, 255, 0), 2)
    
    # Draw overlap regions
    overlap_h = int(slice_height * overlap)
    overlap_w = int(slice_width * overlap)
    
    for row in range(rows):
        for col in range(cols):
            x1 = col * slice_width
            y1 = row * slice_height
            x2 = (col + 1) * slice_width
            y2 = (row + 1) * slice_height
            
            # Draw overlap regions in semi-transparent yellow
            if col > 0:  # Left overlap
                cv2.rectangle(overlay, (x1 - overlap_w, y1), (x1, y2), (0, 255, 255), -1)
            if row > 0:  # Top overlap
                cv2.rectangle(overlay, (x1, y1 - overlap_h), (x2, y1), (0, 255, 255), -1)
    
    # Blend with original
    frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
    
    # Add text
    cv2.putText(frame, f"SAHI Grid: {rows}x{cols}", (10, frame.shape[0] - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Overlap: {overlap*100:.0f}%", (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    return frame


def test_sahi_comparison(config_path, image_path):
    """Compare detection with and without SAHI."""
    # Load config
    config = ConfigLoader(config_path).config
    
    # Initialize detector
    detector = EnhancedObjectDetector(
        model_id=config['models']['player_detection_model_id'],
        api_key=config['api_keys']['roboflow_api_key'],
        confidence_threshold=config['detection']['confidence_threshold'],
        device=config['performance']['device']
    )
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    # Test without SAHI
    print("Testing without SAHI...")
    detections_normal, _, _ = detector.detect_with_pose_and_segmentation(frame)
    
    # Test with SAHI
    print("\nTesting with SAHI...")
    config['sahi']['enable'] = True
    sahi_processor = SAHIProcessor(config)
    detections_sahi, _, _ = sahi_processor.process_with_sahi(frame, detector, False, False)
    
    # Compare results
    print("\n=== Detection Comparison ===")
    print(f"Without SAHI - Players: {len(detections_normal['players'])}")
    print(f"With SAHI    - Players: {len(detections_sahi['players'])}")
    print(f"\nWithout SAHI - Total: {sum(len(d) for d in detections_normal.values())}")
    print(f"With SAHI    - Total: {sum(len(d) for d in detections_sahi.values())}")
    
    # Visualize side by side
    annotator = FootballAnnotator(config)
    
    normal_annotated = annotator.annotate_frame(frame.copy(), detections_normal)
    sahi_annotated = annotator.annotate_frame(frame.copy(), detections_sahi)
    
    # Add labels
    cv2.putText(normal_annotated, "Without SAHI", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(sahi_annotated, "With SAHI", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Combine
    combined = np.hstack([normal_annotated, sahi_annotated])
    
    # Save and display
    output_path = image_path.replace('.', '_sahi_comparison.')
    cv2.imwrite(output_path, combined)
    print(f"\nSaved comparison to: {output_path}")
    
    # Display
    cv2.imshow('SAHI Comparison', cv2.resize(combined, (1920, 540)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SAHI functionality')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--compare', action='store_true', help='Compare with/without SAHI')
    
    args = parser.parse_args()
    
    if args.compare:
        test_sahi_comparison(args.config, args.image)
    else:
        test_sahi_on_frame(args.config, args.image)