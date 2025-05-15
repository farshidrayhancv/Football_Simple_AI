#!/usr/bin/env python3
"""
Test processing resolution functionality
"""

import cv2
import argparse
import sys
import os
import numpy as np
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from models.detector import EnhancedObjectDetector


def test_resolution_comparison(config_path, image_path):
    """Compare processing at different resolutions."""
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
    
    original_height, original_width = frame.shape[:2]
    print(f"Original image size: {original_width}x{original_height}")
    
    # Test at different resolutions
    resolutions = [
        (640, 640),
        (960, 540),
        (1280, 720),
        (1920, 1080),
        (original_width, original_height)  # Native
    ]
    
    results = []
    
    for width, height in resolutions:
        print(f"\nTesting at {width}x{height}...")
        
        # Resize frame
        if (width, height) != (original_width, original_height):
            test_frame = cv2.resize(frame, (width, height))
            scale_x = original_width / width
            scale_y = original_height / height
        else:
            test_frame = frame
            scale_x = scale_y = 1.0
        
        # Measure processing time
        start_time = time.time()
        detections, _, _ = detector.detect_with_pose_and_segmentation(test_frame)
        processing_time = time.time() - start_time
        
        # Scale detections back to original resolution
        if scale_x != 1.0 or scale_y != 1.0:
            for category in detections:
                if isinstance(detections[category], sv.Detections) and len(detections[category]) > 0:
                    detections[category].xyxy[:, [0, 2]] *= scale_x
                    detections[category].xyxy[:, [1, 3]] *= scale_y
        
        # Count detections
        total_detections = sum(len(d) for d in detections.values() if hasattr(d, '__len__'))
        
        results.append({
            'resolution': f'{width}x{height}',
            'processing_time': processing_time,
            'total_detections': total_detections,
            'players': len(detections['players']),
            'goalkeepers': len(detections['goalkeepers']),
            'ball': len(detections['ball']),
            'referees': len(detections['referees'])
        })
        
        print(f"  Processing time: {processing_time:.3f}s")
        print(f"  Total detections: {total_detections}")
    
    # Print comparison table
    print("\n=== Resolution Comparison ===")
    print(f"{'Resolution':12} {'Time (s)':10} {'Total':8} {'Players':8} {'Goalkeepers':12} {'Ball':6} {'Referees':8}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['resolution']:12} {result['processing_time']:10.3f} "
              f"{result['total_detections']:8} {result['players']:8} "
              f"{result['goalkeepers']:12} {result['ball']:6} {result['referees']:8}")
    
    # Find optimal resolution
    best_ratio = max(results, key=lambda x: x['total_detections'] / x['processing_time'])
    print(f"\nBest detection/time ratio: {best_ratio['resolution']}")
    
    # Visualize comparison
    visualize_resolution_comparison(frame, results, image_path)


def visualize_resolution_comparison(original_frame, results, image_path):
    """Create visual comparison of different resolutions."""
    # Create grid visualization
    grid_rows = 2
    grid_cols = 3
    cell_width = 640
    cell_height = 360
    
    grid_width = cell_width * grid_cols
    grid_height = cell_height * grid_rows
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Place results in grid
    for i, result in enumerate(results[:6]):  # Only first 6 resolutions
        row = i // grid_cols
        col = i % grid_cols
        
        # Process frame at this resolution
        res_str = result['resolution']
        width, height = map(int, res_str.split('x'))
        
        if (width, height) != original_frame.shape[1::-1]:
            processed_frame = cv2.resize(original_frame, (width, height))
        else:
            processed_frame = original_frame.copy()
        
        # Resize to cell size
        cell_frame = cv2.resize(processed_frame, (cell_width, cell_height))
        
        # Add text overlay
        overlay = cell_frame.copy()
        cv2.rectangle(overlay, (0, 0), (cell_width, 60), (0, 0, 0), -1)
        cell_frame = cv2.addWeighted(cell_frame, 0.7, overlay, 0.3, 0)
        
        cv2.putText(cell_frame, res_str, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(cell_frame, f"Time: {result['processing_time']:.3f}s", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(cell_frame, f"Detections: {result['total_detections']}", (cell_width - 150, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Place in grid
        y1 = row * cell_height
        y2 = (row + 1) * cell_height
        x1 = col * cell_width
        x2 = (col + 1) * cell_width
        grid[y1:y2, x1:x2] = cell_frame
    
    # Save comparison
    output_path = image_path.replace('.', '_resolution_comparison.')
    cv2.imwrite(output_path, grid)
    print(f"\nSaved comparison to: {output_path}")
    
    # Display
    cv2.imshow('Resolution Comparison', cv2.resize(grid, (1920, 1080)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_processing_resolution(config_path, image_path):
    """Test with configured processing resolution."""
    # Load config
    config = ConfigLoader(config_path).config
    processing_res = config.get('processing', {}).get('resolution', None)
    
    if not processing_res:
        print("No processing resolution configured in config file")
        return
    
    print(f"Testing with processing resolution: {processing_res[0]}x{processing_res[1]}")
    
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
    
    original_height, original_width = frame.shape[:2]
    print(f"Original image size: {original_width}x{original_height}")
    
    # Resize to processing resolution
    proc_width, proc_height = processing_res
    processing_frame = cv2.resize(frame, (proc_width, proc_height))
    
    # Detect
    print(f"Processing at {proc_width}x{proc_height}...")
    start_time = time.time()
    detections, poses, segmentations = detector.detect_with_pose_and_segmentation(processing_frame)
    processing_time = time.time() - start_time
    
    print(f"Processing time: {processing_time:.3f}s")
    
    # Scale back to original resolution
    scale_x = original_width / proc_width
    scale_y = original_height / proc_height
    
    # Scale detections
    for category in detections:
        if isinstance(detections[category], sv.Detections) and len(detections[category]) > 0:
            detections[category].xyxy[:, [0, 2]] *= scale_x
            detections[category].xyxy[:, [1, 3]] *= scale_y
    
    # Print results
    print(f"\nDetection Results:")
    print(f"  Players: {len(detections['players'])}")
    print(f"  Goalkeepers: {len(detections['goalkeepers'])}")
    print(f"  Ball: {len(detections['ball'])}")
    print(f"  Referees: {len(detections['referees'])}")
    
    # Visualize
    from visualization.annotators import FootballAnnotator
    annotator = FootballAnnotator(config)
    
    if segmentations:
        annotated = annotator.annotate_frame_with_pose_and_segmentation(
            frame, detections, poses, segmentations
        )
    elif poses:
        annotated = annotator.annotate_frame_with_pose(frame, detections, poses)
    else:
        annotated = annotator.annotate_frame(frame, detections)
    
    # Add resolution info
    cv2.putText(annotated, f"Original: {original_width}x{original_height}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated, f"Processed at: {proc_width}x{proc_height}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Save and display
    output_path = image_path.replace('.', f'_processed_{proc_width}x{proc_height}.')
    cv2.imwrite(output_path, annotated)
    print(f"\nSaved to: {output_path}")
    
    cv2.imshow('Processing Resolution Test', cv2.resize(annotated, (1280, 720)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test processing resolution')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--compare', action='store_true', help='Compare different resolutions')
    
    args = parser.parse_args()
    
    if args.compare:
        test_resolution_comparison(args.config, args.image)
    else:
        test_processing_resolution(args.config, args.image)