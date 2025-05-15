#!/usr/bin/env python3
"""
Test script for adaptive padding with comprehensive visualization
"""

import argparse
import sys
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from models.detector import EnhancedObjectDetector
from processing.sahi_processor import SAHIProcessor
from visualization.annotators import FootballAnnotator


def test_adaptive_padding(config_path, image_path, compare=True, save_output=True):
    """Test and visualize adaptive padding on a single image"""
    # Load configuration
    config = ConfigLoader(config_path).config
    
    print(f"Testing with image: {image_path}")
    print(f"Configuration loaded from: {config_path}")
    
    # Initialize detector
    detector = EnhancedObjectDetector(
        model_id=config['models']['player_detection_model_id'],
        api_key=config['api_keys']['roboflow_api_key'],
        confidence_threshold=config['detection']['confidence_threshold'],
        enable_pose=config.get('display', {}).get('show_pose', True),
        enable_segmentation=config.get('display', {}).get('show_segmentation', True),
        device=config['performance']['device']
    )
    
    # Initialize annotator and SAHI processor
    annotator = FootballAnnotator(config)
    sahi_processor = SAHIProcessor(config)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Get original image dimensions
    original_height, original_width = image.shape[:2]
    print(f"Image size: {original_width}x{original_height}")
    
    if compare:
        # Run detection with and without adaptive padding
        # To do this, we'll store the original method temporarily
        original_apply_padding = detector._apply_adaptive_padding
        
        # Test both approaches
        results = {}
        
        # Test 1: With fixed padding (old method)
        print("\nTesting with fixed padding...")
        # Override adaptive padding method with a simple fixed padding
        detector._apply_adaptive_padding = lambda boxes, frame_shape, base_padding, padding_ratio: np.array([
            [max(0, x1 - base_padding), max(0, y1 - base_padding), 
             min(frame_shape[1], x2 + base_padding), min(frame_shape[0], y2 + base_padding)] 
            for x1, y1, x2, y2 in boxes
        ])
        
        start_time = time.time()
        detections_fixed, poses_fixed, segmentations_fixed = detector.detect_with_pose_and_segmentation(image.copy())
        time_fixed = time.time() - start_time
        print(f"Fixed padding processing time: {time_fixed:.3f} seconds")
        
        results['fixed'] = {
            'detections': detections_fixed,
            'poses': poses_fixed,
            'segmentations': segmentations_fixed,
            'time': time_fixed
        }
        
        # Test 2: With adaptive padding (new method)
        print("\nTesting with adaptive padding...")
        # Restore original method
        detector._apply_adaptive_padding = original_apply_padding
        
        start_time = time.time()
        detections_adaptive, poses_adaptive, segmentations_adaptive = detector.detect_with_pose_and_segmentation(image.copy())
        time_adaptive = time.time() - start_time
        print(f"Adaptive padding processing time: {time_adaptive:.3f} seconds")
        
        results['adaptive'] = {
            'detections': detections_adaptive,
            'poses': poses_adaptive,
            'segmentations': segmentations_adaptive,
            'time': time_adaptive
        }
        
        # Create comparison visualization
        visualize_comparison(image, results, annotator, config, image_path if save_output else None)
        
    else:
        # Just run with adaptive padding and SAHI if enabled
        if config.get('sahi', {}).get('enable', False):
            print("\nRunning with SAHI and adaptive padding...")
            start_time = time.time()
            detections, poses, segmentations = sahi_processor.process_with_sahi(
                image.copy(), detector, 
                enable_pose=config.get('display', {}).get('show_pose', True),
                enable_segmentation=config.get('display', {}).get('show_segmentation', True)
            )
            processing_time = time.time() - start_time
            print(f"SAHI + adaptive padding processing time: {processing_time:.3f} seconds")
        else:
            print("\nRunning with adaptive padding (SAHI disabled)...")
            start_time = time.time()
            detections, poses, segmentations = detector.detect_with_pose_and_segmentation(image.copy())
            processing_time = time.time() - start_time
            print(f"Adaptive padding processing time: {processing_time:.3f} seconds")
        
        # Visualize results
        visualize_results(image, detections, poses, segmentations, annotator, config, image_path if save_output else None)


def visualize_comparison(image, results, annotator, config, save_path=None):
    """Create side-by-side comparison of fixed vs adaptive padding"""
    # Create two side-by-side frames
    img_fixed = image.copy()
    img_adaptive = image.copy()
    
    # Annotate both images
    if config.get('display', {}).get('show_segmentation', True):
        img_fixed = annotator.annotate_frame_with_pose_and_segmentation(
            img_fixed, 
            results['fixed']['detections'], 
            results['fixed']['poses'], 
            results['fixed']['segmentations']
        )
        
        img_adaptive = annotator.annotate_frame_with_pose_and_segmentation(
            img_adaptive, 
            results['adaptive']['detections'], 
            results['adaptive']['poses'], 
            results['adaptive']['segmentations']
        )
    elif config.get('display', {}).get('show_pose', True):
        img_fixed = annotator.annotate_frame_with_pose(
            img_fixed, 
            results['fixed']['detections'], 
            results['fixed']['poses']
        )
        
        img_adaptive = annotator.annotate_frame_with_pose(
            img_adaptive, 
            results['adaptive']['detections'], 
            results['adaptive']['poses']
        )
    else:
        img_fixed = annotator.annotate_frame(img_fixed, results['fixed']['detections'])
        img_adaptive = annotator.annotate_frame(img_adaptive, results['adaptive']['detections'])
    
    # Calculate stats
    stats_fixed = calculate_stats(results['fixed'])
    stats_adaptive = calculate_stats(results['adaptive'])
    
    # Add stats text to each image
    y_offset = 30
    for key, value in stats_fixed.items():
        text = f"{key}: {value}"
        cv2.putText(img_fixed, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img_fixed, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
    
    # Also display processing time
    cv2.putText(img_fixed, f"Time: {results['fixed']['time']:.3f}s", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img_fixed, f"Time: {results['fixed']['time']:.3f}s", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add title
    cv2.putText(img_fixed, "Fixed Padding", (img_fixed.shape[1]//2 - 100, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(img_fixed, "Fixed Padding", (img_fixed.shape[1]//2 - 100, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    y_offset = 30
    for key, value in stats_adaptive.items():
        text = f"{key}: {value}"
        cv2.putText(img_adaptive, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img_adaptive, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
    
    # Also display processing time
    cv2.putText(img_adaptive, f"Time: {results['adaptive']['time']:.3f}s", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img_adaptive, f"Time: {results['adaptive']['time']:.3f}s", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add title
    cv2.putText(img_adaptive, "Adaptive Padding", (img_adaptive.shape[1]//2 - 120, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(img_adaptive, "Adaptive Padding", (img_adaptive.shape[1]//2 - 120, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Concatenate images side by side
    h1, w1 = img_fixed.shape[:2]
    h2, w2 = img_adaptive.shape[:2]
    
    # Create side-by-side image
    combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined[:h1, :w1] = img_fixed
    combined[:h2, w1:w1+w2] = img_adaptive
    
    # Add dividing line
    cv2.line(combined, (w1, 0), (w1, combined.shape[0]), (255, 255, 255), 2)
    
    # Add comparison text at the top
    improvement_text = f"Improved detections: {stats_adaptive['Total Detections'] - stats_fixed['Total Detections']}"
    cv2.putText(combined, improvement_text, (combined.shape[1]//2 - 150, combined.shape[0] - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(combined, improvement_text, (combined.shape[1]//2 - 150, combined.shape[0] - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Create zoomed-in comparisons of interesting players
    if config.get('display', {}).get('show_pose', True) or config.get('display', {}).get('show_segmentation', True):
        create_zoom_comparison(image, results, save_path)
    
    # Display the result
    cv2.imshow('Fixed vs Adaptive Padding', cv2.resize(combined, (1600, 900)))
    cv2.waitKey(0)
    
    # Save combined image
    if save_path:
        output_path = save_path.replace('.', '_comparison.')
        cv2.imwrite(output_path, combined)
        print(f"Comparison saved to: {output_path}")


def create_zoom_comparison(image, results, save_path=None):
    """Create detailed zoom comparisons of pose estimation and segmentation"""
    # Find some interesting examples by comparing fixed and adaptive padding results
    adaptive_detections = results['adaptive']['detections']
    fixed_detections = results['fixed']['detections']
    
    # Get all players and goalkeepers
    adaptive_humans = []
    if len(adaptive_detections['players']) > 0:
        adaptive_humans.extend(adaptive_detections['players'].xyxy)
    if len(adaptive_detections['goalkeepers']) > 0:
        adaptive_humans.extend(adaptive_detections['goalkeepers'].xyxy)
    
    fixed_humans = []
    if len(fixed_detections['players']) > 0:
        fixed_humans.extend(fixed_detections['players'].xyxy)
    if len(fixed_detections['goalkeepers']) > 0:
        fixed_humans.extend(fixed_detections['goalkeepers'].xyxy)
    
    # If no humans detected, just return
    if not adaptive_humans or not fixed_humans:
        return
    
    # Find a few players to compare (preferably small/distant ones)
    # Sort by area (smallest first)
    adaptive_humans = sorted(adaptive_humans, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
    
    # Get the smallest players
    zoom_boxes = adaptive_humans[:min(3, len(adaptive_humans))]
    
    # Create figure with subplots
    fig, axs = plt.subplots(len(zoom_boxes), 2, figsize=(12, 4*len(zoom_boxes)))
    if len(zoom_boxes) == 1:
        axs = [axs]  # Make it iterable for a single player
    
    fig.suptitle('Zoomed Comparison: Fixed vs Adaptive Padding', fontsize=16)
    
    # For each box, show fixed vs adaptive
    for i, box in enumerate(zoom_boxes):
        # Expand the box a bit for context
        x1, y1, x2, y2 = box
        width, height = x2-x1, y2-y1
        x1 = max(0, x1 - width//2)
        y1 = max(0, y1 - height//2)
        x2 = min(image.shape[1], x2 + width//2)
        y2 = min(image.shape[0], y2 + height//2)
        
        # Extract crops from the annotated images
        fixed_crop = results['fixed'].get('poses_visual', image.copy())[int(y1):int(y2), int(x1):int(x2)]
        adaptive_crop = results['adaptive'].get('poses_visual', image.copy())[int(y1):int(y2), int(x1):int(x2)]
        
        # Display crops
        axs[i][0].imshow(cv2.cvtColor(fixed_crop, cv2.COLOR_BGR2RGB))
        axs[i][0].set_title('Fixed Padding')
        axs[i][0].axis('off')
        
        axs[i][1].imshow(cv2.cvtColor(adaptive_crop, cv2.COLOR_BGR2RGB))
        axs[i][1].set_title('Adaptive Padding')
        axs[i][1].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        zoom_path = save_path.replace('.', '_zoom_comparison.')
        plt.savefig(zoom_path)
        print(f"Zoom comparison saved to: {zoom_path}")
    
    plt.show()


def visualize_results(image, detections, poses, segmentations, annotator, config, save_path=None):
    """Visualize detection, pose, and segmentation results"""
    # Make copies for different visualizations
    detection_img = image.copy()
    result_img = image.copy()
    
    # 1. Detections only
    detection_img = annotator.annotate_frame(detection_img, detections)
    
    # 2. Full result with pose estimation and segmentation if enabled
    if config.get('display', {}).get('show_segmentation', True):
        result_img = annotator.annotate_frame_with_pose_and_segmentation(
            result_img, detections, poses, segmentations
        )
    elif config.get('display', {}).get('show_pose', True):
        result_img = annotator.annotate_frame_with_pose(
            result_img, detections, poses
        )
    
    # Add stats
    stats = calculate_stats({'detections': detections, 'poses': poses, 'segmentations': segmentations})
    
    # Add stats text
    y_offset = 30
    for key, value in stats.items():
        text = f"{key}: {value}"
        cv2.putText(result_img, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(result_img, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        y_offset += 30
    
    # Show configurations used
    features = []
    if config.get('sahi', {}).get('enable', False):
        features.append("SAHI")
    if config.get('display', {}).get('show_pose', False):
        features.append("Pose")
    if config.get('display', {}).get('show_segmentation', False):
        features.append("Segmentation")
    
    feature_text = "Features: " + ", ".join(features)
    cv2.putText(result_img, feature_text, (10, result_img.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(result_img, feature_text, (10, result_img.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    
    # Display results
    cv2.imshow('Detection Only', cv2.resize(detection_img, (1280, 720)))
    cv2.imshow('Full Result', cv2.resize(result_img, (1280, 720)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save results
    if save_path:
        detection_path = save_path.replace('.', '_detection.')
        result_path = save_path.replace('.', '_full_result.')
        
        cv2.imwrite(detection_path, detection_img)
        cv2.imwrite(result_path, result_img)
        
        print(f"Detection image saved to: {detection_path}")
        print(f"Full result saved to: {result_path}")


def calculate_stats(results):
    """Calculate statistics from results"""
    stats = {}
    
    detections = results['detections']
    
    # Count detections by category
    stats['Players'] = len(detections['players'])
    stats['Goalkeepers'] = len(detections['goalkeepers'])
    stats['Referees'] = len(detections['referees'])
    stats['Ball'] = len(detections['ball'])
    stats['Total Detections'] = sum(len(detections[k]) for k in ['players', 'goalkeepers', 'referees', 'ball'])
    
    # Add pose stats if available
    if 'poses' in results and results['poses']:
        poses = results['poses']
        player_poses = sum(1 for p in poses.get('players', []) if p is not None)
        goalkeeper_poses = sum(1 for p in poses.get('goalkeepers', []) if p is not None)
        referee_poses = sum(1 for p in poses.get('referees', []) if p is not None)
        
        stats['Player Poses'] = player_poses
        stats['Goalkeeper Poses'] = goalkeeper_poses
        stats['Referee Poses'] = referee_poses
        stats['Total Poses'] = player_poses + goalkeeper_poses + referee_poses
    
    # Add segmentation stats if available
    if 'segmentations' in results and results['segmentations']:
        segmentations = results['segmentations']
        player_segs = sum(1 for s in segmentations.get('players', []) if s is not None)
        goalkeeper_segs = sum(1 for s in segmentations.get('goalkeepers', []) if s is not None)
        referee_segs = sum(1 for s in segmentations.get('referees', []) if s is not None)
        
        stats['Player Segments'] = player_segs
        stats['Goalkeeper Segments'] = goalkeeper_segs
        stats['Referee Segments'] = referee_segs
        stats['Total Segments'] = player_segs + goalkeeper_segs + referee_segs
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test adaptive padding on a single image")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--compare", action="store_true", help="Compare fixed vs adaptive padding")
    parser.add_argument("--no-save", action="store_true", help="Don't save output images")
    
    args = parser.parse_args()
    
    test_adaptive_padding(args.config, args.image, args.compare, not args.no_save)