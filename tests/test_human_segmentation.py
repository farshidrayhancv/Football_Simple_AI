#!/usr/bin/env python3
"""
Test script for human segmentation with foreground/background controls
"""

import argparse
import sys
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from models.detector import EnhancedObjectDetector, SegmentationDetector


def test_human_segmentation(config_path, image_path, save_output=True):
    """Test and visualize human-focused segmentation on a single image"""
    # Load configuration
    config = ConfigLoader(config_path).config
    
    print(f"Testing human segmentation with image: {image_path}")
    print(f"Configuration loaded from: {config_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Get image dimensions
    height, width = image.shape[:2]
    print(f"Image size: {width}x{height}")
    
    # Initialize detector
    detector = EnhancedObjectDetector(
        model_id=config['models']['player_detection_model_id'],
        api_key=config['api_keys']['roboflow_api_key'],
        confidence_threshold=config['detection']['confidence_threshold'],
        enable_pose=True,
        enable_segmentation=True,
        sam_model=config['models']['sam_model'],
        device=config['performance']['device']
    )
    
    # Detect humans
    print("Detecting humans...")
    detections = detector.detect_categories(image)
    
    # Collect human bounding boxes
    humans = []
    if len(detections['players']) > 0:
        humans.extend(detections['players'].xyxy)
    if len(detections['goalkeepers']) > 0:
        humans.extend(detections['goalkeepers'].xyxy)
    if len(detections['referees']) > 0:
        humans.extend(detections['referees'].xyxy)
    
    print(f"Found {len(humans)} humans in the image")
    
    if not humans:
        print("No humans detected, cannot test segmentation")
        return
    
    # Initialize visualization image
    vis_image = image.copy()
    
    # Draw bounding boxes around detected humans
    for i, box in enumerate(humans):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_image, f"Human {i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Create segmentation detector
    segmenter = SegmentationDetector(
        model_name=config['models']['sam_model'],
        device=config['performance']['device']
    )
    
    # For visual comparison, test with different approaches
    # 1. Original (just bbox prompts)
    # 2. Enhanced (box + fg/bg points)
    
    # Test original approach (bbox only)
    print("\nTesting original segmentation approach (box prompts only)...")
    original_masks = []
    
    # Choose one human for demonstration
    demo_human_idx = 0  # Use first human
    if demo_human_idx < len(humans):
        demo_box = humans[demo_human_idx]
        
        # Run SAM with just box prompts
        result = segmenter.model(
            image, 
            bboxes=[demo_box.tolist()],
            verbose=False,
            device=segmenter.device
        )
        
        # Extract mask
        if result and result[0].masks is not None:
            mask_data = result[0].masks.data.cpu().numpy()
            if mask_data.ndim >= 2:
                original_masks.append(mask_data[0])
    
    # Test enhanced approach
    print("\nTesting enhanced segmentation approach (box + points)...")
    enhanced_results = []
    
    # Debug visualizations for each human
    for i, box in enumerate(humans[:min(3, len(humans))]):  # Test first 3 humans
        print(f"Processing human {i+1}...")
        mask, vis = segmenter.debug_segment_box(image, box)
        enhanced_results.append((mask, vis))
    
    # Create comparison visualization
    if original_masks and enhanced_results:
        fig, axs = plt.subplots(len(enhanced_results), 2, figsize=(12, 4*len(enhanced_results)))
        if len(enhanced_results) == 1:
            axs = [axs]  # Make iterable for single human
        
        fig.suptitle('Human Segmentation Comparison: Original vs Enhanced', fontsize=16)
        
        # Original approach
        for i in range(len(enhanced_results)):
            if i < len(original_masks):
                # Show original mask
                orig_vis = image.copy()
                orig_mask = original_masks[i]
                orig_vis[orig_mask > 0.5] = orig_vis[orig_mask > 0.5] * 0.7 + np.array([0, 0, 255]) * 0.3
                
                axs[i][0].imshow(cv2.cvtColor(orig_vis, cv2.COLOR_BGR2RGB))
                axs[i][0].set_title('Original (Box Only)')
                axs[i][0].axis('off')
                
                # Show enhanced mask
                _, enhanced_vis = enhanced_results[i]
                axs[i][1].imshow(cv2.cvtColor(enhanced_vis, cv2.COLOR_BGR2RGB))
                axs[i][1].set_title('Enhanced (Box + Points)')
                axs[i][1].axis('off')
        
        plt.tight_layout()
        
        # Save comparison
        if save_output:
            comparison_path = image_path.replace('.', '_segmentation_comparison.')
            plt.savefig(comparison_path + '.png')
            print(f"Saved comparison to: {comparison_path}.png")
        
        plt.show()
    
    # Run full detection pipeline to visualize final results
    print("\nRunning full detection pipeline...")
    detections, poses, segmentations = detector.detect_with_pose_and_segmentation(image)
    
    # Create full result visualization
    result_image = image.copy()
    
    # Apply segmentation masks
    if segmentations:
        # Draw segmentation masks for players
        if 'players' in segmentations and segmentations['players']:
            for i, mask in enumerate(segmentations['players']):
                if mask is not None:
                    color = (0, 0, 255)  # Red for players
                    mask_bool = mask > 0.5 if isinstance(mask, np.ndarray) else mask
                    result_image[mask_bool] = result_image[mask_bool] * 0.7 + np.array(color) * 0.3
        
        # Draw segmentation masks for goalkeepers
        if 'goalkeepers' in segmentations and segmentations['goalkeepers']:
            for i, mask in enumerate(segmentations['goalkeepers']):
                if mask is not None:
                    color = (0, 255, 255)  # Yellow for goalkeepers
                    mask_bool = mask > 0.5 if isinstance(mask, np.ndarray) else mask
                    result_image[mask_bool] = result_image[mask_bool] * 0.7 + np.array(color) * 0.3
        
        # Draw segmentation masks for referees
        if 'referees' in segmentations and segmentations['referees']:
            for i, mask in enumerate(segmentations['referees']):
                if mask is not None:
                    color = (255, 0, 0)  # Blue for referees
                    mask_bool = mask > 0.5 if isinstance(mask, np.ndarray) else mask
                    result_image[mask_bool] = result_image[mask_bool] * 0.7 + np.array(color) * 0.3
    
    # Draw poses if available
    if poses:
        from models.detector import PoseDetector
        pose_drawer = PoseDetector()
        
        # Draw player poses
        if 'players' in poses:
            for pose in poses['players']:
                if pose is not None:
                    result_image = pose_drawer.draw_pose(result_image, pose, (0, 255, 0))
        
        # Draw goalkeeper poses
        if 'goalkeepers' in poses:
            for pose in poses['goalkeepers']:
                if pose is not None:
                    result_image = pose_drawer.draw_pose(result_image, pose, (0, 255, 255))
        
        # Draw referee poses
        if 'referees' in poses:
            for pose in poses['referees']:
                if pose is not None:
                    result_image = pose_drawer.draw_pose(result_image, pose, (255, 0, 0))
    
    # Display the result
    cv2.imshow('Original Image', cv2.resize(image, (800, 600)))
    cv2.imshow('Human Detection', cv2.resize(vis_image, (800, 600)))
    cv2.imshow('Final Result', cv2.resize(result_image, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save results
    if save_output:
        detection_path = image_path.replace('.', '_humans_detected.')
        result_path = image_path.replace('.', '_final_result.')
        
        cv2.imwrite(detection_path + '.jpg', vis_image)
        cv2.imwrite(result_path + '.jpg', result_image)
        
        print(f"Saved detection image to: {detection_path}.jpg")
        print(f"Saved final result to: {result_path}.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test human segmentation with foreground/background controls")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--no-save", action="store_true", help="Don't save output images")
    
    args = parser.parse_args()
    
    test_human_segmentation(args.config, args.image, not args.no_save)