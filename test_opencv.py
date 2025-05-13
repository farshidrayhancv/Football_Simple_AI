#!/usr/bin/env python3
"""
Test script to verify OpenCV display functionality
"""

import cv2
import numpy as np
import sys

def test_opencv_display():
    """Test if OpenCV can create and display windows."""
    print("Testing OpenCV display functionality...")
    
    try:
        # Create a simple test image
        print("Creating test image...")
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "OpenCV Test - Press 'q' to quit", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Create window
        print("Creating window...")
        cv2.namedWindow('Test Window', cv2.WINDOW_NORMAL)
        
        # Display image
        print("Displaying image...")
        cv2.imshow('Test Window', test_image)
        
        # Wait for key press
        print("Waiting for key press (press 'q' to quit)...")
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                print("Quit key pressed")
                break
            elif key != 255:
                print(f"Key pressed: {key}")
        
        # Cleanup
        cv2.destroyAllWindows()
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Check for display
    import os
    if 'DISPLAY' not in os.environ:
        print("WARNING: No DISPLAY environment variable set")
        print("Try running: export DISPLAY=:0")
    else:
        print(f"DISPLAY is set to: {os.environ['DISPLAY']}")
    
    success = test_opencv_display()
    sys.exit(0 if success else 1)
