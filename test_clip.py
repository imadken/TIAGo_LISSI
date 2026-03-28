#!/usr/bin/env python3
"""
Test CLIP-based Open-Vocabulary Detection
Run this to verify CLIP installation and functionality
"""

import rospy
import cv2
import numpy as np
from perception_manager import PerceptionManager


def test_clip_detection():
    """Test CLIP open-vocabulary detection."""
    rospy.init_node('clip_detection_test', anonymous=True)

    print("\n" + "=" * 60)
    print("CLIP Open-Vocabulary Detection Test")
    print("=" * 60)

    # Initialize perception with CLIP enabled
    print("\nInitializing perception manager with CLIP...")
    try:
        perception = PerceptionManager(use_clip=True)
    except Exception as e:
        print("ERROR: Failed to initialize perception manager")
        print("Error: {}".format(e))
        print("\nMake sure you've installed CLIP:")
        print("  pip3 install ftfy regex tqdm")
        print("  pip3 install git+https://github.com/openai/CLIP.git")
        return

    # Wait for camera
    rospy.sleep(2)

    # Test 1: Standard YOLO detection
    print("\n" + "-" * 60)
    print("Test 1: Standard YOLO Detection (COCO classes)")
    print("-" * 60)

    yolo_detections = perception.detect_objects()
    print("YOLO detected {} objects:".format(len(yolo_detections)))
    for det in yolo_detections:
        print("  - {} at {} (conf: {:.2f})".format(
            det['class_name'], det['bbox'], det['confidence']))

    # Test 2: Open-vocabulary detection with custom object names
    print("\n" + "-" * 60)
    print("Test 2: Open-Vocabulary Detection (Custom Objects)")
    print("-" * 60)

    # Try to detect objects that might not be in COCO
    custom_objects = [
        "bottle", "mug", "cup",
        "phone", "smartphone", "mobile phone",
        "book", "notebook",
        "laptop", "computer",
        "person", "human"
    ]

    print("Looking for: {}".format(custom_objects))
    clip_detections = perception.detect_open_vocabulary(
        object_names=custom_objects,
        confidence_threshold=0.15
    )

    print("\nCLIP detected {} objects:".format(len(clip_detections)))
    for det in clip_detections:
        print("  - {} at {} (conf: {:.2f})".format(
            det['class_name'], det['bbox'], det['confidence']))

    # Test 3: Specific object search
    print("\n" + "-" * 60)
    print("Test 3: Specific Object Search")
    print("-" * 60)

    specific_objects = ["bottle", "water bottle", "plastic bottle"]
    print("Searching specifically for: {}".format(specific_objects))

    bottle_detections = perception.detect_open_vocabulary(
        object_names=specific_objects,
        confidence_threshold=0.15
    )

    print("\nFound {} matching objects:".format(len(bottle_detections)))
    for det in bottle_detections:
        print("  - {} at {} (conf: {:.2f})".format(
            det['class_name'], det['bbox'], det['confidence']))

    # Save visualization
    print("\n" + "-" * 60)
    print("Saving Visualizations")
    print("-" * 60)

    image = perception.get_latest_rgb()

    if yolo_detections:
        yolo_vis = perception.draw_detections(image, yolo_detections)
        cv2.imwrite('/workspace/yolo_detections.jpg', yolo_vis)
        print("YOLO visualization: /workspace/yolo_detections.jpg")

    if clip_detections:
        clip_vis = perception.draw_detections(image, clip_detections)
        cv2.imwrite('/workspace/clip_detections.jpg', clip_vis)
        print("CLIP visualization: /workspace/clip_detections.jpg")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("YOLO detected: {} objects (COCO classes only)".format(len(yolo_detections)))
    print("CLIP detected: {} objects (open vocabulary)".format(len(clip_detections)))
    print("\nCLIP advantages:")
    print("  ✓ Can detect ANY object by name")
    print("  ✓ Not limited to COCO classes")
    print("  ✓ Works with synonyms (bottle, water bottle, etc.)")
    print("\nNext: Try 'grab the [any object]' in embodied agent!")
    print("=" * 60)


if __name__ == '__main__':
    try:
        test_clip_detection()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print("\nERROR: Test failed")
        print("Error: {}".format(e))
        import traceback
        traceback.print_exc()
