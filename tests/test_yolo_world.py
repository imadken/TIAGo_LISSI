#!/usr/bin/env python3
"""
Test YOLO-World Open-Vocabulary Detection
Quick test to verify YOLO-World works with TIAGo
"""

import rospy
import cv2
from tiago_lissi.perception.perception_manager_v2 import PerceptionManager


def test_yolo_world():
    """Test YOLO-World detection."""
    rospy.init_node('yolo_world_test', anonymous=True)

    print("\n" + "=" * 60)
    print("YOLO-World Open-Vocabulary Detection Test")
    print("=" * 60)

    # Initialize with YOLO-World
    print("\nInitializing perception with YOLO-World...")
    try:
        perception = PerceptionManager(use_yolo_world=True)
    except Exception as e:
        print("ERROR: Failed to initialize perception")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return

    rospy.sleep(2)

    # Test objects
    test_objects = [
        'bottle', 'cup', 'mug',
        'phone', 'smartphone', 'cell phone',
        'laptop', 'computer',
        'book', 'notebook',
        'person', 'human',
        'box', 'package'
    ]

    print(f"\nLooking for: {test_objects[:6]}...")
    print(f"             {test_objects[6:]}...")

    # Detect
    detections = perception.detect_open_vocabulary(
        object_names=test_objects,
        confidence_threshold=0.1
    )

    print(f"\n✓ YOLO-World detected {len(detections)} objects:")
    for det in detections:
        print(f"  - {det['class_name']}: {det['confidence']:.2f} at {det['bbox']}")

    # Save visualization
    image = perception.get_latest_rgb()
    if detections and image is not None:
        annotated = perception.draw_detections(image, detections)
        cv2.imwrite('/workspace/yoloworld_test.jpg', annotated)
        print(f"\n✓ Saved visualization: /workspace/yoloworld_test.jpg")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == '__main__':
    try:
        test_yolo_world()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"\nERROR: Test failed")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
