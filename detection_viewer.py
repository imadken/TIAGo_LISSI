#!/usr/bin/env python3
"""
Real-time Detection Viewer for TIAGo
Shows YOLO detections and optional CLIP classifications with bounding boxes
"""

import rospy
import cv2
import argparse
from perception_manager import PerceptionManager


def main():
    """Run real-time detection viewer."""
    parser = argparse.ArgumentParser(description='View robot camera with object detections')
    parser.add_argument('--clip', action='store_true', help='Enable CLIP open-vocabulary detection')
    parser.add_argument('--objects', type=str, default='',
                       help='Comma-separated list of objects for CLIP (e.g., "phone,mug,bottle")')
    parser.add_argument('--save', action='store_true', help='Save each frame to /workspace/')
    args = parser.parse_args()

    # Initialize ROS
    rospy.init_node('detection_viewer', anonymous=True)

    print("\n" + "=" * 60)
    print("TIAGo Detection Viewer")
    print("=" * 60)

    # Initialize perception
    print("\nInitializing perception...")
    use_clip = args.clip
    perception = PerceptionManager(use_clip=use_clip)

    # Parse object list for CLIP
    object_names = None
    if args.objects:
        object_names = [obj.strip() for obj in args.objects.split(',')]
        print("CLIP objects: {}".format(object_names))

    print("\nViewer ready!")
    print("=" * 60)
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press 'c' to toggle CLIP (if enabled)")
    print("=" * 60)

    if use_clip and object_names:
        print("\nCLIP Mode: Detecting custom objects: {}".format(object_names))
    elif use_clip:
        print("\nCLIP Mode: Ready (specify --objects for custom detection)")
    else:
        print("\nYOLO-only Mode: Detecting COCO classes")
        print("  (Use --clip to enable open-vocabulary detection)")
    print()

    # Display settings
    window_name = "TIAGo Detection View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    clip_enabled = use_clip and object_names is not None
    frame_count = 0
    rate = rospy.Rate(5)  # 5 Hz update rate

    try:
        while not rospy.is_shutdown():
            # Get latest image
            image = perception.get_latest_rgb()
            if image is None:
                rospy.logwarn_throttle(5, "[Viewer] No image available")
                rate.sleep()
                continue

            # Run detection
            if clip_enabled and object_names:
                # CLIP detection
                detections = perception.detect_open_vocabulary(
                    object_names=object_names,
                    confidence_threshold=0.15
                )
                mode_text = "CLIP: {}".format(', '.join(object_names[:3]))
                if len(object_names) > 3:
                    mode_text += "..."
            else:
                # YOLO-only detection
                detections = perception.detect_objects()
                mode_text = "YOLO: COCO classes"

            # Draw detections
            annotated = perception.draw_detections(image, detections)

            # Add info overlay
            h, w = annotated.shape[:2]

            # Semi-transparent overlay for text background
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

            # Detection info
            info_text = "Mode: {} | Detections: {}".format(mode_text, len(detections))
            cv2.putText(annotated, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Controls help
            help_text = "Press 'q' to quit | 's' to save | 'c' to toggle CLIP"
            cv2.putText(annotated, help_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Detection list (bottom)
            if detections:
                y_offset = h - 20
                for i, det in enumerate(detections[-5:]):  # Show last 5
                    det_text = "{}: {:.2f}".format(det['class_name'], det['confidence'])
                    cv2.putText(annotated, det_text, (10, y_offset - i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Display
            cv2.imshow(window_name, annotated)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break

            elif key == ord('s'):
                # Save frame
                filename = "/workspace/detection_frame_{}.jpg".format(frame_count)
                cv2.imwrite(filename, annotated)
                print("Saved: {}".format(filename))
                frame_count += 1

            elif key == ord('c') and use_clip and object_names:
                # Toggle CLIP
                clip_enabled = not clip_enabled
                mode_str = "CLIP" if clip_enabled else "YOLO-only"
                print("Switched to {} mode".format(mode_str))

            # Auto-save if requested
            if args.save:
                filename = "/workspace/detection_stream_{}.jpg".format(frame_count)
                cv2.imwrite(filename, annotated)
                frame_count += 1

            rate.sleep()

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cv2.destroyAllWindows()
        print("\nViewer closed")
        print("=" * 60)


if __name__ == '__main__':
    main()
