#!/usr/bin/env python3
"""
YOLO-World Open-Vocabulary Object Detector
Real-time open-vocabulary detection using YOLO-World from Ultralytics
"""

import cv2
import numpy as np
from typing import List, Dict
from ultralytics import YOLOWorld


class YOLOWorldDetector:
    """YOLO-World detector for open-vocabulary object detection."""

    def __init__(self, model_size: str = 's'):
        """
        Initialize YOLO-World detector.

        Args:
            model_size: Model size ('s', 'm', 'l') - smaller = faster, larger = more accurate
        """
        print(f"[YOLOWorld] Loading YOLO-World-{model_size} model...")
        self.model = YOLOWorld(f'yolov8{model_size}-worldv2.pt')
        print("[YOLOWorld] Model loaded successfully")

        # Track current vocabulary
        self.current_classes = []

    def set_classes(self, object_names: List[str]):
        """
        Set custom object classes for detection.

        Args:
            object_names: List of object names to detect
        """
        if object_names != self.current_classes:
            self.model.set_classes(object_names)
            self.current_classes = object_names
            print(f"[YOLOWorld] Set classes: {object_names}")

    def detect(self, image: np.ndarray, object_names: List[str],
               confidence_threshold: float = 0.1) -> List[Dict]:
        """
        Detect objects in image using open vocabulary.

        Args:
            image: OpenCV BGR image
            object_names: List of object names to detect
            confidence_threshold: Minimum confidence (0-1)

        Returns:
            List of detections: [{'class_name': str, 'bbox': [x,y,w,h], 'confidence': float}, ...]
        """
        # Set vocabulary if changed
        self.set_classes(object_names)

        # Run detection
        results = self.model.predict(image, conf=confidence_threshold, verbose=False)

        # Parse results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

                # Convert to xywh format
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)

                # Get class and confidence
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                class_name = object_names[class_id] if class_id < len(object_names) else 'unknown'

                detections.append({
                    'class_name': class_name,
                    'bbox': [x, y, w, h],
                    'confidence': confidence
                })

        print(f"[YOLOWorld] Detected {len(detections)} objects: {[d['class_name'] for d in detections]}")
        return detections


if __name__ == '__main__':
    """Test YOLO-World detector."""
    import sys

    # Create detector
    detector = YOLOWorldDetector(model_size='s')

    # Test image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python3 yolo_world_detector.py <image_path>")
        print("Example: python3 yolo_world_detector.py /workspace/test.jpg")
        sys.exit(1)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        sys.exit(1)

    # Test with custom objects
    test_objects = [
        'bottle', 'cup', 'mug',
        'phone', 'smartphone', 'cell phone',
        'laptop', 'computer',
        'book', 'notebook',
        'person', 'human'
    ]

    print(f"\nDetecting: {test_objects}")
    detections = detector.detect(image, test_objects, confidence_threshold=0.1)

    # Draw results
    for det in detections:
        x, y, w, h = det['bbox']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save result
    output_path = '/workspace/yoloworld_test.jpg'
    cv2.imwrite(output_path, image)
    print(f"\nSaved result to: {output_path}")
    print(f"Total detections: {len(detections)}")
