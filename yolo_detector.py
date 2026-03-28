#!/usr/bin/env python3
"""
YOLO Object Detection node for TIAGo robot.
Uses OpenCV DNN with YOLOv3-tiny (no ultralytics needed).
"""

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
import numpy as np
import cv2
import os

class YoloDetector:
    def __init__(self):
        rospy.init_node('yolo_detector', anonymous=True)

        # Paths to YOLO files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights = os.path.join(script_dir, 'yolov4-tiny.weights')
        config = os.path.join(script_dir, 'yolov4-tiny.cfg')
        names_file = os.path.join(script_dir, 'coco.names')

        # Load class names
        with open(names_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Load YOLO network
        rospy.loginfo("Loading YOLOv3-tiny model...")
        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Get output layer names
        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        if unconnected.ndim == 1:
            self.output_layers = [layer_names[i - 1] for i in unconnected]
        else:
            self.output_layers = [layer_names[i[0] - 1] for i in unconnected]
        rospy.loginfo("Model loaded.")

        # Publisher for detected object positions
        self.detection_pub = rospy.Publisher('/detected_objects', PointStamped, queue_size=10)

        # Publisher for annotated image
        self.image_pub = rospy.Publisher('/yolo/image', Image, queue_size=1)

        # Subscribe to RGB camera
        self.image_sub = rospy.Subscriber(
            '/xtion/rgb/image_raw',
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )

        # Subscribe to depth image for 3D coordinates
        self.depth_sub = rospy.Subscriber(
            '/xtion/depth/image_raw',
            Image,
            self.depth_callback,
            queue_size=1,
            buff_size=2**24
        )

        self.depth_image = None
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

        rospy.loginfo("YOLO detector ready. Waiting for images...")

    def depth_callback(self, msg):
        """Store latest depth image."""
        try:
            if msg.encoding == '32FC1':
                self.depth_image = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            elif msg.encoding == '16UC1':
                self.depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                self.depth_image = self.depth_image.astype(np.float32) / 1000.0
        except Exception as e:
            rospy.logwarn("Depth conversion error: {}".format(e))

    def image_callback(self, msg):
        """Process RGB image with YOLO."""
        try:
            # Convert ROS Image to numpy array
            if msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                img_bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            else:
                rospy.logwarn("Unsupported encoding: {}".format(msg.encoding))
                return

            height, width = img_bgr.shape[:2]

            # Create blob and run YOLO
            blob = cv2.dnn.blobFromImage(img_bgr, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)

            # Parse detections
            boxes = []
            confidences = []
            class_ids = []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > self.confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    cx = x + w // 2
                    cy = y + h // 2
                    label = self.classes[class_ids[i]]
                    conf = confidences[i]

                    # Get depth at center pixel
                    depth = 0.0
                    if self.depth_image is not None:
                        if 0 <= cy < self.depth_image.shape[0] and 0 <= cx < self.depth_image.shape[1]:
                            depth = self.depth_image[cy, cx]
                            if np.isnan(depth) or np.isinf(depth):
                                depth = 0.0

                    # Draw bounding box
                    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = "{} {:.2f} d={:.2f}m".format(label, conf, depth)
                    cv2.putText(img_bgr, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Publish detection
                    point_msg = PointStamped()
                    point_msg.header = msg.header
                    point_msg.point.x = float(cx)
                    point_msg.point.y = float(cy)
                    point_msg.point.z = depth
                    self.detection_pub.publish(point_msg)

                    rospy.loginfo("Detected: {} at pixel ({}, {}) depth={:.2f}m conf={:.2f}".format(
                        label, cx, cy, depth, conf))

            # Show image with detections
            cv2.imshow("YOLO Detections", img_bgr)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr("Detection error: {}".format(e))

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = YoloDetector()
        detector.run()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
