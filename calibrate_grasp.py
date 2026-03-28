#!/usr/bin/env python3
"""
Grasp offset calibration tool.
1. Detects bottle position with YOLO + depth
2. Waits for you to guide arm to desired grasp pose (gravity comp)
3. Reads arm_tool_link TF
4. Computes and prints the offset (bottle_pos - arm_tool_link_pos)
Run multiple times and average for best results.
"""

import rospy
import numpy as np
import cv2
import os
import subprocess
import re

from sensor_msgs.msg import Image, CameraInfo


class GraspCalibrator:
    def __init__(self):
        rospy.init_node('grasp_calibrator', anonymous=True)

        # YOLO setup
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights = os.path.join(script_dir, 'yolov4-tiny.weights')
        config = os.path.join(script_dir, 'yolov4-tiny.cfg')
        names_file = os.path.join(script_dir, 'coco.names')

        with open(names_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        if unconnected.ndim == 1:
            self.output_layers = [layer_names[i - 1] for i in unconnected]
        else:
            self.output_layers = [layer_names[i[0] - 1] for i in unconnected]

        # Camera intrinsics
        rospy.loginfo("Waiting for camera info...")
        cam_info = rospy.wait_for_message('/xtion/rgb/camera_info', CameraInfo, timeout=10)
        self.fx = cam_info.K[0]
        self.fy = cam_info.K[4]
        self.cx_cam = cam_info.K[2]
        self.cy_cam = cam_info.K[5]

        # Depth
        self.depth_image = None
        self.depth_sub = rospy.Subscriber('/xtion/depth/image_raw', Image,
                                          self.depth_callback, queue_size=1, buff_size=2**24)

        self.target_class = 'bottle'
        self.confidence_threshold = 0.4
        self.samples = []

        rospy.loginfo("Calibrator ready.")

    def depth_callback(self, msg):
        try:
            if msg.encoding == '32FC1':
                self.depth_image = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            elif msg.encoding == '16UC1':
                self.depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                self.depth_image = self.depth_image.astype(np.float32) / 1000.0
        except:
            pass

    def run_tf_echo(self, parent, child):
        """Get TF between two frames."""
        try:
            proc = subprocess.Popen(
                ['rosrun', 'tf', 'tf_echo', parent, child],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            output = ''
            for _ in range(30):
                line = proc.stdout.readline().decode('utf-8')
                output += line
                if 'Quaternion' in line:
                    break
            proc.kill()
            proc.wait()

            t_match = re.search(r'Translation: \[([^,]+), ([^,]+), ([^\]]+)\]', output)
            q_match = re.search(r'Quaternion \[([^,]+), ([^,]+), ([^,]+), ([^\]]+)\]', output)

            if t_match and q_match:
                t = [float(t_match.group(i)) for i in range(1, 4)]
                q = [float(q_match.group(i)) for i in range(1, 5)]
                return t, q
        except Exception as e:
            rospy.logerr("tf_echo failed: {}".format(e))
        return None, None

    def quat_rotate_vec(self, q, v):
        qx, qy, qz, qw = q
        vx, vy, vz = v
        t0 = -qx*vx - qy*vy - qz*vz
        t1 = qw*vx + qy*vz - qz*vy
        t2 = qw*vy - qx*vz + qz*vx
        t3 = qw*vz + qx*vy - qy*vx
        rx = -t0*qx + t1*qw - t2*qz + t3*qy
        ry = -t0*qy + t1*qz + t2*qw - t3*qx
        rz = -t0*qz - t1*qy + t2*qx + t3*qw
        return np.array([rx, ry, rz])

    def detect_bottle(self):
        """Returns (base_x, base_y, base_z) of bottle or None."""
        rospy.loginfo("Detecting bottle...")
        img_msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image, timeout=5)

        if img_msg.encoding == 'rgb8':
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img_msg.encoding == 'bgr8':
            img_bgr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
        else:
            return None

        height, width = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(img_bgr, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold and self.classes[class_id] == self.target_class:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        if not boxes:
            return None

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.4)
        if len(indices) == 0:
            return None

        best = indices.flatten()[0]
        bx, by, bw, bh = boxes[best]
        cx = bx + bw // 2
        cy = by + bh // 2

        if self.depth_image is None:
            return None

        region = self.depth_image[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
        valid = region[~np.isnan(region) & ~np.isinf(region) & (region > 0)]
        if len(valid) == 0:
            return None

        depth = np.median(valid)

        # Pixel to camera frame
        cam_x = (cx - self.cx_cam) * depth / self.fx
        cam_y = (cy - self.cy_cam) * depth / self.fy
        cam_z = depth

        # Camera to base frame
        cam_t, cam_q = self.run_tf_echo('base_footprint', 'xtion_rgb_optical_frame')
        if cam_t is None:
            return None

        rotated = self.quat_rotate_vec(cam_q, np.array([cam_x, cam_y, cam_z]))
        base_x = rotated[0] + cam_t[0]
        base_y = rotated[1] + cam_t[1]
        base_z = rotated[2] + cam_t[2]

        rospy.loginfo("Bottle detected at base frame: x={:.3f} y={:.3f} z={:.3f}".format(
            base_x, base_y, base_z))
        return base_x, base_y, base_z

    def read_arm_pose(self):
        """Read current arm_tool_link pose in base frame."""
        t, q = self.run_tf_echo('base_footprint', 'arm_tool_link')
        if t is None:
            return None, None
        return t, q

    def run(self):
        rospy.sleep(1)

        print("\n" + "="*60)
        print("GRASP OFFSET CALIBRATION")
        print("="*60)

        # Step 1: Detect bottle
        bottle_pos = None
        while bottle_pos is None and not rospy.is_shutdown():
            bottle_pos = self.detect_bottle()
            if bottle_pos is None:
                rospy.loginfo("No bottle found, retrying in 2s...")
                rospy.sleep(2)

        bx, by, bz = bottle_pos
        print("\nBottle position (base frame):")
        print("  x={:.4f}  y={:.4f}  z={:.4f}".format(bx, by, bz))

        # Step 2: Wait for user to guide arm
        print("\n" + "-"*60)
        print("NOW: Guide the arm to the desired grasp pose.")
        print("  - Use gravity compensation / impedance mode")
        print("  - Position arm_tool_link where you want it for grasping")
        print("  - Press ENTER when arm is in position")
        print("-"*60)

        try:
            raw_input("\nPress ENTER when arm is positioned...")
        except NameError:
            input("\nPress ENTER when arm is positioned...")

        # Step 3: Read arm pose
        arm_t, arm_q = self.read_arm_pose()
        if arm_t is None:
            print("ERROR: Could not read arm_tool_link TF!")
            return

        ax, ay, az = arm_t
        print("\nArm tool_link position (base frame):")
        print("  x={:.4f}  y={:.4f}  z={:.4f}".format(ax, ay, az))
        print("Arm tool_link orientation (quaternion):")
        print("  x={:.4f}  y={:.4f}  z={:.4f}  w={:.4f}".format(*arm_q))

        # Step 4: Compute offset
        dx = bx - ax
        dy = by - ay
        dz = bz - az

        print("\n" + "="*60)
        print("OFFSET (bottle_pos - arm_tool_link_pos):")
        print("  dx={:.4f}  dy={:.4f}  dz={:.4f}".format(dx, dy, dz))
        print("="*60)
        print("\nTo use in reach_object.py:")
        print("  grasp_x = x - {:.4f}".format(dx))
        print("  grasp_y = y - {:.4f}".format(dy))
        print("  grasp_z = z - {:.4f}  (or keep z unchanged)".format(dz))
        print("\nOrientation to use:")
        print("  Quaternion(x={:.4f}, y={:.4f}, z={:.4f}, w={:.4f})".format(*arm_q))

        # Store sample
        self.samples.append({'dx': dx, 'dy': dy, 'dz': dz, 'q': arm_q})

        # Ask for more samples
        while not rospy.is_shutdown():
            print("\n" + "-"*60)
            try:
                ans = raw_input("Take another sample? (y/n): ")
            except NameError:
                ans = input("Take another sample? (y/n): ")

            if ans.lower() != 'y':
                break

            # Re-detect bottle (it may have moved)
            print("Re-detecting bottle...")
            bottle_pos = self.detect_bottle()
            if bottle_pos is None:
                print("No bottle found!")
                continue
            bx, by, bz = bottle_pos
            print("Bottle: x={:.4f} y={:.4f} z={:.4f}".format(bx, by, bz))

            try:
                raw_input("Guide arm and press ENTER...")
            except NameError:
                input("Guide arm and press ENTER...")

            arm_t, arm_q = self.read_arm_pose()
            if arm_t is None:
                print("Could not read arm TF!")
                continue

            ax, ay, az = arm_t
            dx = bx - ax
            dy = by - ay
            dz = bz - az
            self.samples.append({'dx': dx, 'dy': dy, 'dz': dz, 'q': arm_q})

            print("Sample {}: dx={:.4f} dy={:.4f} dz={:.4f}".format(
                len(self.samples), dx, dy, dz))

        # Print summary
        if len(self.samples) > 1:
            dxs = [s['dx'] for s in self.samples]
            dys = [s['dy'] for s in self.samples]
            dzs = [s['dz'] for s in self.samples]
            qs = np.array([s['q'] for s in self.samples])

            print("\n" + "="*60)
            print("AVERAGE OVER {} SAMPLES:".format(len(self.samples)))
            print("  dx={:.4f} (std={:.4f})".format(np.mean(dxs), np.std(dxs)))
            print("  dy={:.4f} (std={:.4f})".format(np.mean(dys), np.std(dys)))
            print("  dz={:.4f} (std={:.4f})".format(np.mean(dzs), np.std(dzs)))
            print("  q_avg: x={:.4f} y={:.4f} z={:.4f} w={:.4f}".format(*np.mean(qs, axis=0)))
            print("="*60)
            print("\nCopy into reach_object.py:")
            print("  grasp_x = x - {:.4f}".format(np.mean(dxs)))
            print("  grasp_y = y - {:.4f}".format(np.mean(dys)))
            print("  self.grasp_orientation = Quaternion(x={:.4f}, y={:.4f}, z={:.4f}, w={:.4f})".format(
                *np.mean(qs, axis=0)))


if __name__ == '__main__':
    try:
        cal = GraspCalibrator()
        cal.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Error: {}".format(e))
