#!/usr/bin/env python3
"""
bottle_pose_estimator.py
Estimates bottle 3D pose by fitting a vertical cylinder to the depth point cloud.

Pipeline:
  YOLO bounding box  →  crop depth region  →  project to base_footprint 3D
  →  RANSAC circle fit (vertical cylinder assumption)
  →  publish bottle-top PoseStamped + RViz cylinder marker

No GPU / no extra packages needed — pure numpy + OpenCV.

Run:  python3 bottle_pose_estimator.py
Topics published:
  /bottle_pose   (geometry_msgs/PoseStamped)  — bottle top centre in base_footprint
  /bottle_marker (visualization_msgs/Marker)  — cylinder for RViz
"""

import os
import re
import subprocess

import cv2
import message_filters
import numpy as np
import rospy

from geometry_msgs.msg import PoseStamped, Pose
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, Image
from visualization_msgs.msg import Marker


# ──────────────────────────────────────────────────────────────────────────────
class BottlePoseEstimator:

    def __init__(self):
        rospy.init_node('bottle_pose_estimator', anonymous=True)

        # ── Camera model ───────────────────────────────────────────────────
        rospy.loginfo("Waiting for camera info…")
        cam_info = rospy.wait_for_message('/xtion/rgb/camera_info', CameraInfo, timeout=10)
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(cam_info)
        # Pre-extract intrinsics for vectorised projection
        self.fx = self.cam_model.fx()
        self.fy = self.cam_model.fy()
        self.cx = self.cam_model.cx()
        self.cy = self.cam_model.cy()

        # ── YOLOv4-tiny ────────────────────────────────────────────────────
        d = os.path.dirname(os.path.abspath(__file__))
        rospy.loginfo("Loading YOLOv4-tiny…")
        self.net = cv2.dnn.readNetFromDarknet(
            os.path.join(d, 'yolov4-tiny.cfg'),
            os.path.join(d, 'yolov4-tiny.weights'))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layers = self.net.getLayerNames()
        self.out_layers = [layers[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        with open(os.path.join(d, 'coco.names')) as f:
            self.classes = [l.strip() for l in f]
        rospy.loginfo("YOLO ready.")

        # ── Sync RGB + depth ───────────────────────────────────────────────
        self.latest_rgb   = None
        self.latest_depth = None
        rgb_sub   = message_filters.Subscriber('/xtion/rgb/image_raw', Image)
        depth_sub = message_filters.Subscriber('/xtion/depth_registered/image_raw', Image)
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 5, 0.1)
        ts.registerCallback(self._sync_cb)
        rospy.loginfo("Waiting for first frame…")
        while self.latest_rgb is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Frames received.")

        # ── Publishers ─────────────────────────────────────────────────────
        self.pose_pub   = rospy.Publisher('/bottle_pose',   PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/bottle_marker', Marker,      queue_size=1)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _sync_cb(self, rgb_msg, depth_msg):
        self.latest_rgb = rgb_msg
        if depth_msg.encoding == '32FC1':
            arr = np.frombuffer(depth_msg.data, dtype=np.float32)
        elif depth_msg.encoding == '16UC1':
            arr = np.frombuffer(depth_msg.data, dtype=np.uint16).astype(np.float32) / 1000.0
        else:
            return
        self.latest_depth = arr.reshape(depth_msg.height, depth_msg.width)

    # ── TF helper ─────────────────────────────────────────────────────────────
    def _get_tf(self, parent, child):
        try:
            proc = subprocess.Popen(['rosrun', 'tf', 'tf_echo', parent, child],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output = ''
            for _ in range(30):
                line = proc.stdout.readline().decode('utf-8')
                output += line
                if 'Quaternion' in line:
                    break
            proc.kill(); proc.wait()
            tm = re.search(r'Translation: \[([^,]+), ([^,]+), ([^\]]+)\]', output)
            qm = re.search(r'Quaternion \[([^,]+), ([^,]+), ([^,]+), ([^\]]+)\]', output)
            if tm and qm:
                t = [float(tm.group(i)) for i in range(1, 4)]
                q = [float(qm.group(i)) for i in range(1, 5)]
                return np.array(t), np.array(q)
        except Exception as e:
            rospy.logerr("tf_echo error: {}".format(e))
        return None, None

    def _rotation_matrix(self, q):
        qx, qy, qz, qw = q
        return np.array([
            [1-2*(qy*qy+qz*qz),  2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),    1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
        ])

    # ── YOLO detection ────────────────────────────────────────────────────────
    def _detect_box(self, img_bgr):
        h, w = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(img_bgr, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.out_layers)
        boxes, confs = [], []
        for out in outputs:
            for det in out:
                scores = det[5:]
                cid = int(np.argmax(scores))
                conf = float(scores[cid])
                if conf > 0.4 and self.classes[cid] == 'bottle':
                    bx = int((det[0] - det[2]/2) * w)
                    by = int((det[1] - det[3]/2) * h)
                    bw = int(det[2] * w)
                    bh = int(det[3] * h)
                    boxes.append([bx, by, bw, bh])
                    confs.append(conf)
        if not boxes:
            return None
        idxs = cv2.dnn.NMSBoxes(boxes, confs, 0.4, 0.4)
        if len(idxs) == 0:
            return None
        return boxes[idxs.flatten()[0]]

    # ── Point cloud projection ────────────────────────────────────────────────
    def _depth_roi_to_base(self, depth, box, R, t, step=2):
        """Vectorised: project depth pixels inside YOLO box → base_footprint XYZ."""
        bx, by, bw, bh = box
        margin = 4
        x0 = max(0, bx + margin);  x1 = min(depth.shape[1], bx + bw - margin)
        y0 = max(0, by + margin);  y1 = min(depth.shape[0], by + bh - margin)

        ys, xs = np.mgrid[y0:y1:step, x0:x1:step]
        ds = depth[ys, xs]
        valid = np.isfinite(ds) & (ds > 0.15) & (ds < 3.5)
        xs_v, ys_v, ds_v = xs[valid].ravel(), ys[valid].ravel(), ds[valid].ravel()
        if len(ds_v) == 0:
            return None

        cam_x = (xs_v - self.cx) / self.fx * ds_v
        cam_y = (ys_v - self.cy) / self.fy * ds_v
        cam_z = ds_v
        cam_pts = np.stack([cam_x, cam_y, cam_z], axis=1)   # N×3
        base_pts = (R @ cam_pts.T).T + t                     # N×3
        return base_pts

    # ── RANSAC vertical cylinder fit ──────────────────────────────────────────
    def _fit_cylinder(self, pts, n_iters=300, thresh=0.015):
        """
        Fit a vertical cylinder (axis = world Z) to 3D points in base_footprint.
        Uses RANSAC circle fitting in the XY plane.
        Returns dict with keys: cx, cy, radius, z_top, z_bottom, inlier_ratio
        or None on failure.
        """
        xy = pts[:, :2]
        z  = pts[:, 2]
        n  = len(xy)
        if n < 15:
            return None

        best = {'inliers': 0}

        for _ in range(n_iters):
            idx = np.random.choice(n, 3, replace=False)
            p1, p2, p3 = xy[idx]

            # Circle through 3 points (algebraic solution)
            ax, ay = p1; bx, by = p2; cx_p, cy_p = p3
            D = 2*(ax*(by - cy_p) + bx*(cy_p - ay) + cx_p*(ay - by))
            if abs(D) < 1e-9:
                continue
            ux = ((ax**2+ay**2)*(by-cy_p) + (bx**2+by**2)*(cy_p-ay) + (cx_p**2+cy_p**2)*(ay-by)) / D
            uy = ((ax**2+ay**2)*(cx_p-bx) + (bx**2+by**2)*(ax-cx_p) + (cx_p**2+cy_p**2)*(bx-ax)) / D
            r  = np.hypot(ax - ux, ay - uy)

            # Plausible bottle radius: 1.5 – 6 cm
            if r < 0.015 or r > 0.06:
                continue

            dists   = np.abs(np.hypot(xy[:,0]-ux, xy[:,1]-uy) - r)
            n_in    = int(np.sum(dists < thresh))
            if n_in > best['inliers']:
                best = {'inliers': n_in, 'cx': ux, 'cy': uy, 'r': r, 'mask': dists < thresh}

        if best['inliers'] < 8:
            return None

        # Least-squares circle refinement on inlier set
        mask = best['mask']
        xi, yi = xy[mask, 0], xy[mask, 1]
        # Algebraic linear LS:  xi^2+yi^2 = 2cx*xi + 2cy*yi + C
        A = np.column_stack([2*xi, 2*yi, np.ones(len(xi))])
        b = xi**2 + yi**2
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx_r, cy_r = sol[0], sol[1]
        r_r = float(np.mean(np.hypot(xi - cx_r, yi - cy_r)))

        # Recompute inlier mask with refined circle
        dists2 = np.abs(np.hypot(xy[:,0]-cx_r, xy[:,1]-cy_r) - r_r)
        mask2  = dists2 < thresh
        zi = z[mask2]

        z_top    = float(np.percentile(zi, 95))
        z_bottom = float(np.percentile(zi, 5))

        return dict(cx=float(cx_r), cy=float(cy_r), radius=r_r,
                    z_top=z_top, z_bottom=z_bottom,
                    inlier_ratio=float(np.sum(mask2)) / n)

    # ── Main estimation step ──────────────────────────────────────────────────
    def estimate_once(self):
        rgb_msg = self.latest_rgb
        depth   = self.latest_depth
        if rgb_msg is None or depth is None:
            return None

        # Decode RGB
        if rgb_msg.encoding == 'rgb8':
            img = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(
                rgb_msg.height, rgb_msg.width, 3)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(
                rgb_msg.height, rgb_msg.width, 3)

        # YOLO
        box = self._detect_box(img_bgr)
        if box is None:
            rospy.loginfo_throttle(2, "No bottle detected.")
            return None
        bx, by, bw, bh = box
        rospy.loginfo("Box: x={} y={} w={} h={}".format(bx, by, bw, bh))

        # TF
        t, q = self._get_tf('base_footprint', 'xtion_rgb_optical_frame')
        if t is None:
            rospy.logerr("TF unavailable.")
            return None
        R = self._rotation_matrix(q)

        # Project depth → 3D base_footprint
        pts = self._depth_roi_to_base(depth, box, R, t, step=2)
        if pts is None or len(pts) < 15:
            rospy.logwarn("Too few depth points ({}).".format(0 if pts is None else len(pts)))
            return None
        rospy.loginfo("Projected {} points into base_footprint.".format(len(pts)))

        # RANSAC cylinder fit
        cyl = self._fit_cylinder(pts)
        if cyl is None:
            rospy.logwarn("Cylinder fit failed.")
            return None

        rospy.loginfo("─── Cylinder fit ─────────────────────────────")
        rospy.loginfo("  Centre XY : ({:.3f}, {:.3f}) m".format(cyl['cx'], cyl['cy']))
        rospy.loginfo("  Radius    : {:.3f} m  = {:.1f} cm".format(cyl['radius'], cyl['radius']*100))
        rospy.loginfo("  Height    : {:.3f} m  = {:.1f} cm".format(
            cyl['z_top']-cyl['z_bottom'], (cyl['z_top']-cyl['z_bottom'])*100))
        rospy.loginfo("  Top Z     : {:.3f} m".format(cyl['z_top']))
        rospy.loginfo("  Bottom Z  : {:.3f} m".format(cyl['z_bottom']))
        rospy.loginfo("  Inliers   : {:.1%}".format(cyl['inlier_ratio']))
        rospy.loginfo("──────────────────────────────────────────────")

        return cyl

    # ── Publishers ────────────────────────────────────────────────────────────
    def _publish(self, cyl):
        stamp = rospy.Time.now()

        # PoseStamped at bottle TOP centre
        ps = PoseStamped()
        ps.header.frame_id = "base_footprint"
        ps.header.stamp    = stamp
        ps.pose.position.x = cyl['cx']
        ps.pose.position.y = cyl['cy']
        ps.pose.position.z = cyl['z_top']
        ps.pose.orientation.w = 1.0
        self.pose_pub.publish(ps)

        # RViz cylinder marker
        m = Marker()
        m.header.frame_id = "base_footprint"
        m.header.stamp    = stamp
        m.ns    = "bottle"; m.id = 0
        m.type  = Marker.CYLINDER
        m.action= Marker.ADD
        m.pose.position.x = cyl['cx']
        m.pose.position.y = cyl['cy']
        m.pose.position.z = (cyl['z_top'] + cyl['z_bottom']) / 2.0
        m.pose.orientation.w = 1.0
        m.scale.x = cyl['radius'] * 2
        m.scale.y = cyl['radius'] * 2
        m.scale.z = cyl['z_top'] - cyl['z_bottom']
        m.color.r = 0.0; m.color.g = 0.8; m.color.b = 0.4; m.color.a = 0.7
        self.marker_pub.publish(m)

    # ── Loop ──────────────────────────────────────────────────────────────────
    def run(self):
        rate = rospy.Rate(2)   # 2 Hz — enough to see live updates without flooding
        while not rospy.is_shutdown():
            cyl = self.estimate_once()
            if cyl:
                self._publish(cyl)
                print("\n>>> Bottle top (base_footprint): "
                      "x={cx:.3f}  y={cy:.3f}  z={z_top:.3f}  r={radius:.3f}\n".format(**cyl))
            rate.sleep()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        BottlePoseEstimator().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Fatal: {}".format(e))
        raise
