#!/usr/bin/env python3
"""
Perception Manager v2 for TIAGo Embodied AI
Responsibilities:
  - Subscribe to Xtion RGB topic and hold the latest frame.
  - Enrich VLM detections of 'person' with face recognition data.

Object detection is now handled by VLMReasoner.detect_objects().
"""

import subprocess
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from typing import List, Dict, Optional, Tuple


class PerceptionManager:
    def __init__(self, yolo_weights=None, yolo_config=None,
                 class_names=None, use_clip=False,
                 use_yolo_world=False, face_manager=None):
        """
        Args:
            face_manager: Optional FaceManager instance for person identification.
            (Other args kept for API compatibility but are no longer used.)
        """
        self.face_manager = face_manager
        self.latest_rgb = None

        self.rgb_subscriber = rospy.Subscriber(
            '/xtion/rgb/image_raw', Image, self._rgb_callback)

        rospy.loginfo("[Perception] Waiting for first RGB image...")
        while self.latest_rgb is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("[Perception] Camera ready.")

        # ── Depth + 3D ────────────────────────────────────────────────────────
        self.latest_depth = None
        self.fx_cam = self.fy_cam = self.cx_cam = self.cy_cam = None
        self._tf_cache = None          # (t, R) cached
        self._tf_cache_time = 0.0

        rospy.Subscriber('/xtion/depth_registered/image_raw', Image,
                         self._depth_callback, queue_size=1)
        rospy.Subscriber('/xtion/rgb/camera_info', CameraInfo,
                         self._camera_info_callback, queue_size=1)

        # Warm up TF cache so first detection cycle doesn't fall back to 2D
        rospy.loginfo("[Perception] Warming up TF cache...")
        self._get_tf()
        rospy.loginfo("[Perception] TF ready: {}".format(
            "OK" if self._tf_cache else "unavailable (will retry)"))

    # ── Camera ────────────────────────────────────────────────────────────────

    def _rgb_callback(self, msg: Image):
        try:
            if msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3)
                self.latest_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                self.latest_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3).copy()
        except Exception as e:
            rospy.logwarn("[Perception] RGB decode error: {}".format(e))

    def get_latest_rgb(self) -> Optional[np.ndarray]:
        return self.latest_rgb.copy() if self.latest_rgb is not None else None

    # ── Depth callbacks ───────────────────────────────────────────────────────

    def _depth_callback(self, msg: Image):
        try:
            if msg.encoding == '32FC1':
                arr = np.frombuffer(msg.data, dtype=np.float32)
            elif msg.encoding == '16UC1':
                arr = np.frombuffer(msg.data, dtype=np.uint16).astype(np.float32) / 1000.0
            else:
                return
            self.latest_depth = arr.reshape(msg.height, msg.width)
        except Exception as e:
            rospy.logwarn_throttle(10, "[Perception] Depth decode error: {}".format(e))

    def _camera_info_callback(self, msg: CameraInfo):
        if self.fx_cam is None:
            self.fx_cam = msg.K[0]
            self.fy_cam = msg.K[4]
            self.cx_cam = msg.K[2]
            self.cy_cam = msg.K[5]

    def _tf_echo(self, parent: str, child: str) -> Tuple:
        """Get TF via subprocess. Returns (t, R) or (None, None)."""
        try:
            proc = subprocess.Popen(
                ['rosrun', 'tf', 'tf_echo', parent, child],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output = ''
            for _ in range(30):
                line = proc.stdout.readline().decode('utf-8')
                output += line
                if 'Quaternion' in line:
                    break
            proc.kill(); proc.wait()
            t_m = __import__('re').search(
                r'Translation:.*?\[([-\d., ]+)\]', output, __import__('re').DOTALL)
            q_m = __import__('re').search(
                r'Quaternion:.*?\[([-\d., ]+)\]', output, __import__('re').DOTALL)
            if not t_m or not q_m:
                return None, None
            t = np.array([float(x) for x in t_m.group(1).split(',')])
            qx, qy, qz, qw = [float(x) for x in q_m.group(1).split(',')]
            R = np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
            ])
            return t, R
        except Exception:
            return None, None

    def _get_tf(self) -> Tuple:
        import time
        now = time.time()
        if self._tf_cache and now - self._tf_cache_time < 0.5:
            return self._tf_cache
        t, R = self._tf_echo('base_footprint', 'xtion_rgb_optical_frame')
        if t is not None:
            self._tf_cache = (t, R)
            self._tf_cache_time = now
        return (t, R)

    def _depth_roi_to_base(self, depth: np.ndarray, bbox: list) -> Tuple:
        """Project depth pixels inside bbox to base_footprint 3D points."""
        if self.fx_cam is None:
            return None, 0
        t, R = self._get_tf()
        if t is None:
            return None, 0
        bx, by, bw, bh = bbox
        margin = max(2, int(min(bw, bh) * 0.08))
        x0 = max(0, bx + margin); x1 = min(depth.shape[1], bx + bw - margin)
        y0 = max(0, by + margin); y1 = min(depth.shape[0], by + bh - margin)
        if x1 <= x0 or y1 <= y0:
            return None, 0
        ys, xs = np.mgrid[y0:y1:2, x0:x1:2]
        ds = depth[ys, xs]
        valid = (ds > 0.15) & (ds < 3.5) & np.isfinite(ds)
        if valid.sum() < 8:
            return None, 0
        xs_v = xs[valid].astype(float); ys_v = ys[valid].astype(float)
        ds_v = ds[valid]
        cam_pts = np.stack([
            (xs_v - self.cx_cam) / self.fx_cam * ds_v,
            (ys_v - self.cy_cam) / self.fy_cam * ds_v,
            ds_v], axis=1)
        base_pts = (R @ cam_pts.T).T + t
        return base_pts, int(valid.sum())

    def get_3d_position(self, bbox: list) -> Tuple:
        """Get median 3D position in base_footprint from depth + TF."""
        if self.latest_depth is None:
            return None, None, 0
        base_pts, n = self._depth_roi_to_base(self.latest_depth, bbox)
        if base_pts is None or n < 8:
            return None, None, 0
        centroid = np.median(base_pts, axis=0)
        return centroid.tolist(), 'base_footprint', n

    # ── Face enrichment ───────────────────────────────────────────────────────

    def enrich_with_faces(self, detections: List[Dict],
                          image: np.ndarray) -> List[Dict]:
        """
        For every 'person' detection, run face recognition and attach
        name / is_known / face_confidence / face_bbox to the dict.
        Returns the same list (mutated in-place).
        """
        if not (self.face_manager and self.face_manager.is_available()):
            return detections

        person_dets = [d for d in detections if d['class_name'] == 'person']
        if not person_dets:
            return detections

        faces = self.face_manager.recognize(image)

        for det in person_dets:
            px, py, pw, ph = det['bbox']
            p_cx, p_cy = px + pw // 2, py + ph // 2
            best_face, best_dist = None, float('inf')

            for face in faces:
                fx, fy, fw, fh = face['bbox']
                f_cx, f_cy = fx + fw // 2, fy + fh // 2
                dist = ((p_cx - f_cx) ** 2 + (p_cy - f_cy) ** 2) ** 0.5
                if fx >= px - 20 and fy >= py - 20 and dist < best_dist:
                    best_dist = dist
                    best_face = face

            if best_face:
                det['name'] = best_face['name']
                det['is_known'] = best_face['is_known']
                det['face_confidence'] = best_face['confidence']
                det['face_bbox'] = best_face['bbox']
            else:
                det.setdefault('name', 'unknown')
                det.setdefault('is_known', False)
                det.setdefault('face_confidence', 0.0)
                det.setdefault('face_bbox', det['bbox'])

        return detections

    # ── Visualization ─────────────────────────────────────────────────────────

    def draw_detections(self, image: np.ndarray,
                        detections: List[Dict]) -> np.ndarray:
        annotated = image.copy()
        for det in detections:
            x, y, w, h = det['bbox']
            label = "{} {:.2f}".format(det['class_name'], det['confidence'])
            if det['class_name'] == 'person' and det.get('name'):
                label = "{} ({})".format(det.get('name', '?'), "{:.0f}%".format(
                    det.get('face_confidence', 0) * 100))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
            lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x, y - lsz[1] - 8),
                          (x + lsz[0], y), (0, 200, 0), -1)
            cv2.putText(annotated, label, (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return annotated
