#!/usr/bin/env python3
"""
Detect an object with VLM and reach for it with TIAGo's arm.
- Gemini VLM for object detection (replaces YOLO)
- Synchronized RGB+depth via message_filters
- Proper TF via tf.TransformListener
- MoveIt arm control via moveit_commander
"""

import sys
import argparse
import rospy
import actionlib
import numpy as np
import cv2
import os
import subprocess
import re
import message_filters
import requests
from image_geometry import PinholeCameraModel

from tiago_lissi.agent.vlm_reasoner import VLMReasoner

from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from moveit_msgs.msg import (MoveGroupAction, MoveGroupGoal, Constraints,
                              PositionConstraint, OrientationConstraint,
                              BoundingVolume, CollisionObject, PlanningScene)
from shape_msgs.msg import SolidPrimitive
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class BottleReacher:
    def __init__(self, target_class='bottle', target_description=''):
        rospy.init_node('bottle_reacher', anonymous=True)

        self.target_class         = target_class
        # Prefer env var (full chain-of-thought from agent, no length limit) over CLI arg
        self.target_description   = os.environ.get('TARGET_DESCRIPTION', target_description)
        self.confidence_threshold = 0.4
        self._cached_bbox         = None  # single VLM call, reused for all 5 depth frames
        self._detection_frame     = None  # RGB frame captured at detection time (for debug overlay)
        self._frame_idx           = 0

        rospy.loginfo("[Reach] Initializing VLM for object detection (target: {})".format(target_class))
        self.vlm = VLMReasoner()

        # ── Camera model (image_geometry handles intrinsics safely) ──────────
        rospy.loginfo("Waiting for camera info...")
        cam_info = rospy.wait_for_message('/xtion/rgb/camera_info', CameraInfo, timeout=10)
        self.cam_model = PinholeCameraModel()
        self.cam_model.fromCameraInfo(cam_info)
        # Intrinsics for vectorised depth-to-3D projection
        self.fx_cam = self.cam_model.fx()
        self.fy_cam = self.cam_model.fy()
        self.cx_cam = self.cam_model.cx()
        self.cy_cam = self.cam_model.cy()
        rospy.loginfo("Camera model loaded: fx={:.1f} fy={:.1f}".format(
            self.fx_cam, self.fy_cam))

        # ── Synchronized RGB + Depth ──────────────────────────────────────────
        self.latest_rgb   = None
        self.latest_depth = None   # numpy float32 array, metres
        self.latest_vis   = None   # annotated frame, shown from main thread

        rgb_sub   = message_filters.Subscriber('/xtion/rgb/image_raw', Image)
        # Use aligned depth — already registered to RGB frame
        depth_sub = message_filters.Subscriber('/xtion/depth_registered/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=0.1)
        self.ts.registerCallback(self._sync_callback)
        rospy.loginfo("Waiting for first synchronised RGB+depth frame...")
        while self.latest_rgb is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Got synchronised frames.")

        # ── play_motion ───────────────────────────────────────────────────────
        rospy.loginfo("Connecting to play_motion...")
        self.play_motion_client = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)
        self.play_motion_client.wait_for_server(timeout=rospy.Duration(10))

        # ── Torso controller (for trunk raise) ───────────────────────────────
        rospy.loginfo("Connecting to torso controller...")
        self.torso_client = actionlib.SimpleActionClient(
            '/torso_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.torso_client.wait_for_server(timeout=rospy.Duration(10))

        # ── Head controller ───────────────────────────────────────────────────
        rospy.loginfo("Connecting to head controller...")
        self.head_client = actionlib.SimpleActionClient(
            '/head_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.head_client.wait_for_server(timeout=rospy.Duration(10))

        # ── MoveIt action client ──────────────────────────────────────────────
        rospy.loginfo("Connecting to move_group...")
        self.move_group_client = actionlib.SimpleActionClient('/move_group', MoveGroupAction)
        self.move_group_client.wait_for_server(timeout=rospy.Duration(10))

        # ── Joint state listener (for torso height) ───────────────────────────
        self.current_torso_height = None
        rospy.Subscriber('/joint_states', JointState, self._joint_state_cb)

        # Planning scene publisher
        self.scene_pub = rospy.Publisher('/planning_scene', PlanningScene, queue_size=1)
        rospy.sleep(0.5)

        # ── Grasp config (calibrated with calibrate_grasp.py — PAL gripper) ───
        self.grasp_orientation = Quaternion(x=0.7030, y=0.0730, z=-0.0340, w=0.7060)

        # Grasp offsets — loaded from grasp_offsets.yaml if it exists,
        # then overridden by env vars (set by calibrate_grasp_interactive.py)
        import yaml as _yaml, os as _os
        _repo_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        _offsets_file = _os.path.join(_repo_root, 'grasp_offsets.yaml')
        if _os.path.exists(_offsets_file):
            with open(_offsets_file) as _f:
                _off = _yaml.safe_load(_f)
            self.grasp_dx = float(_off.get('grasp_dx', 0.16))
            self.grasp_dy = float(_off.get('grasp_dy', -0.04))
            self.grasp_dz = float(_off.get('grasp_dz', -0.04))
            rospy.loginfo("Loaded grasp offsets from {}".format(_offsets_file))
        else:
            self.grasp_dx =  0.16
            self.grasp_dy = -0.04
            self.grasp_dz = -0.04
        # Env var override (used by calibrate_grasp_interactive.py)
        if _os.environ.get('GRASP_DX'): self.grasp_dx = float(_os.environ['GRASP_DX'])
        if _os.environ.get('GRASP_DY'): self.grasp_dy = float(_os.environ['GRASP_DY'])
        if _os.environ.get('GRASP_DZ'): self.grasp_dz = float(_os.environ['GRASP_DZ'])
        rospy.loginfo("Grasp offsets: dx={:.3f} dy={:.3f} dz={:.3f}".format(
            self.grasp_dx, self.grasp_dy, self.grasp_dz))

        self.confidence_threshold  = 0.4

        rospy.loginfo("Ready — place a {} in front of the robot.".format(self.target_class))

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _joint_state_cb(self, msg):
        if 'torso_lift_joint' in msg.name:
            idx = msg.name.index('torso_lift_joint')
            self.current_torso_height = float(msg.position[idx])

    def _sync_callback(self, rgb_msg, depth_msg):
        """Store latest time-synchronised RGB image and depth array.
        No imshow here — OpenCV GUI must run on the main thread."""
        self.latest_rgb = rgb_msg
        try:
            if depth_msg.encoding == '32FC1':
                arr = np.frombuffer(depth_msg.data, dtype=np.float32)
            elif depth_msg.encoding == '16UC1':
                arr = np.frombuffer(depth_msg.data, dtype=np.uint16).astype(np.float32) / 1000.0
            else:
                return
            self.latest_depth = arr.reshape(depth_msg.height, depth_msg.width)
        except Exception as e:
            rospy.logwarn("Depth decode error: {}".format(e))

    # ── YOLO bounding-box detection ───────────────────────────────────────────
    def _decode_rgb(self, rgb_msg):
        if rgb_msg.encoding == 'rgb8':
            img = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(
                rgb_msg.height, rgb_msg.width, 3)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif rgb_msg.encoding == 'bgr8':
            return np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(
                rgb_msg.height, rgb_msg.width, 3)
        return None

    def _detect_box(self, img_bgr, frame_idx=0):
        """Return [bx, by, bw, bh] of target object using VLM detection.
        Result is cached after first successful call — reused for all 5 depth-fit frames.
        A debug image is saved to /tmp/ on first detection for inspection."""
        # Reuse cached bbox for subsequent depth-fit frames (single VLM call per grasp)
        if self._cached_bbox is not None:
            rospy.logdebug("[Reach] Using cached bbox {}".format(self._cached_bbox))
            return self._cached_bbox

        graspable_kw = ('bottle', 'cup', 'mug', 'can', 'phone', 'box', 'book', 'remote',
                         'case', 'trunk', 'container', 'bag', 'object', 'item', 'package',
                         'jar', 'bowl', 'plate', 'tool', 'toy')

        # Build rich target hint using agent rationale + ontology context
        if self.target_description:
            target_hint = (
                '{cls}. The operator selected this specific object: "{desc}". '
                'Return ONLY that one object. If multiple {cls}s are visible, '
                'return the one that best matches the description.'
            ).format(cls=self.target_class, desc=self.target_description[:200])
        else:
            target_hint = self.target_class

        # Try the specific target class first.
        # Primary: VLM may return descriptive labels ("small bottle") — match exactly.
        # Fallback: VLM returns generic labels ("bottle") — match on base noun.
        base_noun = self.target_class.split()[-1].lower()
        detections = self.vlm.detect_objects(img_bgr, target_classes=[target_hint])
        matched = [d for d in detections
                   if self.target_class.lower() in d['class_name'].lower()
                   or d['class_name'].lower() in self.target_class.lower()]
        if not matched:
            # Base noun fallback (VLM returned generic labels)
            matched = [d for d in detections if base_noun in d['class_name'].lower()]

        if not matched:
            rospy.loginfo("[Reach] '{}' not found, trying any graspable object...".format(
                self.target_class))
            detections = self.vlm.detect_objects(img_bgr)
            matched = [d for d in detections
                       if any(kw in d['class_name'].lower() for kw in graspable_kw)]

        if not matched:
            self._save_debug_image(img_bgr, [], frame_idx, found=False)
            return None

        # Filter out implausibly large bboxes (>300px wide = likely background/furniture)
        plausible = [d for d in matched if d['bbox'][2] < 300]
        if not plausible:
            plausible = matched

        # When multiple candidates remain and target has a qualifier, ask the VLM to pick
        if len(plausible) > 1 and self.target_description:
            best = self._vlm_pick_best(img_bgr, plausible, self.target_class,
                                       self.target_description)
        else:
            best = max(plausible, key=lambda d: d['confidence'])
        rospy.loginfo("[Reach] VLM detected '{}' @ bbox {} area={}px² — caching".format(
            best['class_name'], best['bbox'], best['bbox'][2] * best['bbox'][3]))

        self._save_debug_image(img_bgr, plausible, frame_idx, highlight=best)
        self._cached_bbox     = best['bbox']
        self._detection_frame = img_bgr.copy()  # freeze this frame for cylinder overlay
        return self._cached_bbox

    def _vlm_pick_best(self, img_bgr, candidates, target_class, description):
        """When multiple candidates match, draw numbered labels and ask VLM to pick one."""
        import re as _re
        annotated = img_bgr.copy()
        for i, d in enumerate(candidates):
            bx, by, bw, bh = d['bbox']
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 200, 255), 2)
            cv2.putText(annotated, str(i + 1), (bx + 4, by + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        from PIL import Image as PILImage
        pil = PILImage.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        prompt = (
            "The image shows {} numbered objects (1 to {}).\n"
            "The robot wants to grasp: '{}'\n\n"
            "Reasoning from the operator's AI agent (use this as context):\n"
            "{}\n\n"
            "Based on the above reasoning, reply with ONLY the number of the correct "
            "object to grasp (e.g. '1' or '2'). No explanation needed."
        ).format(len(candidates), len(candidates), target_class, description)

        try:
            text = self.vlm._call_api([prompt, pil], temperature=0.0, max_tokens=8, timeout=10)
            nums = _re.findall(r'\d+', text.strip())
            if nums:
                idx = int(nums[0]) - 1
                if 0 <= idx < len(candidates):
                    rospy.loginfo("[Reach] VLM picked candidate {} for '{}'".format(
                        idx + 1, target_class))
                    return candidates[idx]
        except Exception as e:
            rospy.logwarn("[Reach] VLM pick failed: {}".format(e))

        # Fallback: highest confidence
        return max(candidates, key=lambda d: d['confidence'])

    def _save_debug_image(self, img_bgr, detections, frame_idx, found=True, highlight=None):
        """Draw all detections + highlight selected one. Save to /tmp/."""
        debug = img_bgr.copy()
        for d in detections:
            bx, by, bw, bh = d['bbox']
            color = (0, 255, 0) if d is highlight else (0, 165, 255)
            cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), color, 2)
            label = '{} {:.0%}'.format(d['class_name'], d.get('confidence', 0))
            cv2.putText(debug, label, (bx, max(by - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        status = 'FOUND' if found else 'NOT_FOUND'
        path = '/tmp/tiago_det_f{}_{}.jpg'.format(frame_idx, status)
        cv2.imwrite(path, debug)
        rospy.loginfo("[Reach] Debug image: {}".format(path))

    # ── TF via tf_echo subprocess ─────────────────────────────────────────────
    def _tf_echo(self, parent, child):
        """Get TF between two frames via rosrun tf tf_echo.
        Returns (t as np.array, R as 3×3 np.array) or (None, None)."""
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
            t_m = re.search(r'Translation: \[([^,]+), ([^,]+), ([^\]]+)\]', output)
            q_m = re.search(r'Quaternion \[([^,]+), ([^,]+), ([^,]+), ([^\]]+)\]', output)
            if t_m and q_m:
                t = np.array([float(t_m.group(i)) for i in range(1, 4)])
                q = [float(q_m.group(i)) for i in range(1, 5)]
                qx, qy, qz, qw = q
                R = np.array([
                    [1-2*(qy*qy+qz*qz),  2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
                    [2*(qx*qy+qz*qw),    1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                    [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
                ])
                rospy.loginfo("TF {}->{}: t={}".format(parent, child, np.round(t, 3)))
                return t, R
        except Exception as e:
            rospy.logerr("tf_echo failed: {}".format(e))
        return None, None

    # ── Point cloud projection ────────────────────────────────────────────────
    def _depth_roi_to_base(self, depth, box, R, t, step=2):
        """Vectorised: project depth pixels inside YOLO box → base_footprint XYZ (N×3)."""
        bx, by, bw, bh = box
        margin = 4
        x0 = max(0, bx+margin);  x1 = min(depth.shape[1], bx+bw-margin)
        y0 = max(0, by+margin);  y1 = min(depth.shape[0], by+bh-margin)
        ys, xs = np.mgrid[y0:y1:step, x0:x1:step]
        ds = depth[ys, xs]
        # 0.15–3.5 m: same as original YOLO version; depth-refinement already isolated the object
        valid = np.isfinite(ds) & (ds > 0.15) & (ds < 3.5)
        xs_v = xs[valid].ravel(); ys_v = ys[valid].ravel(); ds_v = ds[valid].ravel()
        if len(ds_v) == 0:
            return None
        cam_pts = np.stack([(xs_v - self.cx_cam) / self.fx_cam * ds_v,
                            (ys_v - self.cy_cam) / self.fy_cam * ds_v,
                            ds_v], axis=1)
        return (R @ cam_pts.T).T + t

    def _save_cylinder_debug(self, cyl, R, t, path='/tmp/tiago_cylinder.jpg'):
        """Project the fitted cylinder back into image space and save annotated frame."""
        # Use the frame captured at detection time so TF and image are consistent
        img = self._detection_frame if self._detection_frame is not None \
              else self._decode_rgb(self.latest_rgb)
        if img is None or self.fx_cam is None:
            return
        debug = img.copy()
        fx = self.fx_cam; fy = self.fy_cam
        cx = self.cx_cam; cy_cam = self.cy_cam

        # Project cylinder axis points (top and bottom) back to camera frame
        def base_to_pixel(world_pt):
            """world_pt in base_footprint → pixel (u,v)."""
            # camera_pt = R^T (world_pt - t)
            cam_pt = R.T @ (np.array(world_pt) - t)
            if cam_pt[2] <= 0.01:
                return None
            u = int(fx * cam_pt[0] / cam_pt[2] + cx)
            v = int(fy * cam_pt[1] / cam_pt[2] + cy_cam)
            return (u, v)

        # Draw cylinder top circle (project 12 points on circle circumference)
        import math
        cyl_cx, cyl_cy, r = cyl['cx'], cyl['cy'], cyl['radius']
        for z_level, color in [(cyl['z_top'], (0, 255, 0)), (cyl['z_bottom'], (255, 128, 0))]:
            pts_img = []
            for i in range(24):
                angle = 2 * math.pi * i / 24
                world_pt = [cyl_cx + r * math.cos(angle),
                            cyl_cy + r * math.sin(angle),
                            z_level]
                px = base_to_pixel(world_pt)
                if px:
                    pts_img.append(px)
            for i in range(len(pts_img)):
                cv2.line(debug, pts_img[i], pts_img[(i + 1) % len(pts_img)], color, 2)

        # Draw vertical lines connecting top and bottom circles at 4 points
        for angle_deg in [0, 90, 180, 270]:
            angle = math.radians(angle_deg)
            top = base_to_pixel([cyl_cx + r * math.cos(angle),
                                  cyl_cy + r * math.sin(angle), cyl['z_top']])
            bot = base_to_pixel([cyl_cx + r * math.cos(angle),
                                  cyl_cy + r * math.sin(angle), cyl['z_bottom']])
            if top and bot:
                cv2.line(debug, top, bot, (0, 200, 255), 1)

        # Draw centre axis dot
        centre_px = base_to_pixel([cyl_cx, cyl_cy, (cyl['z_top'] + cyl['z_bottom']) / 2])
        if centre_px:
            cv2.circle(debug, centre_px, 5, (0, 0, 255), -1)

        # Label
        top_px = base_to_pixel([cyl_cx, cyl_cy, cyl['z_top']])
        if top_px:
            label = 'r={:.1f}cm h={:.1f}cm'.format(r * 100,
                                                     (cyl['z_top'] - cyl['z_bottom']) * 100)
            cv2.putText(debug, label, (top_px[0] + 5, top_px[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(path, debug)
        rospy.loginfo("[Reach] Cylinder debug image: {}".format(path))

    # ── RANSAC vertical cylinder fit ──────────────────────────────────────────
    def _fit_cylinder(self, pts, n_iters=300, thresh=0.015):
        """Fit a vertical cylinder (axis = world Z) via RANSAC circle fitting in XY.
        Radius range 0.01–0.25m handles bottles (~3cm) and boxes (~20cm).
        Falls back to centroid pose if RANSAC finds no cylinder (flat-faced objects).
        Returns dict {cx, cy, radius, z_top, z_bottom, inlier_ratio} or None."""
        xy = pts[:, :2]; z = pts[:, 2]; n = len(xy)
        if n < 15:
            return None
        best = {'inliers': 0, 'mask': None}
        for _ in range(n_iters):
            idx = np.random.choice(n, 3, replace=False)
            ax, ay = xy[idx[0]]; bx2, by2 = xy[idx[1]]; cx2, cy2 = xy[idx[2]]
            D = 2*(ax*(by2-cy2) + bx2*(cy2-ay) + cx2*(ay-by2))
            if abs(D) < 1e-9:
                continue
            ux = ((ax**2+ay**2)*(by2-cy2) + (bx2**2+by2**2)*(cy2-ay) + (cx2**2+cy2**2)*(ay-by2)) / D
            uy = ((ax**2+ay**2)*(cx2-bx2) + (bx2**2+by2**2)*(ax-cx2) + (cx2**2+cy2**2)*(bx2-ax)) / D
            r = np.hypot(ax-ux, ay-uy)
            if r < 0.015 or r > 0.06:   # was 0.015–0.06 (bottles only); now covers boxes too
                continue
            mask = np.abs(np.hypot(xy[:,0]-ux, xy[:,1]-uy) - r) < thresh
            if mask.sum() > best['inliers']:
                best = {'inliers': int(mask.sum()), 'ux': ux, 'uy': uy, 'r': r, 'mask': mask}
        if best['inliers'] >= 8:
            # Algebraic LS refinement on inliers
            xi, yi = xy[best['mask'], 0], xy[best['mask'], 1]
            A = np.column_stack([2*xi, 2*yi, np.ones(len(xi))])
            sol, *_ = np.linalg.lstsq(A, xi**2 + yi**2, rcond=-1)
            cx_r, cy_r = float(sol[0]), float(sol[1])
            r_r = float(np.mean(np.hypot(xi-cx_r, yi-cy_r)))
            mask2 = np.abs(np.hypot(xy[:,0]-cx_r, xy[:,1]-cy_r) - r_r) < thresh
            zi = z[mask2]
            if len(zi) == 0:
                zi = z
            return dict(cx=cx_r, cy=cy_r, radius=r_r,
                        z_top=float(np.percentile(zi, 95)),
                        z_bottom=float(np.percentile(zi, 5)),
                        inlier_ratio=float(mask2.sum()) / n)
        # ── Centroid fallback for flat/box-shaped objects ──────────────────────
        # RANSAC found no cylinder — use the centroid of the near-depth cluster
        # (closest 50% of points = front face of the object, not background)
        depth_thresh = float(np.percentile(pts[:, 0], 50))  # median X (forward depth)
        front = pts[pts[:, 0] <= depth_thresh + 0.05]
        if len(front) < 5:
            front = pts
        cx_r = float(np.median(front[:, 0]))
        cy_r = float(np.median(front[:, 1]))
        z_vals = front[:, 2]
        # Approximate object half-width as radius for collision object
        r_r = float(np.std(front[:, 1]) + 0.05)
        r_r = float(np.clip(r_r, 0.04, 0.20))
        rospy.loginfo("[Reach] Cylinder RANSAC failed — using centroid fallback "
                      "(cx={:.3f} cy={:.3f} r={:.3f})".format(cx_r, cy_r, r_r))
        return dict(cx=cx_r, cy=cy_r, radius=r_r,
                    z_top=float(np.percentile(z_vals, 90)),
                    z_bottom=float(np.percentile(z_vals, 10)),
                    inlier_ratio=0.5)

    # ── Cylinder-based bottle pose estimation ─────────────────────────────────
    def _estimate_table_z(self, depth, box, R, t):
        """Estimate table Z in base_footprint by sampling depth just below the YOLO box bottom.
        The bottom edge of the YOLO box is where the bottle meets the table surface."""
        bx, by, bw, bh = box
        # Strip of pixels just below the box bottom (table surface)
        y0 = min(by + bh,     depth.shape[0] - 1)
        y1 = min(by + bh + 8, depth.shape[0])
        x0 = max(0, bx + bw // 4)
        x1 = min(depth.shape[1], bx + 3 * bw // 4)
        strip = depth[y0:y1, x0:x1]
        valid = strip[np.isfinite(strip) & (strip > 0.15) & (strip < 3.5)]
        if len(valid) < 5:
            return None
        d = float(np.median(valid))
        px = (x0 + x1) // 2
        py = (y0 + y1) // 2
        cam_pt = np.array([(px - self.cx_cam) / self.fx_cam * d,
                           (py - self.cy_cam) / self.fy_cam * d, d])
        return float((R @ cam_pt + t)[2])

    def _refine_bbox_by_depth(self, depth, bbox):
        """Shrink a loose VLM bbox to the nearest-depth cluster within it.
        This gives the same tight-bbox quality that YOLO provided natively."""
        bx, by, bw, bh = bbox
        margin = 4
        x0 = max(0, bx + margin);  x1 = min(depth.shape[1], bx + bw - margin)
        y0 = max(0, by + margin);  y1 = min(depth.shape[0], by + bh - margin)
        if x1 <= x0 or y1 <= y0:
            return bbox
        roi = depth[y0:y1, x0:x1]
        valid = np.isfinite(roi) & (roi > 0.30) & (roi < 1.10)
        if valid.sum() < 10:
            return bbox  # can't refine, fall back to original
        # Nearest cluster = foreground object (10th percentile depth within ROI)
        near_d = float(np.percentile(roi[valid], 10))
        near_mask = valid & (roi < near_d + 0.12)  # 12 cm tolerance around front surface
        ys, xs = np.where(near_mask)
        if len(ys) < 5:
            return bbox
        rx0 = int(xs.min()) + x0;  rx1 = int(xs.max()) + x0
        ry0 = int(ys.min()) + y0;  ry1 = int(ys.max()) + y0
        refined = [rx0, ry0, max(1, rx1 - rx0), max(1, ry1 - ry0)]
        rospy.loginfo("[Reach] Depth-refined bbox: {}×{} → {}×{}".format(
            bw, bh, refined[2], refined[3]))
        return refined

    def detect_bottle_pose(self, R, t):
        """Single-frame: VLM box → depth-refined box → cylinder fit → base_footprint pose.
        R, t: cached TF rotation matrix and translation (call _tf_echo once externally).
        Returns dict {cx, cy, z_top, z_bottom, z_table, radius, inlier_ratio} or None."""
        img_bgr = self._decode_rgb(self.latest_rgb)
        if img_bgr is None:
            return None
        self._frame_idx += 1
        box = self._detect_box(img_bgr, frame_idx=self._frame_idx)
        if box is None:
            return None
        depth = self.latest_depth
        # Shrink loose VLM bbox to nearest-depth cluster — restores YOLO-like point cloud quality
        box = self._refine_bbox_by_depth(depth, box)
        pts = self._depth_roi_to_base(depth, box, R, t)
        if pts is None or len(pts) < 15:
            return None
        cyl = self._fit_cylinder(pts)
        if cyl is None:
            return None
        # Table height: sample depth just below YOLO box bottom (most direct estimate)
        z_table = self._estimate_table_z(depth, box, R, t)
        cyl['z_table'] = z_table if z_table is not None else cyl['z_bottom'] - 0.01
        return cyl

    def detect_bottle_pose_stable(self, R, t, n_frames=5, max_attempts=30):
        """Average n_frames cylinder fits for a stable pose. Returns averaged dict or None."""
        samples = []
        attempts = 0
        rospy.loginfo("Collecting {} cylinder-fit samples...".format(n_frames))
        while len(samples) < n_frames and attempts < max_attempts and not rospy.is_shutdown():
            attempts += 1
            cyl = self.detect_bottle_pose(R, t)
            if cyl and cyl['inlier_ratio'] > 0.3:
                # Reachability sanity check — reject poses outside arm workspace
                if cyl['cx'] > 1.05 or abs(cyl['cy']) > 0.65:
                    rospy.logwarn("[Reach] Pose out of reach (x={:.3f} y={:.3f}) — skipping".format(
                        cyl['cx'], cyl['cy']))
                else:
                    samples.append(cyl)
                    rospy.loginfo("  Sample {}/{}: XY=({:.3f},{:.3f}) r={:.3f} z_top={:.3f} inliers={:.0%}".format(
                        len(samples), n_frames, cyl['cx'], cyl['cy'],
                        cyl['radius'], cyl['z_top'], cyl['inlier_ratio']))
            else:
                rospy.sleep(0.1)
        if not samples:
            rospy.logwarn("[Reach] No valid depth samples after {} attempts.".format(attempts))
            return None
        keys = ['cx', 'cy', 'radius', 'z_top', 'z_bottom', 'z_table']
        result = {k: float(np.mean([s[k] for s in samples])) for k in keys}
        rospy.loginfo("Stable bottle: XY=({:.3f}±{:.3f}, {:.3f}±{:.3f}) r={:.3f} z_top={:.3f}".format(
            result['cx'], float(np.std([s['cx'] for s in samples])),
            result['cy'], float(np.std([s['cy'] for s in samples])),
            result['radius'], result['z_top']))
        return result

    # ── Planning scene ────────────────────────────────────────────────────────
    def add_table_collision(self, table_x, table_z):
        """Add table as a box collision object."""
        co = CollisionObject()
        co.header.frame_id = "base_footprint"
        co.header.stamp = rospy.Time.now()
        co.id = "table"
        co.operation = CollisionObject.ADD
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.8, 1.2, 0.02]
        box_pose = Pose()
        box_pose.position.x = table_x
        box_pose.position.y = 0.0
        box_pose.position.z = table_z
        box_pose.orientation.w = 1.0
        co.primitives.append(box)
        co.primitive_poses.append(box_pose)
        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)
        self.scene_pub.publish(ps)
        rospy.loginfo("Table collision added at z={:.3f}".format(table_z))
        rospy.sleep(0.5)

    def add_bottle_collision(self, cx, cy, z_bottom, z_top, radius):
        """Add bottle as a cylinder collision object so the arm plans around it."""
        co = CollisionObject()
        co.header.frame_id = "base_footprint"
        co.header.stamp = rospy.Time.now()
        co.id = "bottle"
        co.operation = CollisionObject.ADD
        cyl = SolidPrimitive()
        cyl.type = SolidPrimitive.CYLINDER
        height = max(z_top - z_bottom, 0.05)
        cyl.dimensions = [height, radius + 0.01]   # add 1 cm safety margin around radius
        cyl_pose = Pose()
        cyl_pose.position.x = cx
        cyl_pose.position.y = cy
        cyl_pose.position.z = z_bottom + height / 2.0
        cyl_pose.orientation.w = 1.0
        co.primitives.append(cyl)
        co.primitive_poses.append(cyl_pose)
        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)
        self.scene_pub.publish(ps)
        rospy.loginfo("Bottle collision added: centre=({:.3f},{:.3f}) r={:.3f} h={:.3f}".format(
            cx, cy, radius, height))
        rospy.sleep(0.5)

    def _add_obstacle_objects(self):
        """Add non-target detected objects as box collision obstacles from OBSTACLE_OBJECTS env var."""
        import json as _json, os as _os
        raw = _os.environ.get('OBSTACLE_OBJECTS', '')
        if not raw:
            return
        try:
            obstacles = _json.loads(raw)
        except Exception:
            return
        for obs in obstacles:
            pos = obs.get('position_3d')
            if not pos or len(pos) < 3:
                continue
            ox, oy, oz = float(pos[0]), float(pos[1]), float(pos[2])
            name = obs.get('name', 'obstacle').replace(' ', '_')
            co = CollisionObject()
            co.header.frame_id = "base_footprint"
            co.header.stamp = rospy.Time.now()
            co.id = "obs_{}".format(name)
            co.operation = CollisionObject.ADD
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [0.12, 0.12, 0.20]  # conservative 12×12×20 cm box
            box_pose = Pose()
            box_pose.position.x = ox
            box_pose.position.y = oy
            box_pose.position.z = oz
            box_pose.orientation.w = 1.0
            co.primitives.append(box)
            co.primitive_poses.append(box_pose)
            ps = PlanningScene()
            ps.is_diff = True
            ps.world.collision_objects.append(co)
            self.scene_pub.publish(ps)
            rospy.loginfo("[Reach] Obstacle '{}' added at ({:.2f},{:.2f},{:.2f})".format(
                name, ox, oy, oz))
        rospy.sleep(0.3)

    def remove_collision_object(self, obj_id):
        co = CollisionObject()
        co.header.frame_id = "base_footprint"
        co.header.stamp = rospy.Time.now()
        co.id = obj_id
        co.operation = CollisionObject.REMOVE
        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)
        self.scene_pub.publish(ps)
        rospy.sleep(0.3)

    # ── Arm control ───────────────────────────────────────────────────────────
    def raise_trunk(self, height_m=0.10):
        """Raise torso_lift_joint to height_m (0–0.35 m).
        Paper recommends d≈0.10 m for optimal manipulability (Bajrami et al. 2024)."""
        rospy.loginfo("Raising trunk to {:.2f} m...".format(height_m))
        height_m = float(np.clip(height_m, 0.0, 0.35))
        traj = JointTrajectory()
        traj.joint_names = ['torso_lift_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [height_m]
        pt.time_from_start = rospy.Duration(3.0)
        traj.points.append(pt)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj
        self.torso_client.send_goal(goal)
        self.torso_client.wait_for_result(rospy.Duration(10))
        rospy.loginfo("Trunk raised.")

    def play_motion(self, name):
        rospy.loginfo("Playing motion: {}".format(name))
        goal = PlayMotionGoal()
        goal.motion_name  = name
        goal.skip_planning = False
        self.play_motion_client.send_goal(goal)
        finished = self.play_motion_client.wait_for_result(rospy.Duration(60))
        state = self.play_motion_client.get_state()
        if not finished or state != actionlib.GoalStatus.SUCCEEDED:
            rospy.logwarn("Motion '{}' ended with state {}".format(name, state))

    def _send_arm_goal(self, x, y, z, planner_id="", path_constraints=None):
        """Send arm_tool_link to (x, y, z) with grasp orientation.
        Optional path_constraints (Constraints) applied to the entire motion. Returns True on success."""
        goal = MoveGroupGoal()
        goal.request.group_name = "arm_torso"
        goal.request.planner_id = planner_id
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 10.0

        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_footprint"
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z
        target_pose.pose.orientation = self.grasp_orientation

        pos_c = PositionConstraint()
        pos_c.header = target_pose.header
        pos_c.link_name = "arm_tool_link"
        bv = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.05]
        bv.primitives.append(sphere)
        bv.primitive_poses.append(target_pose.pose)
        pos_c.constraint_region = bv
        pos_c.weight = 1.0

        ori_c = OrientationConstraint()
        ori_c.header = target_pose.header
        ori_c.link_name = "arm_tool_link"
        ori_c.orientation = self.grasp_orientation
        ori_c.absolute_x_axis_tolerance = 0.2
        ori_c.absolute_y_axis_tolerance = 0.2
        ori_c.absolute_z_axis_tolerance = 0.2
        ori_c.weight = 0.5

        constraints = Constraints()
        constraints.position_constraints.append(pos_c)
        constraints.orientation_constraints.append(ori_c)
        goal.request.goal_constraints.append(constraints)
        if path_constraints is not None:
            goal.request.path_constraints = path_constraints

        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        self.move_group_client.send_goal(goal)
        self.move_group_client.wait_for_result(timeout=rospy.Duration(30))
        return self.move_group_client.get_state() == actionlib.GoalStatus.SUCCEEDED

    def _torso_descent(self, delta_z):
        """Lower torso_lift_joint by delta_z metres for a straight vertical descent.
        The torso is a prismatic joint along world Z, so arm_tool_link drops exactly
        delta_z in world space — no IK, no planner, guaranteed straight line."""
        if self.current_torso_height is None:
            rospy.logerr("Torso joint state not yet received — cannot descend.")
            return False
        target = float(np.clip(self.current_torso_height - delta_z, 0.0, 0.35))
        rospy.loginfo("Torso descent: {:.3f} → {:.3f} (delta={:.3f} m)".format(
            self.current_torso_height, target, delta_z))
        self.raise_trunk(target)
        return True

    def move_arm_to_pose(self, bx, by, bz, z_bottom, bottle_radius=0.035):
        """Top-down approach: OMPL to above-bottle (avoiding it), then torso descent to grasp."""
        grasp_x = bx - self.grasp_dx
        grasp_y = by - self.grasp_dy
        grasp_z = bz - self.grasp_dz

        descent = 0.25                        # metres to descend with torso
        above_z = grasp_z + descent +0.025          # pre-grasp height (30 cm above grasp)

        rospy.loginfo("Above:  ({:.3f}, {:.3f}, {:.3f})".format(grasp_x, grasp_y, above_z))
        rospy.loginfo("Grasp:  ({:.3f}, {:.3f}, {:.3f})".format(grasp_x, grasp_y, grasp_z))

        # Add bottle so OMPL routes around it on the way to the pre-grasp pose
        self.add_bottle_collision(bx, by, z_bottom, bz, bottle_radius)

        # Add other detected objects as box obstacles
        self._add_obstacle_objects()

        rospy.loginfo("Step 6a: OMPL to above-bottle pose (bottle as obstacle)...")
        if not self._send_arm_goal(grasp_x, grasp_y, above_z):
            rospy.logwarn("Above-bottle motion failed.")
            self.remove_collision_object("bottle")
            return False

        # Remove bottle before descent — the gripper intentionally enters that space
        self.remove_collision_object("bottle")

        rospy.loginfo("Step 6b: Torso descent {:.0f} cm straight down...".format(descent * 100))
        if not self._torso_descent(descent):
            rospy.logwarn("Torso descent failed.")
            return False

        rospy.loginfo("Arm reached grasp pose.")
        return True

    # ── Head control ──────────────────────────────────────────────────────────
    def _tilt_head(self, tilt=-0.5, duration=2.0):
        """Tilt head to the given tilt angle, preserving the current pan (left-right)."""
        try:
            js = rospy.wait_for_message('/joint_states', JointState, timeout=2.0)
            pan = js.position[js.name.index('head_1_joint')] \
                if 'head_1_joint' in js.name else 0.0
        except Exception:
            pan = 0.0
        traj = JointTrajectory()
        traj.joint_names = ['head_1_joint', 'head_2_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [pan, tilt]
        pt.time_from_start = rospy.Duration(duration)
        traj.points.append(pt)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj
        self.head_client.send_goal(goal)
        self.head_client.wait_for_result(rospy.Duration(duration + 2.0))

    # ── Main pipeline ─────────────────────────────────────────────────────────
    def run(self):
        rospy.sleep(1)

        rospy.loginfo("=== Step 1: Tilt head toward table ===")
        self._tilt_head(tilt=-0.5)

        rospy.loginfo("=== Step 2: Unfold arm ===")
        self.play_motion('unfold_arm')
        rospy.sleep(1)

        rospy.loginfo("=== Step 3: Open hand ===")
        self.play_motion('open')
        rospy.sleep(1)

        rospy.loginfo("=== Step 4: Get camera TF (once) ===")
        tf_t, tf_R = self._tf_echo('base_footprint', 'xtion_rgb_optical_frame')
        if tf_t is None:
            rospy.logerr("TF not available — aborting.")
            return False

        rospy.loginfo("=== Step 5: Detect '{}' — cylinder fit (5-frame average) ===".format(
            self.target_class))
        cyl = self.detect_bottle_pose_stable(tf_R, tf_t, n_frames=5)

        if cyl is None:
            rospy.logwarn("'{}' not found in depth data. Aborting.".format(self.target_class))
            self.play_motion('home')
            return False

        # Save annotated image with projected cylinder
        self._save_cylinder_debug(cyl, tf_R, tf_t, '/tmp/tiago_cylinder.jpg')

        bx = cyl['cx']
        by = cyl['cy']
        bz = cyl['z_top']
        bottle_radius = cyl['radius']
        rospy.loginfo("Bottle pose: x={:.3f} y={:.3f} z_top={:.3f} r={:.3f}".format(
            bx, by, bz, bottle_radius))

        rospy.loginfo("=== Step 6: Planning scene ===")
        # z_table: depth sampled just below the YOLO box bottom = table surface.
        # Center the 2 cm collision box 1 cm below table surface so its top face aligns with it.
        self.add_table_collision(bx, cyl['z_table'] - 0.01)

        rospy.loginfo("=== Step 7: Move arm ===")
        success = self.move_arm_to_pose(bx, by, bz, cyl['z_bottom'], bottle_radius)

        if success:
            rospy.sleep(1)
            rospy.loginfo("=== Step 8: Close hand ===")
            self.play_motion('close')
            # Raise torso back up so the arm clears the table before going home.
            # This is safer than asking MoveIt to plan from below-table height.
            rospy.loginfo("=== Step 9: Raise torso to clear table ===")
            self.raise_trunk(0.30)
            rospy.loginfo("=== Grasp complete! ===")
        else:
            rospy.logwarn("Could not reach bottle. Going home.")
            self.remove_collision_object("table")
            self.play_motion('home')
            self._tilt_head(tilt=-0.5)

        return success


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='bottle',
                        help='Object class to detect and grasp (default: bottle)')
    parser.add_argument('--description', type=str, default='',
                        help='Textual description from agent first VLM call for disambiguation')
    args, _ = parser.parse_known_args()
    try:
        success = BottleReacher(target_class=args.target,
                                target_description=args.description).run()
        sys.exit(0 if success else 1)
    except rospy.ROSInterruptException:
        sys.exit(1)
    except Exception as e:
        rospy.logerr("Fatal error: {}".format(e))
        sys.exit(1)
