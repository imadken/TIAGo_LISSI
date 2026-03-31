#!/usr/bin/env python3
"""
Handover To Person Skill
Detects person with YOLO + depth, computes their 3D position,
and extends the arm toward them at handover height — same pipeline as reach_object_v5.
"""

import os
import sys
import re
import subprocess
import numpy as np
import rospy
import actionlib
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from moveit_msgs.msg import (MoveGroupAction, MoveGroupGoal, Constraints,
                              PositionConstraint, OrientationConstraint,
                              BoundingVolume, CollisionObject, PlanningScene)
from shape_msgs.msg import SolidPrimitive
from std_srvs.srv import Empty as EmptySrv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiago_lissi.skills.base_skill import BaseSkill


# Height (m above ground) at which to present the object for handover
HANDOVER_HEIGHT = 0.85

# How far to extend the arm toward the person (TIAGo arm reach ~0.65 m from base_footprint)
ARM_REACH = 0.62

# Orientation to hold the gripper during handover — same as v5 grasp so the
# bottle stays stable (already being held in this orientation)
HANDOVER_ORIENTATION = Quaternion(x=0.7030, y=0.0730, z=-0.0340, w=0.7060)


class HandoverSkill(BaseSkill):
    """Extend arm toward detected person and hand over held object."""

    def __init__(self, perception_manager=None, tts_speak_fn=None, navigation_client=None,
                 vlm_reasoner=None):
        """
        Args:
            perception_manager: Shared PerceptionManager (for RGB frames).
            tts_speak_fn: Callable speak(text) from EmbodiedAgent.
            vlm_reasoner: VLMReasoner instance for object/person detection.
            navigation_client: Reserved for future navigation support.
        """
        super().__init__('handover_to_person')

        self.speak_fn = tts_speak_fn
        self.perception = perception_manager
        self.vlm = vlm_reasoner

        # ── Camera intrinsics ────────────────────────────────────────────────
        rospy.loginfo("[Handover] Waiting for camera info...")
        cam_info = rospy.wait_for_message('/xtion/rgb/camera_info', CameraInfo, timeout=10)
        model = PinholeCameraModel()
        model.fromCameraInfo(cam_info)
        self.fx = model.fx()
        self.fy = model.fy()
        self.cx = model.cx()
        self.cy = model.cy()
        rospy.loginfo("[Handover] Camera: fx={:.1f} fy={:.1f}".format(self.fx, self.fy))

        # ── Depth subscriber ─────────────────────────────────────────────────
        self.latest_depth = None
        rospy.Subscriber('/xtion/depth_registered/image_raw', Image, self._depth_cb)

        # ── MoveIt ──────────────────────────────────────────────────────────
        self.moveit_client = actionlib.SimpleActionClient('/move_group', MoveGroupAction)
        rospy.loginfo("[Handover] Waiting for MoveIt move_group...")
        self.moveit_client.wait_for_server(timeout=rospy.Duration(15))
        rospy.loginfo("[Handover] MoveIt ready")

        # ── play_motion ──────────────────────────────────────────────────────
        self.play_motion_client = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)
        self.play_motion_client.wait_for_server(timeout=rospy.Duration(10))

        # ── Planning scene publisher (for collision objects) ──────────────────
        self.scene_pub = rospy.Publisher('/planning_scene', PlanningScene, queue_size=1)
        rospy.sleep(0.5)  # let publisher register

        rospy.loginfo("[Handover] Ready")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _depth_cb(self, msg):
        try:
            if msg.encoding == '32FC1':
                arr = np.frombuffer(msg.data, dtype=np.float32)
            elif msg.encoding == '16UC1':
                arr = np.frombuffer(msg.data, dtype=np.uint16).astype(np.float32) / 1000.0
            else:
                return
            self.latest_depth = arr.reshape(msg.height, msg.width)
        except Exception as e:
            rospy.logwarn("[Handover] Depth decode error: {}".format(e))

    # ── TF (same subprocess approach as v5) ───────────────────────────────────

    def _tf_echo(self, parent, child):
        """Get TF between two frames. Returns (t as np.array, R as 3x3) or (None, None)."""
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
                qx, qy, qz, qw = [float(q_m.group(i)) for i in range(1, 5)]
                R = np.array([
                    [1-2*(qy*qy+qz*qz),  2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
                    [2*(qx*qy+qz*qw),    1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                    [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
                ])
                rospy.loginfo("[Handover] TF {}->{}: t={}".format(parent, child, np.round(t, 3)))
                return t, R
        except Exception as e:
            rospy.logerr("[Handover] tf_echo failed: {}".format(e))
        return None, None

    # ── Person localisation ───────────────────────────────────────────────────

    def _person_3d_position(self, detection, depth, R, t):
        """
        Project the person's torso region from depth → base_footprint (x, y, z).
        Uses the centre-strip of the YOLO bounding box to avoid head/feet noise.
        Returns np.array([x, y, z]) or None.
        """
        x, y, w, h = detection['bbox']

        # Torso strip: middle third horizontally, middle half vertically
        mx = w // 3
        my = h // 4
        x0 = max(0, x + mx);  x1 = x + w - mx
        y0 = max(0, y + my);  y1 = y + h - my

        region = depth[y0:y1, x0:x1]
        valid = region[np.isfinite(region) & (region > 0.3) & (region < 5.0)]
        if len(valid) < 5:
            rospy.logwarn("[Handover] Not enough depth pixels in person bbox")
            return None

        d = float(np.median(valid))
        u = x + w // 2
        v = y + h // 2

        cam_pt = np.array([(u - self.cx) / self.fx * d,
                            (v - self.cy) / self.fy * d,
                            d])
        base_pt = R @ cam_pt + t
        rospy.loginfo("[Handover] Person 3D: ({:.2f}, {:.2f}, {:.2f}) m".format(*base_pt))
        return base_pt

    # ── Arm control (mirrors v5 _send_arm_goal) ───────────────────────────────

    def _move_arm_toward_person(self, px, py, timeout=25.0):
        """
        Extend arm toward person at ARM_REACH distance in their direction.
        Scales to a fixed reachable distance regardless of how far away the person is.
        Uses PositionConstraint + soft OrientationConstraint — same pattern as v5.
        """
        dist = float(np.hypot(px, py))
        if dist < 0.1:
            direction = np.array([1.0, 0.0])
        else:
            direction = np.array([px, py]) / dist

        reach = min(dist - 0.10, ARM_REACH)  # stop 10 cm before person if very close
        reach = max(reach, 0.3)              # never retract arm behind 30 cm

        tx = float(direction[0] * reach)
        ty = float(direction[1] * reach)
        tz = HANDOVER_HEIGHT

        rospy.loginfo("[Handover] Person at ({:.2f}, {:.2f}), arm extending {:.2f} m in direction ({:.2f}, {:.2f})".format(
            px, py, reach, direction[0], direction[1]))
        rospy.loginfo("[Handover] Arm target: ({:.2f}, {:.2f}, {:.2f})".format(tx, ty, tz))

        goal = MoveGroupGoal()
        goal.request.group_name = "arm_torso"
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 12.0
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.2

        target_pose = PoseStamped()
        target_pose.header.frame_id = "base_footprint"
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = tx
        target_pose.pose.position.y = ty
        target_pose.pose.position.z = tz
        target_pose.pose.orientation = HANDOVER_ORIENTATION

        # Position constraint — 8 cm sphere (generous: just need to be near the person)
        pos_c = PositionConstraint()
        pos_c.header = target_pose.header
        pos_c.link_name = "arm_tool_link"
        bv = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.08]
        bv.primitives.append(sphere)
        bv.primitive_poses.append(target_pose.pose)
        pos_c.constraint_region = bv
        pos_c.weight = 1.0

        # Soft orientation constraint — keep bottle roughly stable
        ori_c = OrientationConstraint()
        ori_c.header = target_pose.header
        ori_c.link_name = "arm_tool_link"
        ori_c.orientation = HANDOVER_ORIENTATION
        ori_c.absolute_x_axis_tolerance = 0.4
        ori_c.absolute_y_axis_tolerance = 0.4
        ori_c.absolute_z_axis_tolerance = 0.4
        ori_c.weight = 0.5

        constraints = Constraints()
        constraints.position_constraints.append(pos_c)
        constraints.orientation_constraints.append(ori_c)
        goal.request.goal_constraints.append(constraints)

        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        self.moveit_client.send_goal(goal)
        finished = self.moveit_client.wait_for_result(rospy.Duration(timeout))

        if not finished:
            self.moveit_client.cancel_goal()
            rospy.logwarn("[Handover] MoveIt timed out")
            return False

        result = self.moveit_client.get_result()
        ok = result and result.error_code.val == 1
        if ok:
            rospy.loginfo("[Handover] Arm reached handover position")
        else:
            code = result.error_code.val if result else 'N/A'
            rospy.logwarn("[Handover] MoveIt failed (code: {})".format(code))
        return ok

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _say(self, text):
        rospy.loginfo("[Handover] {}".format(text))
        if self.speak_fn:
            self.speak_fn(text)

    def _play_motion(self, motion_name, timeout=30):
        goal = PlayMotionGoal()
        goal.motion_name = motion_name
        goal.skip_planning = False
        self.play_motion_client.send_goal(goal)
        finished = self.play_motion_client.wait_for_result(rospy.Duration(timeout))
        return finished and self.play_motion_client.get_state() == actionlib.GoalStatus.SUCCEEDED

    # ── Planning scene helpers ────────────────────────────────────────────────

    def _clear_octomap(self):
        """Reset MoveIt's depth-sensor occupancy map so stale grab-pose voxels don't block planning."""
        try:
            rospy.wait_for_service('/clear_octomap', timeout=2.0)
            clear = rospy.ServiceProxy('/clear_octomap', EmptySrv)
            clear()
            rospy.loginfo("[Handover] Octomap cleared")
        except Exception as e:
            rospy.logwarn("[Handover] Could not clear octomap: {}".format(e))

    def _remove_collision_object(self, obj_id):
        """Remove a named collision object from the MoveIt planning scene."""
        co = CollisionObject()
        co.header.frame_id = "base_footprint"
        co.header.stamp = rospy.Time.now()
        co.id = obj_id
        co.operation = CollisionObject.REMOVE
        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)
        self.scene_pub.publish(ps)
        rospy.loginfo("[Handover] Removed collision object '{}'".format(obj_id))

    def _add_person_collision(self, px, py):
        """
        Add person as a tall box collision object so MoveIt keeps the arm away from them.
        Placed at their detected XY position. Removed after handover completes.
        """
        co = CollisionObject()
        co.header.frame_id = "base_footprint"
        co.header.stamp = rospy.Time.now()
        co.id = "person"
        co.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.50, 0.50, 1.80]   # 50x50cm footprint, 180cm tall

        pose = Pose()
        pose.position.x = px
        pose.position.y = py
        pose.position.z = 0.90                 # centre of 1.8m person
        pose.orientation.w = 1.0

        co.primitives.append(box)
        co.primitive_poses.append(pose)

        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)
        self.scene_pub.publish(ps)
        rospy.loginfo("[Handover] Person collision box added at ({:.2f}, {:.2f})".format(px, py))
        rospy.sleep(0.3)

    # ── BaseSkill interface ───────────────────────────────────────────────────

    def check_affordance(self, params, state):
        gripper = state.get('gripper', 'empty')
        if not gripper.startswith('holding'):
            return (False, "Gripper is empty — nothing to hand over")
        detected = state.get('detected_objects', [])
        has_person = any('person' in obj.lower() for obj in detected)
        if not has_person:
            return (False, "No person detected in scene")
        return (True, "Ready to hand over {}".format(gripper.replace('holding:', '')))

    def execute(self, params):
        """
        1. Go home (clean arm config)
        2. Get TF camera→base_footprint
        3. Detect person with YOLO, project to 3D via depth
        4. Move arm toward person at HANDOVER_HEIGHT
        5. Announce, wait, open gripper, go home
        """
        held_object = params.get('object', 'item')

        # Step 1: Go home for clean starting configuration
        rospy.loginfo("[Handover] Going to home pose...")
        self._play_motion('home', timeout=30)

        # Step 1b: Clear stale planning scene objects left by grab_bottle (table, bottle)
        # and reset octomap so depth voxels from grab pose don't block handover planning
        for obj in ['table', 'bottle']:
            self._remove_collision_object(obj)
        self._clear_octomap()
        rospy.sleep(0.5)  # let MoveIt process the scene update

        # Step 2: Get camera TF (same as v5)
        rospy.loginfo("[Handover] Getting camera TF...")
        tf_t, tf_R = self._tf_echo('base_footprint', 'xtion_rgb_optical_frame')

        # Step 3: Detect person and compute 3D position
        px, py = 0.8, 0.0  # safe default if detection fails
        if tf_t is not None and self.vlm is not None and self.latest_depth is not None:
            img = self.perception.get_latest_rgb() if self.perception else None
            if img is not None:
                detections = self.vlm.detect_objects(img, target_classes=['person'])
            else:
                detections = []
            person = next((d for d in detections if 'person' in d['class_name'].lower()), None)
            if person is not None:
                pos = self._person_3d_position(person, self.latest_depth, tf_R, tf_t)
                if pos is not None:
                    px, py = float(pos[0]), float(pos[1])
            else:
                rospy.logwarn("[Handover] No person detected — using default position")
        else:
            rospy.logwarn("[Handover] TF/perception/depth not ready — using default position")

        rospy.loginfo("[Handover] Handover target person at ({:.2f}, {:.2f})".format(px, py))

        # Step 4: Add person as collision box (prevents arm from poking them),
        # then extend arm toward them, then remove box after gripper opens
        self._add_person_collision(px, py)
        rospy.loginfo("[Handover] Moving arm toward person...")
        arm_ok = self._move_arm_toward_person(px, py, timeout=25.0)
        if not arm_ok:
            rospy.logwarn("[Handover] Arm did not reach target — continuing anyway")

        # Step 5: Announce and wait for person to take object
        self._say("Please take the {} from my hand.".format(held_object))
        rospy.loginfo("[Handover] Waiting 4s for person to take object...")
        rospy.sleep(4.0)

        # Step 6: Open gripper — remove person collision box first so the arm
        # can retract freely without MoveIt thinking the person is still there
        self._remove_collision_object('person')
        rospy.loginfo("[Handover] Opening gripper...")
        self._play_motion('open', timeout=15)
        self._say("Thank you.")

        # Step 7: Return home
        rospy.loginfo("[Handover] Returning arm to home...")
        self._play_motion('home', timeout=30)

        self.on_success()
        return True

    def get_description(self):
        return (
            "handover_to_person(object='bottle') - "
            "Detect person with YOLO+depth, extend arm toward them, and hand over held object. "
            "Use after grab_object when the goal is to deliver an object to someone."
        )

    def get_expected_outcome(self, params):
        return "Gripper is empty, person has received the object"
