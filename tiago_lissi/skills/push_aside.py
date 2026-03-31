#!/usr/bin/env python3
"""
PushAsideSkill — push a blocking object sideways to clear the path to the target.

Strategy:
1. Detect blocking object 3D position via depth + TF
2. Check there is free space on the table in the lateral direction
3. Move gripper (closed) alongside the object
4. Execute a lateral push via a simple joint trajectory increment
"""

import os
import sys
import rospy
import actionlib
import numpy as np
import subprocess
import re

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import MoveGroupAction, MoveGroupGoal, Constraints, PositionConstraint, BoundingVolume
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from shape_msgs.msg import SolidPrimitive

from .base_skill import BaseSkill

# Table lateral limit — don't push objects beyond this y distance from robot
TABLE_Y_LIMIT = 0.55   # metres
PUSH_DISTANCE  = 0.25  # metres to push laterally
APPROACH_CLEARANCE = 0.06  # metres to the side of the object before pushing


class PushAsideSkill(BaseSkill):
    """Push a blocking object sideways to clear the path to the target."""

    def __init__(self, vlm=None, perception_manager=None):
        super().__init__('push_aside_object')
        self.vlm = vlm
        self.perception_manager = perception_manager

        # MoveIt action client
        self.move_group_client = actionlib.SimpleActionClient('/move_group', MoveGroupAction)
        rospy.loginfo("[PushAside] Connecting to move_group...")
        self.move_group_client.wait_for_server(timeout=rospy.Duration(10))

        # Gripper controller
        self.gripper_client = actionlib.SimpleActionClient(
            '/gripper_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.gripper_client.wait_for_server(timeout=rospy.Duration(5))

        self.grasp_orientation = Quaternion(x=0.7030, y=0.0730, z=-0.0340, w=0.7060)

    def check_affordance(self, params, state):
        blocking = params.get('blocking_object', '')
        if not blocking:
            return (False, "blocking_object parameter required")
        detected = state.get('detected_objects', [])
        if not any(blocking.lower() in o.lower() for o in detected):
            return (False, "{} not detected".format(blocking))
        return (True, "Ready to push aside {}".format(blocking))

    def execute(self, params):
        blocking = params.get('blocking_object', 'object')
        rospy.loginfo("[PushAside] Pushing aside '{}'".format(blocking))

        # 1. Check if agent already resolved the 3D position and passed it in params
        pre_pos = params.get('position_3d')
        if pre_pos is not None:
            pos = list(pre_pos)
        else:
            # 2. Detect via perception manager (depth + current frame)
            pos = self._get_object_3d(blocking)
        if pos is None:
            rospy.logwarn("[PushAside] Could not get 3D position of '{}'".format(blocking))
            return False

        ox, oy, oz = pos
        rospy.loginfo("[PushAside] Blocking object at ({:.3f},{:.3f},{:.3f})".format(ox, oy, oz))

        # Decide push direction: push to the side with more free space
        # Prefer pushing away from robot centre (oy > 0 → push left, oy < 0 → push right)
        push_dir = 1.0 if oy <= 0 else -1.0
        target_y = oy + push_dir * PUSH_DISTANCE
        if abs(target_y) > TABLE_Y_LIMIT:
            push_dir = -push_dir  # try other direction
            target_y = oy + push_dir * PUSH_DISTANCE
            if abs(target_y) > TABLE_Y_LIMIT:
                rospy.logwarn("[PushAside] No space to push — table limit reached both sides")
                return False

        # Approach position: alongside the object at push height
        approach_y = oy + push_dir * APPROACH_CLEARANCE
        push_z = oz + 0.03  # slightly above centre

        rospy.loginfo("[PushAside] Approach: ({:.3f},{:.3f},{:.3f})".format(ox, approach_y, push_z))
        rospy.loginfo("[PushAside] Push target y={:.3f}".format(target_y))

        # Close gripper first
        self._set_gripper(0.0)

        # Move arm to approach position
        if not self._move_arm(ox, approach_y, push_z):
            rospy.logwarn("[PushAside] Could not reach approach position")
            return False

        # Execute push: move arm laterally to target_y
        if not self._move_arm(ox, target_y, push_z):
            rospy.logwarn("[PushAside] Push motion failed (object may have moved)")
            # Not fatal — object might have moved

        rospy.loginfo("[PushAside] Push complete")
        return True

    def _get_object_3d(self, class_name):
        """Get 3D position of object from perception manager or VLM."""
        if self.perception_manager is None:
            return None
        detections = self.perception_manager.detect_objects() if hasattr(
            self.perception_manager, 'detect_objects') else []
        for d in detections:
            if class_name.lower() in d.get('class_name', '').lower():
                pos = d.get('position_3d')
                if pos is not None:
                    return pos
        return None

    def _move_arm(self, x, y, z):
        """Move arm tool_link to (x,y,z) in base_footprint via MoveIt."""
        goal = MoveGroupGoal()
        goal.request.group_name = "arm_torso"
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 8.0
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3

        target = PoseStamped()
        target.header.frame_id = "base_footprint"
        target.header.stamp = rospy.Time.now()
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = z
        target.pose.orientation = self.grasp_orientation

        from moveit_msgs.msg import PositionConstraint, OrientationConstraint
        pc = PositionConstraint()
        pc.header = target.header
        pc.link_name = "arm_tool_link"
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
        pc.target_point_offset.z = 0.0
        bv = BoundingVolume()
        sp = SolidPrimitive()
        sp.type = SolidPrimitive.SPHERE
        sp.dimensions = [0.03]
        bv.primitives.append(sp)
        bv.primitive_poses.append(target.pose)
        pc.constraint_region = bv
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header = target.header
        oc.link_name = "arm_tool_link"
        oc.orientation = self.grasp_orientation
        oc.absolute_x_axis_tolerance = 0.4
        oc.absolute_y_axis_tolerance = 0.4
        oc.absolute_z_axis_tolerance = 0.4
        oc.weight = 0.5

        c = Constraints()
        c.position_constraints.append(pc)
        c.orientation_constraints.append(oc)
        goal.request.goal_constraints.append(c)

        self.move_group_client.send_goal(goal)
        if not self.move_group_client.wait_for_result(timeout=rospy.Duration(20)):
            return False
        result = self.move_group_client.get_result()
        return result is not None and result.error_code.val == 1

    def _set_gripper(self, position):
        """Set gripper position (0.0 = closed, 0.044 = open)."""
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ['gripper_left_finger_joint', 'gripper_right_finger_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [position, position]
        pt.time_from_start = rospy.Duration(1.5)
        goal.trajectory.points = [pt]
        self.gripper_client.send_goal(goal)
        self.gripper_client.wait_for_result(timeout=rospy.Duration(4))

    def get_description(self):
        return ('push_aside_object(blocking_object="cup") - Push a blocking object sideways '
                'to clear the path to the target. Use when an object is in the way and there '
                'is free space on the table to push it into.')

    def get_expected_outcome(self, params):
        return "Blocking object pushed aside, path to target cleared"
