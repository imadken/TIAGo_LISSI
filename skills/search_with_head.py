#!/usr/bin/env python3
"""
Search for Object by Moving Head
Scans surroundings by moving head to different positions
"""

import os
import sys
import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

# Add workspace to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.base_skill import BaseSkill


class SearchWithHeadSkill(BaseSkill):
    """Search for object by moving head to scan surroundings."""

    def __init__(self, vlm=None, face_manager=None, perception_manager=None, state_manager=None):
        """
        Args:
            vlm: VLMReasoner instance for object detection.
            face_manager: FaceManager for face-visibility check when searching for persons.
            perception_manager: PerceptionManager for camera image access.
            state_manager: StateManager for object memory (last-known head position).
        """
        self.name = "search_with_head"
        self.vlm = vlm
        self.face_manager = face_manager
        self.perception_manager = perception_manager
        self.state_manager = state_manager

        # Head controller client
        self.head_client = actionlib.SimpleActionClient(
            '/head_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )
        rospy.loginfo("[SearchHead] Waiting for head controller...")
        self.head_client.wait_for_server(timeout=rospy.Duration(5))
        rospy.loginfo("[SearchHead] Head controller ready")

        # Head scan positions: (label, pan, tilt) in radians
        # pan: positive = left, negative = right
        # tilt: positive = down, negative = up
        self.scan_positions = [
            ('center',          0.0,  0.0),   # Center, level
            ('center_down',     0.0,  -0.45),  # Center, table level
            ('left',            0.8,  0.0),   # Left
            ('left_down',       0.8,  -0.45),  # Left, table level
            ('far_left',        1.4,  0.0),   # Far left
            ('far_left_down',   1.4,  -0.45),  # Far left, table level
            ('right',          -0.8,  0.0),   # Right
            ('right_down',     -0.8,  -0.45),  # Right, table level
            ('far_right',      -1.4,  0.0),   # Far right
            ('far_right_down', -1.4,  -0.45),  # Far right, table level
            ('up',              0.0, 0.3),   # Look up (shelves)
            ('center',          0.0,  0.0),   # Return to center
        ]

    def get_description(self):
        """Get skill description for VLM."""
        return "search_with_head(object_name='bottle') - Move head to scan surroundings for object"

    def check_affordance(self, params, state):
        """
        Check if skill can be executed.

        Always available - no prerequisites.
        """
        return True, "Head search always available"

    def execute(self, params):
        """
        Execute head search for object.

        Args:
            params: Dict with 'object_name' key

        Returns:
            bool: True if object found, False otherwise
        """
        object_name = params.get('object_name', 'bottle')
        searching_for_person = 'person' in object_name.lower()

        rospy.loginfo("[SearchHead] Searching for '{}' by moving head...".format(object_name))

        # Build scan list: last-known head position first (only if previously seen within 10 min)
        scan_positions = list(self.scan_positions)
        if self.state_manager:
            mem = self.state_manager.get_object_memory(object_name)
            if mem and 'head_pan' in mem:
                import time as _time
                age = _time.time() - mem.get('last_seen', 0)
                if age < 600:  # 10 minutes
                    pan_mem  = mem['head_pan']
                    tilt_mem = mem['head_tilt']
                    rospy.loginfo(
                        "[SearchHead] Memory: last saw '{}' {:.0f}s ago at head=({:.2f}, {:.2f}) — trying first".format(
                            object_name, age, pan_mem, tilt_mem))
                    scan_positions = [('memory_position', pan_mem, tilt_mem)] + scan_positions
                else:
                    rospy.loginfo(
                        "[SearchHead] Memory for '{}' too old ({:.0f}s) — full scan".format(
                            object_name, age))

        # Scan each position
        for i, (name, pan, tilt) in enumerate(scan_positions):
            rospy.loginfo("[SearchHead] Position {}/{}: {}".format(
                i + 1, len(self.scan_positions), name))

            if not self._move_head(pan, tilt):
                rospy.logwarn("[SearchHead] Failed to move head to {}".format(name))
                continue

            # Wait for head to settle and camera to deliver a fresh frame
            rospy.sleep(2)

            # Get current image and run VLM detection
            image = self.perception_manager.get_latest_rgb() if self.perception_manager else None
            if image is None:
                rospy.logwarn("[SearchHead] No camera image available")
                continue

            detections = (self.vlm.detect_objects(image, target_classes=[object_name])
                          if self.vlm else [])

            # Look for target object
            matched = [d for d in detections
                       if object_name.lower() in d['class_name'].lower()]

            if not matched:
                continue

            # ── Person-specific check: is the face visible? ───────────────────
            if searching_for_person:
                face_visible = self._check_face_visible(image, matched)
                if not face_visible:
                    rospy.loginfo(
                        "[SearchHead] Person detected at '{}' but face not visible "
                        "(foot/body only) — continuing search upward.".format(name))
                    # Tilt head upward by 0.3 rad and try same pan
                    self._move_head(pan, tilt + 0.3)
                    rospy.sleep(1.5)
                    image2 = self.perception_manager.get_latest_rgb()
                    if image2 is not None:
                        d2 = (self.vlm.detect_objects(image2, target_classes=['person'])
                              if self.vlm else [])
                        if self._check_face_visible(image2, d2):
                            rospy.loginfo("[SearchHead] ✓ Face now visible after tilt up!")
                            return True
                    continue  # proceed to next scan position

            rospy.loginfo("[SearchHead] ✓ Found '{}' at {}!".format(
                matched[0]['class_name'], name))
            return True

        # Not found after full scan
        rospy.loginfo("[SearchHead] '{}' not found after full scan".format(object_name))
        self._move_head(0.0, 0.0)
        return False

    def _check_face_visible(self, image, person_detections):
        """
        Returns True if at least one person detection has a visible face
        (face bbox centroid is in the upper 60% of the person bbox).
        Uses face_manager if available, otherwise checks bbox geometry.
        """
        if not person_detections:
            return False

        # Try face_manager recognition first
        if self.face_manager and self.face_manager.is_available():
            faces = self.face_manager.recognize(image)
            if not faces:
                return False
            # Confirm at least one face overlaps a person box in the upper region
            for det in person_detections:
                px, py, pw, ph = det['bbox']
                for face in faces:
                    fx, fy, fw, fh = face['bbox']
                    f_cy = fy + fh // 2
                    # Face centroid must be in upper 60% of person box
                    if py <= f_cy <= py + int(ph * 0.6):
                        return True
            return False

        # Fallback: check if person bbox has enough height above mid-point
        # (if person is very small/partial, assume foot-only)
        for det in person_detections:
            _, py, pw, ph = det['bbox']
            # If height is large enough relative to width, likely full body seen
            if ph > pw * 1.2:
                return True
        return False

    def _move_head(self, pan, tilt):
        """
        Move head to specified pan/tilt position.

        Args:
            pan: Pan angle in radians (positive = left)
            tilt: Tilt angle in radians (positive = down)

        Returns:
            bool: True if successful
        """
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ['head_1_joint', 'head_2_joint']

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [pan, tilt]
        point.time_from_start = rospy.Duration(2.0)

        goal.trajectory.points = [point]

        rospy.loginfo("[SearchHead] Moving head to pan={:.2f}, tilt={:.2f}".format(pan, tilt))

        self.head_client.send_goal(goal)

        # Wait for result
        finished = self.head_client.wait_for_result(timeout=rospy.Duration(5.0))

        if not finished:
            rospy.logwarn("[SearchHead] Head movement timed out")
            self.head_client.cancel_goal()
            return False

        state = self.head_client.get_state()
        success = (state == actionlib.GoalStatus.SUCCEEDED)

        if success:
            rospy.loginfo("[SearchHead] Head moved successfully")
        else:
            rospy.logwarn("[SearchHead] Head movement failed")

        return success


if __name__ == '__main__':
    """Test head search skill."""
    rospy.init_node('test_search_head', anonymous=True)

    print("\n" + "=" * 60)
    print("Testing Head Search Skill")
    print("=" * 60)

    skill = SearchWithHeadSkill()

    # Test: search for bottle
    print("\nSearching for 'bottle'...")
    params = {'object_name': 'bottle'}
    success = skill.execute(params)

    if success:
        print("\n✓ Success: Found bottle!")
    else:
        print("\n✗ Bottle not found after full scan")

    print("=" * 60)
