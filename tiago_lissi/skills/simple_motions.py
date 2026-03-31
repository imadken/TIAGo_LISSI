#!/usr/bin/env python3
"""
Simple Motion Skills - Basic robot motions via play_motion
"""

import rospy
import actionlib
from typing import Dict, Any
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from .base_skill import BaseSkill


class SimpleMotionSkill(BaseSkill):
    """
    Base class for simple motion skills that use play_motion.
    """

    _shared_client = None   # Shared across all skill instances
    _server_ready  = False  # True once wait_for_server succeeded

    @classmethod
    def _get_client(cls):
        if cls._shared_client is None:
            cls._shared_client = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)
        if not cls._server_ready:
            rospy.loginfo("[SimpleMotionSkill] Connecting to play_motion server...")
            if cls._shared_client.wait_for_server(timeout=rospy.Duration(3)):
                cls._server_ready = True
                rospy.loginfo("[SimpleMotionSkill] play_motion server ready")
            else:
                rospy.logwarn("[SimpleMotionSkill] play_motion server not available")
        return cls._shared_client

    def __init__(self, name: str, motion_name: str, description: str):
        super().__init__(name)
        self.motion_name = motion_name
        self.description_text = description
        # Client created immediately (no blocking wait — lazy connect on first execute)
        if SimpleMotionSkill._shared_client is None:
            SimpleMotionSkill._shared_client = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)

    def check_affordance(self, params: Dict[str, Any], state: Dict[str, Any]) -> tuple:
        """
        Simple motions are always available.

        Args:
            params: Skill parameters (none)
            state: Current robot state

        Returns:
            Tuple (True, "Always available")
        """
        return (True, "Always available")

    def execute(self, params: Dict[str, Any]) -> bool:
        """
        Execute play_motion.

        Args:
            params: Skill parameters (none)

        Returns:
            True if motion succeeded, False otherwise
        """
        try:
            client = SimpleMotionSkill._get_client()
            if not SimpleMotionSkill._server_ready:
                self.on_failure("play_motion server not available")
                return False

            goal = PlayMotionGoal()
            goal.motion_name = self.motion_name
            goal.skip_planning = False

            client.send_goal(goal)
            finished = client.wait_for_result(timeout=rospy.Duration(60))

            if finished and client.get_state() == actionlib.GoalStatus.SUCCEEDED:
                self.on_success()
                return True
            else:
                self.on_failure("Motion did not complete successfully")
                return False

        except Exception as e:
            self.on_failure(str(e))
            return False

    def get_description(self) -> str:
        """Get skill description for VLM."""
        return f"{self.name}() - {self.description_text}"

    def get_expected_outcome(self, params: Dict[str, Any]) -> str:
        """Get expected outcome for verification."""
        return f"Robot should have completed the {self.motion_name} motion"


class GoHomeSkill(SimpleMotionSkill):
    """Move arm to home/rest position."""

    def __init__(self):
        super().__init__(
            name='go_home',
            motion_name='home',
            description='Move arm to home/rest position'
        )


class WaveSkill(SimpleMotionSkill):
    """Wave at person."""

    def __init__(self):
        super().__init__(
            name='wave',
            motion_name='wave',
            description='Wave at person as a greeting gesture'
        )


class OpenHandSkill(SimpleMotionSkill):
    """Open gripper."""

    def __init__(self):
        super().__init__(
            name='open_hand',
            motion_name='open',
            description='Open the robot gripper hand'
        )


class CloseHandSkill(SimpleMotionSkill):
    """Close gripper."""

    def __init__(self):
        super().__init__(
            name='close_hand',
            motion_name='close',
            description='Close the robot gripper hand'
        )

    def check_affordance(self, params: Dict[str, Any], state: Dict[str, Any]) -> tuple:
        """
        Close hand is available, but warn if gripper already holding something.

        Args:
            params: Skill parameters
            state: Current robot state

        Returns:
            Tuple (can_execute: bool, reason: str)
        """
        if state.get('gripper', '').startswith('holding'):
            return (True, f"Gripper already holding {state['gripper'].split(':')[1]}, but can still close")

        return (True, "Ready to close gripper")


if __name__ == '__main__':
    # Test simple motion skills
    rospy.init_node('simple_motions_test', anonymous=True)

    # Test go_home
    go_home = GoHomeSkill()
    print(f"Testing: {go_home.get_description()}")

    can_exec, reason = go_home.check_affordance({}, {})
    print(f"Can execute: {can_exec}, Reason: {reason}")

    if can_exec:
        print("Executing go_home...")
        success = go_home.execute({})
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")

    # Test wave
    wave = WaveSkill()
    print(f"\nTesting: {wave.get_description()}")
    success = wave.execute({})
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
