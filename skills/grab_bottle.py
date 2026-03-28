#!/usr/bin/env python3
"""
Grab Bottle Skill - Wraps reach_object.py grasping pipeline
"""

import os
import subprocess
from typing import Dict, Any
from .base_skill import BaseSkill


class GrabBottleSkill(BaseSkill):
    def __init__(self, reach_object_script: str = None):
        """
        Initialize grab object skill.

        Args:
            reach_object_script: Path to reach_object.py script
        """
        super().__init__('grab_object')

        # Default to workspace directory
        if reach_object_script is None:
            workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            reach_object_script = os.path.join(workspace_dir, 'reach_object_v5_torso_descent_working.py')

        self.reach_object_script = reach_object_script

    def check_affordance(self, params: Dict[str, Any], state: Dict[str, Any]) -> tuple:
        """
        Check if we can grab an object.

        Requires:
        - Gripper must be empty
        - Any graspable object must be detected in scene

        Args:
            params: Skill parameters (can include 'target_object' for specific object)
            state: Current robot state

        Returns:
            Tuple (can_execute: bool, reason: str)
        """
        # Check gripper is empty
        if state.get('gripper', '').startswith('holding'):
            return (False, "Gripper already holding {}".format(state['gripper'].split(':')[1]))

        # Check if any graspable object is detected
        detected_objects = state.get('detected_objects', [])
        target_object = params.get('target_object', None)

        if target_object:
            has_target = any(target_object.lower() in obj.lower() for obj in detected_objects)
            if not has_target:
                return (False, "No {} detected in scene".format(target_object))
        else:
            graspable_keywords = ['bottle', 'phone', 'cup', 'mug', 'can', 'box', 'book', 'remote', 'object']
            has_graspable = any(
                any(keyword in obj.lower() for keyword in graspable_keywords)
                for obj in detected_objects
            )
            if not has_graspable:
                return (False, "No graspable object detected in scene (looking for: {})".format(
                    ', '.join(graspable_keywords)))

        # Check reach_object.py script exists
        if not os.path.exists(self.reach_object_script):
            return (False, "reach_object.py not found at {}".format(self.reach_object_script))

        return (True, "Ready to grab object")

    def execute(self, params: Dict[str, Any]) -> bool:
        """
        Execute bottle grasping by calling reach_object.py as subprocess.

        Args:
            params: Skill parameters (none)

        Returns:
            True if grasp succeeded, False otherwise
        """
        try:
            target = params.get('target_object', 'bottle')
            cmd = ['python3', self.reach_object_script, '--target', target]

            # Pass full description via env var (no length limit, preserves chain-of-thought)
            env = os.environ.copy()
            description = params.get('target_description', '')
            if description:
                env['TARGET_DESCRIPTION'] = description

            # Pass obstacle objects as env var for MoveIt collision avoidance
            obstacles = params.get('obstacle_objects', '')
            if obstacles:
                env['OBSTACLE_OBJECTS'] = obstacles

            # Run reach_object.py as subprocess with 3-minute timeout
            # stdout/stderr go directly to console so we can see what's happening
            result = subprocess.run(
                cmd,
                env=env,
                timeout=180,  # 3 minutes
            )

            if result.returncode == 0:
                self.on_success()
                return True
            else:
                self.on_failure("reach_object exited with code {}".format(result.returncode))
                return False

        except subprocess.TimeoutExpired:
            self.on_failure("Grasp timed out after 3 minutes")
            return False
        except Exception as e:
            self.on_failure(str(e))
            return False

    def get_description(self) -> str:
        """Get skill description for VLM."""
        return ('grab_object(target_object="box") - Grasp any detected object in front of the robot. '
                'ALWAYS set target_object to the exact class name of the object to grab (e.g. "bottle", "box", "cup"). '
                'Works with any graspable object.')

    def get_expected_outcome(self, params: Dict[str, Any]) -> str:
        """Get expected outcome for verification."""
        return "Gripper should be closed around the object, robot holding the object"


if __name__ == '__main__':
    # Test grab_bottle skill
    import rospy
    rospy.init_node('grab_bottle_test', anonymous=True)

    skill = GrabBottleSkill()

    # Test affordance check
    test_state = {
        'gripper': 'empty',
        'detected_objects': ['bottle at (320, 240)']
    }

    can_exec, reason = skill.check_affordance({}, test_state)
    print(f"Can execute: {can_exec}, Reason: {reason}")

    if can_exec:
        print("Executing grab_bottle...")
        success = skill.execute({})
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
