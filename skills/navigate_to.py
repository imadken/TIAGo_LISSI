#!/usr/bin/env python3
"""
Navigate To Skill - Navigate base to named location
"""

import os
import sys
import rospy
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.base_skill import BaseSkill
from navigation_client import NavigationClient


class NavigateToSkill(BaseSkill):
    def __init__(self, navigation_client=None):
        """
        Initialize navigate skill.

        Args:
            navigation_client: NavigationClient instance (optional)
        """
        super().__init__('navigate_to')

        # Use provided navigation client or create new one
        if navigation_client is None:
            self.nav_client = NavigationClient()
        else:
            self.nav_client = navigation_client

        rospy.loginfo("[NavigateTo] Ready with locations: {}".format(
            self.nav_client.get_location_names()))

    def check_affordance(self, params: Dict[str, Any], state: Dict[str, Any]) -> tuple:
        """
        Check if we can navigate to a location.

        Args:
            params: {'location': 'table'} - target location
            state: Current robot state

        Returns:
            Tuple (can_execute: bool, reason: str)
        """
        # Check location parameter provided
        if 'location' not in params:
            return (False, "No location specified")

        location_name = params['location']

        # Check location exists
        if location_name not in self.nav_client.locations:
            available = ', '.join(self.nav_client.get_location_names())
            return (False, "Unknown location '{}'. Available: {}".format(
                location_name, available))

        # Check navigation available
        if self.nav_client.client is None:
            return (False, "Navigation not available (move_base not running)")

        return (True, "Ready to navigate to '{}'".format(location_name))

    def execute(self, params: Dict[str, Any]) -> bool:
        """
        Execute navigation to target location.

        Args:
            params: {'location': 'table'}

        Returns:
            True if navigation succeeded
        """
        location_name = params['location']

        rospy.loginfo("[NavigateTo] Navigating to '{}'...".format(location_name))

        # Navigate with 30s timeout
        success = self.nav_client.navigate_to(location_name, timeout=30.0)

        if success:
            rospy.loginfo("[NavigateTo] Reached '{}'".format(location_name))
            self.on_success()
        else:
            rospy.logwarn("[NavigateTo] Failed to reach '{}'".format(location_name))
            self.on_failure("Navigation failed")

        return success

    def get_description(self) -> str:
        """Get skill description for VLM."""
        locations = ', '.join(self.nav_client.get_location_names())
        return "navigate_to(location='table') - Navigate robot base to a named location. Available locations: {}".format(locations)

    def get_expected_outcome(self, params: Dict[str, Any]) -> str:
        """Get expected outcome for verification."""
        location = params.get('location', 'target')
        return "Robot should be at location '{}'".format(location)


if __name__ == '__main__':
    # Test navigate skill
    rospy.init_node('navigate_to_test', anonymous=True)

    skill = NavigateToSkill()

    # Test affordance check
    test_state = {'gripper': 'empty'}
    test_params = {'location': 'table'}

    can_exec, reason = skill.check_affordance(test_params, test_state)
    print("Can execute: {}, Reason: {}".format(can_exec, reason))

    if can_exec:
        print("Executing navigate_to...")
        success = skill.execute(test_params)
        print("Result: {}".format('SUCCESS' if success else 'FAILED'))
