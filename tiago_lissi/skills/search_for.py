#!/usr/bin/env python3
"""
Search For Object Skill - Navigate to waypoints to find objects
"""

import rospy
from typing import Dict, Any

from tiago_lissi.skills.base_skill import BaseSkill
from tiago_lissi.navigation.navigation_client import NavigationClient


class SearchForSkill(BaseSkill):
    def __init__(self, navigation_client=None):
        """
        Initialize search skill with navigation.

        Args:
            navigation_client: NavigationClient instance (optional)
        """
        super().__init__('search_for')

        # Use provided navigation client or create new one
        if navigation_client is None:
            self.nav_client = NavigationClient()
        else:
            self.nav_client = navigation_client

        # Search waypoints - predefined locations to check
        self.search_waypoints = ['table', 'shelf', 'corner1', 'corner2']
        rospy.loginfo("[SearchFor] Ready with waypoints: {}".format(self.search_waypoints))

    def check_affordance(self, params: Dict[str, Any], state: Dict[str, Any]) -> tuple:
        """
        Check if we can search for an object.

        Args:
            params: {'object': 'bottle'} - object to search for
            state: Current robot state

        Returns:
            Tuple (can_execute: bool, reason: str)
        """
        # Check object parameter provided
        if 'object' not in params:
            return (False, "No object specified to search for")

        # Check if object already detected
        detected_objects = state.get('detected_objects', [])
        target_object = params['object'].lower()

        already_detected = any(target_object in obj.lower() for obj in detected_objects)
        if already_detected:
            return (False, "Object '{}' already visible".format(target_object))

        # Check navigation available
        if self.nav_client.client is None:
            return (False, "Navigation not available (move_base not running)")

        return (True, "Ready to search for '{}'".format(target_object))

    def execute(self, params: Dict[str, Any]) -> bool:
        """
        Execute search by navigating to waypoints.

        The embodied agent will run YOLO detection after this skill
        to check if the object is now visible.

        Args:
            params: {'object': 'bottle'}

        Returns:
            True if search completed (even if object not found)
        """
        target_object = params.get('object', 'object')

        rospy.loginfo("[SearchFor] Searching for '{}'...".format(target_object))
        rospy.loginfo("[SearchFor] Will visit waypoints: {}".format(self.search_waypoints))

        # Navigate to each waypoint
        for waypoint in self.search_waypoints:
            # Skip waypoints that don't exist
            if waypoint not in self.nav_client.locations:
                rospy.logwarn("[SearchFor] Waypoint '{}' not in locations, skipping".format(waypoint))
                continue

            rospy.loginfo("[SearchFor] Navigating to '{}'...".format(waypoint))

            # Navigate with 30s timeout
            success = self.nav_client.navigate_to(waypoint, timeout=30.0)

            if not success:
                rospy.logwarn("[SearchFor] Failed to reach '{}', continuing search".format(waypoint))
                continue

            # Wait for base to settle
            rospy.loginfo("[SearchFor] Reached '{}', waiting for stabilization".format(waypoint))
            rospy.sleep(2.0)

            # The embodied agent will run YOLO here and check if object found
            # If found, VLM will detect it in next iteration and stop search

        rospy.loginfo("[SearchFor] Search pattern complete")
        self.on_success()
        return True

    def get_description(self) -> str:
        """Get skill description for VLM."""
        return "search_for(object='bottle') - Navigate to different locations to search for an object that is not currently visible"

    def get_expected_outcome(self, params: Dict[str, Any]) -> str:
        """Get expected outcome for verification."""
        target = params.get('object', 'object')
        return "Robot navigated to search waypoints, object '{}' may now be visible".format(target)


if __name__ == '__main__':
    # Test search skill
    rospy.init_node('search_for_test', anonymous=True)

    skill = SearchForSkill()

    # Test affordance check
    test_state = {
        'gripper': 'empty',
        'detected_objects': []  # No objects detected
    }

    test_params = {'object': 'bottle'}

    can_exec, reason = skill.check_affordance(test_params, test_state)
    print("Can execute: {}, Reason: {}".format(can_exec, reason))

    if can_exec:
        print("Executing search_for...")
        success = skill.execute(test_params)
        print("Result: {}".format('SUCCESS' if success else 'FAILED'))
