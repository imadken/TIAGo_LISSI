#!/usr/bin/env python3
"""
Navigation Client for TIAGo Embodied AI
Interface with move_base for base navigation (ROS 1/Melodic)
"""

import rospy
import actionlib
import yaml
import os
import numpy as np
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from typing import Dict, Tuple, Optional


class NavigationClient:
    def __init__(self, locations_file: str = None):
        """
        Initialize navigation client.

        Args:
            locations_file: Path to YAML file with named locations
        """
        # Load locations from YAML or use defaults
        if locations_file and os.path.exists(locations_file):
            with open(locations_file, 'r') as f:
                config = yaml.safe_load(f)
                self.locations = config.get('locations', {})
            rospy.loginfo(f"[Nav] Loaded {len(self.locations)} locations from {locations_file}")
        else:
            # Default predefined locations (x, y, theta in map frame)
            self.locations = {
                'table': {'x': 1.5, 'y': 0.0, 'theta': 0.0, 'description': 'Main table'},
                'shelf': {'x': 2.0, 'y': 1.5, 'theta': 1.57, 'description': 'Storage shelf'},
                'person_handover_spot': {'x': 1.0, 'y': -0.5, 'theta': 0.0, 'description': 'Handover position'},
                'home': {'x': 0.0, 'y': 0.0, 'theta': 0.0, 'description': 'Starting position'}
            }
            rospy.loginfo("[Nav] Using default locations (no config file provided)")

        # Connect to move_base action server
        rospy.loginfo("[Nav] Connecting to move_base...")
        self.client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)

        # Wait for server (with timeout)
        if not self.client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logwarn("[Nav] move_base server not available - navigation will not work")
            rospy.logwarn("[Nav] Make sure navigation stack is running")
            self.client = None
        else:
            rospy.loginfo("[Nav] Connected to move_base")

    def _create_goal(self, x: float, y: float, theta: float) -> MoveBaseGoal:
        """
        Create MoveBaseGoal from (x, y, theta).

        Args:
            x: X position in map frame (meters)
            y: Y position in map frame (meters)
            theta: Yaw angle in map frame (radians)

        Returns:
            MoveBaseGoal
        """
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0

        # Convert theta to quaternion
        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = np.sin(theta / 2.0)
        goal.target_pose.pose.orientation.w = np.cos(theta / 2.0)

        return goal

    def navigate_to(self, location_name: str, timeout: float = 30.0) -> bool:
        """
        Navigate to a named location.

        Args:
            location_name: Name of location (must be in self.locations)
            timeout: Timeout in seconds

        Returns:
            True if navigation succeeded, False otherwise
        """
        if self.client is None:
            rospy.logerr("[Nav] move_base client not initialized - cannot navigate")
            return False

        if location_name not in self.locations:
            rospy.logerr(f"[Nav] Unknown location: {location_name}")
            rospy.loginfo(f"[Nav] Available locations: {list(self.locations.keys())}")
            return False

        loc = self.locations[location_name]
        x = loc['x']
        y = loc['y']
        theta = loc['theta']

        rospy.loginfo(f"[Nav] Navigating to {location_name}: ({x:.2f}, {y:.2f}, {theta:.2f})")

        # Create and send goal
        goal = self._create_goal(x, y, theta)
        self.client.send_goal(goal)

        # Wait for result
        finished = self.client.wait_for_result(timeout=rospy.Duration(timeout))

        if not finished:
            rospy.logwarn(f"[Nav] Navigation to {location_name} timed out after {timeout}s")
            self.client.cancel_goal()
            return False

        # Check result
        state = self.client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo(f"[Nav] Reached {location_name}")
            return True
        else:
            rospy.logwarn(f"[Nav] Navigation to {location_name} failed with state {state}")
            return False

    def navigate_to_pose(self, x: float, y: float, theta: float, timeout: float = 30.0) -> bool:
        """
        Navigate to a specific pose (not a named location).

        Args:
            x: X position in map frame (meters)
            y: Y position in map frame (meters)
            theta: Yaw angle in map frame (radians)
            timeout: Timeout in seconds

        Returns:
            True if navigation succeeded, False otherwise
        """
        if self.client is None:
            rospy.logerr("[Nav] move_base client not initialized - cannot navigate")
            return False

        rospy.loginfo(f"[Nav] Navigating to pose: ({x:.2f}, {y:.2f}, {theta:.2f})")

        # Create and send goal
        goal = self._create_goal(x, y, theta)
        self.client.send_goal(goal)

        # Wait for result
        finished = self.client.wait_for_result(timeout=rospy.Duration(timeout))

        if not finished:
            rospy.logwarn(f"[Nav] Navigation to pose timed out after {timeout}s")
            self.client.cancel_goal()
            return False

        # Check result
        state = self.client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("[Nav] Reached target pose")
            return True
        else:
            rospy.logwarn(f"[Nav] Navigation to pose failed with state {state}")
            return False

    def cancel_navigation(self):
        """Cancel current navigation goal."""
        if self.client is not None:
            self.client.cancel_goal()
            rospy.loginfo("[Nav] Navigation cancelled")

    def get_location_names(self) -> list:
        """Get list of available named locations."""
        return list(self.locations.keys())

    def get_location_pose(self, location_name: str) -> Optional[Tuple[float, float, float]]:
        """
        Get (x, y, theta) for a named location.

        Args:
            location_name: Name of location

        Returns:
            Tuple (x, y, theta) or None if location doesn't exist
        """
        if location_name in self.locations:
            loc = self.locations[location_name]
            return (loc['x'], loc['y'], loc['theta'])
        return None

    def add_location(self, name: str, x: float, y: float, theta: float, description: str = ""):
        """
        Add a new named location.

        Args:
            name: Location name
            x: X position in map frame
            y: Y position in map frame
            theta: Yaw angle in map frame
            description: Human-readable description
        """
        self.locations[name] = {
            'x': x,
            'y': y,
            'theta': theta,
            'description': description
        }
        rospy.loginfo(f"[Nav] Added location '{name}' at ({x:.2f}, {y:.2f}, {theta:.2f})")


if __name__ == '__main__':
    # Test navigation client
    rospy.init_node('nav_test', anonymous=True)

    # Explicitly load config file
    import os
    config_file = os.path.join(os.path.dirname(__file__), 'config', 'locations.yaml')
    nav_client = NavigationClient(locations_file=config_file)

    rospy.loginfo(f"Available locations: {nav_client.get_location_names()}")

    # Test navigation to home
    if len(nav_client.get_location_names()) > 0:
        test_location = nav_client.get_location_names()[0]
        rospy.loginfo(f"Testing navigation to: {test_location}")

        # Note: This will only work if move_base and navigation stack are running
        success = nav_client.navigate_to(test_location, timeout=10.0)

        if success:
            rospy.loginfo("Navigation test succeeded!")
        else:
            rospy.loginfo("Navigation test failed (expected if navigation stack not running)")
