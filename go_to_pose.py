#!/usr/bin/env python3
"""
Send robot to a specific pose using move_base navigation
"""

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion
import sys


def go_to_pose(x, y, theta):
    """
    Send robot to target pose (x, y, theta) in map frame.

    Args:
        x: X position in meters
        y: Y position in meters
        theta: Orientation in radians (0 = facing forward)
    """
    rospy.init_node('go_to_pose', anonymous=True)

    # Create action client
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    rospy.loginfo("Waiting for move_base action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to move_base server")

    # Create goal
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    # Set position
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.position.z = 0.0

    # Set orientation (convert theta to quaternion)
    import math
    goal.target_pose.pose.orientation.x = 0.0
    goal.target_pose.pose.orientation.y = 0.0
    goal.target_pose.pose.orientation.z = math.sin(theta / 2.0)
    goal.target_pose.pose.orientation.w = math.cos(theta / 2.0)

    rospy.loginfo("Sending goal: x={:.2f}, y={:.2f}, theta={:.2f}".format(x, y, theta))

    # Send goal
    client.send_goal(goal)

    # Wait for result (with timeout)
    rospy.loginfo("Waiting for robot to reach goal...")
    finished = client.wait_for_result(rospy.Duration(60))

    if finished:
        state = client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("SUCCESS! Robot reached the goal")
            return True
        else:
            rospy.logwarn("Navigation failed with state: {}".format(state))
            return False
    else:
        rospy.logwarn("Navigation timed out")
        client.cancel_goal()
        return False


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("GO TO POSE - Navigation")
    print("=" * 60)

    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  python3 go_to_pose.py <x> <y> [theta]")
        print("\nExamples:")
        print("  python3 go_to_pose.py 0 0 0        # Go to origin")
        print("  python3 go_to_pose.py 1.5 2.0 1.57 # Go to (1.5, 2.0) facing left")
        print("=" * 60 + "\n")
        sys.exit(1)

    x = float(sys.argv[1])
    y = float(sys.argv[2])
    theta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    try:
        success = go_to_pose(x, y, theta)
        if success:
            print("\n" + "=" * 60)
            print("Navigation complete!")
            print("=" * 60 + "\n")
        else:
            print("\n" + "=" * 60)
            print("Navigation failed - check for obstacles or path issues")
            print("=" * 60 + "\n")

    except rospy.ROSInterruptException:
        print("\nNavigation interrupted")


if __name__ == '__main__':
    main()
