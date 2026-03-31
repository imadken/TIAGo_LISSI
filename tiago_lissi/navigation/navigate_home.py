#!/usr/bin/env python3
"""
Send robot back to starting position (0, 0, 0) using navigation
"""

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import math


def navigate_home():
    """Send robot to home position (0, 0, 0)."""
    rospy.init_node('navigate_home', anonymous=True)

    # Create action client
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    rospy.loginfo("[GoHome] Waiting for move_base action server...")
    client.wait_for_server(timeout=rospy.Duration(10))
    rospy.loginfo("[GoHome] Connected to move_base server")

    # Create goal for home position
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    # Home position: (0, 0, 0)
    goal.target_pose.pose.position.x = 0.0
    goal.target_pose.pose.position.y = 0.0
    goal.target_pose.pose.position.z = 0.0

    # Facing forward (theta = 0)
    goal.target_pose.pose.orientation.x = 0.0
    goal.target_pose.pose.orientation.y = 0.0
    goal.target_pose.pose.orientation.z = 0.0
    goal.target_pose.pose.orientation.w = 1.0

    rospy.loginfo("[GoHome] Sending robot to home position (0, 0, 0)...")

    # Send goal
    client.send_goal(goal)

    # Wait for result
    rospy.loginfo("[GoHome] Navigating... (this may take a while)")
    finished = client.wait_for_result(rospy.Duration(120))  # 2 minute timeout

    if finished:
        state = client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("[GoHome] ✓ Robot successfully returned home!")
            return True
        else:
            rospy.logwarn("[GoHome] Navigation failed with state: {}".format(state))
            return False
    else:
        rospy.logwarn("[GoHome] Navigation timed out")
        client.cancel_goal()
        return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("NAVIGATE HOME")
    print("=" * 60)
    print("\nSending robot to starting position (0, 0, 0)...")
    print("Make sure navigation is running!")
    print("=" * 60 + "\n")

    try:
        success = navigate_home()

        print("\n" + "=" * 60)
        if success:
            print("✓ Robot is home!")
        else:
            print("✗ Failed to return home")
        print("=" * 60 + "\n")

    except rospy.ROSInterruptException:
        print("\nNavigation interrupted by user")
    except Exception as e:
        print("\nError: {}".format(e))
        print("\nMake sure navigation is running:")
        print("  roslaunch tiago_2dnav navigation.launch")
