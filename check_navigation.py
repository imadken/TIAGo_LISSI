#!/usr/bin/env python3
"""
Check navigation status and current goal
"""

import rospy
from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseActionFeedback
from actionlib_msgs.msg import GoalStatusArray
import math


def quaternion_to_yaw(q):
    """Convert quaternion to yaw angle."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class NavigationChecker:
    def __init__(self):
        self.current_goal = None
        self.feedback = None
        self.status = None

        rospy.Subscriber('/move_base/goal', MoveBaseActionGoal, self.goal_callback)
        rospy.Subscriber('/move_base/feedback', MoveBaseActionFeedback, self.feedback_callback)
        rospy.Subscriber('/move_base/status', GoalStatusArray, self.status_callback)

    def goal_callback(self, msg):
        self.current_goal = msg.goal.target_pose.pose

    def feedback_callback(self, msg):
        self.feedback = msg.feedback.base_position.pose

    def status_callback(self, msg):
        if msg.status_list:
            self.status = msg.status_list[-1]

    def print_status(self):
        print("\n" + "=" * 60)
        print("NAVIGATION STATUS")
        print("=" * 60)

        if self.current_goal:
            x = self.current_goal.position.x
            y = self.current_goal.position.y
            yaw = quaternion_to_yaw(self.current_goal.orientation)
            print("\n🎯 Current Goal:")
            print("   X: {:.3f} m".format(x))
            print("   Y: {:.3f} m".format(y))
            print("   Theta: {:.3f} rad ({:.1f}°)".format(yaw, math.degrees(yaw)))
        else:
            print("\n⚠️  No active goal")

        if self.feedback:
            x = self.feedback.position.x
            y = self.feedback.position.y
            yaw = quaternion_to_yaw(self.feedback.orientation)
            print("\n📍 Current Position:")
            print("   X: {:.3f} m".format(x))
            print("   Y: {:.3f} m".format(y))
            print("   Theta: {:.3f} rad ({:.1f}°)".format(yaw, math.degrees(yaw)))

            if self.current_goal:
                dx = self.current_goal.position.x - x
                dy = self.current_goal.position.y - y
                dist = math.sqrt(dx*dx + dy*dy)
                print("\n📏 Distance to goal: {:.3f} m".format(dist))
        else:
            print("\n⚠️  No feedback available")

        if self.status:
            status_text = {
                0: "PENDING",
                1: "ACTIVE",
                2: "PREEMPTED",
                3: "SUCCEEDED",
                4: "ABORTED",
                5: "REJECTED",
                6: "PREEMPTING",
                7: "RECALLING",
                8: "RECALLED",
                9: "LOST"
            }
            status_name = status_text.get(self.status.status, "UNKNOWN")
            print("\n🚦 Goal Status: {} ({})".format(status_name, self.status.status))
            if self.status.text:
                print("   Message: {}".format(self.status.text))

        print("\n" + "=" * 60 + "\n")


if __name__ == '__main__':
    rospy.init_node('check_navigation', anonymous=True)

    checker = NavigationChecker()

    print("\nWaiting for navigation data...")
    rospy.sleep(2.0)

    checker.print_status()
