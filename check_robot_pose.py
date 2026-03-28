#!/usr/bin/env python3
"""
Check robot's current position in the map
"""

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import math


def quaternion_to_yaw(q):
    """Convert quaternion to yaw angle."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PoseChecker:
    def __init__(self):
        self.amcl_pose = None
        self.odom_pose = None

        # Subscribe to AMCL pose (robot position in map frame)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_callback)

        # Subscribe to odometry (robot position in odom frame)
        rospy.Subscriber('/mobile_base_controller/odom', Odometry, self.odom_callback)

    def amcl_callback(self, msg):
        self.amcl_pose = msg.pose.pose

    def odom_callback(self, msg):
        self.odom_pose = msg.pose.pose

    def print_poses(self):
        print("\n" + "=" * 60)
        print("ROBOT CURRENT POSITION")
        print("=" * 60)

        if self.amcl_pose:
            x = self.amcl_pose.position.x
            y = self.amcl_pose.position.y
            yaw = quaternion_to_yaw(self.amcl_pose.orientation)
            print("\n📍 Position in MAP frame (for navigation):")
            print("   X: {:.3f} m".format(x))
            print("   Y: {:.3f} m".format(y))
            print("   Theta: {:.3f} rad ({:.1f}°)".format(yaw, math.degrees(yaw)))
            print("\n💡 To send robot HERE, use:")
            print("   python3 go_to_pose.py {:.3f} {:.3f} {:.3f}".format(x, y, yaw))
        else:
            print("\n⚠️  No AMCL pose available")
            print("   Is localization running?")

        if self.odom_pose:
            x = self.odom_pose.position.x
            y = self.odom_pose.position.y
            yaw = quaternion_to_yaw(self.odom_pose.orientation)
            print("\n📍 Position in ODOM frame:")
            print("   X: {:.3f} m".format(x))
            print("   Y: {:.3f} m".format(y))
            print("   Theta: {:.3f} rad ({:.1f}°)".format(yaw, math.degrees(yaw)))
        else:
            print("\n⚠️  No odometry available")

        print("\n" + "=" * 60)
        print("TIP: The MAP frame position is what you use for navigation.")
        print("     (0, 0, 0) might not be where the robot started!")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    rospy.init_node('check_robot_pose', anonymous=True)

    checker = PoseChecker()

    print("\nWaiting for pose data...")
    rospy.sleep(1.0)

    checker.print_poses()
