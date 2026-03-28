#!/usr/bin/env python3
"""
Turn robot 180 degrees and drive forward to escape corner
Uses odometry for accurate rotation
"""

import rospy
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class TurnAndGo:
    def __init__(self):
        rospy.init_node('turn_and_go', anonymous=True)

        self.cmd_vel_pub = rospy.Publisher('/key_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/mobile_base_controller/odom', Odometry, self.odom_callback)

        self.current_yaw = None
        self.start_yaw = None

        rospy.loginfo("[TurnAndGo] Waiting for odometry...")
        rospy.sleep(1.0)

    def odom_callback(self, msg):
        """Get current heading from odometry."""
        q = msg.pose.pose.orientation
        # Convert quaternion to yaw
        t3 = 2.0 * (q.w * q.z + q.x * q.y)
        t4 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(t3, t4)

    def normalize_angle(self, angle):
        """Keep angle in -pi to pi range."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def stop(self):
        """Stop the robot."""
        self.cmd_vel_pub.publish(Twist())

    def turn_180(self):
        """Turn robot 180 degrees using odometry feedback."""
        rospy.loginfo("[TurnAndGo] Starting 180-degree turn...")

        # Record starting orientation
        self.start_yaw = self.current_yaw
        target_yaw = self.normalize_angle(self.start_yaw + math.pi)  # +180 degrees

        twist = Twist()
        twist.angular.z = 0.4  # rad/s (adjust if needed)

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            # Calculate remaining angle
            angle_diff = self.normalize_angle(target_yaw - self.current_yaw)

            rospy.loginfo_throttle(1.0, "[TurnAndGo] Remaining: {:.1f}°".format(
                math.degrees(abs(angle_diff))))

            # Are we there yet?
            if abs(angle_diff) < 0.1:  # ~5.7 degrees tolerance
                rospy.loginfo("[TurnAndGo] ✓ 180-degree turn complete!")
                self.stop()
                break

            # Keep turning
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        rospy.sleep(0.5)

    def go_forward(self, distance=2.0, speed=0.2):
        """Drive forward for specified distance."""
        rospy.loginfo("[TurnAndGo] Moving forward {:.1f}m...".format(distance))

        twist = Twist()
        twist.linear.x = speed

        duration = distance / speed
        rate = rospy.Rate(10)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()

            if elapsed >= duration:
                rospy.loginfo("[TurnAndGo] ✓ Forward motion complete!")
                self.stop()
                break

            remaining = distance - (elapsed * speed)
            rospy.loginfo_throttle(1.0, "[TurnAndGo] Distance remaining: {:.1f}m".format(remaining))

            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        rospy.sleep(0.5)

    def execute(self, forward_distance=2.0):
        """Execute full sequence: turn 180° then go forward."""
        print("\n" + "=" * 60)
        print("TURN AND GO - Escape Corner")
        print("=" * 60)
        print("\nThis will:")
        print("  1. Turn 180 degrees (using odometry)")
        print("  2. Drive forward {:.1f} meters".format(forward_distance))
        print("\nStarting in 2 seconds...")
        print("=" * 60 + "\n")

        rospy.sleep(2)

        # Execute sequence
        self.turn_180()
        self.go_forward(distance=forward_distance)

        print("\n" + "=" * 60)
        print("✓ Sequence complete! Robot should be clear of corner.")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    import sys

    try:
        robot = TurnAndGo()

        # Get forward distance from command line (default 2.0m)
        distance = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0

        robot.execute(forward_distance=distance)

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print("\nError: {}".format(e))
