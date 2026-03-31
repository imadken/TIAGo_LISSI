#!/usr/bin/env python3
"""
Help robot get unstuck from corner with simple movements
"""

import rospy
from geometry_msgs.msg import Twist
import sys


def move(linear_x, angular_z, duration):
    """
    Send velocity command for specified duration.

    Args:
        linear_x: Forward/backward speed (m/s) - positive = forward, negative = backward
        angular_z: Rotation speed (rad/s) - positive = left, negative = right
        duration: How long to move (seconds)
    """
    pub = rospy.Publisher('/key_vel', Twist, queue_size=10)
    rospy.sleep(0.5)  # Wait for publisher to connect

    twist = Twist()
    twist.linear.x = linear_x
    twist.angular.z = angular_z

    rate = rospy.Rate(10)  # 10 Hz
    start_time = rospy.Time.now()

    print("Moving: linear={:.2f} m/s, angular={:.2f} rad/s for {:.1f}s".format(
        linear_x, angular_z, duration))

    while (rospy.Time.now() - start_time).to_sec() < duration:
        pub.publish(twist)
        rate.sleep()

    # Stop
    twist.linear.x = 0
    twist.angular.z = 0
    pub.publish(twist)
    print("Movement complete")


def unstuck_sequence():
    """Execute a sequence to get out of corner."""
    rospy.init_node('unstuck_robot', anonymous=True)

    print("\n" + "=" * 60)
    print("UNSTUCK ROBOT - Getting out of corner")
    print("=" * 60)
    print("\nThis will:")
    print("1. Rotate 180 degrees to face away from corner")
    print("2. Move backward slowly")
    print("3. Rotate to clear orientation")
    print("\nStarting in 2 seconds...")
    print("=" * 60 + "\n")

    rospy.sleep(2)

    # Step 1: Rotate 180 degrees (turn around)
    print("\n[1/3] Rotating to face away from corner...")
    move(linear_x=0.0, angular_z=0.5, duration=6.0)  # ~180 degrees at 0.5 rad/s
    rospy.sleep(1)

    # Step 2: Move backward to clear the corner
    print("\n[2/3] Moving backward...")
    move(linear_x=-0.15, angular_z=0.0, duration=4.0)  # Move back 0.6m
    rospy.sleep(1)

    # Step 3: Rotate to adjust orientation
    print("\n[3/3] Final rotation adjustment...")
    move(linear_x=0.0, angular_z=-0.3, duration=3.0)
    rospy.sleep(1)

    print("\n" + "=" * 60)
    print("✓ Unstuck sequence complete!")
    print("Robot should be clear of corner now.")
    print("=" * 60 + "\n")


def custom_move(linear, angular, duration):
    """Custom movement with user-specified parameters."""
    rospy.init_node('unstuck_robot', anonymous=True)
    print("\nExecuting custom movement...")
    move(linear, angular, duration)
    print("\n✓ Done")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'auto':
            # Automatic unstuck sequence
            unstuck_sequence()
        elif len(sys.argv) == 4:
            # Custom movement: python3 unstuck_robot.py <linear> <angular> <duration>
            linear = float(sys.argv[1])
            angular = float(sys.argv[2])
            duration = float(sys.argv[3])
            custom_move(linear, angular, duration)
        else:
            print("\nUsage:")
            print("  python3 unstuck_robot.py auto")
            print("    - Run automatic unstuck sequence")
            print("\n  python3 unstuck_robot.py <linear> <angular> <duration>")
            print("    - Custom movement")
            print("\nExamples:")
            print("  python3 unstuck_robot.py auto")
            print("  python3 unstuck_robot.py -0.2 0.0 3.0   # Move backward")
            print("  python3 unstuck_robot.py 0.0 0.5 4.0    # Rotate left")
            print("  python3 unstuck_robot.py 0.0 -0.5 4.0   # Rotate right")
    else:
        print("\n" + "=" * 60)
        print("UNSTUCK ROBOT")
        print("=" * 60)
        print("\nUsage:")
        print("  python3 unstuck_robot.py auto")
        print("    - Run automatic unstuck sequence (recommended)")
        print("\n  python3 unstuck_robot.py <linear> <angular> <duration>")
        print("    - Manual control:")
        print("      linear:  forward(+) / backward(-) speed (m/s)")
        print("      angular: left(+) / right(-) rotation (rad/s)")
        print("      duration: time in seconds")
        print("\nExamples:")
        print("  python3 unstuck_robot.py auto")
        print("  python3 unstuck_robot.py -0.2 0.0 3.0   # Backward 3 sec")
        print("  python3 unstuck_robot.py 0.0 0.5 4.0    # Rotate left 4 sec")
        print("=" * 60 + "\n")
