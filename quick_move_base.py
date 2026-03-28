#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Simple script to move TIAGo's base forward/backward
Usage: python2 quick_move_base.py [distance_meters]
Example: python2 quick_move_base.py 0.5  # Move 50cm forward
"""

import rospy
from geometry_msgs.msg import Twist

def move_base_forward(distance):
    """Move base forward by specified distance (negative for backward)"""
    rospy.init_node('quick_move_base', anonymous=True)
    pub = rospy.Publisher('/nav_vel', Twist, queue_size=1)

    rospy.sleep(0.5)  # Wait for publisher

    velocity = 0.1 if distance > 0 else -0.1  # 10 cm/s
    duration = abs(distance) / abs(velocity)

    print("Moving base {:.2f}m at {:.2f} m/s (duration: {:.1f}s)".format(distance, velocity, duration))

    twist = Twist()
    twist.linear.x = velocity

    rate = rospy.Rate(10)  # 10 Hz
    start_time = rospy.Time.now()

    while (rospy.Time.now() - start_time).to_sec() < duration:
        pub.publish(twist)
        rate.sleep()

    # Stop
    twist.linear.x = 0.0
    pub.publish(twist)
    print("Movement complete!")

if __name__ == '__main__':
    import sys

    if len(sys.argv) == 2:
        distance = float(sys.argv[1])
    else:
        distance = 0.5  # Default: 50cm forward

    print("Moving base: {:.2f}m".format(distance))

    try:
        move_base_forward(distance)
    except rospy.ROSInterruptException:
        print("Interrupted")
