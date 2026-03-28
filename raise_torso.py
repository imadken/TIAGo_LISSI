#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Raise TIAGo's torso to maximum height to extend arm reach
Usage: python2 raise_torso.py [height]
Example: python2 raise_torso.py 0.35  # Max height
"""

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def raise_torso(height):
    """
    Raise torso to specified height

    Args:
        height: torso_lift_joint position (0.0 to 0.35 meters)
    """
    rospy.init_node('raise_torso', anonymous=True)

    pub = rospy.Publisher('/torso_controller/command', JointTrajectory, queue_size=1)
    rospy.sleep(0.5)

    traj = JointTrajectory()
    traj.joint_names = ['torso_lift_joint']

    point = JointTrajectoryPoint()
    point.positions = [height]
    point.velocities = [0.0]
    point.time_from_start = rospy.Duration(3.0)

    traj.points = [point]

    print("Raising torso to {:.2f}m height...".format(height))
    pub.publish(traj)

    rospy.sleep(3.5)
    print("Torso raised!")

if __name__ == '__main__':
    import sys

    if len(sys.argv) == 2:
        height = float(sys.argv[1])
    else:
        height = 0.35  # Max height

    # Clamp to valid range
    height = max(0.0, min(0.35, height))

    print("TIAGo Torso Control")
    print("=" * 40)
    print("Target height: {:.2f}m".format(height))
    print("=" * 40)

    try:
        raise_torso(height)
    except rospy.ROSInterruptException:
        print("Interrupted")
