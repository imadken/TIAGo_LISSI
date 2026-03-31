#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Simple script to move TIAGo's head to look at a specific position
Usage: python2 move_head.py [head_1_joint] [head_2_joint]
Example: python2 move_head.py 0.0 -0.3
"""

import rospy
import sys
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def move_head(pan, tilt):
    """
    Move TIAGo's head to specified joint positions

    Args:
        pan: head_1_joint position (yaw, left/right) in radians
        tilt: head_2_joint position (pitch, up/down) in radians
    """
    rospy.init_node('move_head', anonymous=True)

    # Publisher for head controller
    pub = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=1)

    # Wait for publisher to connect
    rospy.sleep(0.5)

    # Create trajectory message
    traj = JointTrajectory()
    traj.joint_names = ['head_1_joint', 'head_2_joint']

    # Create trajectory point
    point = JointTrajectoryPoint()
    point.positions = [pan, tilt]
    point.velocities = [0.0, 0.0]
    point.time_from_start = rospy.Duration(2.0)  # 2 seconds to reach position

    traj.points = [point]

    # Publish command
    print("Moving head to: pan={:.2f} rad, tilt={:.2f} rad".format(pan, tilt))
    pub.publish(traj)

    rospy.sleep(2.5)
    print("Head movement complete!")


if __name__ == '__main__':
    # Parse command line arguments
    if len(sys.argv) == 3:
        try:
            pan = float(sys.argv[1])
            tilt = float(sys.argv[2])
        except ValueError:
            print("Error: Arguments must be numbers")
            print("Usage: python2 move_head.py [pan] [tilt]")
            sys.exit(1)
    else:
        # Default: PAL official detection angle from inspect_surroundings motion
        pan = 0.0
        tilt = -0.85

    print("TIAGo Head Control")
    print("=" * 40)
    print("Pan (head_1_joint):  {:.2f} rad".format(pan))
    print("Tilt (head_2_joint): {:.2f} rad".format(tilt))
    print("=" * 40)

    try:
        move_head(pan, tilt)
    except rospy.ROSInterruptException:
        print("Interrupted")
    except Exception as e:
        print("Error: {}".format(e))
