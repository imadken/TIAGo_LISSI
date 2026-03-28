#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Simple test script to verify MoveIt connection to TIAGo robot
Run this first to check if everything is configured correctly
"""

import rospy
import sys
import moveit_commander


def test_moveit_connection():
    """Test MoveIt connection and print robot state"""

    rospy.init_node('test_moveit', anonymous=True)

    print("\n" + "=" * 60)
    print("TESTING MOVEIT CONNECTION")
    print("=" * 60)

    try:
        # Initialize MoveIt
        print("\n[1/5] Initializing MoveIt commander...")
        moveit_commander.roscpp_initialize(sys.argv)
        print("      OK MoveIt initialized")

        # Connect to robot
        print("\n[2/5] Connecting to robot...")
        robot = moveit_commander.RobotCommander()
        print("      OK Robot name: {}".format(robot.get_robot_name()))

        # Get planning scene
        print("\n[3/5] Connecting to planning scene...")
        scene = moveit_commander.PlanningSceneInterface()
        print("      OK Planning scene connected")

        # Get arm group
        print("\n[4/5] Connecting to arm_torso group...")
        arm = moveit_commander.MoveGroupCommander("arm_torso")
        print("      OK Planning group: {}".format(arm.get_name()))
        print("      OK End effector link: {}".format(arm.get_end_effector_link()))
        print("      OK Planning frame: {}".format(arm.get_planning_frame()))

        # Get gripper group
        print("\n[5/5] Connecting to gripper group...")
        gripper = moveit_commander.MoveGroupCommander("gripper")
        print("      OK Gripper group: {}".format(gripper.get_name()))
        print("      OK Gripper joints: {}".format(gripper.get_active_joints()))

        # Print current joint values
        print("\n" + "=" * 60)
        print("CURRENT ROBOT STATE")
        print("=" * 60)
        print("\nArm joints:")
        joint_values = arm.get_current_joint_values()
        for i, joint_name in enumerate(arm.get_active_joints()):
            print("  {}: {:.3f}".format(joint_name, joint_values[i]))

        print("\nCurrent pose of end effector:")
        current_pose = arm.get_current_pose().pose
        print("  Position: x={:.3f}, y={:.3f}, z={:.3f}".format(
              current_pose.position.x,
              current_pose.position.y,
              current_pose.position.z))
        print("  Orientation: x={:.3f}, y={:.3f}, z={:.3f}, w={:.3f}".format(
              current_pose.orientation.x,
              current_pose.orientation.y,
              current_pose.orientation.z,
              current_pose.orientation.w))

        # Test named targets
        print("\nAvailable named targets for arm:")
        print("  {}".format(arm.get_named_targets()))

        print("\nAvailable named targets for gripper:")
        print("  {}".format(gripper.get_named_targets()))

        print("\n" + "=" * 60)
        print("OK MOVEIT CONNECTION TEST PASSED!")
        print("=" * 60)
        print("\nYou can now run improved_grasp_bottle.py")
        print("=" * 60 + "\n")

        return True

    except Exception as e:
        print("\nERROR: {}".format(e))
        print("\nTroubleshooting:")
        print("  1. Is the robot turned on?")
        print("  2. Is ROS_MASTER_URI set correctly? (export ROS_MASTER_URI=http://<ROBOT_IP>:11311)")
        print("  3. Is ROS_IP set correctly? (export ROS_IP=<YOUR_HOST_IP>)")
        print("  4. Can you ping the robot? (ping $ROBOT_IP)")
        print("  5. Are ROS topics visible? (rostopic list)")
        return False

    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == '__main__':
    try:
        test_moveit_connection()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
