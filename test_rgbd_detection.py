#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test RGB-D 3D position detection without moving the robot
Useful for debugging perception before trying to grasp
"""

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


def test_rgbd_detection():
    """Test getting 3D position from RGB-D camera"""

    rospy.init_node('test_rgbd_detection', anonymous=True)

    print("\n" + "=" * 60)
    print("TESTING RGB-D 3D POSITION DETECTION")
    print("=" * 60)

    try:
        # Step 1: Get point cloud
        print("\n[1/3] Waiting for point cloud from Xtion camera...")
        print("        Topic: /xtion/depth_registered/points")
        print("        (This may take a few seconds...)")

        pointcloud = rospy.wait_for_message(
            '/xtion/depth_registered/points',
            PointCloud2,
            timeout=10.0
        )

        print("        OK Received point cloud: {}x{} points".format(pointcloud.width, pointcloud.height))
        print("        Frame ID: {}".format(pointcloud.header.frame_id))

        # Step 2: Sample points in center region (where bottle should be)
        print("\n[2/3] Sampling 3D points in center region...")

        cx, cy = pointcloud.width // 2, pointcloud.height // 2
        sample_region = 50  # pixels around center

        points_3d = []
        valid_points = 0
        nan_points = 0

        for u in range(cx - sample_region, cx + sample_region, 5):
            for v in range(cy - sample_region, cy + sample_region, 5):
                # Read point at (u, v)
                gen = pc2.read_points(pointcloud, uvs=[(u, v)], skip_nans=False)
                for p in gen:
                    if np.isnan(p[0]) or np.isnan(p[1]) or np.isnan(p[2]):
                        nan_points += 1
                    elif 0.3 < p[2] < 2.0:  # Valid depth range (30cm to 2m)
                        points_3d.append([p[0], p[1], p[2]])
                        valid_points += 1

        print("        Valid 3D points: {}".format(valid_points))
        print("        NaN points (no depth): {}".format(nan_points))

        if len(points_3d) < 10:
            print("\n        ERROR Not enough valid points!")
            print("\n        Possible issues:")
            print("          - Nothing in front of camera")
            print("          - Object too close (<30cm) or too far (>2m)")
            print("          - Camera not working")
            print("\n        Try:")
            print("          1. Place a bottle 50-100cm in front of robot")
            print("          2. Check camera: rostopic hz /xtion/depth_registered/points")
            return False

        # Step 3: Compute median 3D position
        print("\n[3/3] Computing 3D position...")

        points_3d = np.array(points_3d)

        # Statistics
        median_pos = np.median(points_3d, axis=0)
        mean_pos = np.mean(points_3d, axis=0)
        std_pos = np.std(points_3d, axis=0)

        print("\n        Median position (robust estimate):")
        print("          x = {:.3f} m  (forward/backward)".format(median_pos[0]))
        print("          y = {:.3f} m  (left/right)".format(median_pos[1]))
        print("          z = {:.3f} m  (up/down)".format(median_pos[2]))

        print("\n        Mean position:")
        print("          x = {:.3f} m".format(mean_pos[0]))
        print("          y = {:.3f} m".format(mean_pos[1]))
        print("          z = {:.3f} m".format(mean_pos[2]))

        print("\n        Standard deviation (spread):")
        print("          x = {:.3f} m".format(std_pos[0]))
        print("          y = {:.3f} m".format(std_pos[1]))
        print("          z = {:.3f} m".format(std_pos[2]))

        # Distance to bottle
        distance = np.linalg.norm(median_pos)
        print("\n        Distance to object: {:.3f} m".format(distance))

        print("\n" + "=" * 60)
        print("OK RGB-D 3D DETECTION TEST PASSED!")
        print("=" * 60)
        print("\nInterpretation:")
        print("  - Object detected at {:.2f}m from robot base".format(distance))
        print("  - Position: {:.2f}m forward, {:.2f}m {}".format(median_pos[0], median_pos[1], 'left' if median_pos[1] > 0 else 'right'))

        if distance < 0.4:
            print("\n  Warning: Object very close (<40cm) - grasping might be difficult")
        elif distance > 1.5:
            print("\n  Warning: Object far (>1.5m) - might be out of reach")
        else:
            print("\n  OK Distance looks good for grasping!")

        print("\n" + "=" * 60 + "\n")

        return True

    except rospy.ROSException as e:
        print("\nERROR: Failed to get point cloud: {}".format(e))
        print("\nTroubleshooting:")
        print("  1. Is the robot on?")
        print("  2. Is Xtion camera working?")
        print("     Check: rostopic list | grep xtion")
        print("     Should see: /xtion/depth_registered/points")
        print("  3. Is ROS_MASTER_URI correct?")
        print("     Check: echo $ROS_MASTER_URI")
        print("     Should be: http://<ROBOT_IP>:11311")
        return False

    except Exception as e:
        print("\nERROR UNEXPECTED ERROR: {}".format(e))
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    try:
        test_rgbd_detection()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
