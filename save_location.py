#!/usr/bin/env python3
"""
Interactive Location Saver
Drive robot to position, press Enter to save location to YAML
"""

import rospy
import yaml
import os
import math
from geometry_msgs.msg import PoseWithCovarianceStamped


class LocationSaver:
    def __init__(self):
        rospy.init_node('location_saver', anonymous=True)

        self.current_pose = None

        # Subscribe to robot pose
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self._pose_callback)

        # Wait for first pose
        print("Waiting for robot position...")
        rospy.sleep(1.0)

        if self.current_pose is None:
            print("ERROR: No pose received. Is /amcl_pose publishing?")
            return

        print("\n" + "=" * 60)
        print("LOCATION SAVER")
        print("=" * 60)
        print("Drive robot to desired position, then run this script")
        print("It will save the current position to config/locations.yaml")
        print("=" * 60 + "\n")

    def _pose_callback(self, msg):
        """Store latest pose."""
        self.current_pose = msg.pose.pose

    def get_current_position(self):
        """Get current x, y, theta."""
        if self.current_pose is None:
            return None

        x = self.current_pose.position.x
        y = self.current_pose.position.y

        # Convert quaternion to theta
        qz = self.current_pose.orientation.z
        qw = self.current_pose.orientation.w
        theta = 2 * math.atan2(qz, qw)

        return (x, y, theta)

    def save_location(self, name, description=""):
        """Save current position to YAML."""
        pos = self.get_current_position()
        if pos is None:
            print("ERROR: No position available")
            return False

        x, y, theta = pos

        # Load existing locations
        yaml_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'config', 'locations.yaml'
        )

        if os.path.exists(yaml_file):
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                locations = data.get('locations', {})
        else:
            locations = {}

        # Add new location
        locations[name] = {
            'x': round(x, 3),
            'y': round(y, 3),
            'theta': round(theta, 3),
            'description': description
        }

        # Save back to YAML
        with open(yaml_file, 'w') as f:
            yaml.dump({'locations': locations}, f, default_flow_style=False)

        print("\n✓ Saved location '{}':".format(name))
        print("  x: {:.3f}".format(x))
        print("  y: {:.3f}".format(y))
        print("  theta: {:.3f} ({:.1f} degrees)".format(theta, math.degrees(theta)))
        print("  File: {}".format(yaml_file))

        return True

    def interactive_save(self):
        """Interactive mode - save multiple locations."""
        print("\nCurrent position:")
        pos = self.get_current_position()
        if pos:
            x, y, theta = pos
            print("  x: {:.3f} m".format(x))
            print("  y: {:.3f} m".format(y))
            print("  theta: {:.3f} rad ({:.1f} deg)".format(theta, math.degrees(theta)))

        name = input("\nLocation name (e.g., 'table'): ").strip()
        if not name:
            print("Cancelled")
            return False

        description = input("Description (optional): ").strip()

        return self.save_location(name, description)


if __name__ == '__main__':
    try:
        saver = LocationSaver()
        saver.interactive_save()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\nCancelled")
