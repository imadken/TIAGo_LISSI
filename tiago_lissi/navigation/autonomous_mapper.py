#!/usr/bin/env python3
"""
Enhanced Autonomous Mapper - Production-grade exploration with patterns
Combines reactive safety (ProductionExplorer) with systematic coverage patterns
"""

import rospy
import math
import random
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry


# State Machine
STATE_EXPLORING = 0
STATE_CLEARING = 1
STATE_CONFIRMING = 2


class EnhancedMapper:
    def __init__(self, mode='reactive', duration=300):
        rospy.init_node('enhanced_mapper', anonymous=True)

        # Publishers and Subscribers
        self.cmd_vel_pub = rospy.Publisher('/nav_vel', Twist, queue_size=10)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.odom_sub = rospy.Subscriber('/mobile_base_controller/odom', Odometry, self.odom_callback)

        # Safety Parameters
        self.max_linear_speed = 0.20
        self.turn_speed = 0.4
        self.warning_distance = 1.2
        self.stop_distance = 0.5
        self.extra_clearance_angle = math.radians(45)

        # State variables
        self.state = STATE_EXPLORING
        self.dist_front = float('inf')
        self.dist_left = float('inf')
        self.dist_right = float('inf')
        self.current_yaw = 0.0
        self.target_yaw = 0.0
        self.turn_direction = 1.0
        self.stuck_start_time = None  # Track how long we've been stuck

        # Exploration mode and timing
        self.mode = mode
        self.duration = duration
        self.start_time = rospy.Time.now()

        # Pattern-specific variables
        self.spiral_segment = 0
        self.max_spiral_segments = 12
        self.last_turn_time = rospy.Time.now()
        self.turn_interval = 10.0  # seconds between turns in random mode

        # Statistics
        self.obstacles_avoided = 0
        self.distance_traveled = 0.0
        self.last_x = 0.0
        self.last_y = 0.0

        rospy.on_shutdown(self.stop)

        rospy.loginfo("[Mapper] ========== Enhanced Autonomous Mapper ==========")
        rospy.loginfo("[Mapper] Mode: {}".format(mode))
        rospy.loginfo("[Mapper] Duration: {} seconds ({} minutes)".format(duration, duration/60))
        rospy.loginfo("[Mapper] Safety: Zoned detection, proportional speed")
        rospy.loginfo("[Mapper] ================================================")
        rospy.sleep(2)

        # Watchdog (start after init)
        self.last_loop_time = rospy.Time.now()
        self.watchdog_timer = rospy.Timer(rospy.Duration(0.1), self.watchdog_callback)

    def get_yaw_from_quaternion(self, qx, qy, qz, qw):
        """Convert quaternion to yaw."""
        t3 = 2.0 * (qw * qz + qx * qy)
        t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(t3, t4)

    def watchdog_callback(self, event):
        """Monitor control loop health."""
        if (rospy.Time.now() - self.last_loop_time).to_sec() > 0.5:
            rospy.logerr_throttle(2.0, "[WATCHDOG] Control loop stalled! Emergency stop.")
            self.cmd_vel_pub.publish(Twist())

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def odom_callback(self, msg):
        """Update position and heading from odometry."""
        q = msg.pose.pose.orientation
        self.current_yaw = self.get_yaw_from_quaternion(q.x, q.y, q.z, q.w)

        # Track distance traveled
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.last_x != 0.0 or self.last_y != 0.0:
            dx = x - self.last_x
            dy = y - self.last_y
            self.distance_traveled += math.sqrt(dx*dx + dy*dy)
        self.last_x = x
        self.last_y = y

    def scan_callback(self, scan):
        """Process laser scan into safety zones."""
        min_front = float('inf')
        min_left = float('inf')
        min_right = float('inf')

        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r) or r < scan.range_min or r > scan.range_max:
                continue

            angle = scan.angle_min + i * scan.angle_increment

            if -math.pi/6 <= angle <= math.pi/6:       # Front ±30°
                min_front = min(min_front, r)
            elif math.pi/6 < angle <= math.pi/2:       # Left 30-90°
                min_left = min(min_left, r)
            elif -math.pi/2 <= angle < -math.pi/6:     # Right -30 to -90°
                min_right = min(min_right, r)

        self.dist_front = min_front
        self.dist_left = min_left
        self.dist_right = min_right

    def stop(self):
        """Emergency stop."""
        self.cmd_vel_pub.publish(Twist())

    def should_continue(self):
        """Check if exploration should continue."""
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        return elapsed < self.duration and not rospy.is_shutdown()

    def reactive_navigation(self, twist):
        """Core reactive obstacle avoidance (state machine)."""
        if self.state == STATE_EXPLORING:
            if self.dist_front <= self.stop_distance:
                rospy.logwarn("[Mapper] Obstacle! Avoiding... (#{})".format(self.obstacles_avoided + 1))
                self.obstacles_avoided += 1
                self.stop()
                self.turn_direction = 1.0 if self.dist_left > self.dist_right else -1.0
                self.state = STATE_CLEARING
            else:
                # Proportional speed control
                ratio = (self.dist_front - self.stop_distance) / (self.warning_distance - self.stop_distance)
                clamped_ratio = max(0.0, min(1.0, ratio))
                twist.linear.x = self.max_linear_speed * clamped_ratio
                twist.angular.z = 0.0  # Go straight, no rotation
                return True  # Allow forward motion

        elif self.state == STATE_CLEARING:
            if self.dist_left < self.stop_distance or self.dist_right < self.stop_distance:
                # Track how long we've been stuck
                if self.stuck_start_time is None:
                    self.stuck_start_time = rospy.Time.now()

                stuck_duration = (rospy.Time.now() - self.stuck_start_time).to_sec()

                # If stuck for more than 3 seconds, back up
                if stuck_duration > 3.0:
                    rospy.logwarn("[Mapper] Stuck for {:.1f}s! Backing up...".format(stuck_duration))
                    twist.linear.x = -0.15  # Reverse slowly
                    twist.angular.z = 0.0
                    self.cmd_vel_pub.publish(twist)

                    # Reset after backing up for 2 seconds
                    if stuck_duration > 5.0:
                        self.stuck_start_time = None
                        self.state = STATE_EXPLORING
                        rospy.loginfo("[Mapper] Recovery complete, resuming exploration")
                else:
                    rospy.logerr_throttle(2.0, "[Mapper] Side collision risk! Waiting... ({:.1f}s)".format(stuck_duration))
                    self.stop()
            else:
                # Clear to turn
                self.stuck_start_time = None
                twist.linear.x = 0.0
                twist.angular.z = self.turn_speed * self.turn_direction
                self.cmd_vel_pub.publish(twist)

            if self.dist_front > self.warning_distance and self.stuck_start_time is None:
                self.target_yaw = self.normalize_angle(
                    self.current_yaw + (self.extra_clearance_angle * self.turn_direction))
                self.state = STATE_CONFIRMING

        elif self.state == STATE_CONFIRMING:
            angle_diff = self.normalize_angle(self.target_yaw - self.current_yaw)
            if abs(angle_diff) < 0.05:
                self.stop()
                self.state = STATE_EXPLORING
            else:
                twist.linear.x = 0.0
                twist.angular.z = self.turn_speed if angle_diff > 0 else -self.turn_speed
                self.cmd_vel_pub.publish(twist)

        return False

    def explore_spiral(self):
        """Spiral pattern with reactive safety."""
        rospy.loginfo("[Mapper] === Starting Spiral Exploration ===")
        rate = rospy.Rate(10)
        twist = Twist()

        while self.should_continue() and self.spiral_segment < self.max_spiral_segments:
            self.last_loop_time = rospy.Time.now()

            # Reactive safety override
            if not self.reactive_navigation(twist):
                rate.sleep()
                continue

            # Spiral pattern: move forward, then turn
            rospy.loginfo_throttle(10.0, "[Mapper] Segment {}/{}, Distance: {:.1f}m, Obstacles: {}".format(
                self.spiral_segment + 1, self.max_spiral_segments,
                self.distance_traveled, self.obstacles_avoided))

            # Just keep moving forward (reactive nav handles obstacles)
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

    def explore_random(self):
        """Random walk with reactive safety."""
        rospy.loginfo("[Mapper] === Starting Random Walk ===")
        rate = rospy.Rate(10)
        twist = Twist()

        while self.should_continue():
            self.last_loop_time = rospy.Time.now()

            # Reactive safety override
            if not self.reactive_navigation(twist):
                rate.sleep()
                continue

            # Random turns every N seconds
            if (rospy.Time.now() - self.last_turn_time).to_sec() > self.turn_interval:
                angle = random.uniform(-math.pi/2, math.pi/2)
                rospy.loginfo("[Mapper] Random turn: {:.0f}°".format(math.degrees(angle)))
                self.target_yaw = self.normalize_angle(self.current_yaw + angle)
                self.last_turn_time = rospy.Time.now()
                self.state = STATE_CONFIRMING

            rospy.loginfo_throttle(10.0, "[Mapper] Distance: {:.1f}m, Obstacles avoided: {}".format(
                self.distance_traveled, self.obstacles_avoided))

            self.cmd_vel_pub.publish(twist)
            rate.sleep()

    def explore_reactive(self):
        """Pure reactive exploration (no pattern, just avoid obstacles and keep moving)."""
        rospy.loginfo("[Mapper] === Starting Reactive Exploration ===")
        rate = rospy.Rate(10)
        twist = Twist()

        while self.should_continue():
            self.last_loop_time = rospy.Time.now()

            self.reactive_navigation(twist)

            rospy.loginfo_throttle(10.0, "[Mapper] Distance: {:.1f}m, Obstacles: {}".format(
                self.distance_traveled, self.obstacles_avoided))

            self.cmd_vel_pub.publish(twist)
            rate.sleep()

    def run(self):
        """Main exploration loop."""
        if self.mode == 'spiral':
            self.explore_spiral()
        elif self.mode == 'random':
            self.explore_random()
        else:  # reactive
            self.explore_reactive()

        # Final stats
        self.stop()
        rospy.loginfo("=" * 60)
        rospy.loginfo("[Mapper] Exploration Complete!")
        rospy.loginfo("[Mapper] Duration: {:.1f} minutes".format(
            (rospy.Time.now() - self.start_time).to_sec() / 60))
        rospy.loginfo("[Mapper] Distance traveled: {:.1f} meters".format(self.distance_traveled))
        rospy.loginfo("[Mapper] Obstacles avoided: {}".format(self.obstacles_avoided))
        rospy.loginfo("=" * 60)


def main():
    """Main entry point."""
    import sys

    print("\n" + "=" * 70)
    print("ENHANCED AUTONOMOUS MAPPER")
    print("=" * 70)

    # Parse arguments
    mode = 'reactive'
    duration = 300  # 5 minutes default

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode not in ['spiral', 'random', 'reactive']:
            print("\nERROR: Invalid mode '{}'".format(mode))
            print("\nUsage:")
            print("  python3 autonomous_mapper.py [mode] [duration_minutes]")
            print("\nModes:")
            print("  reactive  - Pure obstacle avoidance (default)")
            print("  spiral    - Expanding spiral pattern")
            print("  random    - Random walk with turns")
            print("\nExamples:")
            print("  python3 autonomous_mapper.py reactive 5")
            print("  python3 autonomous_mapper.py spiral 10")
            print("  python3 autonomous_mapper.py random 3")
            print("=" * 70 + "\n")
            sys.exit(1)

    if len(sys.argv) > 2:
        duration = int(sys.argv[2]) * 60  # Convert minutes to seconds

    print("\nMode: {}".format(mode.upper()))
    print("Duration: {} minutes".format(duration / 60))
    print("\nMake sure mapping is running:")
    print("  roslaunch tiago_2dnav mapping.launch")
    print("\nPress Ctrl+C to stop early and save map")
    print("=" * 70 + "\n")

    try:
        mapper = EnhancedMapper(mode=mode, duration=duration)
        mapper.run()

        print("\n" + "=" * 70)
        print("SAVE THE MAP NOW:")
        print("  rosrun map_server map_saver -f ~/my_room_map")
        print("=" * 70 + "\n")

    except rospy.ROSInterruptException:
        print("\nMapping interrupted by user")
    except KeyboardInterrupt:
        print("\nMapping stopped by user")


if __name__ == '__main__':
    main()
