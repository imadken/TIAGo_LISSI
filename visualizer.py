#!/usr/bin/env python3
"""
Live Visualizer for TIAGo Embodied Agent
Shows annotated camera feed + current skill + robot state in an OpenCV window.

Run alongside the agent:
  docker exec -it tiago_ros bash -c "
    export ROS_MASTER_URI=http://${ROBOT_IP:-10.68.0.1}:11311 &&
    export ROS_IP=${HOST_IP:-$(hostname -I | awk '{print $1}')} &&
    export DISPLAY=:0 &&
    source /opt/ros/melodic/setup.bash &&
    source /workspace/pal_ws/devel/setup.bash &&
    cd /workspace &&
    python3 visualizer.py"
"""

import json
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String


class AgentVisualizer:
    def __init__(self):
        rospy.init_node('agent_visualizer', anonymous=True)

        self.latest_frame = None
        self.latest_status = {
            'skill': 'idle',
            'gripper': 'unknown',
            'location': 'unknown',
            'detected': [],
            'history': []
        }

        rospy.Subscriber('/agent/camera_annotated', Image, self._image_cb)
        rospy.Subscriber('/agent/status', String, self._status_cb)

        rospy.loginfo("[Visualizer] Subscribed to /agent/camera_annotated and /agent/status")
        rospy.loginfo("[Visualizer] Press Q in the window to quit")

    def _image_cb(self, msg):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            self.latest_frame = arr.reshape(msg.height, msg.width, 3)
        except Exception as e:
            rospy.logwarn("[Visualizer] Image decode error: {}".format(e))

    def _status_cb(self, msg):
        try:
            self.latest_status = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn("[Visualizer] Status decode error: {}".format(e))

    def _draw_status_panel(self, frame):
        """Draw status panel below the camera image."""
        s = self.latest_status
        panel_h = 120
        h, w = frame.shape[:2]
        panel = np.zeros((panel_h, w, 3), dtype=np.uint8)

        # Skill indicator — green if active, grey if idle
        skill = s.get('skill', 'idle')
        skill_color = (0, 255, 0) if skill != 'idle' else (150, 150, 150)
        cv2.putText(panel, "SKILL: {}".format(skill.upper()),
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, skill_color, 2)

        # Gripper + location
        gripper = s.get('gripper', 'unknown')
        gripper_color = (0, 200, 255) if 'holding' in gripper else (200, 200, 200)
        cv2.putText(panel, "GRIPPER: {}".format(gripper),
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gripper_color, 1)

        # Detected objects
        detected = s.get('detected', [])
        det_str = "DETECTED: {}".format(', '.join(detected) if detected else 'none')
        cv2.putText(panel, det_str, (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 1)

        # Task history (last 5)
        history = s.get('history', [])
        hist_str = "HISTORY: {}".format('  '.join(history[-4:]) if history else 'none')
        cv2.putText(panel, hist_str, (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

        # Divider
        cv2.line(panel, (0, 0), (w, 0), (80, 80, 80), 1)

        return np.vstack([frame, panel])

    def run(self):
        rate = rospy.Rate(15)  # 15 Hz display
        cv2.namedWindow("TIAGo Agent", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("TIAGo Agent", 800, 700)

        while not rospy.is_shutdown():
            if self.latest_frame is not None:
                display = self._draw_status_panel(self.latest_frame.copy())
                cv2.imshow("TIAGo Agent", display)
            else:
                # Waiting screen
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for agent...",
                            (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
                cv2.imshow("TIAGo Agent", placeholder)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        AgentVisualizer().run()
    except rospy.ROSInterruptException:
        pass
