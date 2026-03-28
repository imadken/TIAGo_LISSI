#!/usr/bin/env python3
"""
Minimal ROS node that grabs ONE frame from the Xtion camera and saves it
to /tmp/eval_frame.jpg, then exits.  Runs inside Docker.
"""
import sys, os
import numpy as np

try:
    import rospy
    from sensor_msgs.msg import Image
except ImportError:
    print('[save_frame] rospy not available', file=sys.stderr)
    sys.exit(1)

try:
    import cv2
except ImportError:
    print('[save_frame] cv2 not available', file=sys.stderr)
    sys.exit(1)

OUT = os.environ.get('EVAL_FRAME_PATH', '/tmp/eval_frame.jpg')

rospy.init_node('eval_save_frame', anonymous=True)
try:
    msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image, timeout=8.0)
except rospy.ROSException as e:
    print(f'[save_frame] timeout: {e}', file=sys.stderr)
    sys.exit(1)

if msg.encoding == 'rgb8':
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
elif msg.encoding == 'bgr8':
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
else:
    print(f'[save_frame] unknown encoding {msg.encoding}', file=sys.stderr)
    sys.exit(1)

cv2.imwrite(OUT, img)
print(f'[save_frame] saved {img.shape} → {OUT}')
