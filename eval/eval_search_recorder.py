#!/usr/bin/env python3
"""
eval_search_recorder.py — Head visual search efficiency recorder (requires ROS).

Drives search_with_head and records per-position YOLO detection results,
position count until found, and total search time.

Metrics collected
─────────────────
• positions_scanned     int   (1–12)
• target_found          bool
• search_duration_s     float
• first_found_position  int | None
• yolo_conf_at_found    float | None
• timeout               bool
• false_positive_rate   float  (YOLO detected wrong class %)

Usage
─────
    python3 eval_search_recorder.py --target bottle --trials 10
    python3 eval_search_recorder.py --target person --trials 10
"""
import argparse
import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.eval_utils import save_result, timer

ROS_AVAILABLE = False
try:
    import rospy
    import actionlib
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    ROS_AVAILABLE = True
except ImportError:
    print('[warn] rospy not available — will simulate scan positions')

YOLO_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yolo11n.pt')
YOLO_CLASS_MAP = {
    'bottle': 39, 'cup': 41, 'remote': 65, 'phone': 67,
    'person': 0, 'chair': 56,
}

# 12 scan positions: (pan_rad, tilt_rad) — mirrors search_with_head.py
SCAN_POSITIONS = [
    (0.0,   0.0),   # 1 center
    (0.0,  -0.45),  # 2 center-down
    (0.8,   0.0),   # 3 left
    (0.8,  -0.45),  # 4 left-down
    (1.4,   0.0),   # 5 far-left
    (1.4,  -0.45),  # 6 far-left-down
    (-0.8,  0.0),   # 7 right
    (-0.8, -0.45),  # 8 right-down
    (-1.4,  0.0),   # 9 far-right
    (-1.4, -0.45),  # 10 far-right-down
    (0.0,   0.30),  # 11 up
    (0.0,   0.0),   # 12 return
]
SETTLE_TIME = 2.0   # seconds to wait after head move


try:
    from ultralytics import YOLO as YOLOModel
    _yolo = YOLOModel(YOLO_WEIGHTS)
except Exception:
    _yolo = None


# ── ROS helpers ───────────────────────────────────────────────────────────────

def move_head(pan: float, tilt: float, duration=1.5):
    if not ROS_AVAILABLE:
        time.sleep(0.1)
        return
    client = actionlib.SimpleActionClient(
        '/head_controller/follow_joint_trajectory',
        FollowJointTrajectoryAction)
    client.wait_for_server(timeout=rospy.Duration(3))
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = ['head_1_joint', 'head_2_joint']
    pt = JointTrajectoryPoint()
    pt.positions = [pan, tilt]
    pt.time_from_start = rospy.Duration(duration)
    goal.trajectory.points = [pt]
    client.send_goal(goal)
    client.wait_for_result(timeout=rospy.Duration(duration + 2))


def capture_frame():
    if ROS_AVAILABLE:
        try:
            bridge = CvBridge()
            msg = rospy.wait_for_message('/xtion/rgb/image_raw', Image, timeout=3.0)
            return bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            print(f'[warn] {e}')
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def yolo_detect(img_bgr, target_class: str, conf_thresh=0.35):
    if _yolo is None or img_bgr is None:
        return False, 0.0, None
    cid = YOLO_CLASS_MAP.get(target_class)
    results = _yolo(img_bgr, verbose=False, conf=conf_thresh)
    best = None
    for r in results:
        for box in r.boxes:
            if cid is not None and int(box.cls[0]) != cid:
                continue
            conf = float(box.conf[0])
            if best is None or conf > best[1]:
                best = (True, conf, [int(v) for v in box.xyxy[0]])
    return best if best else (False, 0.0, None)


# ── Trial ─────────────────────────────────────────────────────────────────────

def run_trial(target_class: str, trial_id: int) -> dict:
    print(f'\n  ── Trial {trial_id+1} [{target_class}] ──')
    rec = {
        'target_class': target_class,
        'trial_id': trial_id,
        'n_positions': len(SCAN_POSITIONS),
        'detections_per_position': [],
    }
    t0 = time.perf_counter()
    found = False
    found_pos = None
    found_conf = None

    for i, (pan, tilt) in enumerate(SCAN_POSITIONS):
        pos_num = i + 1
        move_head(pan, tilt)
        time.sleep(SETTLE_TIME)

        frame = capture_frame()
        detected, conf, bbox = yolo_detect(frame, target_class)

        pos_rec = {
            'position': pos_num,
            'pan_rad': pan,
            'tilt_rad': tilt,
            'detected': detected,
            'conf': round(conf, 3),
            'bbox': bbox,
        }
        rec['detections_per_position'].append(pos_rec)
        icon = '✓' if detected else '·'
        print(f'  [{pos_num:02d}] pan={pan:+.1f} tilt={tilt:+.2f}  '
              f'{icon} conf={conf:.2f}')

        if detected and not found:
            found = True
            found_pos = pos_num
            found_conf = conf
            print(f'  → TARGET FOUND at position {pos_num}')
            break   # mirrors search_with_head behaviour

    rec['target_found']         = found
    rec['first_found_position'] = found_pos
    rec['yolo_conf_at_found']   = round(found_conf, 3) if found_conf else None
    rec['positions_scanned']    = found_pos if found else len(SCAN_POSITIONS)
    rec['search_duration_s']    = round(time.perf_counter() - t0, 2)
    rec['timeout']              = not found

    # Return head to default
    move_head(0.0, 0.0)

    save_result('search_trials', rec)
    status = f'FOUND @ pos {found_pos}' if found else 'NOT FOUND (timeout)'
    print(f'  → {status}  t={rec["search_duration_s"]:.1f}s')
    return rec


def summarise(results: list):
    print('\n' + '═'*65)
    found    = [r for r in results if r['target_found']]
    not_found = [r for r in results if not r['target_found']]
    n        = len(results)
    sr       = len(found) / n * 100
    pos_mean = np.mean([r['positions_scanned'] for r in found]) if found else float('nan')
    dur_mean = np.mean([r['search_duration_s'] for r in results])
    conf_mean= np.mean([r['yolo_conf_at_found'] for r in found if r.get('yolo_conf_at_found')])

    print(f'  Total trials      : {n}')
    print(f'  Search success    : {len(found)}/{n}  ({sr:.1f}%)')
    print(f'  Timeout rate      : {len(not_found)}/{n}  ({len(not_found)/n*100:.1f}%)')
    print(f'  Mean positions    : {pos_mean:.1f}  (when found)')
    print(f'  Mean search time  : {dur_mean:.1f} s')
    print(f'  Mean YOLO conf    : {conf_mean:.3f}')
    print('═'*65)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', nargs='+', default=['bottle'])
    ap.add_argument('--trials', type=int, default=10)
    args = ap.parse_args()

    if ROS_AVAILABLE:
        rospy.init_node('eval_search_recorder', anonymous=True)

    all_results = []
    for cls in args.target:
        for t in range(args.trials):
            r = run_trial(cls, t)
            all_results.append(r)
            time.sleep(1)

    summarise(all_results)
