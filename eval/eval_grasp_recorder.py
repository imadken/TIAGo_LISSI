#!/usr/bin/env python3
"""
eval_grasp_recorder.py — Live grasp trial recorder (requires ROS).

Runs INDEPENDENTLY of EmbodiedAgent.  For each trial it:
  1. Calls reach_object_v5 as a subprocess (same as grab_bottle skill)
  2. Optionally uses YOLO to verify object detection before/after
  3. Records outcome, timing, and pose estimates to data/grasp_trials.json

Requirements
────────────
  HOST:   python3 yolo_service.py   (listens on :5001)
  DOCKER: source ROS, then run this script with -t flag for input() prompts

Usage
─────
    # Inside Docker (needs -t for interactive input prompts):
    docker exec -it tiago_ros bash -c \
      "source /opt/ros/melodic/setup.bash && \
       source /workspace/pal_ws/devel/setup.bash && \
       cd /workspace && python3 eval/eval_grasp_recorder.py --object bottle --trials 15"

    # Dry run (no robot motion)
    python3 eval_grasp_recorder.py --object bottle --trials 5 --dry_run

    # Override YOLO service URL (default: http://<docker-gateway>:5001/detect)
    YOLO_SERVICE_URL=http://192.168.1.10:5001/detect python3 eval_grasp_recorder.py ...

Metrics collected
─────────────────
• grasp_success         bool
• detection_success     bool   (YOLO detected object before grasp)
• attempt_duration_s    float  (seconds from command to outcome)
• pre_grasp_conf        float  (YOLO confidence before grasp)
• post_grasp_detected   bool   (object still visible = grasp failed)
• error_type            str    (timeout / no_detection / moveit_fail / ok)
• object_class          str
• trial_id              int
"""
import argparse
import json
import os
import subprocess
import sys
import time
import cv2
import numpy as np

# Source ROS environment if not already loaded (allows running directly from Docker shell)
if 'ROS_DISTRO' not in os.environ:
    for setup in ['/opt/ros/melodic/setup.bash', '/workspace/pal_ws/devel/setup.bash']:
        if os.path.exists(setup):
            out = subprocess.check_output(
                ['bash', '-c', 'source {} && env'.format(setup)],
                stderr=subprocess.DEVNULL)
            for line in out.decode().splitlines():
                if '=' in line:
                    k, _, v = line.partition('=')
                    os.environ.setdefault(k, v)
    sys.path = [p for p in os.environ.get('PYTHONPATH', '').split(':') if p] + sys.path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.eval_utils import save_result, timer

ROS_AVAILABLE = False
try:
    import rospy
    from sensor_msgs.msg import Image
    import cv_bridge
    ROS_AVAILABLE = True
except ImportError:
    print('[warn] rospy not available — live camera feed disabled')

REACH_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'tiago_lissi', 'manipulation', 'reach_object_v5_torso_descent_working.py')

# YOLO service (python3 yolo_service.py) — reachable via localhost since
# the Docker container runs with --network=host.
YOLO_SERVICE_URL = os.environ.get('YOLO_SERVICE_URL', 'http://localhost:5001/detect')


def capture_frame():
    """Grab a single frame from the robot camera (ROS) or webcam fallback."""
    if ROS_AVAILABLE:
        try:
            import rospy
            from sensor_msgs.msg import Image as ROSImage
            from cv_bridge import CvBridge
            bridge = CvBridge()
            msg = rospy.wait_for_message('/xtion/rgb/image_raw', ROSImage,
                                         timeout=5.0)
            return bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            print(f'[warn] ROS frame failed: {e}')
    # webcam fallback for testing
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def yolo_detect(img_bgr, object_class: str, conf_thresh=0.25):
    """
    Send frame to the host YOLO service and return (detected, conf, bbox).
    Requires yolo_service.py running on the host (python3 yolo_service.py).
    """
    if img_bgr is None:
        return False, 0.0, None
    try:
        import urllib.request
        _, buf = cv2.imencode('.jpg', img_bgr)
        req = urllib.request.Request(
            YOLO_SERVICE_URL,
            data=buf.tobytes(),
            headers={
                'Content-Type': 'image/jpeg',
                'X-Classes': object_class,
            },
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        # find best detection matching the requested class
        best = None
        for det in data.get('detections', []):
            # match base noun (e.g. "bottle" matches "small bottle")
            if object_class.lower() not in det['class_name'].lower() and \
               det['class_name'].lower() not in object_class.lower():
                continue
            if det['confidence'] < conf_thresh:
                continue
            if best is None or det['confidence'] > best[1]:
                # bbox from service is [x, y, w, h] format
                best = (True, det['confidence'], det['bbox'])
        return best if best else (False, 0.0, None)
    except Exception as e:
        print(f'[warn] YOLO service call failed ({YOLO_SERVICE_URL}): {e}')
        return False, 0.0, None


def run_reach_object(object_class: str, dry_run=False, timeout=180) -> dict:
    """
    Call reach_object_v5 as subprocess and parse result.
    Returns dict with: success, duration_s, error_type, stdout.
    """
    if dry_run:
        time.sleep(0.5)
        return {'success': True, 'duration_s': 0.5, 'error_type': 'dry_run',
                'stdout': 'DRY RUN'}
    cmd = [
        sys.executable, REACH_SCRIPT,
        '--target', object_class,
        '--description', f'evaluate {object_class}',
    ]
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout)
        dur = time.perf_counter() - t0
        stdout = result.stdout + result.stderr
        success = result.returncode == 0 and 'success' in stdout.lower()
        if result.returncode != 0:
            error_type = 'crash'
        elif 'moveit' in stdout.lower() and 'fail' in stdout.lower():
            error_type = 'moveit_fail'
        elif 'no detection' in stdout.lower() or 'not found' in stdout.lower():
            error_type = 'no_detection'
        else:
            error_type = 'ok' if success else 'unknown'
        return {'success': success, 'duration_s': dur,
                'error_type': error_type, 'stdout': stdout[-500:]}
    except subprocess.TimeoutExpired:
        return {'success': False, 'duration_s': timeout,
                'error_type': 'timeout', 'stdout': ''}


def run_trial(object_class: str, trial_id: int, dry_run=False) -> dict:
    print(f'\n  ── Trial {trial_id+1} [{object_class}] ──')
    rec = {'object_class': object_class, 'trial_id': trial_id}

    # Pre-grasp: YOLO detection check
    frame_pre = capture_frame()
    detected, conf_pre, bbox_pre = yolo_detect(frame_pre, object_class)
    rec['pre_detection'] = detected
    rec['pre_conf']      = round(conf_pre, 3)
    rec['pre_bbox']      = bbox_pre
    print(f'  Pre-grasp YOLO: detected={detected}  conf={conf_pre:.2f}')

    if not detected and not dry_run:
        input('  Object not detected.  Position it and press ENTER …')
        frame_pre = capture_frame()
        detected, conf_pre, bbox_pre = yolo_detect(frame_pre, object_class)
        rec['pre_detection'] = detected
        rec['pre_conf'] = round(conf_pre, 3)

    # Execute grasp
    print(f'  Executing reach_object_v5 …')
    res = run_reach_object(object_class, dry_run=dry_run)
    rec.update(res)

    # Post-grasp: object still visible? (if yes → grasp likely failed)
    time.sleep(1.5)
    frame_post = capture_frame()
    det_post, conf_post, _ = yolo_detect(frame_post, object_class)
    rec['post_detection']  = det_post
    rec['post_conf']       = round(conf_post, 3)
    # If object no longer visible and grasp succeeded → good
    rec['grasp_success']   = res['success'] and not det_post

    status = '✓ SUCCESS' if rec['grasp_success'] else '✗ FAIL'
    print(f'  {status}  t={res["duration_s"]:.1f}s  err={res["error_type"]}')

    # Manual confirmation override
    if not dry_run:
        ans = input('  Override result? [y=success/n=fail/Enter=keep]: ').strip().lower()
        if ans == 'y':
            rec['grasp_success'] = True; rec['manual_override'] = True
        elif ans == 'n':
            rec['grasp_success'] = False; rec['manual_override'] = True

    save_result('grasp_trials', rec)
    return rec


def summarise(results: list, object_classes: list):
    print('\n' + '═'*60)
    print(f'{"Class":<12} {"Trials":>6} {"Success%":>9} {"Avg t(s)":>9} {"Det%":>7}')
    print('─'*60)
    for cls in object_classes:
        rows = [r for r in results if r['object_class'] == cls]
        if not rows: continue
        n   = len(rows)
        sr  = np.mean([r.get('grasp_success', False) for r in rows]) * 100
        dur = np.mean([r.get('duration_s', 0) for r in rows])
        det = np.mean([r.get('pre_detection', False) for r in rows]) * 100
        print(f'{cls:<12} {n:>6}  {sr:>8.1f}%  {dur:>8.1f}   {det:>6.1f}%')
    all_sr = np.mean([r.get('grasp_success', False) for r in results]) * 100
    print('─'*60)
    print(f'{"OVERALL":<12} {len(results):>6}  {all_sr:>8.1f}%')
    print('═'*60)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--object', nargs='+', default=['bottle'],
                    help='object classes to evaluate')
    ap.add_argument('--trials', type=int, default=15,
                    help='trials per class')
    ap.add_argument('--dry_run', action='store_true',
                    help='skip actual robot motion')
    args = ap.parse_args()

    print(f'[eval] YOLO service → {YOLO_SERVICE_URL}')

    if ROS_AVAILABLE:
        rospy.init_node('eval_grasp_recorder', anonymous=True)

    all_results = []
    for cls in args.object:
        print(f'\n{"═"*60}')
        print(f' Evaluating class: {cls}  ({args.trials} trials)')
        print(f'{"═"*60}')
        for t in range(args.trials):
            rec = run_trial(cls, t, dry_run=args.dry_run)
            all_results.append(rec)
            time.sleep(2)

    summarise(all_results, args.object)
