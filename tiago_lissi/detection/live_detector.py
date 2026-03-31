#!/usr/bin/env python3
"""
Live YOLO Detector — smooth streaming, async detection
Camera thread updates stream at full camera rate (~15 Hz).
Detection thread calls YOLO service independently at whatever speed it can.
Boxes from the last detection are overlaid on every new camera frame.

Run inside Docker:
  docker exec -it tiago_ros bash -c "
    export ROS_MASTER_URI=http://${ROBOT_IP:-10.68.0.1}:11311 &&
    export ROS_IP=${HOST_IP:-$(hostname -I | awk '{print $1}')} &&
    source /opt/ros/melodic/setup.bash &&
    source /workspace/pal_ws/devel/setup.bash &&
    cd /workspace &&
    python3 -m tiago_lissi.detection.live_detector"

Then open http://localhost:8081
"""

import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import requests
import rospy
from sensor_msgs.msg import Image

YOLO_SERVICE_URL = 'http://localhost:5001'
STREAM_PORT = 8081

# ---------------------------------------------------------------------------
# Shared state — written by camera thread, read by HTTP server
# ---------------------------------------------------------------------------
_stream_lock  = threading.Lock()
_latest_jpeg  = None          # current annotated frame for the stream

# Written by detection thread, read by camera thread
_det_lock     = threading.Lock()
_last_dets    = []            # last known detections
_det_stats    = {'model': 'YOLO', 'inference_ms': 0, 'det_fps': 0.0}

# Colour palette
_colours = {}
_rng = np.random.default_rng(42)

def _colour(cls):
    if cls not in _colours:
        r, g, b = _rng.integers(80, 255, size=3).tolist()
        _colours[cls] = (int(b), int(g), int(r))
    return _colours[cls]


def _annotate(frame, dets, stats, stream_fps):
    out = frame.copy()
    for d in dets:
        x, y, w, h = d['bbox']
        c = _colour(d['class_name'])
        label = "{} {:.0f}%".format(d['class_name'], d['confidence'] * 100)
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
        lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x, y - lsz[1] - 8), (x + lsz[0] + 4, y), c, -1)
        cv2.putText(out, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    hud = "{}  stream {:.0f}fps  det {}ms  {} obj".format(
        stats['model'], stream_fps, stats['inference_ms'], len(dets))
    cv2.rectangle(out, (0, 0), (len(hud) * 10 + 10, 28), (0, 0, 0), -1)
    cv2.putText(out, hud, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Camera thread — runs at full camera rate, overlays last known boxes
# ---------------------------------------------------------------------------

class _CameraThread:
    def __init__(self):
        self._latest_raw = None
        self._raw_lock   = threading.Lock()
        self._fps_times  = []

        rospy.Subscriber('/xtion/rgb/image_raw', Image, self._cam_cb,
                         queue_size=1, buff_size=2 ** 24)
        rospy.loginfo("[LiveDet] Waiting for camera...")
        while self._latest_raw is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("[LiveDet] Camera ready")

        threading.Thread(target=self._publish_loop, daemon=True).start()

    def _cam_cb(self, msg):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = arr.reshape(msg.height, msg.width, -1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) \
                if msg.encoding == 'rgb8' else frame.copy()
            with self._raw_lock:
                self._latest_raw = frame
        except Exception as e:
            rospy.logwarn_throttle(5, "[LiveDet] Cam error: {}".format(e))

    def _publish_loop(self):
        global _latest_jpeg
        while not rospy.is_shutdown():
            with self._raw_lock:
                frame = self._latest_raw.copy() if self._latest_raw is not None else None
            if frame is None:
                time.sleep(0.05)
                continue

            with _det_lock:
                dets  = list(_last_dets)
                stats = dict(_det_stats)

            # Stream FPS
            now = time.time()
            self._fps_times = [t for t in self._fps_times if now - t < 2.0]
            self._fps_times.append(now)
            stream_fps = len(self._fps_times) / 2.0

            annotated = _annotate(frame, dets, stats, stream_fps)
            ok, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with _stream_lock:
                    _latest_jpeg = bytes(buf)

            time.sleep(1.0 / 15)   # cap stream at 15 fps


# ---------------------------------------------------------------------------
# Detection thread — calls YOLO service as fast as it can independently
# ---------------------------------------------------------------------------

class _DetectionThread:
    def __init__(self, camera):
        self._camera = camera
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        global _last_dets, _det_stats
        det_fps_times = []

        while not rospy.is_shutdown():
            with self._camera._raw_lock:
                frame = self._camera._latest_raw.copy() \
                    if self._camera._latest_raw is not None else None
            if frame is None:
                time.sleep(0.1)
                continue

            try:
                ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                t0 = time.time()
                resp = requests.post(
                    YOLO_SERVICE_URL + '/detect',
                    data=bytes(buf),
                    headers={'Content-Type': 'image/jpeg'},
                    timeout=3.0)
                data = resp.json()
                inf_ms = int((time.time() - t0) * 1000)

                now = time.time()
                det_fps_times = [t for t in det_fps_times if now - t < 4.0]
                det_fps_times.append(now)

                with _det_lock:
                    _last_dets = data.get('detections', [])
                    _det_stats = {
                        'model':        data.get('model', 'YOLO'),
                        'inference_ms': data.get('inference_ms', inf_ms),
                        'det_fps':      len(det_fps_times) / 4.0
                    }
            except Exception as e:
                rospy.logwarn_throttle(5, "[LiveDet] Service error: {}".format(e))
                time.sleep(0.5)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

HTML_PAGE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>TIAGo Live Detector</title>
  <style>
    body {{ background:#111; color:#eee; font-family:monospace;
            display:flex; flex-direction:column; align-items:center;
            margin:0; padding:16px; }}
    h2 {{ color:#4af; margin:8px 0; }}
    img {{ border:2px solid #333; border-radius:4px; max-width:100%; }}
  </style>
</head>
<body>
  <h2>TIAGo — Live YOLO Detector</h2>
  <img src="/stream" />
</body>
</html>
"""


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            placeholder = None
            try:
                while True:
                    with _stream_lock:
                        jpeg = _latest_jpeg
                    if jpeg is None:
                        if placeholder is None:
                            img = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(img, "Waiting...", (200, 240),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                        (100, 100, 100), 2)
                            _, buf = cv2.imencode('.jpg', img)
                            placeholder = bytes(buf)
                        jpeg = placeholder
                    self.wfile.write(
                        b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                        + jpeg + b'\r\n')
                    self.wfile.flush()
                    time.sleep(1.0 / 15)
            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self.send_response(404)
            self.end_headers()


def main():
    rospy.init_node('live_detector', anonymous=True)

    try:
        r = requests.get(YOLO_SERVICE_URL + '/health', timeout=2.0)
        rospy.loginfo("[LiveDet] YOLO service: {}".format(r.json().get('model')))
    except Exception:
        rospy.logwarn("[LiveDet] YOLO service not reachable — start yolo_service.py first")

    cam = _CameraThread()
    _DetectionThread(cam)

    server = _ThreadedHTTPServer(('0.0.0.0', STREAM_PORT), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    rospy.loginfo("[LiveDet] Stream at http://localhost:{}".format(STREAM_PORT))
    rospy.spin()
    server.shutdown()


if __name__ == '__main__':
    main()
