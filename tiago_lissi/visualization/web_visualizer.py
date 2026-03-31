#!/usr/bin/env python3
"""
Web-based Live Visualizer for TIAGo Embodied Agent
Serves MJPEG stream + status dashboard on http://localhost:8080

Run inside Docker alongside the agent:
  docker exec -it tiago_ros bash -c "
    export ROS_MASTER_URI=http://${ROBOT_IP:-10.68.0.1}:11311 &&
    export ROS_IP=${HOST_IP:-$(hostname -I | awk '{print $1}')} &&
    source /opt/ros/melodic/setup.bash &&
    source /workspace/pal_ws/devel/setup.bash &&
    cd /workspace &&
    python3 web_visualizer.py"

Then open http://localhost:8080 in your browser on the host.
"""

import json
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

PORT = 8080

# ---------------------------------------------------------------------------
# Shared state (written by ROS callbacks, read by HTTP server)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_latest_jpeg = None   # bytes: latest JPEG-encoded frame with overlay
_latest_status = {
    'skill': 'idle',
    'gripper': 'unknown',
    'location': 'unknown',
    'detected': [],
    'history': []
}


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------

def _encode_and_store(frame, s):
    """Draw overlay and store as JPEG. Call from any image callback."""
    global _latest_jpeg
    frame = _draw_overlay(frame, s)
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        with _lock:
            _latest_jpeg = bytes(buf)


def _raw_image_cb(msg):
    """Continuous feed from raw camera — always produces frames."""
    try:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = arr.reshape(msg.height, msg.width, -1)
        if msg.encoding == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame = frame.copy()
        with _lock:
            s = dict(_latest_status)
        _encode_and_store(frame, s)
    except Exception as e:
        rospy.logwarn("[WebVis] Raw image error: {}".format(e))


def _annotated_image_cb(msg):
    """Annotated frames from agent (with YOLO boxes) — override raw when available."""
    try:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = arr.reshape(msg.height, msg.width, 3).copy()
        with _lock:
            s = dict(_latest_status)
        _encode_and_store(frame, s)
    except Exception as e:
        rospy.logwarn("[WebVis] Annotated image error: {}".format(e))


def _status_cb(msg):
    global _latest_status
    try:
        data = json.loads(msg.data)
        with _lock:
            _latest_status = data
    except Exception as e:
        rospy.logwarn("[WebVis] Status error: {}".format(e))


def _draw_overlay(frame, s):
    """Draw status panel below the camera image."""
    panel_h = 130
    h, w = frame.shape[:2]
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)

    skill = s.get('skill', 'idle')
    skill_color = (0, 255, 0) if skill != 'idle' else (150, 150, 150)
    cv2.putText(panel, "SKILL: {}".format(skill.upper()),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, skill_color, 2)

    gripper = s.get('gripper', 'unknown')
    gripper_color = (0, 200, 255) if 'holding' in gripper else (200, 200, 200)
    cv2.putText(panel, "GRIPPER: {}".format(gripper),
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, gripper_color, 1)

    detected = s.get('detected', [])
    det_str = "DETECTED: {}".format(', '.join(detected) if detected else 'none')
    cv2.putText(panel, det_str, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 1)

    history = s.get('history', [])
    hist_str = "HISTORY: {}".format('  >  '.join(history[-5:]) if history else 'none')
    cv2.putText(panel, hist_str, (10, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

    cv2.line(panel, (0, 0), (w, 0), (80, 80, 80), 2)
    return np.vstack([frame, panel])


def _make_placeholder_jpeg():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Waiting for agent...",
                (110, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
    _, buf = cv2.imencode('.jpg', img)
    return bytes(buf)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

HTML_PAGE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>TIAGo Agent Live</title>
  <style>
    body {{ background:#111; color:#eee; font-family:monospace;
            display:flex; flex-direction:column; align-items:center;
            margin:0; padding:16px; }}
    h2   {{ color:#4af; margin:8px 0; }}
    img  {{ border:2px solid #333; border-radius:4px; max-width:100%; }}
    #status {{ margin-top:12px; background:#1e1e1e; border:1px solid #333;
               border-radius:6px; padding:12px 20px; width:640px; }}
    .row {{ display:flex; justify-content:space-between; margin:4px 0; }}
    .label {{ color:#888; }}
    .active {{ color:#0f0; font-weight:bold; }}
    .idle   {{ color:#888; }}
    .holding {{ color:#0cf; font-weight:bold; }}
  </style>
</head>
<body>
  <h2>TIAGo Embodied Agent</h2>
  <img src="/stream" />
  <div id="status">
    <div class="row"><span class="label">SKILL</span>   <span id="skill" class="idle">idle</span></div>
    <div class="row"><span class="label">GRIPPER</span> <span id="gripper">unknown</span></div>
    <div class="row"><span class="label">DETECTED</span><span id="detected">none</span></div>
    <div class="row"><span class="label">HISTORY</span> <span id="history">none</span></div>
  </div>
  <script>
    function poll() {{
      fetch('/status').then(r => r.json()).then(s => {{
        const skill = s.skill || 'idle';
        const el = document.getElementById('skill');
        el.textContent = skill.toUpperCase();
        el.className = skill !== 'idle' ? 'active' : 'idle';

        const gr = s.gripper || 'unknown';
        const gel = document.getElementById('gripper');
        gel.textContent = gr;
        gel.className = gr.includes('holding') ? 'holding' : '';

        document.getElementById('detected').textContent =
          (s.detected && s.detected.length) ? s.detected.join(', ') : 'none';

        const hist = s.history || [];
        document.getElementById('history').textContent =
          hist.length ? hist.slice(-5).join(' > ') : 'none';
      }}).catch(() => {{}});
      setTimeout(poll, 500);
    }}
    poll();
  </script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request logs

    def do_GET(self):
        if self.path == '/':
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == '/status':
            with _lock:
                data = json.dumps(_latest_status).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with _lock:
                        jpeg = _latest_jpeg
                    if jpeg is None:
                        jpeg = _make_placeholder_jpeg()

                    frame_bytes = (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n'
                        + jpeg + b'\r\n'
                    )
                    self.wfile.write(frame_bytes)
                    self.wfile.flush()
                    time.sleep(1.0 / 15)   # 15 fps cap
            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self.send_response(404)
            self.end_headers()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    rospy.init_node('web_visualizer', anonymous=True)
    # Raw camera: continuous feed so the stream never stalls between agent commands
    rospy.Subscriber('/xtion/rgb/image_raw', Image, _raw_image_cb,
                     queue_size=1, buff_size=2**24)
    # Annotated feed from agent: shows YOLO boxes when agent is active
    rospy.Subscriber('/agent/camera_annotated', Image, _annotated_image_cb,
                     queue_size=1, buff_size=2**24)
    rospy.Subscriber('/agent/status', String, _status_cb)

    server = _ThreadedHTTPServer(('0.0.0.0', PORT), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    rospy.loginfo("[WebVis] Dashboard at http://localhost:{}".format(PORT))
    rospy.loginfo("[WebVis] Subscribed to /agent/camera_annotated + /agent/status")

    rospy.spin()

    server.shutdown()


if __name__ == '__main__':
    main()
