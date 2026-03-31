#!/usr/bin/env python3
"""
YOLO Detection Service
Runs on the HOST (Python 3.10) and serves detections over HTTP on port 5001.
The Docker agent (Python 3.6) calls this instead of running YOLOv4-tiny locally.

Usage:
    python3 -m tiago_lissi.services.yolo_service                          # YOLOE-26s (open-vocab, recommended)
    python3 -m tiago_lissi.services.yolo_service --model yoloe-26s-seg    # same, explicit
    python3 -m tiago_lissi.services.yolo_service --model yoloe-26m-seg    # larger, more accurate
    python3 -m tiago_lissi.services.yolo_service --model yoloe-26s-seg-pf # prompt-free, 4585 classes, no CLIP
    python3 -m tiago_lissi.services.yolo_service --model yolo11n          # fixed 80 COCO classes, fastest

API:
    POST /detect
        Content-Type: image/jpeg          — body: raw JPEG bytes
        X-Classes: bottle,person,cup      — optional: text prompt classes (YOLOE only)
        returns: {"detections": [...], "model": "...", "inference_ms": N, "segmentation": bool}

        detection format:
            {"class_name": str, "confidence": float, "bbox": [x,y,w,h]}
            + "mask_bbox": [x,y,w,h]  (tighter mask bounding box, if segmentation model)
            + "centroid": [cx, cy]    (mask centroid in pixels, if segmentation model)

    GET  /health
        returns: {"status": "ok", "model": "...", "open_vocab": bool, "segmentation": bool}
"""

import argparse
import json
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
from ultralytics import YOLO

PORT = 5001

_model = None
_model_name = None
_is_open_vocab = False   # supports set_classes() / text prompts
_is_seg = False          # returns segmentation masks
_default_classes = None  # current active classes for open-vocab model
_model_lock = threading.Lock()


def _load_model(name):
    global _model, _model_name, _is_open_vocab, _is_seg, _default_classes
    print("[YOLO Service] Loading {}...".format(name))
    _model = YOLO("{}.pt".format(name))
    _model_name = name
    _is_open_vocab = 'yoloe' in name and 'pf' not in name
    _is_seg = 'seg' in name

    # Default classes for open-vocab models
    if _is_open_vocab:
        _default_classes = ['bottle', 'person', 'cup', 'chair', 'table',
                             'bowl', 'cell phone', 'backpack', 'book']
        _model.set_classes(_default_classes)
        print("[YOLO Service] Open-vocab mode, default classes: {}".format(
            _default_classes))
    else:
        print("[YOLO Service] Fixed-class mode ({} classes)".format(
            len(_model.names)))

    # Warm up
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    _model(dummy, verbose=False)
    print("[YOLO Service] {} ready on port {} | seg={} open_vocab={}".format(
        name, PORT, _is_seg, _is_open_vocab))


def _run_inference(jpeg_bytes, classes=None):
    """Run inference. Optionally override classes for open-vocab model."""
    global _default_classes

    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return []

    with _model_lock:
        # Update classes if requested and model supports it
        if _is_open_vocab and classes and classes != _default_classes:
            _model.set_classes(classes)
            _default_classes = classes

        results = _model(frame, verbose=False)[0]

    detections = []
    boxes = results.boxes
    masks = results.masks  # None for non-seg models

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = results.names[cls_id]

        det = {
            'class_name': class_name,
            'confidence': round(conf, 4),
            'bbox': [x, y, w, h]
        }

        # Add mask centroid if segmentation model
        if masks is not None and i < len(masks.xy):
            pts = masks.xy[i]  # polygon points
            if len(pts) > 0:
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                det['centroid'] = [cx, cy]
                # Tight mask bbox
                mx1, my1 = int(pts[:, 0].min()), int(pts[:, 1].min())
                mx2, my2 = int(pts[:, 0].max()), int(pts[:, 1].max())
                det['mask_bbox'] = [mx1, my1, mx2 - mx1, my2 - my1]

        detections.append(det)

    return detections


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == '/health':
            body = json.dumps({
                'status': 'ok',
                'model': _model_name,
                'open_vocab': _is_open_vocab,
                'segmentation': _is_seg,
                'active_classes': _default_classes
            }).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != '/detect':
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            self.send_response(400)
            self.end_headers()
            return

        jpeg_bytes = self.rfile.read(length)

        # Optional text-prompt classes from header: "X-Classes: bottle,person,cup"
        classes_header = self.headers.get('X-Classes', '')
        classes = [c.strip() for c in classes_header.split(',') if c.strip()] \
            if classes_header else None

        t0 = time.time()
        try:
            detections = _run_inference(jpeg_bytes, classes=classes)
        except Exception as e:
            detections = []
            print("[YOLO Service] Inference error: {}".format(e))
        elapsed_ms = int((time.time() - t0) * 1000)

        body = json.dumps({
            'detections': detections,
            'model': _model_name,
            'inference_ms': elapsed_ms,
            'segmentation': _is_seg
        }).encode()

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yoloe-26s-seg',
                        help='Model: yoloe-26s-seg (default), yoloe-26m-seg, '
                             'yoloe-26s-seg-pf (prompt-free), yolo11n')
    parser.add_argument('--port', type=int, default=PORT)
    args = parser.parse_args()

    _load_model(args.model)

    server = _ThreadedHTTPServer(('0.0.0.0', args.port), _Handler)
    print("[YOLO Service] Listening on port {}".format(args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[YOLO Service] Shutting down.")
        server.shutdown()


if __name__ == '__main__':
    main()
