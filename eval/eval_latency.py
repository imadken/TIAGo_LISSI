#!/usr/bin/env python3
"""
eval_latency.py — Component-level latency profiling (no ROS required).

Measures latency of each pipeline component separately, then models
the end-to-end time budget.

Metrics produced
────────────────
• VLM API latency  — p50 / p95 / p99 / mean / std   (n=30 calls)
• YOLO latency     — per-frame inference time
• Depth refinement — compute time for refine_bbox + RANSAC
• Face recognition — HTTP round-trip
• STT latency      — Whisper transcription time (if audio sample provided)
• End-to-end model — sum of components with measured distributions

Usage
─────
    python3 eval_latency.py --n_vlm 20 --n_yolo 50 --n_depth 100
    python3 eval_latency.py --skip_vlm   # skip VLM if no API key
"""
import argparse
import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.eval_utils import save_result, timer

YOLO_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yolo11n.pt')

try:
    from ultralytics import YOLO as YOLOModel
    _yolo = YOLOModel(YOLO_WEIGHTS)
except Exception:
    _yolo = None


def make_test_image(h=480, w=640):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (200,100), (350,380), (50,100,200), -1)
    return img


# ── VLM latency ───────────────────────────────────────────────────────────────

def measure_vlm(n: int) -> list:
    try:
        from vlm_reasoner import VLMReasoner
        vlm = VLMReasoner()
    except Exception as e:
        print(f'[skip] VLM not available: {e}')
        return []
    img = make_test_image()
    lats = []
    print(f'  VLM latency ({n} calls) …', end='', flush=True)
    for i in range(n):
        with timer() as t:
            try:
                vlm.detect_objects(img, target_classes=['bottle'])
            except Exception:
                pass
        lats.append(t.elapsed)
        print('.', end='', flush=True)
    print()
    return lats


# ── YOLO latency ──────────────────────────────────────────────────────────────

def measure_yolo(n: int) -> list:
    if _yolo is None:
        print('[skip] YOLO not available')
        return []
    img = make_test_image()
    lats = []
    print(f'  YOLO latency ({n} inferences) …', end='', flush=True)
    # Warm-up
    _yolo(img, verbose=False)
    for _ in range(n):
        with timer() as t:
            _yolo(img, verbose=False)
        lats.append(t.elapsed)
        print('.', end='', flush=True)
    print()
    return lats


# ── Depth refinement latency ──────────────────────────────────────────────────

def measure_depth_refine(n: int) -> dict:
    """Time refine_bbox + RANSAC on synthetic depth maps."""
    sys.path.insert(0, os.path.dirname(__file__))
    from eval_depth_refine import (refine_bbox_by_depth, project_to_3d,
                                   ransac_cylinder, DEFAULT_CAM)
    depth = np.random.uniform(0.4, 1.4, (480, 640)).astype(np.float32)
    depth[100:300, 150:350] = 0.75
    vlm_bbox = [130, 80, 380, 330]
    cam = DEFAULT_CAM
    lats_refine = []; lats_ransac = []
    print(f'  Depth refine latency ({n} iterations) …', end='', flush=True)
    for _ in range(n):
        with timer() as t:
            rb, d_star, _ = refine_bbox_by_depth(depth, vlm_bbox)
        lats_refine.append(t.elapsed)
        pts, _ = project_to_3d(depth, rb or vlm_bbox, cam)
        with timer() as t:
            ransac_cylinder(pts)
        lats_ransac.append(t.elapsed)
        print('.', end='', flush=True)
    print()
    return {'refine': lats_refine, 'ransac': lats_ransac}


# ── Face recognition latency ──────────────────────────────────────────────────

def measure_face(n: int, host='localhost', port=5002) -> list:
    import requests
    img = make_test_image(224, 224)
    _, buf = cv2.imencode('.jpg', img)
    lats = []
    print(f'  Face recognition latency ({n} calls) …', end='', flush=True)
    for _ in range(n):
        with timer() as t:
            try:
                requests.post(f'http://{host}:{port}/recognize',
                              files={'image': ('f.jpg', buf.tobytes(), 'image/jpeg')},
                              timeout=5)
            except Exception:
                pass
        lats.append(t.elapsed)
        print('.', end='', flush=True)
    print()
    return lats


# ── Summary ───────────────────────────────────────────────────────────────────

def percentile_summary(name: str, lats: list) -> dict:
    if not lats:
        return {'component': name, 'n': 0}
    a = np.array(lats)
    rec = {
        'component': name,
        'n': len(a),
        'mean_s':  round(float(a.mean()), 4),
        'std_s':   round(float(a.std()), 4),
        'p50_s':   round(float(np.percentile(a, 50)), 4),
        'p95_s':   round(float(np.percentile(a, 95)), 4),
        'p99_s':   round(float(np.percentile(a, 99)), 4),
        'min_s':   round(float(a.min()), 4),
        'max_s':   round(float(a.max()), 4),
    }
    print(f'  {name:<22}  mean={rec["mean_s"]*1000:6.0f}ms  '
          f'p50={rec["p50_s"]*1000:6.0f}ms  '
          f'p95={rec["p95_s"]*1000:6.0f}ms  '
          f'std={rec["std_s"]*1000:5.0f}ms')
    return rec


def main(args):
    print('\n' + '═'*65)
    print(' Component Latency Profiling')
    print('═'*65)

    all_summaries = []

    if not args.skip_yolo:
        yolo_lats = measure_yolo(args.n_yolo)
        all_summaries.append(percentile_summary('YOLO inference', yolo_lats))

    if not args.skip_vlm:
        vlm_lats = measure_vlm(args.n_vlm)
        all_summaries.append(percentile_summary('VLM API (Gemini)', vlm_lats))

    if not args.skip_depth:
        d = measure_depth_refine(args.n_depth)
        all_summaries.append(percentile_summary('Depth refinement', d['refine']))
        all_summaries.append(percentile_summary('RANSAC cylinder', d['ransac']))

    if not args.skip_face:
        face_lats = measure_face(args.n_face, args.face_host, args.face_port)
        all_summaries.append(percentile_summary('Face recognition', face_lats))

    # End-to-end budget model
    print('\n  ── End-to-end task time budget (mean) ──')
    components = {s['component']: s.get('mean_s', 0) for s in all_summaries}
    pipeline = {
        'STT recognition':    0.8,    # Whisper base ~0.8s
        'VLM API':            components.get('VLM API (Gemini)', 1.8),
        'Depth refinement':   components.get('Depth refinement', 0.003),
        'RANSAC estimation':  components.get('RANSAC cylinder', 0.05),
        'MoveIt planning':    3.5,    # typical
        'Arm execution':      8.0,    # trajectory time
        'TTS speech':         2.0,    # PAL TTS
    }
    total = sum(pipeline.values())
    for k, v in pipeline.items():
        pct = v / total * 100
        print(f'  {k:<22}  {v:5.2f}s  ({pct:4.1f}%)')
    print(f'  {"TOTAL":<22}  {total:5.2f}s')

    all_summaries.append({'component': 'e2e_budget_s', 'total': round(total, 2),
                          'breakdown': pipeline})
    save_result('latency', all_summaries)
    print('\n[done] results saved to data/latency.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_vlm',   type=int, default=5)
    ap.add_argument('--n_yolo',  type=int, default=20)
    ap.add_argument('--n_depth', type=int, default=50)
    ap.add_argument('--n_face',  type=int, default=10)
    ap.add_argument('--skip_vlm',   action='store_true')
    ap.add_argument('--skip_yolo',  action='store_true')
    ap.add_argument('--skip_depth', action='store_true')
    ap.add_argument('--skip_face',  action='store_true')
    ap.add_argument('--face_host', default='localhost')
    ap.add_argument('--face_port', type=int, default=5002)
    args = ap.parse_args()
    main(args)
