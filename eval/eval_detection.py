#!/usr/bin/env python3
"""
eval_detection.py — Offline detection quality evaluation.

Compares YOLO (ground-truth proxy) with VLM bounding boxes on a set
of saved RGB images.  No ROS required.

Metrics produced
────────────────
• Detection rate    — % of images where target found by YOLO / VLM
• Confidence        — mean YOLO confidence score
• Bbox IoU          — overlap between YOLO bbox and VLM bbox
• Bbox area ratio   — VLM bbox area / YOLO bbox area (>1 = VLM is looser)
• VLM latency       — per-call API response time (seconds)
• Per-class breakdown

Usage
─────
    # 1. Drop JPEGs into eval/data/images/{class}/img001.jpg …
    # 2. Optionally pre-save VLM responses in eval/data/vlm_cache.json
    python3 eval_detection.py --image_dir data/images --classes bottle cup remote
    python3 eval_detection.py --demo          # generates synthetic test images
"""
import argparse
import json
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.eval_utils import save_result, bbox_iou, timer

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print('[warn] ultralytics not installed — YOLO metrics skipped')

# COCO classes that map to our eval targets
YOLO_CLASS_MAP = {
    'bottle': 39, 'cup': 41, 'remote': 65, 'phone': 67,
    'bowl': 45, 'book': 73, 'person': 0, 'chair': 56,
}

YOLO_WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yolo11n.pt')


def run_yolo(model, img_bgr, target_class: str, conf_thresh=0.25):
    """Return best detection [x1,y1,x2,y2,conf] or None."""
    class_id = YOLO_CLASS_MAP.get(target_class)
    results = model(img_bgr, verbose=False, conf=conf_thresh)
    best = None
    for r in results:
        for box in r.boxes:
            cid = int(box.cls[0])
            conf = float(box.conf[0])
            if class_id is not None and cid != class_id:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            if best is None or conf > best[4]:
                best = [x1, y1, x2, y2, conf]
    return best


def run_vlm(img_bgr, target_class: str, vlm=None):
    """Return VLM bbox [x1,y1,x2,y2] and latency_s, or None."""
    if vlm is None:
        return None, 0.0
    h, w = img_bgr.shape[:2]
    with timer() as t:
        dets = vlm.detect_objects(img_bgr, target_classes=[target_class])
    lat = t.elapsed
    if not dets:
        return None, lat
    d = dets[0]
    # VLM returns normalised [0,1000] coords
    x1 = int(d.get('x1', d.get('x', 0)) / 1000 * w)
    y1 = int(d.get('y1', d.get('y', 0)) / 1000 * h)
    x2 = int(d.get('x2', d.get('x', 0) + d.get('w', 0)) / 1000 * w)
    y2 = int(d.get('y2', d.get('y', 0) + d.get('h', 0)) / 1000 * h)
    return [x1, y1, x2, y2], lat


def make_synthetic_images(out_dir: str):
    """Create simple synthetic test images for demo mode."""
    os.makedirs(out_dir, exist_ok=True)
    classes = ['bottle', 'cup', 'remote']
    paths = []
    rng = np.random.default_rng(42)
    for cls in classes:
        cls_dir = os.path.join(out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(5):
            img = np.ones((480, 640, 3), dtype=np.uint8) * 200
            # Draw a coloured rectangle to simulate object
            x, y = rng.integers(100, 400), rng.integers(80, 300)
            w, h = rng.integers(60, 180), rng.integers(100, 250)
            col = {'bottle': (50,100,200), 'cup': (80,180,80),
                   'remote': (200,80,50)}[cls]
            cv2.rectangle(img, (x, y), (x+w, y+h), col, -1)
            # Add noise
            img = cv2.add(img, rng.integers(0, 20, img.shape, dtype=np.uint8))
            path = os.path.join(cls_dir, f'{cls}_{i:03d}.jpg')
            cv2.imwrite(path, img)
            paths.append((cls, path, [x, y, x+w, y+h]))  # GT bbox stored too
    return paths


def evaluate(image_dir: str, classes: list, use_vlm=False, conf_thresh=0.25):
    model = YOLO(YOLO_WEIGHTS) if YOLO_AVAILABLE else None
    vlm = None
    if use_vlm:
        from vlm_reasoner import VLMReasoner
        vlm = VLMReasoner()

    results = []
    for cls in classes:
        cls_dir = os.path.join(image_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f'[skip] no directory: {cls_dir}')
            continue
        images = sorted([f for f in os.listdir(cls_dir)
                         if f.lower().endswith(('.jpg','.jpeg','.png'))])
        print(f'\n[{cls}] evaluating {len(images)} images …')
        for fname in images:
            path = os.path.join(cls_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue

            rec = {'class': cls, 'image': fname}

            # YOLO detection
            if model:
                with timer() as t:
                    yolo_det = run_yolo(model, img, cls, conf_thresh)
                rec['yolo_latency_s'] = t.elapsed
                rec['yolo_detected']  = yolo_det is not None
                rec['yolo_conf']      = yolo_det[4] if yolo_det else None
                rec['yolo_bbox']      = yolo_det[:4] if yolo_det else None

            # VLM detection
            if vlm:
                vlm_bbox, vlm_lat = run_vlm(img, cls, vlm)
                rec['vlm_latency_s'] = vlm_lat
                rec['vlm_detected']  = vlm_bbox is not None
                rec['vlm_bbox']      = vlm_bbox

                # IoU between YOLO and VLM
                if rec.get('yolo_bbox') and vlm_bbox:
                    rec['iou_yolo_vlm'] = bbox_iou(rec['yolo_bbox'], vlm_bbox)
                    yolo_area = ((rec['yolo_bbox'][2]-rec['yolo_bbox'][0]) *
                                 (rec['yolo_bbox'][3]-rec['yolo_bbox'][1]))
                    vlm_area  = ((vlm_bbox[2]-vlm_bbox[0]) *
                                 (vlm_bbox[3]-vlm_bbox[1]))
                    rec['vlm_area_ratio'] = vlm_area / yolo_area if yolo_area else None

            results.append(rec)
            status = '✓' if rec.get('yolo_detected') else '✗'
            print(f'  {status} {fname}  conf={rec.get("yolo_conf","—"):.2f}'
                  f'  lat={rec.get("yolo_latency_s",0)*1000:.0f}ms')

    save_result('detection', results)
    _print_summary(results, classes)


def _print_summary(results, classes):
    print('\n' + '─'*60)
    print(f'{"Class":<12} {"Det%":>6} {"Conf":>6} {"IoU":>6} {"VLMlat":>8}')
    print('─'*60)
    for cls in classes:
        rows = [r for r in results if r['class'] == cls]
        if not rows: continue
        det_rate = np.mean([r.get('yolo_detected', False) for r in rows]) * 100
        confs     = [r['yolo_conf'] for r in rows if r.get('yolo_conf')]
        ious      = [r.get('iou_yolo_vlm', np.nan) for r in rows]
        lats      = [r.get('vlm_latency_s', np.nan) for r in rows]
        print(f'{cls:<12} {det_rate:>5.1f}%'
              f'  {np.mean(confs) if confs else float("nan"):>5.3f}'
              f'  {np.nanmean(ious):>5.3f}'
              f'  {np.nanmean(lats):>6.2f}s')
    print('─'*60)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_dir', default='data/images')
    ap.add_argument('--classes', nargs='+',
                    default=['bottle', 'cup', 'remote', 'phone'])
    ap.add_argument('--vlm', action='store_true', help='also run VLM detection')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--demo', action='store_true',
                    help='generate synthetic images and run demo')
    args = ap.parse_args()

    if args.demo:
        print('[demo] generating synthetic images …')
        make_synthetic_images(args.image_dir)

    evaluate(args.image_dir, args.classes, use_vlm=args.vlm, conf_thresh=args.conf)
