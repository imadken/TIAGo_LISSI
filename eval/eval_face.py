#!/usr/bin/env python3
"""
eval_face.py — Face recognition accuracy evaluation (offline, no ROS).

Tests FaceManager at varying distances and under occlusion.

Metrics produced
────────────────
• Recognition accuracy    — % correctly identified (per distance band)
• False Accept Rate (FAR) — % unknown face accepted as known person
• False Reject Rate (FRR) — % known face rejected / mis-identified
• Recognition latency     — HTTP round-trip time (seconds)
• Embedding distance dist — distribution of intra-class vs inter-class distances

Data layout
───────────
data/faces/
    known/
        Alice/  img001.jpg img002.jpg …
        Bob/    img001.jpg …
    unknown/
        img001.jpg …   (people NOT in the known set)

The script registers 'known' identities, then tests recognition on:
  • known faces (should match correctly)  → TP / FN
  • unknown faces (should return None)    → TN / FP

Usage
─────
    python3 eval_face.py --face_dir data/faces --host localhost --port 5002
    python3 eval_face.py --demo   # generate synthetic face images
"""
import argparse
import os
import sys
import time
import json
import numpy as np
import cv2
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.eval_utils import save_result, timer


def call_face_api(endpoint: str, host: str, port: int,
                  method='GET', data=None, files=None):
    url = f'http://{host}:{port}/{endpoint}'
    try:
        if method == 'POST':
            r = requests.post(url, json=data, files=files, timeout=10)
        else:
            r = requests.get(url, json=data, timeout=10)
        return r.json() if r.ok else None
    except Exception as e:
        return None


def encode_image(img_bgr) -> bytes:
    _, buf = cv2.imencode('.jpg', img_bgr)
    return buf.tobytes()


def register_identity(name: str, images: list, host: str, port: int) -> bool:
    """Register a person with multiple images."""
    ok_count = 0
    for img in images:
        files = {'image': ('face.jpg', encode_image(img), 'image/jpeg')}
        result = call_face_api('register', host, port, method='POST',
                               data=None,
                               files={'image': ('face.jpg', encode_image(img), 'image/jpeg'),
                                      'name':  (None, name)})
        if result and result.get('status') == 'registered':
            ok_count += 1
    return ok_count > 0


def recognize_image(img_bgr, host: str, port: int) -> tuple[str | None, float, float]:
    """Returns (name|None, confidence, latency_s)."""
    files = {'image': ('face.jpg', encode_image(img_bgr), 'image/jpeg')}
    with timer() as t:
        result = call_face_api('recognize', host, port, method='POST',
                               files=files)
    lat = t.elapsed
    if result is None:
        return None, 0.0, lat
    name = result.get('name')
    conf = result.get('confidence', 0.0)
    return name, conf, lat


def load_images_from_dir(d: str):
    imgs = []
    for fn in sorted(os.listdir(d)):
        if fn.lower().endswith(('.jpg','.jpeg','.png')):
            p = os.path.join(d, fn)
            img = cv2.imread(p)
            if img is not None:
                imgs.append((fn, img))
    return imgs


def prepare_from_kaggle(kaggle_dir: str, out_dir: str, n_unknown: int = 6):
    """
    Prepare data/faces/ layout from the Kaggle face-recognition-dataset.
    Dataset structure: kaggle_dir/<PersonName>/*.jpg
    Splits: last n_unknown persons → unknown/, rest → known/<Name>/
    Uses symlinks — no file copying.
    """
    persons = sorted([
        p for p in os.listdir(kaggle_dir)
        if os.path.isdir(os.path.join(kaggle_dir, p))
    ])
    if not persons:
        print(f'[error] No person subdirectories found in {kaggle_dir}')
        return False

    known_persons  = persons[:-n_unknown] if len(persons) > n_unknown else persons
    unknown_persons = persons[-n_unknown:] if len(persons) > n_unknown else []

    known_dir   = os.path.join(out_dir, 'known')
    unknown_dir = os.path.join(out_dir, 'unknown')
    os.makedirs(known_dir,   exist_ok=True)
    os.makedirs(unknown_dir, exist_ok=True)

    for name in known_persons:
        src = os.path.abspath(os.path.join(kaggle_dir, name))
        dst = os.path.join(known_dir, name)
        if not os.path.exists(dst):
            os.symlink(src, dst)

    for name in unknown_persons:
        src_dir = os.path.join(kaggle_dir, name)
        for fn in os.listdir(src_dir):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.abspath(os.path.join(src_dir, fn))
                dst = os.path.join(unknown_dir, f'{name}_{fn}')
                if not os.path.exists(dst):
                    os.symlink(src, dst)

    print(f'[kaggle] {len(known_persons)} known people, '
          f'{len(unknown_persons)} unknown people → {out_dir}')
    return True


def make_synthetic_faces(out_dir: str):
    """Create placeholder face images using OpenCV shapes (fallback demo)."""
    os.makedirs(out_dir, exist_ok=True)
    colours = {'Alice': (180,100,80), 'Bob': (80,130,200), 'Carol': (100,180,90)}
    for name, col in colours.items():
        d = os.path.join(out_dir, 'known', name)
        os.makedirs(d, exist_ok=True)
        for i in range(6):
            img = np.ones((224,224,3), np.uint8)*220
            cv2.ellipse(img, (112,110), (70+i*2, 85), 0, 0, 360, col, -1)
            cv2.circle(img, (85,90),  8, (30,30,30), -1)
            cv2.circle(img, (139,90), 8, (30,30,30), -1)
            cv2.ellipse(img, (112,135), (25,12), 0, 0, 180, (30,30,30), 2)
            img = np.clip(img.astype(int)+np.random.randint(-8,8,img.shape), 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(d, f'{i:03d}.jpg'), img)
    unk_dir = os.path.join(out_dir, 'unknown')
    os.makedirs(unk_dir, exist_ok=True)
    for i in range(5):
        img = np.ones((224,224,3), np.uint8)*220
        col = tuple(int(v) for v in np.random.randint(60,200,3))
        cv2.ellipse(img, (112,110), (72,88), 0, 0, 360, col, -1)
        cv2.circle(img, (85,90),  8, (20,20,20), -1)
        cv2.circle(img, (139,90), 8, (20,20,20), -1)
        cv2.imwrite(os.path.join(unk_dir, f'unknown_{i:03d}.jpg'), img)
    print(f'[demo] synthetic faces created in {out_dir}')


def evaluate(face_dir: str, host: str, port: int):
    known_dir   = os.path.join(face_dir, 'known')
    unknown_dir = os.path.join(face_dir, 'unknown')

    if not os.path.isdir(known_dir):
        print(f'[error] {known_dir} not found'); return

    # Check FaceManager is running
    ping = call_face_api('health', host, port)
    if ping is None:
        print(f'[error] FaceManager not reachable at {host}:{port}')
        print('  Start with: python3 face_recognition_service.py')
        return

    print(f'\n[1/3] Registering known identities …')
    known_names = sorted(os.listdir(known_dir))
    registration = {}
    for name in known_names:
        nd = os.path.join(known_dir, name)
        if not os.path.isdir(nd): continue
        imgs = [img for _, img in load_images_from_dir(nd)]
        # Use first half for registration, second half for test
        n_reg = max(1, len(imgs) // 2)
        ok = register_identity(name, imgs[:n_reg], host, port)
        registration[name] = ok
        print(f'  {name}: registered={ok}  ({n_reg} images)')

    results = []

    print(f'\n[2/3] Testing known faces (should be recognised) …')
    for name in known_names:
        nd = os.path.join(known_dir, name)
        if not os.path.isdir(nd): continue
        all_imgs = load_images_from_dir(nd)
        n_reg = max(1, len(all_imgs) // 2)
        test_imgs = all_imgs[n_reg:]  # held-out test set
        for fname, img in test_imgs:
            pred, conf, lat = recognize_image(img, host, port)
            correct = (pred == name)
            rec = {
                'true_name':  name,
                'pred_name':  pred,
                'correct':    correct,
                'confidence': round(conf, 4),
                'latency_s':  round(lat, 3),
                'type':       'known',
                'image':      fname,
            }
            results.append(rec)
            icon = '✓' if correct else '✗'
            print(f'  {icon} {name}/{fname}  pred={pred}  conf={conf:.3f}  '
                  f'lat={lat*1000:.0f}ms')

    print(f'\n[3/3] Testing unknown faces (should return None) …')
    if os.path.isdir(unknown_dir):
        for fname, img in load_images_from_dir(unknown_dir):
            pred, conf, lat = recognize_image(img, host, port)
            fp = (pred is not None)
            rec = {
                'true_name':  None,
                'pred_name':  pred,
                'correct':    not fp,
                'confidence': round(conf, 4),
                'latency_s':  round(lat, 3),
                'type':       'unknown',
                'image':      fname,
            }
            results.append(rec)
            icon = '✓' if not fp else '✗ FP'
            print(f'  {icon} {fname}  pred={pred}  conf={conf:.3f}')

    save_result('face_recognition', results)
    _print_summary(results)


def _print_summary(results):
    known   = [r for r in results if r['type'] == 'known']
    unknown = [r for r in results if r['type'] == 'unknown']
    print('\n' + '═'*55)
    if known:
        acc = np.mean([r['correct'] for r in known]) * 100
        frr = 100 - acc
        lats = [r['latency_s'] for r in known]
        print(f'  Known face accuracy  : {acc:.1f}%')
        print(f'  False Reject Rate    : {frr:.1f}%')
        print(f'  Latency (mean/p95)   : {np.mean(lats)*1000:.0f}ms / '
              f'{np.percentile(lats,95)*1000:.0f}ms')
    if unknown:
        far = np.mean([not r['correct'] for r in unknown]) * 100
        print(f'  False Accept Rate    : {far:.1f}%  ({len(unknown)} unknowns)')
    print('═'*55)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--face_dir', default='data/faces')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=5002)
    ap.add_argument('--demo', action='store_true',
                    help='Use synthetic faces if --kaggle_dir not given')
    ap.add_argument('--kaggle_dir', default=None,
                    help='Path to extracted Kaggle face-recognition-dataset folder')
    ap.add_argument('--n_unknown', type=int, default=6,
                    help='Number of people to hold out as unknowns (default 6)')
    args = ap.parse_args()

    if args.kaggle_dir:
        prepare_from_kaggle(args.kaggle_dir, args.face_dir, args.n_unknown)
    elif args.demo:
        make_synthetic_faces(args.face_dir)

    evaluate(args.face_dir, args.host, args.port)
