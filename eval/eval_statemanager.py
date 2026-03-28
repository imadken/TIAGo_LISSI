#!/usr/bin/env python3
"""
eval_statemanager.py — Spatial relation accuracy evaluation (no ROS required).

Tests the StateManager's scene graph inference (left_of, right_of, above,
below, near, on_table) against hand-annotated ground-truth JSON files.

Ground-truth format (data/scene_annotations/<scene_id>.json)
─────────────────────────────────────────────────────────────
{
  "scene_id": "scene_001",
  "image_file": "scene_001.jpg",      # optional, not used for metric
  "objects": [
    {"id": "A", "class": "bottle", "bbox": [x1, y1, x2, y2], "depth_m": 0.82},
    {"id": "B", "class": "cup",    "bbox": [x1, y1, x2, y2], "depth_m": 0.90}
  ],
  "relations": [
    {"subject": "A", "relation": "left_of",  "object": "B", "truth": true},
    {"subject": "A", "relation": "near",     "object": "B", "truth": true},
    {"subject": "A", "relation": "on_table", "object": null, "truth": true},
    {"subject": "B", "relation": "on_table", "object": null, "truth": true}
  ]
}

Usage
─────
    # Evaluate against hand-annotated scenes
    python3 eval_statemanager.py --data_dir data/scene_annotations

    # Generate synthetic scenes and run self-test
    python3 eval_statemanager.py --demo
"""
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.eval_utils import save_result

# ── Inline spatial relation logic (mirrors StateManager) ──────────────────────
# Kept self-contained so the test doesn't depend on ROS or embodied_agent stack

IMG_W, IMG_H = 640, 480
ON_TABLE_TILT_THRESH = 0.75   # bbox bottom must be in lower 75% of image
NEAR_THRESH_PX = 150          # pixel distance between centroids
DEPTH_ABOVE_THRESH = 0.15     # depth difference (m) to call "above"

SUPPORTED_RELATIONS = ['left_of', 'right_of', 'above', 'below', 'near', 'on_table']


def centroid(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def infer_relation(subject: dict, relation: str, obj: dict | None) -> bool:
    """
    Infer a single spatial relation using the same heuristics as StateManager.
    subject / obj: dicts with keys 'bbox' and optionally 'depth_m'.
    """
    sx, sy = centroid(subject['bbox'])
    sd = subject.get('depth_m', 1.0)

    if relation == 'on_table':
        # Object bottom edge below 60% of image height (table-level)
        bottom_y = subject['bbox'][3]
        return bottom_y > IMG_H * 0.60

    if obj is None:
        return False

    ox, oy = centroid(obj['bbox'])
    od = obj.get('depth_m', 1.0)

    if relation == 'left_of':
        return sx < ox - 20        # 20px hysteresis
    if relation == 'right_of':
        return sx > ox + 20
    if relation == 'above':
        # Above: higher in image AND closer (smaller depth) or clearly higher
        return sy < oy - 20 or (sd < od - DEPTH_ABOVE_THRESH)
    if relation == 'below':
        return sy > oy + 20 or (sd > od + DEPTH_ABOVE_THRESH)
    if relation == 'near':
        dist = np.hypot(sx - ox, sy - oy)
        return dist < NEAR_THRESH_PX

    return False


# ── Annotation loading ─────────────────────────────────────────────────────────

def load_annotation(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Demo scene generator ───────────────────────────────────────────────────────

def make_demo_annotations(out_dir: str) -> list:
    """
    Create a small set of synthetic scene annotations that exercise every
    relation type.  Ground truth is derived from the geometry of the bboxes.
    """
    os.makedirs(out_dir, exist_ok=True)
    scenes = []

    # Scene 1: bottle left of cup, both on table, both near each other
    scenes.append({
        "scene_id": "demo_001",
        "description": "bottle left of cup, near each other, both on table",
        "objects": [
            {"id": "A", "class": "bottle", "bbox": [120, 160, 200, 380], "depth_m": 0.80},
            {"id": "B", "class": "cup",    "bbox": [280, 170, 360, 370], "depth_m": 0.82},
        ],
        "relations": [
            {"subject": "A", "relation": "left_of",  "object": "B", "truth": True},
            {"subject": "B", "relation": "right_of", "object": "A", "truth": True},
            {"subject": "A", "relation": "near",     "object": "B", "truth": True},
            {"subject": "A", "relation": "on_table", "object": None, "truth": True},
            {"subject": "B", "relation": "on_table", "object": None, "truth": True},
        ]
    })

    # Scene 2: bottle above cup (on shelf vs table level)
    scenes.append({
        "scene_id": "demo_002",
        "description": "bottle above cup (shelf vs table), far apart",
        "objects": [
            {"id": "A", "class": "bottle", "bbox": [200, 40, 280, 180], "depth_m": 0.60},
            {"id": "B", "class": "cup",    "bbox": [180, 280, 260, 430], "depth_m": 0.85},
        ],
        "relations": [
            {"subject": "A", "relation": "above",    "object": "B", "truth": True},
            {"subject": "B", "relation": "below",    "object": "A", "truth": True},
            {"subject": "A", "relation": "near",     "object": "B", "truth": False},
            {"subject": "A", "relation": "on_table", "object": None, "truth": False},
            {"subject": "B", "relation": "on_table", "object": None, "truth": True},
        ]
    })

    # Scene 3: three objects — bottle, cup, remote in a row
    scenes.append({
        "scene_id": "demo_003",
        "description": "three objects in a row at table level",
        "objects": [
            {"id": "A", "class": "bottle", "bbox": [60,  200, 140, 420], "depth_m": 0.78},
            {"id": "B", "class": "cup",    "bbox": [240, 210, 310, 400], "depth_m": 0.80},
            {"id": "C", "class": "remote", "bbox": [430, 220, 530, 400], "depth_m": 0.81},
        ],
        "relations": [
            {"subject": "A", "relation": "left_of",  "object": "B", "truth": True},
            {"subject": "A", "relation": "left_of",  "object": "C", "truth": True},
            {"subject": "B", "relation": "left_of",  "object": "C", "truth": True},
            {"subject": "C", "relation": "right_of", "object": "A", "truth": True},
            {"subject": "A", "relation": "near",     "object": "B", "truth": False},
            {"subject": "A", "relation": "near",     "object": "C", "truth": False},
            {"subject": "A", "relation": "on_table", "object": None, "truth": True},
            {"subject": "B", "relation": "on_table", "object": None, "truth": True},
            {"subject": "C", "relation": "on_table", "object": None, "truth": True},
        ]
    })

    # Scene 4: near pair only
    scenes.append({
        "scene_id": "demo_004",
        "description": "cup and bottle very close together",
        "objects": [
            {"id": "A", "class": "bottle", "bbox": [280, 220, 340, 400], "depth_m": 0.79},
            {"id": "B", "class": "cup",    "bbox": [345, 225, 405, 395], "depth_m": 0.80},
        ],
        "relations": [
            {"subject": "A", "relation": "near",     "object": "B", "truth": True},
            {"subject": "A", "relation": "left_of",  "object": "B", "truth": True},
            {"subject": "A", "relation": "on_table", "object": None, "truth": True},
        ]
    })

    for sc in scenes:
        path = os.path.join(out_dir, f'{sc["scene_id"]}.json')
        with open(path, 'w') as f:
            json.dump(sc, f, indent=2)
    print(f'[demo] wrote {len(scenes)} scene annotations to {out_dir}')
    return scenes


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_scene(ann: dict) -> list:
    """
    Evaluate all annotated relations in one scene.
    Returns list of per-relation result dicts.
    """
    obj_map = {o['id']: o for o in ann['objects']}
    results = []

    for rel_entry in ann['relations']:
        subj_id  = rel_entry['subject']
        relation = rel_entry['relation']
        obj_id   = rel_entry.get('object')
        truth    = rel_entry['truth']

        subj = obj_map.get(subj_id)
        obj  = obj_map.get(obj_id) if obj_id else None

        if subj is None:
            print(f'  [warn] subject "{subj_id}" not in objects — skipping')
            continue

        predicted = infer_relation(subj, relation, obj)
        correct   = (predicted == truth)

        results.append({
            'scene_id':    ann.get('scene_id', '?'),
            'subject':     subj_id,
            'subject_cls': subj.get('class', ''),
            'relation':    relation,
            'object':      obj_id,
            'object_cls':  obj.get('class', '') if obj else None,
            'predicted':   predicted,
            'truth':       truth,
            'correct':     correct,
        })

        icon = '✓' if correct else '✗'
        print(f'  {icon}  {subj_id}({subj.get("class","?")})'
              f' {relation}'
              f' {obj_id+"("+obj.get("class","?")+")  " if obj else "         "}'
              f'pred={predicted}  gt={truth}')

    return results


def summarise(all_results: list):
    n  = len(all_results)
    if n == 0:
        print('[warn] no results')
        return
    ok = sum(1 for r in all_results if r['correct'])
    print(f'\n{"═"*65}')
    print(f'  Overall accuracy: {ok}/{n}  ({ok/n*100:.1f}%)')

    # Per-relation breakdown
    print(f'\n  {"Relation":<14}  {"Correct":>7}  {"Total":>5}  {"Acc%":>6}')
    print(f'  {"─"*14}  {"─"*7}  {"─"*5}  {"─"*6}')
    for rel in SUPPORTED_RELATIONS:
        rows = [r for r in all_results if r['relation'] == rel]
        if not rows:
            continue
        r_ok = sum(1 for r in rows if r['correct'])
        print(f'  {rel:<14}  {r_ok:>7}  {len(rows):>5}  {r_ok/len(rows)*100:>5.1f}%')

    # Confusion stats (TP/FP/TN/FN)
    tp = sum(1 for r in all_results if r['correct'] and r['truth'])
    tn = sum(1 for r in all_results if r['correct'] and not r['truth'])
    fp = sum(1 for r in all_results if not r['correct'] and r['predicted'])
    fn = sum(1 for r in all_results if not r['correct'] and not r['predicted'])
    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    rec  = tp / (tp + fn) if (tp + fn) else float('nan')
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) else float('nan')
    print(f'\n  TP={tp}  TN={tn}  FP={fp}  FN={fn}')
    print(f'  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}')
    print(f'{"═"*65}')


def main(args):
    data_dir = args.data_dir

    if args.demo:
        print('[demo] generating synthetic annotations …')
        make_demo_annotations(data_dir)

    annotation_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.json')
    ]) if os.path.isdir(data_dir) else []

    if not annotation_files:
        print(f'[error] no JSON annotation files found in {data_dir}')
        print('  Run with --demo to generate synthetic scenes first.')
        return

    print(f'\n{"═"*65}')
    print(f' Spatial Relation Accuracy Evaluation  ({len(annotation_files)} scenes)')
    print(f'{"═"*65}')

    all_results = []
    for path in annotation_files:
        ann = load_annotation(path)
        print(f'\n  Scene: {ann.get("scene_id", path)}  '
              f'— {ann.get("description", "")}')
        scene_results = evaluate_scene(ann)
        all_results.extend(scene_results)

    summarise(all_results)

    agg = {
        'n_scenes': len(annotation_files),
        'n_relations': len(all_results),
        'accuracy': round(sum(r['correct'] for r in all_results) / max(1, len(all_results)), 4),
        'per_relation': {},
        'raw': all_results,
    }
    for rel in SUPPORTED_RELATIONS:
        rows = [r for r in all_results if r['relation'] == rel]
        if rows:
            agg['per_relation'][rel] = {
                'acc': round(sum(r['correct'] for r in rows) / len(rows), 4),
                'n': len(rows),
            }

    save_result('statemanager', agg)
    print('\n[done] results saved to data/statemanager.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/scene_annotations',
                    help='directory containing *scene*.json annotation files')
    ap.add_argument('--demo', action='store_true',
                    help='generate synthetic scenes and evaluate')
    args = ap.parse_args()
    main(args)
