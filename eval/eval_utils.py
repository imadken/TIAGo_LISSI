"""Shared utilities for the evaluation suite."""
import json
import os
import time
from datetime import datetime
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_result(name, data):  # type: (str, object) -> None
    """Append a result entry to data/<name>.json."""
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    existing = []
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    if isinstance(data, dict):
        data['_timestamp'] = datetime.now().isoformat()
        existing.append(data)
    else:
        existing.extend(data)
    with open(path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'[eval] saved → {path}  ({len(existing)} entries)')


def load_result(name: str) -> list:
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def bbox_iou(a, b) -> float:
    """Compute IoU between two bboxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_area_reduction(loose, tight) -> float:
    """Return fractional area reduction (positive = tight is smaller)."""
    def area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    a_loose = area(loose); a_tight = area(tight)
    if a_loose == 0:
        return 0.0
    return (a_loose - a_tight) / a_loose


def centroid_3d_distance(p1, p2) -> float:
    """Euclidean 3D distance in metres."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def timer():
    """Context manager that returns elapsed seconds."""
    class _Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *_):
            self.elapsed = time.perf_counter() - self.start
    return _Timer()
