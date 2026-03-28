#!/usr/bin/env python3
"""
eval_depth_refine.py — Depth-based bounding box refinement evaluation.

Works entirely offline on saved RGBD pairs.  No ROS, no VLM calls.

Metrics produced
────────────────
• Bbox area reduction     — (loose_area − tight_area) / loose_area
• Refined bbox IoU        — overlap with YOLO ground-truth bbox
• RANSAC convergence rate — % of frames where RANSAC finds a valid cylinder
• Centroid fallback rate  — % of frames using centroid instead of RANSAC
• 3D centroid error       — distance from estimated to known GT (if available)
• Pose stability (σ)      — std-dev of centroid over a 5-frame window
• Depth pixel purity      — % of refined bbox pixels that belong to the object

Data format
───────────
data/rgbd/
    <scene_id>/
        rgb.jpg          640×480 RGB
        depth.npy        640×480 float32, metres (0 = invalid)
        annotation.json  {
            "class": "bottle",
            "yolo_bbox": [x1,y1,x2,y2],   # pixel, tight
            "vlm_bbox":  [x1,y1,x2,y2],   # pixel, loose (what VLM returned)
            "gt_3d": [X, Y, Z],            # metres in robot base frame (optional)
            "camera_info": {fx,fy,cx,cy}   # optional; uses TIAGo defaults otherwise
        }

Usage
─────
    python3 eval_depth_refine.py --rgbd_dir data/rgbd
    python3 eval_depth_refine.py --demo       # generate synthetic RGBD scenes
"""
import argparse
import json
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.eval_utils import save_result, bbox_iou, bbox_area_reduction, centroid_3d_distance

# TIAGo Xtion Pro Live defaults (metres, pixels)
DEFAULT_CAM = dict(fx=554.25, fy=554.25, cx=320.5, cy=240.5)

# RANSAC cylinder params (must match reach_object_v5)
RANSAC_ITERS     = 300
RANSAC_THRESH    = 0.015   # m
RANSAC_R_MIN     = 0.01
RANSAC_R_MAX     = 0.25
RANSAC_MIN_INLIERS = 8
DEPTH_MIN, DEPTH_MAX = 0.30, 1.10  # m


# ── Depth refinement (mirrors reach_object_v5 logic) ─────────────────────────

def refine_bbox_by_depth(depth_img, bbox, margin=4):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 + margin); y1 = max(0, y1 + margin)
    x2 = min(depth_img.shape[1]-1, x2 - margin)
    y2 = min(depth_img.shape[0]-1, y2 - margin)
    roi = depth_img[y1:y2, x1:x2]
    valid = roi[(roi >= DEPTH_MIN) & (roi <= DEPTH_MAX)]
    if len(valid) < 20:
        return None, None, None
    d_star = np.percentile(valid, 10)
    mask = (roi >= d_star) & (roi <= d_star + 0.12)
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None, None, None
    rx1, rx2 = xs.min() + x1, xs.max() + x1
    ry1, ry2 = ys.min() + y1, ys.max() + y1
    return [rx1, ry1, rx2, ry2], d_star, mask


def project_to_3d(depth_img, bbox, cam):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    roi = depth_img[y1:y2, x1:x2]
    valid_mask = (roi >= DEPTH_MIN) & (roi <= DEPTH_MAX)
    if valid_mask.sum() < 5:
        return None, None
    d_star = np.percentile(roi[valid_mask], 10)
    fore_mask = valid_mask & (roi <= d_star + 0.12)
    ys, xs = np.where(fore_mask)
    ds = roi[fore_mask]
    # Camera projection
    X = (xs + x1 - cam['cx']) * ds / cam['fx']
    Y = (ys + y1 - cam['cy']) * ds / cam['fy']
    Z = ds
    pts = np.stack([X, Y, Z], axis=1)
    return pts, d_star


def ransac_cylinder(pts):
    """Fit vertical cylinder; return (cx,cy,r,n_inliers) or None."""
    if pts is None or len(pts) < RANSAC_MIN_INLIERS:
        return None
    best = None; best_n = 0
    xy = pts[:, :2]
    rng = np.random.default_rng(0)
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(len(xy), 3, replace=False)
        a, b, c = xy[idx]
        D = 2 * (a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
        if abs(D) < 1e-9:
            continue
        ux = ((a[0]**2+a[1]**2)*(b[1]-c[1]) + (b[0]**2+b[1]**2)*(c[1]-a[1]) +
              (c[0]**2+c[1]**2)*(a[1]-b[1])) / D
        uy = ((a[0]**2+a[1]**2)*(c[0]-b[0]) + (b[0]**2+b[1]**2)*(a[0]-c[0]) +
              (c[0]**2+c[1]**2)*(b[0]-a[0])) / D
        r  = np.sqrt((a[0]-ux)**2 + (a[1]-uy)**2)
        if not (RANSAC_R_MIN <= r <= RANSAC_R_MAX):
            continue
        dists = np.abs(np.sqrt((xy[:,0]-ux)**2 + (xy[:,1]-uy)**2) - r)
        inliers = np.sum(dists < RANSAC_THRESH)
        if inliers > best_n:
            best_n = inliers
            best = (ux, uy, r, inliers)
    if best and best[3] >= RANSAC_MIN_INLIERS:
        return best
    return None


def centroid_fallback(pts):
    if pts is None or len(pts) == 0:
        return None
    med_d = np.percentile(pts[:,2], 50)
    near  = pts[pts[:,2] <= med_d + 0.05]
    return near.mean(axis=0) if len(near) > 0 else pts.mean(axis=0)


# ── Purity helper ─────────────────────────────────────────────────────────────

def depth_purity(depth_img, refined_bbox, d_star, tolerance=0.12):
    """% of pixels in refined bbox that are within [d_star, d_star+tol]."""
    x1, y1, x2, y2 = [int(v) for v in refined_bbox]
    roi = depth_img[y1:y2, x1:x2]
    total = roi.size
    if total == 0:
        return 0.0
    ok = np.sum((roi >= d_star) & (roi <= d_star + tolerance))
    return ok / total


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate_scene(scene_dir: str, cam: dict) -> dict:
    rgb_path   = os.path.join(scene_dir, 'rgb.jpg')
    depth_path = os.path.join(scene_dir, 'depth.npy')
    ann_path   = os.path.join(scene_dir, 'annotation.json')

    rgb   = cv2.imread(rgb_path)
    depth = np.load(depth_path)
    with open(ann_path) as f:
        ann = json.load(f)

    if 'camera_info' in ann:
        cam = ann['camera_info']

    vlm_bbox  = ann.get('vlm_bbox')   # [x1,y1,x2,y2] pixels
    yolo_bbox = ann.get('yolo_bbox')  # ground truth tight bbox
    gt_3d     = ann.get('gt_3d')      # [X,Y,Z] metres, optional

    rec = {
        'scene':  os.path.basename(scene_dir),
        'class':  ann.get('class', 'unknown'),
    }

    if vlm_bbox is None:
        rec['error'] = 'no vlm_bbox in annotation'
        return rec

    # ── Depth refinement ──────────────────────────────────────────────────────
    refined_bbox, d_star, _ = refine_bbox_by_depth(depth, vlm_bbox)
    if refined_bbox:
        rec['refined_bbox']    = refined_bbox
        rec['area_reduction']  = bbox_area_reduction(vlm_bbox, refined_bbox)
        rec['vlm_bbox_area']   = ((vlm_bbox[2]-vlm_bbox[0]) *
                                  (vlm_bbox[3]-vlm_bbox[1]))
        rec['refined_bbox_area'] = ((refined_bbox[2]-refined_bbox[0]) *
                                    (refined_bbox[3]-refined_bbox[1]))
        if yolo_bbox:
            rec['iou_vlm_yolo']     = bbox_iou(vlm_bbox, yolo_bbox)
            rec['iou_refined_yolo'] = bbox_iou(refined_bbox, yolo_bbox)
        rec['depth_purity'] = depth_purity(depth, refined_bbox, d_star)
    else:
        rec['refined_bbox']   = None
        rec['area_reduction'] = None

    # ── 3D pose estimation ────────────────────────────────────────────────────
    pts, _ = project_to_3d(depth, refined_bbox or vlm_bbox, cam)
    cyl = ransac_cylinder(pts)

    if cyl:
        cx, cy, r, n_inliers = cyl
        med_z = np.median(pts[:,2]) if pts is not None else None
        centroid = np.array([cx, cy, med_z]) if med_z else None
        rec['ransac_success']   = True
        rec['ransac_n_inliers'] = n_inliers
        rec['centroid_method']  = 'ransac'
        rec['cylinder_radius']  = r
    else:
        centroid = centroid_fallback(pts)
        rec['ransac_success']   = False
        rec['ransac_n_inliers'] = 0
        rec['centroid_method']  = 'centroid_fallback'
        rec['cylinder_radius']  = None

    if centroid is not None:
        rec['estimated_3d'] = centroid.tolist()
        if gt_3d:
            rec['error_3d_m'] = centroid_3d_distance(centroid, gt_3d)

    return rec


def evaluate_stability(scene_dir: str, cam: dict, n_frames=5) -> dict:
    """Evaluate pose stability over n_frames by adding small noise."""
    ann_path = os.path.join(scene_dir, 'annotation.json')
    depth_path = os.path.join(scene_dir, 'depth.npy')
    if not (os.path.exists(ann_path) and os.path.exists(depth_path)):
        return {}
    with open(ann_path) as f:
        ann = json.load(f)
    depth = np.load(depth_path)
    vlm_bbox = ann.get('vlm_bbox')
    if not vlm_bbox:
        return {}

    rng = np.random.default_rng(1)
    centroids = []
    for _ in range(n_frames):
        noise_depth = depth + rng.normal(0, 0.005, depth.shape)
        refined, d_star, _ = refine_bbox_by_depth(noise_depth, vlm_bbox)
        pts, _ = project_to_3d(noise_depth, refined or vlm_bbox, cam)
        cyl = ransac_cylinder(pts)
        if cyl:
            cx, cy, r, _ = cyl
            med_z = np.median(pts[:,2]) if pts is not None else None
            if med_z:
                centroids.append([cx, cy, med_z])
        elif pts is not None and len(pts) > 0:
            c = centroid_fallback(pts)
            if c is not None:
                centroids.append(c.tolist())

    if len(centroids) < 2:
        return {}
    arr = np.array(centroids)
    return {
        'scene': os.path.basename(scene_dir),
        'n_frames': len(centroids),
        'mean_centroid': arr.mean(axis=0).tolist(),
        'std_centroid': arr.std(axis=0).tolist(),
        'pose_stability_cm': float(arr.std(axis=0).mean() * 100),
    }


def make_synthetic_rgbd(out_dir: str):
    """Generate synthetic RGBD scenes for demo/testing."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(99)
    scenes = [
        ('bottle_01', 'bottle', [200, 100, 260, 280], [0.45, -0.1, 0.85]),
        ('cup_01',    'cup',    [300, 150, 370, 280], [0.52,  0.0, 0.75]),
        ('remote_01', 'remote', [150, 200, 250, 240], [0.38,  0.1, 0.65]),
    ]
    created = []
    for name, cls, obj_bbox, gt_3d in scenes:
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)

        # RGB
        img = np.ones((480, 640, 3), dtype=np.uint8) * 180
        x1,y1,x2,y2 = obj_bbox
        col = {'bottle': (50,100,220), 'cup': (80,200,80),
               'remote': (200,80,50)}.get(cls, (120,120,120))
        cv2.rectangle(img, (x1,y1), (x2,y2), col, -1)
        cv2.imwrite(os.path.join(d, 'rgb.jpg'), img)

        # Depth (object at 0.75m, background at 1.2m)
        depth = np.full((480,640), 1.2, dtype=np.float32)
        depth[y1:y2, x1:x2] = 0.75
        depth += rng.normal(0, 0.008, depth.shape).astype(np.float32)
        np.save(os.path.join(d, 'depth.npy'), depth)

        # VLM bbox is looser than object bbox
        pad = 30
        vlm_bbox = [max(0,x1-pad), max(0,y1-pad),
                    min(639,x2+pad), min(479,y2+pad)]
        ann = {
            'class': cls, 'yolo_bbox': obj_bbox,
            'vlm_bbox': vlm_bbox, 'gt_3d': gt_3d,
        }
        with open(os.path.join(d, 'annotation.json'), 'w') as f:
            json.dump(ann, f, indent=2)
        created.append(d)
        print(f'  created: {d}')
    return created


def run(rgbd_dir: str):
    cam = DEFAULT_CAM
    scenes = sorted([
        os.path.join(rgbd_dir, d)
        for d in os.listdir(rgbd_dir)
        if os.path.isdir(os.path.join(rgbd_dir, d))
    ])
    if not scenes:
        print(f'[error] no scene directories in {rgbd_dir}')
        return

    print(f'\nEvaluating {len(scenes)} scenes …')
    per_scene = []
    stability = []
    for s in scenes:
        if not os.path.exists(os.path.join(s, 'annotation.json')):
            continue
        rec = evaluate_scene(s, cam)
        per_scene.append(rec)
        stab = evaluate_stability(s, cam)
        if stab:
            stability.append(stab)

        print(f'  {rec["scene"]:20s}  '
              f'area_red={rec.get("area_reduction", float("nan")):.2%}  '
              f'iou_ref={rec.get("iou_refined_yolo", float("nan")):.3f}  '
              f'ransac={rec.get("ransac_success")}  '
              f'err3d={rec.get("error_3d_m", float("nan")):.3f}m')

    save_result('depth_refinement', per_scene)
    if stability:
        save_result('pose_stability', stability)
    _print_summary(per_scene, stability)


def _print_summary(per_scene, stability):
    print('\n' + '─'*70)
    ar  = [r['area_reduction'] for r in per_scene if r.get('area_reduction')]
    iou_vlm = [r['iou_vlm_yolo'] for r in per_scene if r.get('iou_vlm_yolo')]
    iou_ref = [r['iou_refined_yolo'] for r in per_scene if r.get('iou_refined_yolo')]
    ransac_ok = [r for r in per_scene if r.get('ransac_success')]
    err3d = [r['error_3d_m'] for r in per_scene if r.get('error_3d_m')]
    purity = [r['depth_purity'] for r in per_scene if r.get('depth_purity')]
    stab   = [s['pose_stability_cm'] for s in stability]

    def mn(lst): return f'{np.mean(lst):.3f}' if lst else '—'
    print(f'  bbox area reduction :  {mn(ar)} ({np.mean(ar)*100:.1f}% mean)' if ar else '  bbox area reduction :  —')
    print(f'  IoU VLM  vs YOLO   :  {mn(iou_vlm)}')
    print(f'  IoU Refined vs YOLO:  {mn(iou_ref)}  ← improvement')
    print(f'  RANSAC convergence :  {len(ransac_ok)}/{len(per_scene)}  ({len(ransac_ok)/len(per_scene)*100:.0f}%)')
    print(f'  Depth purity       :  {mn(purity)}')
    print(f'  3D centroid error  :  {mn(err3d)} m')
    print(f'  Pose stability σ   :  {mn(stab)} cm (over 5-frame window)')
    print('─'*70)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgbd_dir', default='data/rgbd')
    ap.add_argument('--demo', action='store_true')
    args = ap.parse_args()
    if args.demo:
        print('[demo] generating synthetic RGBD scenes …')
        make_synthetic_rgbd(args.rgbd_dir)
    run(args.rgbd_dir)
