#!/usr/bin/env python3
"""
capture_data.py — Save RGB images and RGBD pairs from the TIAGo robot.

Run this INSIDE docker with ROS sourced.

Modes
─────
  --mode images   Save RGB frames for eval_detection.py
                  → eval/data/images/<class>/img001.jpg …

  --mode rgbd     Save RGB + depth pairs for eval_depth_refine.py
                  → eval/data/rgbd/<scene_id>/rgb.jpg + depth.npy + annotation.json

Usage
─────
  # Capture 10 bottle images (press SPACE to capture, Q to quit)
  python3 eval/capture_data.py --mode images --class bottle --n 10

  # Capture an RGBD pair with annotation (one shot per press)
  python3 eval/capture_data.py --mode rgbd --class bottle --scene scene_001

  # Capture face images for a person
  python3 eval/capture_data.py --mode faces --name Alice --n 6
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# ── Inline depth pipeline (mirrors reach_object_v5 + eval_depth_refine) ───────
DEPTH_MIN, DEPTH_MAX = 0.30, 1.10
RANSAC_ITERS, RANSAC_THRESH = 300, 0.015
RANSAC_R_MIN,  RANSAC_R_MAX = 0.01, 0.25
RANSAC_MIN_INLIERS = 8


def _refine_bbox(depth, bbox, margin=4):
    x1,y1,x2,y2 = [int(v) for v in bbox]
    x1=max(0,x1+margin); y1=max(0,y1+margin)
    x2=min(depth.shape[1]-1,x2-margin); y2=min(depth.shape[0]-1,y2-margin)
    roi = depth[y1:y2, x1:x2]
    valid = roi[(roi>=DEPTH_MIN)&(roi<=DEPTH_MAX)]
    if len(valid) < 20:
        return None, None, None
    d_star = np.percentile(valid, 10)
    mask = (roi>=d_star)&(roi<=d_star+0.12)
    ys,xs = np.where(mask)
    if len(xs) < 10:
        return None, None, None
    return [xs.min()+x1, ys.min()+y1, xs.max()+x1, ys.max()+y1], d_star, mask


def _project_3d(depth, bbox, cam):
    x1,y1,x2,y2 = [int(v) for v in bbox]
    roi = depth[y1:y2, x1:x2]
    vmask = (roi>=DEPTH_MIN)&(roi<=DEPTH_MAX)
    if vmask.sum() < 5:
        return None, None
    d_star = np.percentile(roi[vmask], 10)
    fmask = vmask & (roi<=d_star+0.12)
    ys,xs = np.where(fmask); ds = roi[fmask]
    X=(xs+x1-cam['cx'])*ds/cam['fx']
    Y=(ys+y1-cam['cy'])*ds/cam['fy']
    return np.stack([X,Y,ds],axis=1), d_star


def _ransac_cylinder(pts):
    if pts is None or len(pts) < RANSAC_MIN_INLIERS:
        return None
    xy = pts[:,:2]; best=None; best_n=0
    rng = np.random.default_rng(42)
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(len(xy),3,replace=False)
        a,b,c = xy[idx]
        D = 2*(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]))
        if abs(D)<1e-9: continue
        ux=((a[0]**2+a[1]**2)*(b[1]-c[1])+(b[0]**2+b[1]**2)*(c[1]-a[1])+(c[0]**2+c[1]**2)*(a[1]-b[1]))/D
        uy=((a[0]**2+a[1]**2)*(c[0]-b[0])+(b[0]**2+b[1]**2)*(a[0]-c[0])+(c[0]**2+c[1]**2)*(b[0]-a[0]))/D
        r=np.sqrt((a[0]-ux)**2+(a[1]-uy)**2)
        if not (RANSAC_R_MIN<=r<=RANSAC_R_MAX): continue
        dists=np.abs(np.sqrt((xy[:,0]-ux)**2+(xy[:,1]-uy)**2)-r)
        n=np.sum(dists<RANSAC_THRESH)
        if n>best_n: best_n=n; best=(ux,uy,r,n)
    return best if best and best[3]>=RANSAC_MIN_INLIERS else None


def show_pipeline(rgb, depth, vlm_bbox, cam, out_path=None):
    """
    Display 4-panel pipeline visualization and optionally save it.

    Panel 1 — VLM bbox (loose, as returned by the model)
    Panel 2 — Depth foreground mask (nearest cluster pixels highlighted)
    Panel 3 — Refined tight bbox (depth-constrained)
    Panel 4 — RANSAC cylinder axis projected back to image plane
    """
    H, W = rgb.shape[:2]
    pad = 6
    panel_w, panel_h = W, H
    canvas = np.zeros((panel_h*2 + pad*3, panel_w*2 + pad*3, 3), np.uint8)
    canvas[:] = (30, 30, 30)

    def place(img, row, col):
        y = row*(panel_h+pad)+pad
        x = col*(panel_w+pad)+pad
        canvas[y:y+panel_h, x:x+panel_w] = img

    def label(img, text, color=(255,255,255)):
        out = img.copy()
        cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        return out

    # ── Panel 1: VLM bbox ─────────────────────────────────────────────────────
    p1 = rgb.copy()
    x1,y1,x2,y2 = [int(v) for v in vlm_bbox]
    cv2.rectangle(p1,(x1,y1),(x2,y2),(0,165,255),2)   # orange
    cx,cy = (x1+x2)//2,(y1+y2)//2
    cv2.drawMarker(p1,(cx,cy),(0,165,255),cv2.MARKER_CROSS,12,2)
    place(label(p1,'1. VLM bounding box (loose)',(0,165,255)), 0, 0)

    # ── Panel 2: Depth foreground cluster ────────────────────────────────────
    p2 = rgb.copy()
    bx1,by1,bx2,by2 = [int(v) for v in vlm_bbox]
    bx1=max(0,bx1+4); by1=max(0,by1+4)
    bx2=min(W-1,bx2-4); by2=min(H-1,by2-4)
    roi = depth[by1:by2, bx1:bx2]
    valid = roi[(roi>=DEPTH_MIN)&(roi<=DEPTH_MAX)]
    if len(valid) >= 20:
        d_star = np.percentile(valid, 10)
        fore_mask = (roi>=d_star)&(roi<=d_star+0.12)
        overlay = np.zeros_like(p2)
        ys,xs = np.where(fore_mask)
        for px,py in zip(xs,ys):
            overlay[py+by1, px+bx1] = (0,255,255)   # cyan
        p2 = cv2.addWeighted(p2, 0.6, overlay, 0.4, 0)
        cv2.rectangle(p2,(bx1,by1),(bx2,by2),(0,165,255),1)
        cv2.putText(p2,f'd*={d_star:.2f}m',(bx1,by1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
    place(label(p2,'2. Nearest depth cluster (foreground)',(0,255,255)), 0, 1)

    # ── Panel 3: Refined tight bbox ──────────────────────────────────────────
    p3 = rgb.copy()
    refined, d_star3, _ = _refine_bbox(depth, vlm_bbox)
    if refined:
        rx1,ry1,rx2,ry2 = refined
        # Draw original VLM faint
        cv2.rectangle(p3,(x1,y1),(x2,y2),(0,165,255),1)
        # Draw refined tight
        cv2.rectangle(p3,(rx1,ry1),(rx2,ry2),(0,255,0),2)   # green
        rcx,rcy = (rx1+rx2)//2,(ry1+ry2)//2
        cv2.drawMarker(p3,(rcx,rcy),(0,255,0),cv2.MARKER_CROSS,12,2)
        area_red = 1.0-(rx2-rx1)*(ry2-ry1)/max(1,(x2-x1)*(y2-y1))
        cv2.putText(p3,f'area -{area_red*100:.0f}%',(rx1,ry1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    place(label(p3,'3. Depth-refined tight bbox',(0,255,0)), 1, 0)

    # ── Panel 4: RANSAC cylinder ──────────────────────────────────────────────
    p4 = rgb.copy()
    use_bbox = refined if refined else vlm_bbox
    pts, _ = _project_3d(depth, use_bbox, cam)
    cyl = _ransac_cylinder(pts)
    if cyl:
        cx3d,cy3d,r3d,n_inl = cyl
        # project cylinder centre back to image
        # Use mean depth from the refined region for back-projection
        bx1r,by1r,bx2r,by2r = [int(v) for v in use_bbox]
        roi_r = depth[by1r:by2r, bx1r:bx2r]
        valid_r = roi_r[(roi_r>=DEPTH_MIN)&(roi_r<=DEPTH_MAX)]
        Z_est = float(np.percentile(valid_r,10)) if len(valid_r)>0 else 0.8
        u_cyl = int(cx3d * cam['fx'] / Z_est + cam['cx'])
        v_cyl = int(cy3d * cam['fy'] / Z_est + cam['cy'])
        r_px  = int(r3d  * cam['fx'] / Z_est)
        if refined:
            rx1,ry1,rx2,ry2 = refined
            cv2.rectangle(p4,(rx1,ry1),(rx2,ry2),(0,255,0),1)
        cv2.circle(p4,(u_cyl,v_cyl),max(2,r_px),(0,0,255),2)   # red circle = cylinder
        cv2.drawMarker(p4,(u_cyl,v_cyl),(0,0,255),cv2.MARKER_CROSS,14,2)
        cv2.putText(p4,f'r={r3d*100:.1f}cm  n={n_inl}',(u_cyl-40,v_cyl-r_px-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),1)
    else:
        # Fallback: centroid
        if pts is not None and len(pts)>0:
            med_d=np.percentile(pts[:,2],50)
            near=pts[pts[:,2]<=med_d+0.05]
            c3d=(near if len(near)>0 else pts).mean(axis=0)
            u_c=int(c3d[0]*cam['fx']/c3d[2]+cam['cx'])
            v_c=int(c3d[1]*cam['fy']/c3d[2]+cam['cy'])
            cv2.drawMarker(p4,(u_c,v_c),(255,0,255),cv2.MARKER_STAR,20,2)
            cv2.putText(p4,'centroid fallback',(u_c-50,v_c-14),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,255),1)
        else:
            cv2.putText(p4,'RANSAC: no pts',(20,60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    place(label(p4,'4. RANSAC cylinder fit (3D axis)',(0,0,255)), 1, 1)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_y = canvas.shape[0] - 12
    cv2.putText(canvas,'Orange=VLM  Cyan=depth cluster  Green=refined  Red=RANSAC cylinder',
                (pad,legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(200,200,200),1)

    if out_path:
        cv2.imwrite(out_path, canvas)
        print(f'  Pipeline visualization saved: {out_path}')

    cv2.imshow('Depth Pipeline — 4 stages (press any key)', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return canvas

# ── ROS imports ────────────────────────────────────────────────────────────────
try:
    import rospy
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    ROS_OK = True
except ImportError:
    print('[error] rospy not found — run this inside the docker container')
    print('  docker exec -it tiago_ros bash')
    print('  source /opt/ros/melodic/setup.bash')
    sys.exit(1)

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR  = os.path.join(WORKSPACE, 'eval')

RGB_TOPIC   = '/xtion/rgb/image_raw'
DEPTH_TOPIC = '/xtion/depth_registered/image_raw'
CAM_TOPIC   = '/xtion/rgb/camera_info'

bridge = CvBridge()


# ── Camera helpers ─────────────────────────────────────────────────────────────

def get_rgb(timeout=5.0):
    msg = rospy.wait_for_message(RGB_TOPIC, Image, timeout=timeout)
    return bridge.imgmsg_to_cv2(msg, 'bgr8')


def get_depth(timeout=5.0):
    msg = rospy.wait_for_message(DEPTH_TOPIC, Image, timeout=timeout)
    depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    depth = depth.astype(np.float32)
    # Convert mm → m if values look like millimetres
    if depth[depth > 0].mean() > 10:
        depth = depth / 1000.0
    return depth


def get_camera_info(timeout=5.0):
    try:
        info = rospy.wait_for_message(CAM_TOPIC, CameraInfo, timeout=timeout)
        k = info.K
        return {'fx': k[0], 'fy': k[4], 'cx': k[2], 'cy': k[5]}
    except Exception:
        # TIAGo Xtion defaults
        return {'fx': 554.25, 'fy': 554.25, 'cx': 320.5, 'cy': 240.5}


# ── Live preview with capture ──────────────────────────────────────────────────

def preview_and_capture(label: str):
    """
    Show live RGB feed.  Returns the frame captured when user presses SPACE.
    Press Q to cancel.
    """
    print(f'  [preview] SPACE=capture  Q=quit  — {label}')
    while not rospy.is_shutdown():
        try:
            frame = get_rgb(timeout=2.0)
        except Exception as e:
            print(f'  [warn] {e}')
            continue
        display = frame.copy()
        cv2.putText(display, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(display, 'SPACE=capture  Q=quit', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow('TIAGo capture', display)
        key = cv2.waitKey(50) & 0xFF
        if key == ord(' '):
            cv2.destroyAllWindows()
            return frame
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None
    return None


# ── Mode: images (for eval_detection.py) ──────────────────────────────────────

def mode_images(cls: str, n: int):
    out_dir = os.path.join(EVAL_DIR, 'data', 'images', cls)
    os.makedirs(out_dir, exist_ok=True)

    # Find next available index
    existing = [f for f in os.listdir(out_dir) if f.endswith('.jpg')]
    start_idx = len(existing)

    print(f'\nCapturing {n} images of class "{cls}"')
    print(f'Saving to: {out_dir}')

    saved = 0
    for i in range(n):
        frame = preview_and_capture(f'{cls}  [{saved+1}/{n}]')
        if frame is None:
            print('  Cancelled.')
            break
        fname = f'img{start_idx + saved + 1:03d}.jpg'
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, frame)
        print(f'  Saved: {fname}')
        saved += 1

    print(f'\n[done] {saved} images saved to {out_dir}')


# ── Mode: rgbd (for eval_depth_refine.py) ─────────────────────────────────────

def mode_rgbd(cls: str, scene_id: str):
    out_dir = os.path.join(EVAL_DIR, 'data', 'rgbd', scene_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\nCapturing RGBD pair for scene "{scene_id}" (class: {cls})')
    print(f'Saving to: {out_dir}')
    print('\nPosition the object in front of the robot, then press SPACE.')

    frame = preview_and_capture(f'RGBD capture — {cls}')
    if frame is None:
        print('Cancelled.')
        return

    print('  Capturing depth …')
    try:
        depth = get_depth(timeout=5.0)
    except Exception as e:
        print(f'  [error] Could not get depth: {e}')
        return

    cam_info = get_camera_info()

    # Save RGB
    rgb_path = os.path.join(out_dir, 'rgb.jpg')
    cv2.imwrite(rgb_path, frame)

    # Save depth
    depth_path = os.path.join(out_dir, 'depth.npy')
    np.save(depth_path, depth)

    # Show the frame with YOLO-style bbox drawing for annotation
    print('\n  Draw the TIGHT bbox around the object:')
    print('  Click top-left corner, then bottom-right corner.')
    bbox_pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox_pts.append((x, y))
            print(f'    Point {len(bbox_pts)}: ({x}, {y})')

    display = frame.copy()
    cv2.imshow('Draw tight bbox (YOLO ground-truth)', display)
    cv2.setMouseCallback('Draw tight bbox (YOLO ground-truth)', on_mouse)
    print('  Press ENTER when done.')
    while len(bbox_pts) < 2:
        cv2.waitKey(50)
    cv2.destroyAllWindows()

    yolo_bbox = [bbox_pts[0][0], bbox_pts[0][1], bbox_pts[1][0], bbox_pts[1][1]]

    print('\n  Now draw the LOOSE bbox (as VLM would return — bigger):')
    bbox_pts2 = []
    display2 = frame.copy()
    cv2.rectangle(display2, tuple(bbox_pts[0]), tuple(bbox_pts[1]), (0,255,0), 2)
    cv2.putText(display2, 'YOLO bbox (green). Draw VLM bbox (loose).',
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
    cv2.imshow('Draw loose VLM bbox', display2)
    cv2.setMouseCallback('Draw loose VLM bbox', lambda e,x,y,f,p: bbox_pts2.append((x,y)) if e==cv2.EVENT_LBUTTONDOWN else None)
    while len(bbox_pts2) < 2:
        cv2.waitKey(50)
    cv2.destroyAllWindows()

    vlm_bbox = [bbox_pts2[0][0], bbox_pts2[0][1], bbox_pts2[1][0], bbox_pts2[1][1]]

    annotation = {
        'class':       cls,
        'yolo_bbox':   yolo_bbox,
        'vlm_bbox':    vlm_bbox,
        'gt_3d':       None,   # fill manually if you have ground truth metres
        'camera_info': cam_info,
    }
    ann_path = os.path.join(out_dir, 'annotation.json')
    with open(ann_path, 'w') as f:
        json.dump(annotation, f, indent=2)

    print(f'\n[done] Saved:')
    print(f'  {rgb_path}')
    print(f'  {depth_path}')
    print(f'  {ann_path}')
    print(f'\n  yolo_bbox: {yolo_bbox}')
    print(f'  vlm_bbox:  {vlm_bbox}')
    print(f'\n  (Edit annotation.json to add gt_3d in metres if you know the real position)')

    # ── Show full pipeline visualization ──────────────────────────────────────
    print('\n  Running depth pipeline visualization …')
    vis_path = os.path.join(out_dir, 'pipeline_vis.jpg')
    show_pipeline(frame, depth, vlm_bbox, cam_info, out_path=vis_path)


# ── Mode: faces (for eval_face.py) ────────────────────────────────────────────

def mode_faces(name: str, n: int):
    out_dir = os.path.join(EVAL_DIR, 'data', 'faces', 'known', name)
    os.makedirs(out_dir, exist_ok=True)

    existing = [f for f in os.listdir(out_dir) if f.endswith('.jpg')]
    start_idx = len(existing)

    print(f'\nCapturing {n} face images for "{name}"')
    print(f'Saving to: {out_dir}')
    print('Tip: vary expression, slight angle, distance between shots.\n')

    saved = 0
    for i in range(n):
        frame = preview_and_capture(f'Face: {name}  [{saved+1}/{n}]')
        if frame is None:
            print('  Cancelled.')
            break
        fname = f'{start_idx + saved + 1:03d}.jpg'
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, frame)
        print(f'  Saved: {fname}')
        saved += 1

    print(f'\n[done] {saved} face images saved to {out_dir}')
    print('  To add unknown faces: save to eval/data/faces/unknown/')


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['images', 'rgbd', 'faces'], required=True)
    ap.add_argument('--class', dest='cls', default='bottle',
                    help='object class (for images/rgbd modes)')
    ap.add_argument('--scene', default=None,
                    help='scene ID for rgbd mode (e.g. scene_001)')
    ap.add_argument('--name', default='Person',
                    help='person name for faces mode')
    ap.add_argument('--n', type=int, default=10,
                    help='number of images to capture (images/faces modes)')
    args = ap.parse_args()

    rospy.init_node('eval_capture', anonymous=True)

    if args.mode == 'images':
        mode_images(args.cls, args.n)

    elif args.mode == 'rgbd':
        scene = args.scene or f'{args.cls}_{int(time.time())}'
        mode_rgbd(args.cls, scene)

    elif args.mode == 'faces':
        mode_faces(args.name, args.n)
