#!/usr/bin/env python3
"""
Standalone face recognition test — no ROS needed.
Captures from webcam (or uses a test image), sends to face service, shows results.

Usage:
    python3 test_face.py                  # live webcam loop
    python3 test_face.py --image path.jpg # test with a single image
    python3 test_face.py --register "Alice" --image path.jpg  # register a person
"""

import sys
import cv2
import requests
import argparse
import numpy as np

FACE_SERVICE_URL = "http://172.17.0.1:5002"


def check_service():
    try:
        r = requests.get(FACE_SERVICE_URL + "/health", timeout=3)
        data = r.json()
        print("[OK] Face service running — {} person(s) in DB: {}".format(
            data.get("num_persons", 0), data.get("persons", [])))
        return True
    except Exception as e:
        print("[ERROR] Cannot reach face service at {}: {}".format(FACE_SERVICE_URL, e))
        return False


def recognize(bgr_image):
    ok, buf = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return []
    r = requests.post(
        FACE_SERVICE_URL + "/recognize",
        data=bytes(buf),
        headers={"Content-Type": "image/jpeg"},
        timeout=5,
    )
    return r.json() if r.status_code == 200 else []


def register(bgr_image, name):
    ok, buf = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return False
    r = requests.post(
        FACE_SERVICE_URL + "/register",
        files={"file": ("face.jpg", bytes(buf), "image/jpeg")},
        data={"name": name},
        timeout=10,
    )
    result = r.json()
    if result.get("success"):
        print("[OK] Registered '{}' — {} sample(s) total".format(
            name, result.get("total_samples", 1)))
    else:
        print("[FAIL] Register error: {}".format(result.get("error", "?")))
    return result.get("success", False)


def draw_faces(bgr_image, faces):
    out = bgr_image.copy()
    for face in faces:
        x, y, w, h = face["bbox"]
        name = face["name"]
        conf = face.get("confidence", 0.0)
        is_known = face.get("is_known", False)
        colour = (0, 200, 0) if is_known else (0, 100, 255)
        label = "{} ({:.0f}%)".format(name, conf * 100) if is_known else "unknown"
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
        lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x, y - lsz[1] - 8), (x + lsz[0] + 4, y), colour, -1)
        cv2.putText(out, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to image file (skip webcam)")
    parser.add_argument("--register", metavar="NAME",
                        help="Register the face in the image/webcam under this name")
    args = parser.parse_args()

    if not check_service():
        sys.exit(1)

    # ── Single image mode ──────────────────────────────────────────────────────
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print("[ERROR] Cannot read image: {}".format(args.image))
            sys.exit(1)

        if args.register:
            register(img, args.register)
        else:
            faces = recognize(img)
            print("Recognized {} face(s):".format(len(faces)))
            for f in faces:
                print("  {} | conf={:.2f} | known={} | bbox={}".format(
                    f["name"], f.get("confidence", 0), f.get("is_known"), f["bbox"]))
            out = draw_faces(img, faces)
            cv2.imwrite("/workspace/face_test_result.jpg", out)
            print("Saved annotated image to /workspace/face_test_result.jpg")
        return

    # ── Live webcam mode ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (device 0)")
        sys.exit(1)

    print("\nLive mode — press:")
    print("  R  → recognize current frame")
    if args.register:
        print("  S  → register current frame as '{}'".format(args.register))
    print("  Q  → quit\n")

    last_faces = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Auto-recognize every 30 frames (~1s at 30fps)
        if frame_count % 30 == 0:
            last_faces = recognize(frame)
            if last_faces:
                print("Auto-recognized: {}".format(
                    ["{} ({:.0f}%)".format(f["name"], f.get("confidence",0)*100)
                     for f in last_faces]))

        display = draw_faces(frame, last_faces)
        cv2.imshow("Face Recognition Test (Q=quit, R=recognize, S=save/register)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            last_faces = recognize(frame)
            print("Recognized: {}".format(
                ["{} ({:.0f}%)".format(f["name"], f.get("confidence",0)*100)
                 for f in last_faces] or ["none"]))
        elif key == ord('s') and args.register:
            register(frame, args.register)
            last_faces = recognize(frame)  # refresh

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
