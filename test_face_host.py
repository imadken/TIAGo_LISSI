#!/usr/bin/env python3
"""
Face recognition test — run on HOST (has webcam access + face service).

Usage:
    python3 test_face_host.py                       # live webcam loop
    python3 test_face_host.py --image photo.jpg     # test single image
    python3 test_face_host.py --register "Alice"    # webcam + register
    python3 test_face_host.py --register "Alice" --image photo.jpg
"""

import sys
import cv2
import requests
import argparse

FACE_SERVICE_URL = "http://localhost:5002"


def check_service():
    try:
        r = requests.get(FACE_SERVICE_URL + "/health", timeout=3)
        data = r.json()
        print("[OK] Face service running — {} person(s) in DB: {}".format(
            data.get("num_persons", 0), data.get("persons", [])))
        return True
    except Exception as e:
        print("[ERROR] Cannot reach face service: {}".format(e))
        print("  Start it with: python3 face_recognition_service.py")
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
        print("[FAIL] {}".format(result.get("error", "unknown error")))
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
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--register", metavar="NAME", help="Register faces under this name")
    args = parser.parse_args()

    if not check_service():
        sys.exit(1)

    # ── Single image mode ──────────────────────────────────────────────────────
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print("[ERROR] Cannot read: {}".format(args.image))
            sys.exit(1)
        if args.register:
            register(img, args.register)
        else:
            faces = recognize(img)
            print("Found {} face(s):".format(len(faces)))
            for f in faces:
                print("  {} | conf={:.0f}% | known={}".format(
                    f["name"], f.get("confidence", 0) * 100, f.get("is_known")))
            out = draw_faces(img, faces)
            cv2.imwrite("face_test_result.jpg", out)
            print("Saved → face_test_result.jpg")
        return

    # ── Live webcam mode ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        sys.exit(1)

    print("\nLive webcam — press:")
    print("  R  → recognize current frame")
    if args.register:
        print("  S  → register current frame as '{}'".format(args.register))
    print("  Q  → quit\n")

    last_faces = []
    frame_n = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_n += 1
        if frame_n % 30 == 0:          # auto-recognize ~1x per second
            last_faces = recognize(frame)
            if last_faces:
                print("  → {}".format(
                    ["{} {:.0f}%".format(f["name"], f.get("confidence",0)*100)
                     for f in last_faces]))

        cv2.imshow("Face Test  [R=recognize  S=register  Q=quit]",
                   draw_faces(frame, last_faces))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            last_faces = recognize(frame)
            print("Recognized: {}".format(
                ["{} {:.0f}%".format(f["name"], f.get("confidence",0)*100)
                 for f in last_faces] or ["none"]))
        elif key == ord('s') and args.register:
            if register(frame, args.register):
                last_faces = recognize(frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
