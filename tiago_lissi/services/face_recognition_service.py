#!/usr/bin/env python3
"""
Face Recognition Service  — host machine, Python 3.10
Runs alongside the YOLO service on port 5002.

Endpoints
---------
GET  /health            → {"status": "ok", "num_persons": N}
POST /recognize         → JPEG body  → [{"name","confidence","bbox","is_known"}, ...]
POST /register          → form: name=<str>, file=<JPEG>  → {"success": bool}
GET  /persons           → {"persons": ["Alice", ...]}
POST /forget            → form: name=<str>  → {"success": bool}

Start with:
    python3 -m tiago_lissi.services.face_recognition_service
"""

import os
import json
import pickle
import logging
import numpy as np
import cv2
from flask import Flask, request, jsonify
import face_recognition

# ── Configuration ────────────────────────────────────────────────────────────
PORT = 5002
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces_db")
DB_FILE = os.path.join(DB_DIR, "faces.pkl")
RECOGNITION_TOLERANCE = 0.50   # lower = stricter  (0.6 = default, 0.5 = tight)
MIN_FACE_SIZE = 40             # ignore tiny faces below this pixel height

logging.basicConfig(level=logging.INFO,
                    format="[FaceService] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── In-memory database ────────────────────────────────────────────────────────
known_encodings: list = []   # list of np.ndarray (128-d)
known_names: list = []       # parallel list of str


def _load_db():
    """Load face database from disk into memory."""
    global known_encodings, known_names
    os.makedirs(DB_DIR, exist_ok=True)
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            data = pickle.load(f)
        known_encodings = data.get("encodings", [])
        known_names = data.get("names", [])
        log.info("Loaded %d face sample(s) for %d person(s)",
                 len(known_encodings), len(set(known_names)))
    else:
        known_encodings, known_names = [], []
        log.info("No existing database — starting fresh.")


def _save_db():
    """Persist current database to disk."""
    os.makedirs(DB_DIR, exist_ok=True)
    with open(DB_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    log.info("Database saved (%d samples, %d persons)",
             len(known_encodings), len(set(known_names)))


def _decode_jpeg(raw_bytes: bytes) -> np.ndarray:
    """JPEG bytes → RGB numpy array for face_recognition."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "num_persons": len(set(known_names)),
        "num_samples": len(known_encodings),
        "persons": list(set(known_names)),
    })


@app.route("/recognize", methods=["POST"])
def recognize():
    """
    Body: raw JPEG bytes
    Returns: list of face dicts with name, confidence, bbox, is_known
    """
    try:
        rgb = _decode_jpeg(request.data)
    except Exception:
        log.exception("Failed to decode /recognize payload")
        return jsonify({"error": "invalid image data"}), 400

    # Locate faces — use 'hog' model (fast, CPU-only); swap to 'cnn' for GPU
    locations = face_recognition.face_locations(rgb, model="hog")
    if not locations:
        return jsonify([])

    encodings = face_recognition.face_encodings(rgb, locations)
    results = []

    for enc, loc in zip(encodings, locations):
        top, right, bottom, left = loc
        face_h = bottom - top
        if face_h < MIN_FACE_SIZE:
            continue

        bbox = [left, top, right - left, bottom - top]  # x, y, w, h  (OpenCV style)

        if not known_encodings:
            results.append({
                "name": "unknown",
                "confidence": 0.0,
                "bbox": bbox,
                "is_known": False,
            })
            continue

        distances = face_recognition.face_distance(known_encodings, enc)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        confidence = max(0.0, 1.0 - best_dist)   # higher = more confident

        if best_dist <= RECOGNITION_TOLERANCE:
            name = known_names[best_idx]
            is_known = True
        else:
            name = "unknown"
            is_known = False

        results.append({
            "name": name,
            "confidence": round(confidence, 3),
            "bbox": bbox,
            "is_known": is_known,
        })

    log.info("Recognized %d face(s): %s",
             len(results), [r["name"] for r in results])
    return jsonify(results)


@app.route("/register", methods=["POST"])
def register():
    """
    Form fields: name=<str>
    File upload OR raw JPEG body
    Stores all detected face encodings for that name (can call multiple times).
    """
    name = (request.form.get("name") or "").strip()
    if not name:
        return jsonify({"success": False, "error": "name required"}), 400

    # Accept file upload or raw body
    if "file" in request.files:
        raw = request.files["file"].read()
    else:
        raw = request.data

    if not raw:
        return jsonify({"success": False, "error": "no image data"}), 400

    try:
        rgb = _decode_jpeg(raw)
    except Exception:
        log.exception("Failed to decode /register payload")
        return jsonify({"success": False, "error": "invalid image data"}), 400

    locations = face_recognition.face_locations(rgb, model="hog")
    if not locations:
        return jsonify({"success": False, "error": "no face detected in image"}), 200

    encodings = face_recognition.face_encodings(rgb, locations)
    # Store every detected face under this name (multiple encodings = more robust)
    for enc in encodings:
        known_encodings.append(enc)
        known_names.append(name)

    _save_db()
    log.info("Registered %d encoding(s) for '%s'", len(encodings), name)
    return jsonify({
        "success": True,
        "name": name,
        "encodings_added": len(encodings),
        "total_samples": known_names.count(name),
    })


@app.route("/persons", methods=["GET"])
def list_persons():
    persons = {}
    for name in known_names:
        persons[name] = persons.get(name, 0) + 1
    return jsonify({
        "persons": list(persons.keys()),
        "samples": persons,
    })


@app.route("/forget", methods=["POST"])
def forget():
    """Remove all encodings for a given name."""
    name = (request.form.get("name") or request.json.get("name", "")
            if request.content_type == "application/json"
            else request.form.get("name", "")).strip()
    if not name:
        return jsonify({"success": False, "error": "name required"}), 400

    global known_encodings, known_names
    before = len(known_names)
    pairs = [(e, n) for e, n in zip(known_encodings, known_names) if n != name]
    if pairs:
        known_encodings, known_names = zip(*pairs)
        known_encodings = list(known_encodings)
        known_names = list(known_names)
    else:
        known_encodings, known_names = [], []

    removed = before - len(known_names)
    if removed > 0:
        _save_db()
    log.info("Forgot '%s': removed %d sample(s)", name, removed)
    return jsonify({"success": removed > 0, "removed": removed})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _load_db()
    log.info("Face Recognition Service starting on port %d", PORT)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
