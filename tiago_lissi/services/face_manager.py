#!/usr/bin/env python3
"""
Face Manager — Docker-side HTTP client (Python 3.6 compatible)
Communicates with face_recognition_service.py running on the host (port 5002).

Usage:
    fm = FaceManager()
    faces = fm.recognize(bgr_image)
    # faces = [{"name": "Alice", "confidence": 0.87, "bbox": [x,y,w,h], "is_known": True}, ...]
    fm.register(bgr_image, "Alice")   # called after person identifies themselves
"""

import os
import cv2
import requests
import numpy as np
from typing import List, Dict, Optional

try:
    import rospy
    def _log_info(msg, *a): rospy.loginfo(msg, *a)
    def _log_warn(msg, *a): rospy.logwarn(msg, *a)
except ImportError:
    def _log_info(msg, *a): print("[FaceManager]", msg % a if a else msg)
    def _log_warn(msg, *a): print("[FaceManager] WARN:", msg % a if a else msg)

FACE_SERVICE_URL = os.environ.get("FACE_SERVICE_URL", "http://localhost:5002")
SERVICE_TIMEOUT = 2.0   # seconds — kept short so detection loop isn't blocked


class FaceManager(object):
    """HTTP client for the face recognition microservice."""

    def __init__(self, service_url=FACE_SERVICE_URL):
        self._url = service_url.rstrip("/")
        self._available = False
        self._check_connection()

    # ── Connection ─────────────────────────────────────────────────────────

    def _check_connection(self):
        try:
            resp = requests.get(self._url + "/health", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                self._available = True
                _log_info(
                    "[FaceManager] Connected — %d person(s) in database: %s",
                    len(data.get("persons", [])),
                    data.get("persons", []),
                )
                return
        except Exception:
            pass
        _log_warn(
            "[FaceManager] Face recognition service not reachable at %s. "
            "Person identification disabled.", self._url
        )
        self._available = False

    def is_available(self):
        return self._available

    # ── Core operations ────────────────────────────────────────────────────

    def recognize(self, bgr_image):
        # type: (np.ndarray) -> List[Dict]
        """
        Send image to service and return list of recognized faces.

        Returns list of dicts:
          {"name": str, "confidence": float, "bbox": [x,y,w,h], "is_known": bool}

        Returns [] on failure (never raises).
        """
        if not self._available:
            return []
        try:
            ok, buf = cv2.imencode(".jpg", bgr_image,
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return []
            resp = requests.post(
                self._url + "/recognize",
                data=bytes(buf),
                headers={"Content-Type": "image/jpeg"},
                timeout=SERVICE_TIMEOUT,
            )
            return resp.json() if resp.status_code == 200 else []
        except Exception as e:
            _log_warn("[FaceManager] recognize() failed: %s", e)
            return []

    def register(self, bgr_image, name):
        # type: (np.ndarray, str) -> bool
        """
        Register one or more faces found in bgr_image under `name`.
        Returns True on success.
        """
        if not self._available:
            return False
        try:
            ok, buf = cv2.imencode(".jpg", bgr_image,
                                   [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                return False
            resp = requests.post(
                self._url + "/register",
                files={"file": ("face.jpg", bytes(buf), "image/jpeg")},
                data={"name": name},
                timeout=SERVICE_TIMEOUT + 2.0,
            )
            result = resp.json()
            if result.get("success"):
                _log_info(
                    "[FaceManager] Registered '%s' (%d sample(s) total)",
                    name, result.get("total_samples", 1),
                )
            else:
                _log_warn("[FaceManager] Register failed: %s",
                              result.get("error", "unknown"))
            return bool(result.get("success"))
        except Exception as e:
            _log_warn("[FaceManager] register() failed: %s", e)
            return False

    def forget(self, name):
        # type: (str) -> bool
        """Remove all encodings for a person by name."""
        if not self._available:
            return False
        try:
            resp = requests.post(
                self._url + "/forget",
                data={"name": name},
                timeout=SERVICE_TIMEOUT,
            )
            result = resp.json()
            return bool(result.get("success"))
        except Exception as e:
            _log_warn("[FaceManager] forget() failed: %s", e)
            return False

    def get_known_persons(self):
        # type: () -> List[str]
        """Return list of names in the database."""
        if not self._available:
            return []
        try:
            resp = requests.get(self._url + "/persons", timeout=SERVICE_TIMEOUT)
            return resp.json().get("persons", []) if resp.status_code == 200 else []
        except Exception:
            return []

    # ── Annotation helper ──────────────────────────────────────────────────

    def draw_faces(self, bgr_image, faces):
        # type: (np.ndarray, List[Dict]) -> np.ndarray
        """Draw face bounding boxes and names on a BGR image."""
        annotated = bgr_image.copy()
        for face in faces:
            x, y, w, h = face["bbox"]
            name = face["name"]
            conf = face.get("confidence", 0.0)
            is_known = face.get("is_known", False)

            colour = (0, 200, 0) if is_known else (0, 100, 255)
            label = "{} ({:.0f}%)".format(name, conf * 100) if is_known else "?"

            cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)
            lsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated,
                          (x, y - lsz[1] - 8), (x + lsz[0] + 4, y),
                          colour, -1)
            cv2.putText(annotated, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return annotated
