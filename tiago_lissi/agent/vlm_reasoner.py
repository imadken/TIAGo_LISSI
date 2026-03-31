#!/usr/bin/env python3
"""
VLM Reasoning Engine for TIAGo Embodied AI
Eden AI API (OpenAI-compatible) — google/gemini-2.0-flash
Python 3.6+ compatible
"""

import os
import json
import time
import re
import base64
import requests
import cv2
import numpy as np
from typing import Dict, List, Optional, Any


class VLMReasoner:
    _EDEN_URL = "https://api.edenai.run/v3/llm/chat/completions"

    def __init__(self, api_key: Optional[str] = None,
                 model_name: str = "google/gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get("EDENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("EDENAI_API_KEY not set.")
        self.model_name = model_name
        self._headers = {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
        }
        self.conversation_history = []
        self.max_history_turns = 20
        print("[VLM] Initialized {} via Eden AI".format(model_name))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def encode_image(self, image_bgr: np.ndarray, max_size: int = 640,
                     quality: int = 85) -> str:
        """Encode BGR image as base64 JPEG string."""
        h, w = image_bgr.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        _, buffer = cv2.imencode('.jpg', image_bgr,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(buffer).decode('utf-8')

    def _to_pil(self, image_bgr: np.ndarray, max_size: int = 640) -> np.ndarray:
        """Return resized BGR array (kept for call-site compatibility)."""
        h, w = image_bgr.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        return image_bgr

    def _call_api(self, parts: List, temperature: float = 0.1,
                  max_tokens: int = 2048, timeout: int = 30,
                  force_json: bool = False) -> str:
        """Call Eden AI with a list of parts (text strings + BGR numpy arrays)."""
        content = []
        for part in parts:
            if isinstance(part, str):
                content.append({"type": "text", "text": part})
            else:
                # numpy BGR image → base64 JPEG data URI
                b64 = self.encode_image(part)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{}".format(b64)}
                })

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if force_json:
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(self._EDEN_URL, headers=self._headers,
                             json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _clean_json_text(self, text: str) -> str:
        text = re.sub(r'```(?:json)?\s*', '', text).strip()
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)
        text = re.sub(r'\bNone\b', 'null', text)
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return text

    def _extract_json_object(self, text: str) -> Dict:
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            raise json.JSONDecodeError("No JSON object", text, 0)
        raw = text[start:end + 1]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return json.loads(self._clean_json_text(raw))

    def _extract_json_array(self, text: str) -> List:
        start = text.find('[')
        end = text.rfind(']')
        if start == -1 or end == -1:
            return []
        raw = text[start:end + 1]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return json.loads(self._clean_json_text(raw))

    def _parse_box2d_detections(self, raw: List, w_img: int, h_img: int) -> List[Dict]:
        """Convert box_2d normalized detections to pixel [x,y,w,h] dicts."""
        detections = []
        for item in raw:
            box = item.get('box_2d', [])
            label = item.get('label', 'object').lower().strip()
            if len(box) == 4:
                y1, x1, y2, x2 = box
                px1 = int(x1 / 1000.0 * w_img)
                py1 = int(y1 / 1000.0 * h_img)
                px2 = int(x2 / 1000.0 * w_img)
                py2 = int(y2 / 1000.0 * h_img)
                detections.append({
                    'class_name': label,
                    'bbox': [px1, py1, max(1, px2 - px1), max(1, py2 - py1)],
                    'confidence': float(item.get('confidence', 0.9))
                })
        return detections

    # ── Prompt builders ───────────────────────────────────────────────────────

    def build_system_prompt(self, robot_state: Dict[str, Any],
                            available_skills: List[str]) -> str:
        derived = robot_state.get('derived_facts', 'none')
        sem     = robot_state.get('semantic_facts', [])
        sem_str = ', '.join(sem) if sem else 'none'
        pos_3d  = robot_state.get('object_positions_3d', {})
        pos_str = '; '.join(
            '{}: ({:.2f}m fwd, {:.2f}m lat, {:.2f}m up)'.format(k, *v)
            for k, v in pos_3d.items()
        ) if pos_3d else 'none'

        return (
            "You are TIAGo, a mobile manipulator robot.\n\n"
            "**Current State:**\n"
            "- Gripper: {gripper}\n"
            "- Base Location: {location}\n"
            "- Detected Objects: {objects}\n"
            "- 3D Positions (fwd/lat/up from robot): {pos_3d}\n"
            "- Recent Tasks: {tasks}\n"
            "- Object Memory: {memory}\n\n"
            "**Symbolic Scene State (rule engine derived from depth+TF):**\n"
            "{derived}\n\n"
            "**VLM Semantic Facts (previous cycle):**\n"
            "{sem}\n\n"
            "**Available Skills:**\n{skills}\n\n"
            "**Instructions:**\n"
            "- Analyze the image to understand the visual scene.\n"
            "- Choose ONE skill to execute next.\n"
            "- If the object is NOT visible, use search_with_head first.\n"
            "- Only choose grab_object if a graspable object is actually visible.\n"
            "- Do ONLY what the user explicitly asked. Do not add steps they did not request.\n"
            "  Example: 'grasp X' means grab it and stop — do NOT handover unless asked.\n"
            "- Return semantic_facts: visual observations geometry cannot compute\n"
            "  (e.g. bottle_is_open, person_is_reaching, path_is_clear, object_near_edge).\n"
            "- Return spatial_corrections: relations from the rule engine that are WRONG\n"
            "  based on what you see in the image. Format each as 'relation(A,B)' or\n"
            "  'relation(A)'. Only list incorrect ones — omit correct ones.\n"
            "  Example: [\"left_of(bottle,cup)\", \"reachable(phone)\"]\n"
            "- Respond ONLY with valid JSON, no prose:\n"
            "{{\n"
            "  \"skill\": \"skill_name\",\n"
            "  \"parameters\": {{}},\n"
            "  \"rationale\": \"brief explanation\",\n"
            "  \"semantic_facts\": [\"fact1\", \"fact2\"],\n"
            "  \"spatial_corrections\": []\n"
            "}}"
        ).format(
            gripper=robot_state.get('gripper', 'empty'),
            location=robot_state.get('base_location', 'unknown'),
            objects=', '.join(robot_state.get('detected_objects', [])) or 'none',
            tasks=', '.join(robot_state.get('task_history', [])[-3:]) or 'none',
            memory=robot_state.get('object_memory', 'none'),
            skills='\n'.join(available_skills),
            pos_3d=pos_str,
            derived=derived,
            sem=sem_str,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_objects(self, image: np.ndarray,
                       target_classes: List[str] = None) -> List[Dict]:
        """
        Detect objects in image. Returns [{class_name, bbox [x,y,w,h], confidence}].
        """
        h_img, w_img = image.shape[:2]
        pil_img = self._to_pil(image)

        hint = ("Focus on detecting: {}.".format(", ".join(target_classes))
                if target_classes else
                "Detect all objects (people, furniture, everyday items).")

        prompt_text = (
            "{hint} Return bounding boxes as a JSON array. "
            "Limit to 20 objects. "
            "Format: [{{\"box_2d\": [y_min, x_min, y_max, x_max], \"label\": \"name\"}}]. "
            "Coordinates normalized 0-1000. Lowercase labels. "
            "If multiple instances of the same type are visible, use descriptive names "
            "(e.g. 'big bottle', 'small bottle', 'red cup', 'left bottle'). "
            "If nothing found return []."
        ).format(hint=hint)

        try:
            text = self._call_api([prompt_text, pil_img],
                                  temperature=0.1, max_tokens=2048, timeout=15)
            try:
                raw = self._extract_json_array(text)
            except Exception:
                raw = []
                for m in re.finditer(
                    r'"box_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
                    r'.*?"label"\s*:\s*"([^"]+)"', text, re.DOTALL):
                    raw.append({"box_2d": [int(m.group(i)) for i in range(1, 5)],
                                "label": m.group(5)})

            detections = self._parse_box2d_detections(raw, w_img, h_img)
            print("[VLM] Detected {} object(s): {}".format(
                len(detections), [d['class_name'] for d in detections]))
            return detections

        except Exception as e:
            print("[VLM] Detection error: {}".format(e))
            return []

    def query_with_detection(self, user_command: str, image: np.ndarray,
                             robot_state: Dict, available_skills: List[str],
                             retry: int = 3) -> Dict:
        """
        Detect objects AND select skill in one API call.
        Returns {detections, skill, parameters, rationale}.
        """
        h_img, w_img = image.shape[:2]
        pil_img = self._to_pil(image)
        system_prompt = self.build_system_prompt(robot_state, available_skills)

        prompt_text = (
            "{system}\n\n"
            "Look at the image and do TWO things:\n"
            "1. Detect all visible objects with bounding boxes.\n"
            "2. Select the best skill for the user request.\n\n"
            "User request: \"{cmd}\"\n\n"
            "IMPORTANT for labels: If multiple instances of the same object type are visible, "
            "use descriptive labels to distinguish them (e.g. 'big bottle', 'small bottle', "
            "'red cup', 'blue cup', 'left bottle', 'right bottle'). "
            "Use plain generic labels only when there is a single instance.\n\n"
            "Respond ONLY with valid JSON:\n"
            "{{\n"
            "  \"detections\": [{{\"label\": \"name\", \"box_2d\": [y_min, x_min, y_max, x_max]}}, ...],\n"
            "  \"skill\": \"skill_name\",\n"
            "  \"parameters\": {{}},\n"
            "  \"rationale\": \"brief explanation\",\n"
            "  \"semantic_facts\": [\"fact1\", \"fact2\"],\n"
            "  \"spatial_corrections\": []\n"
            "}}\n"
            "box_2d normalized 0-1000. Limit 20 detections. Empty scene: \"detections\": []."
        ).format(system=system_prompt, cmd=user_command)

        for attempt in range(retry):
            try:
                text = self._call_api([prompt_text, pil_img],
                                      temperature=0.1, max_tokens=2048,
                                      timeout=30, force_json=True)
                result = self._extract_json_object(text)
                result['detections'] = self._parse_box2d_detections(
                    result.get('detections', []), w_img, h_img)

                self.conversation_history.append({
                    'user_command': user_command,
                    'response': {'skill': result.get('skill'),
                                 'rationale': result.get('rationale')},
                    'timestamp': time.time()
                })
                if len(self.conversation_history) > self.max_history_turns:
                    self.conversation_history = \
                        self.conversation_history[-self.max_history_turns:]

                print("[VLM] Detected {} object(s): {}".format(
                    len(result['detections']),
                    [d['class_name'] for d in result['detections']]))
                print("[VLM] Selected skill: {} — {}".format(
                    result.get('skill'), result.get('rationale')))
                sem_facts = result.get('semantic_facts', [])
                if sem_facts:
                    print("[VLM] Semantic facts: {}".format(sem_facts))
                corrections = result.get('spatial_corrections', [])
                if corrections:
                    print("[VLM] Spatial corrections: {}".format(corrections))
                return result

            except json.JSONDecodeError as e:
                print("[VLM] Combined query JSON error ({}/{}): {}".format(
                    attempt + 1, retry, e))
                if attempt == retry - 1:
                    return {'detections': [], 'skill': 'go_home',
                            'parameters': {}, 'rationale': 'VLM parse error'}
                time.sleep(1)
            except Exception as e:
                print("[VLM] Combined query API error ({}/{}): {}".format(
                    attempt + 1, retry, e))
                if attempt == retry - 1:
                    raise
                time.sleep(2)

    def query(self, user_command: str, image: np.ndarray,
              detections: List[Dict], robot_state: Dict,
              available_skills: List[str], retry: int = 3) -> Dict:
        """Skill selection from image + detections (kept for compatibility)."""
        system_prompt = self.build_system_prompt(robot_state, available_skills)
        pil_img = self._to_pil(image)

        det_str = ("\n".join([
            "- {} @ bbox({},{},{},{}) conf={:.2f}".format(
                d['class_name'], *d['bbox'], d['confidence'])
            for d in detections]) if detections else "No objects detected")

        prompt_text = (
            "{system}\n\n"
            "User request: \"{cmd}\"\n\n"
            "Detected objects:\n{dets}\n\n"
            "What skill should I execute?"
        ).format(system=system_prompt, cmd=user_command, dets=det_str)

        for attempt in range(retry):
            try:
                text = self._call_api([prompt_text, pil_img],
                                      timeout=30, force_json=True)
                result = self._extract_json_object(text)

                self.conversation_history.append({
                    'user_command': user_command,
                    'response': result,
                    'timestamp': time.time()
                })
                if len(self.conversation_history) > self.max_history_turns:
                    self.conversation_history = \
                        self.conversation_history[-self.max_history_turns:]

                print("[VLM] Selected skill: {} - {}".format(
                    result.get('skill'), result.get('rationale')))
                return result

            except json.JSONDecodeError as e:
                print("[VLM] query JSON error ({}/{}): {}".format(
                    attempt + 1, retry, e))
                if attempt == retry - 1:
                    return {'skill': 'go_home', 'parameters': {},
                            'rationale': 'VLM parse error'}
                time.sleep(1)
            except Exception as e:
                print("[VLM] query API error ({}/{}): {}".format(
                    attempt + 1, retry, e))
                if attempt == retry - 1:
                    raise
                time.sleep(2)

    def verify(self, skill_name: str, parameters: Dict, expected_outcome: str,
               pre_image: np.ndarray, post_image: np.ndarray,
               retry: int = 3) -> Dict:
        """Verify skill execution by comparing before/after images."""
        param_str = ', '.join('{}={}'.format(k, v) for k, v in parameters.items())
        prompt_text = (
            "I executed: {skill}({params})\n"
            "Expected: {outcome}\n\n"
            "Compare the BEFORE image and AFTER image.\n"
            "Respond ONLY with valid JSON:\n"
            "{{\"success\": true/false, \"observation\": \"...\", \"issue\": \"...\"}}"
        ).format(skill=skill_name, params=param_str, outcome=expected_outcome)

        pre_pil = self._to_pil(pre_image)
        post_pil = self._to_pil(post_image)

        for attempt in range(retry):
            try:
                text = self._call_api(
                    ["BEFORE image:", pre_pil, "AFTER image:", post_pil, prompt_text],
                    timeout=30, force_json=True)
                result = self._extract_json_object(text)
                print("[VLM] Verify: {} — {}".format(
                    'OK' if result.get('success') else 'FAIL',
                    result.get('observation', '')))
                return result
            except json.JSONDecodeError:
                if attempt == retry - 1:
                    return {'success': True,
                            'observation': 'Could not verify', 'issue': ''}
                time.sleep(1)
            except Exception as e:
                if attempt == retry - 1:
                    return {'success': True,
                            'observation': 'Verify error', 'issue': str(e)}
                time.sleep(2)

    def extract_name(self, speech_text: str) -> Optional[str]:
        """Extract a person's first name from free-form speech."""
        try:
            text = self._call_api(
                ["Extract the person's first name from this speech. "
                 "Reply with ONLY the first name, nothing else. "
                 "If no name, reply UNKNOWN.\n\nSpeech: \"{}\"".format(speech_text)],
                temperature=0.0, max_tokens=16, timeout=10)
            name = text.strip('"\'').strip()
            if name.upper() == 'UNKNOWN' or not name.isalpha():
                return None
            return name.capitalize()
        except Exception:
            return None


if __name__ == '__main__':
    import sys
    reasoner = VLMReasoner()
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "Test", (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    dets = reasoner.detect_objects(test_image)
    print("Detections:", dets)
