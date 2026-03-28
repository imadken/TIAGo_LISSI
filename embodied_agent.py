#!/usr/bin/env python3
"""
Embodied Agent - VILA-style Vision-Language-Action System for TIAGo
Main control loop integrating VLM reasoning, perception, and skills
WITH COMPREHENSIVE SAFETY FEATURES
"""

import os
import sys
import json
import signal
import numpy as np
import cv2
import rospy
import actionlib
import speech_recognition as sr
from typing import Dict, List, Any
from sensor_msgs.msg import Image
from std_msgs.msg import String

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pal_interaction_msgs.msg import TtsAction, TtsGoal

from vlm_reasoner import VLMReasoner
from perception_manager_v2 import PerceptionManager
from state_manager import StateManager
from face_manager import FaceManager
from skills.base_skill import BaseSkill
from skills.grab_bottle import GrabBottleSkill  # file kept as grab_bottle.py, skill name is grab_object
from skills.simple_motions import GoHomeSkill, WaveSkill, OpenHandSkill, CloseHandSkill
from skills.search_with_head import SearchWithHeadSkill
from skills.handover import HandoverSkill
from skills.push_aside import PushAsideSkill


class EmbodiedAgent:
    def __init__(self):
        """Initialize embodied agent with all components and safety features."""
        rospy.init_node('embodied_agent', anonymous=True)

        # Safety state tracking
        self.safe_mode = False  # True if in emergency/error state
        self.consecutive_failures = 0  # Track consecutive skill failures
        self.max_consecutive_failures = 3  # Trigger safe mode after this many failures

        rospy.loginfo("=" * 60)
        rospy.loginfo("VILA Embodied Agent Initializing...")
        rospy.loginfo("=" * 60)

        # Initialize VLM reasoner
        rospy.loginfo("[Agent] Initializing VLM reasoner...")
        self.vlm = VLMReasoner()

        # Initialize face recognition (optional — gracefully disabled if service is down)
        rospy.loginfo("[Agent] Initializing face recognition...")
        self.face_manager = FaceManager()
        # Track persons seen this session to avoid repeated greetings
        self._greeted_this_session = set()

        # Initialize perception (YOLO only — CLIP not used in detect_objects())
        rospy.loginfo("[Agent] Initializing perception...")
        self.perception = PerceptionManager(use_clip=False,
                                            face_manager=self.face_manager)

        # Initialize state manager
        rospy.loginfo("[Agent] Initializing state manager...")
        self.state = StateManager()

        # Initialize skill library
        rospy.loginfo("[Agent] Loading skills...")
        self.skills = {}
        self._load_skills()

        rospy.loginfo("[Agent] Ready! Available skills: {}".format(list(self.skills.keys())))
        rospy.loginfo("=" * 60)

        # Initialize TTS
        rospy.loginfo("[Agent] Connecting to TTS server...")
        self.tts_client = actionlib.SimpleActionClient('/tts', TtsAction)
        tts_ok = self.tts_client.wait_for_server(timeout=rospy.Duration(10))
        if tts_ok:
            rospy.loginfo("[Agent] TTS ready.")
        else:
            rospy.logwarn("[Agent] TTS server not found — speech output disabled.")
            self.tts_client = None

        # Initialize STT
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.mic_index = self._find_microphone()

        # Live visualisation publishers
        self.vis_image_pub = rospy.Publisher('/agent/camera_annotated', Image, queue_size=1)
        self.vis_status_pub = rospy.Publisher('/agent/status', String, queue_size=1)
        self._current_skill = 'idle'

        # Register safety shutdown handlers
        self._register_shutdown_handlers()

    def _publish_visualization(self, rgb_image, detections, current_skill):
        """Publish annotated camera image and status JSON for the live visualizer."""
        try:
            # Draw YOLO boxes on a copy of the frame
            annotated = self.perception.draw_detections(rgb_image.copy(), detections)

            # Overlay current skill name
            label = "Skill: {}".format(current_skill)
            cv2.rectangle(annotated, (0, 0), (len(label) * 11 + 10, 28), (0, 0, 0), -1)
            cv2.putText(annotated, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Publish as ROS Image (BGR8)
            h, w = annotated.shape[:2]
            msg = Image()
            msg.header.stamp = rospy.Time.now()
            msg.height = h
            msg.width = w
            msg.encoding = 'bgr8'
            msg.step = w * 3
            msg.data = annotated.tobytes()
            self.vis_image_pub.publish(msg)
        except Exception as e:
            rospy.logwarn("[Vis] Image publish failed: {}".format(e))

        try:
            state = self.state.get_state()
            status = {
                'skill': current_skill,
                'gripper': state.get('gripper', 'unknown'),
                'location': state.get('base_location', 'unknown'),
                'detected': state.get('detected_objects', []),
                'history': state.get('task_history', [])[-5:],  # last 5 tasks
            }
            self.vis_status_pub.publish(String(data=json.dumps(status)))
        except Exception as e:
            rospy.logwarn("[Vis] Status publish failed: {}".format(e))

    def _find_microphone(self):
        """Find a suitable microphone index."""
        # When PulseAudio is active, use the default device (None) — raw ALSA hw:x,x conflicts
        if os.environ.get('PULSE_SERVER'):
            rospy.loginfo("[Agent] PulseAudio detected — using default input device.")
            return None
        try:
            names = sr.Microphone.list_microphone_names()
            for i, name in enumerate(names):
                if any(k in name for k in ('Andrea', 'USB Audio', 'HDA Intel PCH: ALC')):
                    rospy.loginfo("[Agent] Microphone: {}".format(name))
                    return i
        except Exception:
            pass
        rospy.logwarn("[Agent] Using default microphone.")
        return None

    def speak(self, text):
        """Speak text via PAL TTS. Also prints to console."""
        rospy.loginfo("TIAGo: {}".format(text))
        if self.tts_client is None:
            return
        try:
            goal = TtsGoal()
            goal.rawtext.text = text
            goal.rawtext.lang_id = "en_GB"
            self.tts_client.send_goal(goal)
            self.tts_client.wait_for_result(timeout=rospy.Duration(30))
        except Exception as e:
            rospy.logwarn("[Agent] TTS error: {}".format(e))

    def listen(self, timeout=8):
        """Listen for voice input. Returns text or None."""
        try:
            with sr.Microphone(device_index=self.mic_index) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                rospy.loginfo("[Agent] Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
            try:
                text = self.recognizer.recognize_whisper(audio, model="base", language="english")
            except Exception:
                text = self.recognizer.recognize_google(audio, language="en-US")
            text = text.strip()
            if text:
                rospy.loginfo("[Agent] Heard: {}".format(text))
                return text
        except sr.WaitTimeoutError:
            pass
        except Exception as e:
            rospy.logwarn("[Agent] STT error: {}".format(e))
        return None

    # ── Spatial correction from VLM ────────────────────────────────────────────

    def _apply_spatial_corrections(self, corrections):
        """Parse VLM spatial_corrections list and retract wrong facts from rule engine.

        Accepts strings like 'left_of(bottle,cup)' or 'reachable(bottle)'.
        """
        import re
        engine = self.state._rule_engine
        for raw in corrections:
            m = re.match(r'(\w+)\(([^)]+)\)', raw.strip())
            if not m:
                continue
            relation = m.group(1)
            args = [a.strip() for a in m.group(2).split(',')]
            # Skip self-referential corrections (e.g. left_of(bottle,bottle)) — meaningless
            if len(args) >= 2 and args[0] == args[1]:
                continue
            engine.retract_fact(relation, *args)
            rospy.loginfo("[Agent] Spatial correction: retracted {}{}".format(
                relation, tuple(args)))

    # ── Face recognition interactions ──────────────────────────────────────────

    def handle_person_detections(self, detections, rgb_image):
        """
        Called after every detection pass.
        - Greets known persons (once per session).
        - Asks unknown persons to identify themselves.
        - Registers them if they consent.

        Returns a dict: {person_name: str or None} for the most prominent person.
        """
        person_dets = [d for d in detections if d.get('class_name') == 'person']
        if not person_dets or not self.face_manager.is_available():
            return {}

        # Use the largest (closest) person detection
        person_dets_sorted = sorted(
            person_dets, key=lambda d: d['bbox'][2] * d['bbox'][3], reverse=True)
        primary = person_dets_sorted[0]

        name = primary.get('name', 'unknown')
        is_known = primary.get('is_known', False)

        if is_known and name not in self._greeted_this_session:
            # Greet known person once per session
            self._greeted_this_session.add(name)
            rospy.loginfo("[Face] Recognised: {}".format(name))
            self.speak("Hello {}! Good to see you again.".format(name))
            return {'person_name': name}

        if not is_known:
            # Ask unknown person to identify themselves
            identified_name = self._identify_unknown_person(rgb_image, primary)
            if identified_name:
                return {'person_name': identified_name}

        return {'person_name': name if is_known else None}

    def _identify_unknown_person(self, rgb_image, person_det):
        """
        Dialogue flow for an unknown person:
          1. Ask for their name.
          2. Listen for reply.
          3. Ask if they want to be remembered.
          4. Register face if yes.
        Returns the identified name, or None if identification failed.
        """
        self.speak("Hello! I don't think we have met before. What is your name?")
        name_response = self.listen(timeout=8)

        if not name_response:
            self.speak("I could not hear you. I will just call you friend for now.")
            return None

        # Use LLM to extract the name from free-form speech
        name = self.vlm.extract_name(name_response)
        if not name:
            # Simple fallback: last word (avoids "my", "I", "am")
            words = [w for w in name_response.split() if w.isalpha() and len(w) > 1]
            name = words[-1].capitalize() if words else None
        if not name:
            return None

        self.speak("Nice to meet you, {}!".format(name))

        # Ask if they want to be remembered
        self.speak("Should I remember your face for next time? Say yes or no.")
        confirm = self.listen(timeout=6)

        if confirm and any(w in confirm.lower() for w in ('yes', 'sure', 'please', 'ok', 'yeah')):
            self.speak("Great! Let me take a look at you.")
            rospy.sleep(0.5)
            # Capture a fresh frame for registration (better than re-using old one)
            fresh = self.perception.get_latest_rgb()
            image_to_register = fresh if fresh is not None else rgb_image
            success = self.face_manager.register(image_to_register, name)
            if success:
                self._greeted_this_session.add(name)
                self.speak("Done! I will recognise you next time, {}.".format(name))
            else:
                self.speak("Sorry, I could not capture your face clearly. "
                           "Please make sure your face is visible.")
        else:
            self.speak("No problem, {}. I will remember you just for this session.".format(name))
            self._greeted_this_session.add(name)

        return name

    def find_person_by_name(self, target_name, detections):
        """
        Given a target name (from user command like 'give it to Alice'),
        return the detection dict of the matching person or None.
        """
        target_lower = target_name.lower()
        for det in detections:
            if det.get('class_name') == 'person':
                det_name = det.get('name', '').lower()
                if det_name and target_lower in det_name:
                    return det
        return None

    # ── Shutdown handlers ───────────────────────────────────────────────────

    def _register_shutdown_handlers(self):
        """Register signal handlers for safe shutdown."""
        # Handle ROS shutdown
        rospy.on_shutdown(self._safe_shutdown)

        # Handle SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        rospy.loginfo("[Agent] Safety shutdown handlers registered")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals (SIGINT, SIGTERM)."""
        rospy.logwarn("\n[SAFETY] Shutdown signal received (signal {})".format(signum))
        rospy.signal_shutdown("User interrupt")

    def _safe_shutdown(self):
        """
        Execute safe shutdown sequence.
        Called automatically on ROS shutdown or program exit.
        """
        rospy.logwarn("=" * 60)
        rospy.logwarn("[SAFETY] INITIATING SAFE SHUTDOWN SEQUENCE")
        rospy.logwarn("=" * 60)

        try:
            # 1. Stop any ongoing movements
            rospy.loginfo("[SAFETY] Stopping any ongoing movements...")

            # 2. Return to home position
            rospy.loginfo("[SAFETY] Returning to home position...")
            if 'go_home' in self.skills:
                try:
                    home_skill = self.skills['go_home']
                    success = home_skill.execute({})
                    if success:
                        rospy.loginfo("[SAFETY] Robot successfully returned to home position")
                    else:
                        rospy.logwarn("[SAFETY] Failed to return to home position")
                except Exception as e:
                    rospy.logerr("[SAFETY] Error during go_home: {}".format(e))

            # 3. Open gripper to release any held objects
            rospy.loginfo("[SAFETY] Opening gripper...")
            if 'open_hand' in self.skills:
                try:
                    open_skill = self.skills['open_hand']
                    open_skill.execute({})
                    rospy.loginfo("[SAFETY] Gripper opened")
                except Exception as e:
                    rospy.logerr("[SAFETY] Error opening gripper: {}".format(e))

            rospy.loginfo("[SAFETY] Shutdown sequence complete")

        except Exception as e:
            rospy.logerr("[SAFETY] Error during shutdown sequence: {}".format(e))

        rospy.logwarn("=" * 60)
        rospy.logwarn("[SAFETY] Robot is now safe to power off")
        rospy.logwarn("=" * 60)

    def _ensure_safe_pose(self, reason="error"):
        """
        Ensure robot is in a safe pose after errors.

        Args:
            reason: Why safe pose is needed (for logging)
        """
        rospy.logwarn("[SAFETY] Ensuring safe pose (reason: {})...".format(reason))

        try:
            # Open gripper first (in case holding something)
            if 'open_hand' in self.skills:
                rospy.loginfo("[SAFETY] Opening gripper...")
                self.skills['open_hand'].execute({})

            # Return to home position
            if 'go_home' in self.skills:
                rospy.loginfo("[SAFETY] Moving to home position...")
                success = self.skills['go_home'].execute({})
                if success:
                    rospy.loginfo("[SAFETY] Robot in safe pose")
                    return True
                else:
                    rospy.logwarn("[SAFETY] Failed to reach safe pose")
                    return False

        except Exception as e:
            rospy.logerr("[SAFETY] Error while ensuring safe pose: {}".format(e))
            return False

    def _enter_safe_mode(self, reason=""):
        """
        Enter safe mode - stop all operations and return to safe pose.

        Args:
            reason: Why safe mode was triggered
        """
        if self.safe_mode:
            return  # Already in safe mode

        rospy.logerr("=" * 60)
        rospy.logerr("[SAFETY] ENTERING SAFE MODE")
        rospy.logerr("Reason: {}".format(reason))
        rospy.logerr("=" * 60)

        self.safe_mode = True
        self._ensure_safe_pose(reason="safe_mode_entry")

        rospy.logwarn("[SAFETY] Safe mode active. Send 'reset' command to exit.")

    def _exit_safe_mode(self):
        """Exit safe mode and resume normal operation."""
        rospy.loginfo("[SAFETY] Exiting safe mode...")
        self.safe_mode = False
        self.consecutive_failures = 0
        rospy.loginfo("[SAFETY] Normal operation resumed")

    def _nav_available(self):
        """Check if move_base is available for navigation."""
        try:
            client = actionlib.SimpleActionClient('/move_base', __import__('move_base_msgs.msg', fromlist=['MoveBaseAction']).MoveBaseAction)
            return client.wait_for_server(timeout=rospy.Duration(2))
        except Exception:
            return False

    def _load_skills(self):
        """Load all available skills."""
        # Manipulation skills
        self.skills['grab_object'] = GrabBottleSkill()

        # Motion skills
        self.skills['go_home'] = GoHomeSkill()
        self.skills['wave'] = WaveSkill()
        self.skills['open_hand'] = OpenHandSkill()
        self.skills['close_hand'] = CloseHandSkill()

        # Perception skills (reuse the agent's already-loaded PerceptionManager)
        self.skills['search_with_head'] = SearchWithHeadSkill(
            vlm=self.vlm,
            face_manager=self.face_manager,
            perception_manager=self.perception,
            state_manager=self.state
        )

        # Delivery skills
        self.skills['handover_to_person'] = HandoverSkill(
            perception_manager=self.perception,
            tts_speak_fn=self.speak,
            vlm_reasoner=self.vlm
        )

        # Manipulation — push blocking objects aside
        self.skills['push_aside_object'] = PushAsideSkill(
            vlm=self.vlm,
            perception_manager=self.perception
        )

        rospy.loginfo("[Agent] Loaded {} skills".format(len(self.skills)))

    def get_skill_descriptions(self) -> List[str]:
        """Get formatted skill descriptions for VLM prompt."""
        return [skill.get_description() for skill in self.skills.values()]

    def execute_command(self, user_command: str, max_steps: int = 5) -> Dict[str, Any]:
        """
        Execute a user command using VLM reasoning and skill execution.
        Supports multi-step tasks by iterating until task is complete.
        WITH SAFETY FEATURES: error recovery, safe pose management, failure tracking.

        Args:
            user_command: Natural language command
            max_steps: Maximum number of skills to execute (prevents infinite loops)

        Returns:
            Dict with execution results
        """
        # Handle special safety commands
        if user_command.lower() in ['reset', 'exit safe mode', 'resume']:
            if self.safe_mode:
                self._exit_safe_mode()
                return {'success': True, 'message': 'Safe mode exited'}
            else:
                return {'success': True, 'message': 'Not in safe mode'}

        # Check if in safe mode
        if self.safe_mode:
            rospy.logwarn("[SAFETY] Cannot execute commands in safe mode. Send 'reset' to exit.")
            return {
                'success': False,
                'error': 'Robot in safe mode. Send \"reset\" command to exit safe mode.'
            }

        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("USER COMMAND: '{}'".format(user_command))
        rospy.loginfo("=" * 60)

        executed_skills = []
        original_command = user_command
        last_affordance_failure = None  # Passed back to VLM when affordance check fails

        # Multi-step loop: continue until task is complete or max steps reached
        for step in range(max_steps):
            if step > 0:
                rospy.loginfo("\n" + "-" * 60)
                rospy.loginfo("STEP {}/{}: Checking if more actions needed...".format(step + 1, max_steps))
                rospy.loginfo("-" * 60)
                self.speak("Let me check if there is more to do.")

            # Step 1: Capture current scene
            rospy.loginfo("[Agent] Capturing scene...")
            self.speak("Looking at the scene.")
            rgb_image = self.perception.get_latest_rgb()
            if rgb_image is None:
                rospy.logerr("[Agent] No camera image available")
                self.speak("I cannot see anything. My camera is not working.")
                return {'success': False, 'error': 'No camera image'}

            # Step 2: Get current robot state (needed for VLM prompt)
            robot_state = self.state.get_state()

            # Build query prompt based on progress
            if last_affordance_failure:
                query_prompt = "Original task: '{}'\nCompleted so far: {}\nCould NOT execute '{}' because: {}\nChoose a different skill to make progress (e.g. search_with_head if object not visible).".format(
                    original_command, ', '.join(executed_skills) or 'nothing yet',
                    last_affordance_failure['skill'], last_affordance_failure['reason'])
                last_affordance_failure = None
            elif executed_skills:
                gripper_state = robot_state.get('gripper', 'empty')
                query_prompt = (
                    "Original task: '{}'\n"
                    "Completed so far: {}\n"
                    "Gripper: {}\n\n"
                    "IMPORTANT: Only do what the original task explicitly asks for. "
                    "Do NOT add extra steps the user did not request (e.g. if the task "
                    "is 'grasp X', stop after grasping — do not handover unless asked). "
                    "If the original task is FULLY complete, respond with skill='done'."
                ).format(original_command, ', '.join(executed_skills), gripper_state)
            else:
                query_prompt = user_command

            # Detect if stuck in loop (same skill 2x in a row = likely done)
            if len(executed_skills) >= 2 and executed_skills[-1] == executed_skills[-2]:
                rospy.loginfo("[Agent] Detected completion: same skill executed twice")
                break

            # Step 3: Single VLM call — detect objects AND select skill simultaneously
            rospy.loginfo("[Agent] Querying VLM (detect + reason in one call)...")
            self.speak("Thinking about what to do.")
            try:
                vlm_response = self.vlm.query_with_detection(
                    user_command=query_prompt,
                    image=rgb_image,
                    robot_state=robot_state,
                    available_skills=self.get_skill_descriptions()
                )
            except Exception as e:
                rospy.logerr("[Agent] VLM query failed: {}".format(e))
                self.speak("My reasoning system failed. I cannot process this request.")
                return {'success': False, 'error': 'VLM query failed: {}'.format(str(e)), 'executed_skills': executed_skills}

            # Extract results from combined response
            detections = vlm_response.get('detections', [])
            skill_name = vlm_response.get('skill')
            parameters = vlm_response.get('parameters', {})
            rationale = vlm_response.get('rationale', '')

            # Face enrichment for any person detections
            detections = self.perception.enrich_with_faces(detections, rgb_image)

            # 3D position enrichment — project each VLM bbox into base_footprint via depth
            for det in detections:
                if det.get('position_3d') is None and det.get('bbox'):
                    pos, frame, n_pts = self.perception.get_3d_position(det['bbox'])
                    det['position_3d'] = pos
                    det['position_frame'] = frame
                    det['n_depth_pts'] = n_pts

            # Announce what we see
            if detections:
                obj_labels = []
                for d in detections:
                    if d['class_name'] == 'person' and d.get('is_known'):
                        obj_labels.append(d.get('name', 'person'))
                    else:
                        obj_labels.append(d['class_name'])
                self.speak("I can see: {}.".format(', '.join(set(obj_labels))))
            else:
                self.speak("I do not see any objects.")

            # Face recognition interaction (greet/identify persons)
            face_context = self.handle_person_detections(detections, rgb_image)

            # Update state with detections + merge VLM semantic facts into rule engine
            self.state.update_detected_objects(detections)
            semantic_facts = vlm_response.get('semantic_facts', [])
            if semantic_facts:
                self.state.merge_semantic_facts(semantic_facts)

            # Apply VLM spatial corrections — retract geometric facts the VLM sees as wrong
            spatial_corrections = vlm_response.get('spatial_corrections', [])
            if spatial_corrections:
                self._apply_spatial_corrections(spatial_corrections)
            if face_context.get('person_name'):
                self.state.set_recognized_person(face_context['person_name'])

            # Refresh state after detection update
            robot_state = self.state.get_state()
            rospy.loginfo("[Agent] Current state: {}".format(robot_state))

            # Publish live visualisation
            self._publish_visualization(rgb_image, detections, self._current_skill)

            # Speak the rationale (truncated for speech)
            if rationale:
                self.speak(rationale[:200])

            # Check if task is complete
            if skill_name and skill_name.lower() in ['done', 'complete', 'finished']:
                rospy.loginfo("[Agent] VLM reports task complete: {}".format(rationale))
                rospy.loginfo("=" * 60)
                rospy.loginfo("TASK COMPLETE: Executed {} skills: {}".format(
                    len(executed_skills), executed_skills))
                rospy.loginfo("=" * 60 + "\n")
                self.speak("Task complete!")
                return {
                    'success': True,
                    'completed': True,
                    'executed_skills': executed_skills,
                    'rationale': rationale
                }

            rospy.loginfo("[Agent] VLM selected: {} - {}".format(skill_name, rationale))

            # Step 5: Validate skill exists
            if skill_name not in self.skills:
                rospy.logerr("[Agent] Unknown skill: {}".format(skill_name))
                rospy.loginfo("[Agent] Available skills: {}".format(list(self.skills.keys())))
                self.speak("I do not know how to do: {}.".format(skill_name))
                return {'success': False, 'error': 'Unknown skill: {}'.format(skill_name), 'executed_skills': executed_skills}

            skill = self.skills[skill_name]

            # Ensure grab_object always has target_object — infer from detections if VLM omitted it
            # Also pass the first-VLM rationale as a description so reach_object_v5 knows
            # exactly which object to look for during its own VLM detection pass.
            if skill_name == 'grab_object' and detections:
                graspable_kw = ('bottle', 'cup', 'mug', 'can', 'phone', 'box', 'book', 'remote',
                                'case', 'trunk', 'container', 'object', 'item')
                target_name = parameters.get('target_object', '')
                matched = [d for d in detections
                           if target_name and target_name.lower() in d['class_name'].lower()]
                if not matched:
                    matched = [d for d in detections
                               if any(kw in d['class_name'].lower() for kw in graspable_kw)]
                if matched:
                    best = max(matched, key=lambda d: d['confidence'])
                    if not target_name:
                        parameters['target_object'] = best['class_name']
                        rospy.loginfo("[Agent] Inferred target_object='{}' from detections".format(
                            best['class_name']))

                    # Build rich description: rationale + ontology spatial context
                    target_cls = parameters.get('target_object', best['class_name'])
                    relations = self.state.get_spatial_relations(target_cls)
                    memory = self.state.get_object_memory(target_cls)
                    desc_parts = [rationale]
                    if relations:
                        desc_parts.append('Spatial context: {}'.format(
                            ', '.join(relations[:4])))
                    if memory and memory.get('seen_count', 0) > 1:
                        desc_parts.append('Seen {} time(s) in this session'.format(
                            memory['seen_count']))
                    parameters['target_description'] = ' | '.join(desc_parts)
                    rospy.loginfo("[Agent] target_description: {}".format(
                        parameters['target_description'][:120]))

                    # Pass other detected objects as obstacles for MoveIt collision avoidance
                    target_cls = parameters.get('target_object', '')
                    obstacles = [
                        {'name': d['class_name'],
                         'position_3d': list(d['position_3d']) if d.get('position_3d') else None}
                        for d in detections
                        if target_cls.lower() not in d['class_name'].lower()
                        and d.get('position_3d') is not None
                    ]
                    if obstacles:
                        import json
                        parameters['obstacle_objects'] = json.dumps(obstacles)
                        rospy.loginfo("[Agent] Passing {} obstacle(s) to grasp script".format(
                            len(obstacles)))

            # Enrich push_aside_object with blocking object's 3D position from current detections
            if skill_name == 'push_aside_object' and detections:
                blocking = parameters.get('blocking_object', '')
                if blocking:
                    matched = [d for d in detections
                               if blocking.lower() in d['class_name'].lower()
                               and d.get('position_3d') is not None]
                    if matched:
                        parameters['position_3d'] = matched[0]['position_3d']
                        rospy.loginfo("[Agent] push_aside: resolved '{}' 3D pos {}".format(
                            blocking, [round(v, 3) for v in matched[0]['position_3d']]))

            # Step 6: Check affordance
            rospy.loginfo("[Agent] Checking affordance for {}...".format(skill_name))
            can_execute, reason = skill.check_affordance(parameters, robot_state)

            if not can_execute:
                rospy.logwarn("[Agent] Cannot execute {}: {}".format(skill_name, reason))
                self.speak("I cannot do that right now. Let me think of another way.")
                last_affordance_failure = {'skill': skill_name, 'reason': reason}
                continue  # Loop back — VLM will pick a different skill with failure context

            rospy.loginfo("[Agent] Affordance check passed: {}".format(reason))

            # Step 7: Execute skill with safety wrapper
            skill_label = skill_name.replace('_', ' ')
            self.speak("Executing: {}.".format(skill_label))
            rospy.loginfo("[Agent] Executing skill: {}".format(skill_name))
            rospy.loginfo("-" * 60)
            self._current_skill = skill_name

            success = False
            error_msg = None

            try:
                success = skill.execute(parameters)
            except Exception as e:
                rospy.logerr("[Agent] Skill execution exception: {}".format(e))
                error_msg = "Execution exception: {}".format(str(e))
                success = False

            rospy.loginfo("-" * 60)

            # Handle skill failure with safety measures
            if not success:
                rospy.logwarn("[Agent] Skill '{}' failed".format(skill_name))
                self.state.add_task(skill_name, success=False)
                self.consecutive_failures += 1
                self.speak("{} failed. Returning to safe position.".format(skill_label))

                # Check if we should enter safe mode
                if self.consecutive_failures >= self.max_consecutive_failures:
                    self._enter_safe_mode(
                        reason="{} consecutive failures".format(self.consecutive_failures)
                    )
                    self.speak("Too many failures. Entering safe mode.")
                    return {
                        'success': False,
                        'skill': skill_name,
                        'error': 'Entered safe mode after {} failures'.format(self.consecutive_failures),
                        'executed_skills': executed_skills,
                        'safe_mode': True
                    }

                # Attempt recovery: ensure robot is in safe pose
                rospy.logwarn("[SAFETY] Skill failed, ensuring safe pose...")
                self._ensure_safe_pose(reason="skill_failure_{}".format(skill_name))

                return {
                    'success': False,
                    'skill': skill_name,
                    'error': error_msg or 'Skill returned failure',
                    'executed_skills': executed_skills,
                    'consecutive_failures': self.consecutive_failures
                }

            # Step 8: Post-execution success handling
            rospy.loginfo("[Agent] Skill '{}' executed successfully".format(skill_name))
            self.speak("{} done.".format(skill_label))

            # Reset failure counter on success
            self.consecutive_failures = 0

            # Update state based on skill
            if skill_name == 'grab_object' and success:
                target = self.skills['grab_object'].last_target if hasattr(self.skills['grab_object'], 'last_target') else 'object'
                self.state.update_gripper_status('holding:{}'.format(target))
            elif skill_name == 'open_hand' and success:
                self.state.update_gripper_status('empty')

            self.state.add_task(skill_name, success=True)
            executed_skills.append(skill_name)

            self._current_skill = 'idle'
            rospy.loginfo("[Agent] Step complete: {} succeeded".format(skill_name))

        # End of loop - max steps reached
        rospy.loginfo("=" * 60)
        rospy.loginfo("EXECUTION COMPLETE: Executed {} skills: {}".format(
            len(executed_skills), executed_skills))
        rospy.loginfo("=" * 60 + "\n")

        return {
            'success': True,
            'executed_skills': executed_skills,
            'note': 'Reached max steps ({})'.format(max_steps) if len(executed_skills) == max_steps else 'Task complete'
        }

    def run_interactive(self):
        """Run interactive interface — type or speak commands."""
        print("\n" + "=" * 60)
        print("EMBODIED AGENT - TIAGo Voice + Text Interface")
        print("=" * 60)
        print("  [v] Voice mode  — speak your command")
        print("  [t] Text mode   — type your command  (default)")
        print("  Type 'quit' or Ctrl+C to exit")
        print("  Type 'reset' to exit safe mode")
        print("=" * 60)

        mode = input("\nChoose mode [v/t, default=t]: ").strip().lower()
        voice_mode = (mode == 'v')

        if voice_mode:
            print("Voice mode active. Speak after the prompt.")
            self.speak("Hello! I am TIAGo. I am ready for your commands.")
        else:
            print("Text mode active. Type your commands.")

        while not rospy.is_shutdown():
            try:
                if self.safe_mode:
                    print("\n[SAFE MODE] Type 'reset' to resume.")

                if voice_mode:
                    print("\n[Speak now...]")
                    user_input = self.listen(timeout=10)
                    if user_input is None:
                        print("(nothing heard, try again)")
                        continue
                    print("You said: {}".format(user_input))
                else:
                    prompt = "[SAFE MODE] > " if self.safe_mode else "> "
                    user_input = input("\n{}".format(prompt)).strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.speak("Goodbye!")
                    break

                # Toggle mode mid-session
                if user_input.lower() in ['voice', 'voice mode', 'speak']:
                    voice_mode = True
                    self.speak("Switching to voice mode.")
                    continue
                if user_input.lower() in ['text', 'text mode', 'type']:
                    voice_mode = False
                    print("Switched to text mode.")
                    continue

                # Execute command
                result = self.execute_command(user_input)

                if result['success']:
                    msg = result.get('message')
                    if not msg:
                        skills = result.get('executed_skills', [])
                        msg = "Done! Executed: {}".format(', '.join(skills)) if skills else "Done!"
                    print("\n OK: {}".format(msg))
                    self.speak(msg)
                else:
                    err = result.get('error', 'Unknown error')
                    print("\n Failed: {}".format(err))
                    self.speak("I could not do that. {}".format(err[:80]))
                    if result.get('safe_mode'):
                        self.speak("I am in safe mode. Say reset to resume.")

            except KeyboardInterrupt:
                rospy.loginfo("\nShutting down...")
                break
            except Exception as e:
                rospy.logerr("Error: {}".format(e))
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    self._enter_safe_mode(reason="unexpected_error: {}".format(str(e)))


def main():
    """Main entry point."""
    try:
        agent = EmbodiedAgent()
        agent.run_interactive()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Fatal error: {}".format(e))
        raise


if __name__ == '__main__':
    main()
