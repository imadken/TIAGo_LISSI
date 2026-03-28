#!/usr/bin/env python3
"""
TIAGo LLM Voice Interface
Microphone → Whisper STT → Groq API (tool use) → PAL TTS

Actions the LLM can trigger:
  - grab_bottle  → runs reach_object.py
  - go_home      → play_motion('home')
  - wave         → play_motion('wave')
  - open_hand    → play_motion('open')
  - close_hand   → play_motion('close')

Requirements (install in Docker):
  pip install SpeechRecognition pyaudio requests
  # requests is almost always pre-installed; groq/openai packages need Python>=3.7
"""

import os
import json
import subprocess
import rospy
import actionlib
import speech_recognition as sr
import requests

from pal_interaction_msgs.msg import TtsAction, TtsGoal
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal

# ── Configuration ─────────────────────────────────────────────────────────────
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL         = "llama-3.3-70b-versatile"   # best tool-use model on Groq
REACH_OBJECT_SCRIPT = "/workspace/reach_object.py"

SYSTEM_PROMPT = (
    "You are TIAGo, a friendly service robot. You speak naturally and concisely — "
    "your responses are played out loud, so keep them short (1-3 sentences). "
    "You can perform physical actions when asked. When you start an action, tell the "
    "user what you are doing before doing it. After an action completes, report briefly."
)

# Groq uses OpenAI-compatible tool format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "grab_bottle",
            "description": (
                "Grab the bottle placed in front of the robot using the arm and gripper. "
                "Use when the user asks to grab, fetch, pick up, or take the bottle."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go_home",
            "description": "Move the robot arm to the home or rest position.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wave",
            "description": "Wave at the person as a greeting gesture.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_hand",
            "description": "Open the robot's gripper hand.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_hand",
            "description": "Close the robot's gripper hand.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class RobotTalker:
    def __init__(self):
        rospy.init_node('robot_talker', anonymous=True)

        # ── PAL TTS ───────────────────────────────────────────────────────────
        rospy.loginfo("Connecting to TTS server...")
        self.tts_client = actionlib.SimpleActionClient('/tts', TtsAction)
        self.tts_client.wait_for_server(timeout=rospy.Duration(15))
        rospy.loginfo("TTS ready.")

        # ── play_motion (for simple robot motions) ────────────────────────────
        self.play_motion_client = actionlib.SimpleActionClient('/play_motion', PlayMotionAction)
        self.play_motion_client.wait_for_server(timeout=rospy.Duration(10))

        # ── Whisper STT ───────────────────────────────────────────────────────
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.mic_index = self._find_microphone()

        # ── Groq LLM (via requests — works on Python 3.6) ────────────────────
        if not GROQ_API_KEY:
            rospy.logwarn("GROQ_API_KEY not set — set it before running.")
        self.groq_headers = {
            "Authorization": "Bearer {}".format(GROQ_API_KEY),
            "Content-Type": "application/json",
        }
        self.conversation = []   # rolling conversation history

        rospy.loginfo("Robot talker ready — model: {}".format(GROQ_MODEL))

    # ── Microphone discovery ──────────────────────────────────────────────────
    def _find_microphone(self):
        """Return the speech_recognition device index for the microphone."""
        if os.environ.get('PULSE_SERVER'):
            rospy.loginfo("PulseAudio detected — using default input device.")
            return None
        names = sr.Microphone.list_microphone_names()
        for i, name in enumerate(names):
            rospy.loginfo("  Mic {:2d}: {}".format(i, name))
            if any(kw in name for kw in ('Andrea', 'andrea', 'USB Audio', 'USB-SA')):
                rospy.loginfo("Selected microphone {}: {}".format(i, name))
                return i
        rospy.logwarn("Andrea headset not found — using system default.")
        return None

    # ── Speech output ─────────────────────────────────────────────────────────
    def speak(self, text):
        """Send text to PAL TTS and wait for it to finish."""
        rospy.loginfo("TIAGo says: {}".format(text))
        goal = TtsGoal()
        goal.rawtext.text = text
        goal.rawtext.lang_id = "en_GB"
        self.tts_client.send_goal(goal)
        self.tts_client.wait_for_result(timeout=rospy.Duration(60))

    # ── Speech input ──────────────────────────────────────────────────────────
    def listen(self, timeout=8, phrase_limit=15):
        """Record from microphone and transcribe with Whisper.
        Returns transcribed text or None if nothing heard."""
        with sr.Microphone(device_index=self.mic_index) as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
            rospy.loginfo("Listening...")
            try:
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_limit)
            except sr.WaitTimeoutError:
                return None
        try:
            # Try Whisper first (offline, more accurate) — falls back to Google if unavailable
            try:
                text = self.recognizer.recognize_whisper(
                    audio, model="base", language="english")
            except (AttributeError, Exception):
                rospy.loginfo_once("Whisper not available — using Google STT")
                text = self.recognizer.recognize_google(audio, language="en-US")
            text = text.strip()
            if text:
                rospy.loginfo("Heard: {}".format(text))
            return text or None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            rospy.logwarn("STT error: {}".format(e))
            return None

    # ── Tool execution ────────────────────────────────────────────────────────
    def _play_motion(self, name, timeout=30):
        goal = PlayMotionGoal()
        goal.motion_name = name
        goal.skip_planning = False
        self.play_motion_client.send_goal(goal)
        self.play_motion_client.wait_for_result(rospy.Duration(timeout))

    def _execute_tool(self, tool_name, tool_args):
        rospy.loginfo("Executing tool: {}".format(tool_name))

        if tool_name == "grab_bottle":
            result = subprocess.run(
                ['python3', REACH_OBJECT_SCRIPT], timeout=180)
            return "success" if result.returncode == 0 else "failed"

        elif tool_name == "go_home":
            self._play_motion('home')
            return "done"

        elif tool_name == "wave":
            self._play_motion('wave')
            return "done"

        elif tool_name == "open_hand":
            self._play_motion('open')
            return "done"

        elif tool_name == "close_hand":
            self._play_motion('close')
            return "done"

        return "unknown tool: {}".format(tool_name)

    # ── LLM conversation ──────────────────────────────────────────────────────
    def _groq_request(self, messages):
        """Call Groq REST API directly (works on Python 3.6)."""
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": 512,
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=self.groq_headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def chat(self, user_text):
        """Send user text to Groq, speak responses, execute any tool calls."""
        self.conversation.append({"role": "user", "content": user_text})

        # Keep history bounded to last 20 turns
        if len(self.conversation) > 20:
            self.conversation = self.conversation[-20:]

        while not rospy.is_shutdown():
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation
            data = self._groq_request(messages)

            choice       = data["choices"][0]
            msg          = choice["message"]
            finish_reason = choice["finish_reason"]
            content      = msg.get("content") or ""
            tool_calls   = msg.get("tool_calls") or []

            # Speak any text content
            if content:
                self.speak(content)

            # Record assistant turn
            assistant_turn = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_turn["tool_calls"] = tool_calls
            self.conversation.append(assistant_turn)

            # No tool calls → done
            if finish_reason != "tool_calls" or not tool_calls:
                break

            # Execute each tool and append results
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                tool_args = json.loads(tc["function"].get("arguments") or "{}")
                result = self._execute_tool(tool_name, tool_args)
                self.conversation.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
            # Loop — Groq will produce a follow-up response after tool results

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        self.speak("Hello! I am TIAGo. How can I help you today?")

        while not rospy.is_shutdown():
            text = self.listen(timeout=10)
            if text and len(text.strip()) > 1:
                try:
                    self.chat(text)
                except Exception as e:
                    rospy.logerr("Chat error: {}".format(e))
                    self.speak("Sorry, I had an error. Please try again.")
            rospy.sleep(0.05)


if __name__ == '__main__':
    try:
        RobotTalker().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Fatal: {}".format(e))
        raise
