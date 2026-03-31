# EmbodiedAgent — All VLM Prompts

All prompts sent to Gemini 2.0 Flash in this project.
Source: `vlm_reasoner.py`, `embodied_agent.py`, `state_manager.py`.

---

## 1. System Prompt (injected into every skill-selection call)

**Source:** `vlm_reasoner.py :: build_system_prompt()`
**Used by:** prompts 3 and 4 below as the `{system}` block.

```
You are TIAGo, a mobile manipulator robot.

**Current State:**
- Gripper: {gripper}
- Base Location: {location}
- Detected Objects: {objects}
- Recent Tasks: {tasks}
- Object Memory: {memory}

**Available Skills:**
{skills}

**Instructions:**
- Analyze the image to understand the visual scene.
- Choose ONE skill to execute next.
- If the object is NOT visible, use search_with_head first.
- Only choose grab_bottle if a graspable object is actually visible.
- Respond ONLY with valid JSON, no prose:
{
  "skill": "skill_name",
  "parameters": {},
  "rationale": "brief explanation"
}
```

**Runtime values injected:**
| Placeholder | Source |
|---|---|
| `{gripper}` | `StateManager.gripper_status` — e.g. `"empty"` or `"holding:bottle"` |
| `{location}` | `StateManager.base_location` |
| `{objects}` | `StateManager.detected_objects[:5]` (5 most recent) |
| `{tasks}` | `StateManager.task_history[-3:]` (last 3 skills + success flag) |
| `{memory}` | `StateManager.memory_summary(max_age_sec=300)` — see §6 |
| `{skills}` | `get_skill_descriptions()` — one line per skill from `get_description()` |

---

## 2. Object Detection Prompt

**Source:** `vlm_reasoner.py :: detect_objects()`
**Model call:** single image + text, `temperature=0.1`, `max_tokens=2048`, timeout 15 s.

```
{hint} Return bounding boxes as a JSON array.
Limit to 20 objects.
Format: [{"box_2d": [y_min, x_min, y_max, x_max], "label": "name"}].
Coordinates normalized 0-1000. Lowercase labels.
If nothing found return [].
```

Where `{hint}` is either:
- **Targeted:** `"Focus on detecting: bottle, cup."`  (when `target_classes` provided)
- **Open-vocab:** `"Detect all objects (people, furniture, everyday items)."`

---

## 3. Combined Detection + Skill Selection Prompt (primary loop)

**Source:** `vlm_reasoner.py :: query_with_detection()`
**Model call:** single image + text, `temperature=0.1`, `max_tokens=2048`, timeout 30 s, `force_json=True`.
**This is the main prompt used every agent cycle.**

```
{system_prompt}

Look at the image and do TWO things:
1. Detect all visible objects with bounding boxes.
2. Select the best skill for the user request.

User request: "{user_command}"

Respond ONLY with valid JSON:
{
  "detections": [{"label": "name", "box_2d": [y_min, x_min, y_max, x_max]}, ...],
  "skill": "skill_name",
  "parameters": {},
  "rationale": "brief explanation"
}
box_2d normalized 0-1000. Limit 20 detections. Empty scene: "detections": [].
```

---

## 4. Skill Selection Prompt (fallback / compatibility)

**Source:** `vlm_reasoner.py :: query()`
**Model call:** single image + text, timeout 30 s, `force_json=True`.
Used when detections are provided separately (legacy path).

```
{system_prompt}

User request: "{user_command}"

Detected objects:
- bottle @ bbox(320, 240, 60, 80) conf=0.92
- cup @ bbox(150, 300, 40, 55) conf=0.85
...

What skill should I execute?
```

---

## 5. Skill Verification Prompt (before/after comparison)

**Source:** `vlm_reasoner.py :: verify()`
**Model call:** two images (BEFORE + AFTER) + text, timeout 30 s, `force_json=True`.

```
I executed: grab_bottle(target_object=bottle)
Expected: Gripper should be closed around the object, robot holding the object

Compare the BEFORE image and AFTER image.
Respond ONLY with valid JSON:
{"success": true/false, "observation": "...", "issue": "..."}
```

---

## 6. Multi-Step Task Query Prompts

**Source:** `embodied_agent.py :: execute_command()` (the `query_prompt` variable).
These replace `{user_command}` in prompt 3 during multi-step execution.

### 6a. First step (pass-through)
```
{original user command verbatim}
```

### 6b. Continuation step (after at least one skill completed)
```
Original task: 'bring me the bottle'
Completed so far: search_with_head, grab_bottle

If the original task is FULLY complete, respond with skill='done'.
Otherwise, what skill should execute next?
```

### 6c. Affordance-retry step (previous skill could not execute)
```
Original task: 'bring me the bottle'
Completed so far: search_with_head
Could NOT execute 'grab_bottle' because: No bottle detected in scene
Choose a different skill to make progress (e.g. search_with_head if object not visible).
```

---

## 7. StateManager Context (injected as `{memory}` in the system prompt)

**Source:** `state_manager.py :: memory_summary(max_age_sec=300.0)`
Objects older than 5 minutes are excluded from the VLM context.

**Format of each entry:**
```
- bottle (last seen: 12s ago, near cup, seen 3 time(s))
- cup (last seen: 1m34s ago, no relations, seen 1 time(s))
```

**Full state dict injected into the system prompt** (`StateManager.get_state()`):
```python
{
  'gripper':           "empty" | "holding:bottle",
  'base_location':     "home" | "room_A" | ...,
  'detected_objects':  ["bottle at (320, 240)", "cup at (150, 300)"],  # max 5
  'task_history':      ["✓ search_with_head", "✓ grab_bottle"],        # last 3
  'spatial_relations': ["bottle near cup", "cup on table"],            # max 10
  'recognized_person': "Alice" | None,
  'object_memory':     <memory_summary string>
}
```

---

## 8. Skill Descriptions (injected as `{skills}` in the system prompt)

One line per skill, from each skill's `get_description()` method:

```
search_with_head() - Scan 12 pan/tilt positions to find an object. Use when target is not visible.
grab_bottle(target_object="box") - Grasp any detected object in front of the robot. ALWAYS set
    target_object to the exact class name of the object to grab (e.g. "bottle", "box", "cup").
    Works with any graspable object.
handover_to_person() - Extend arm toward detected person at 0.85m height and release object.
wave() - Wave the robot's arm as a greeting gesture.
go_home() - Move arm to safe home position and clear MoveIt planning scene.
open_hand() - Open the robot gripper.
close_hand() - Close the robot gripper.
```
