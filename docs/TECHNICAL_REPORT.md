# Technical Report: TIAGo Embodied AI Agent

**A Neuro-Symbolic Vision-Language System for Real-World Robot Manipulation**

---

## Abstract

We present an embodied AI system for the PAL Robotics TIAGo robot that combines Vision-Language Model (VLM) reasoning with a forward-chaining symbolic rule engine to enable natural-language-driven manipulation in unstructured environments. The architecture tightly couples neural perception (RGB-D camera, open-vocabulary YOLO, Gemini 2.0 Flash) with symbolic state representation (typed fact base, geometric 3D relations, affordance ontology) in a bidirectional loop: symbolic facts ground VLM reasoning, and VLM semantic outputs enrich the symbolic world model. A modular skill library (grasp, search, handover, navigation, push-aside) is selected by the VLM and executed via MoveIt. The system achieves reliable multi-step task execution from commands like "grab the small bottle and hand it to Alice" with safety mechanisms preventing robot damage on failure. An evaluation suite of 8 scripts covers detection, depth accuracy, spatial reasoning, face recognition, latency, and live grasp trials.

---

## 1. System Architecture

The system is structured as a perception-reason-act loop executing at human interaction timescales (~3–8 s per step):

```
┌──────────────────────────────────────────────────────────────────────────┐
│  SENSE                                                                   │
│  RGB frame (/xtion/rgb/image_raw)                                        │
│  Depth frame (/xtion/depth_registered/image_raw) + TF                   │
│  Joint states (/joint_states) · Odometry (/mobile_base_controller/odom) │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PERCEIVE  (perception_manager_v2.py)                                    │
│  · YOLO open-vocab detections via HTTP service                           │
│  · 3D position: depth ROI → back-project via camera intrinsics + TF     │
│  · Face recognition enrichment (HTTP microservice, port 5002)           │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │  detections with 3D positions
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  REASON  (vlm_reasoner.py + state_manager.py)                           │
│                                                                          │
│  StateManager._infer_3d_relations()                                      │
│    → geometric base facts: left_of, above, near_3d, reachable, on_table │
│                                                                          │
│  SymbolicRuleEngine.forward_chain()                                      │
│    → derived facts: can_grasp, handover_possible, stacked_on, ...       │
│                                                                          │
│  VLMReasoner.query_with_detection(image, command, state + derived_facts) │
│    → detections with descriptive labels, selected skill, rationale,     │
│      semantic_facts (bottle_is_open, person_is_reaching, ...)           │
│                                                                          │
│  semantic_facts merged → SymbolicRuleEngine re-fires                    │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │  skill + parameters
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ACT  (skills/)                                                          │
│  Affordance check → Skill.execute() → ROS action clients                │
│  (MoveIt, play_motion, head/torso controllers, navigation)              │
└──────────────────────────────────────────────────────────────────────────┘
```

### The Neuro-Symbolic Loop

The key architectural innovation is the **bidirectional coupling** between the neural and symbolic layers:

| Direction | Mechanism | Effect |
|-----------|-----------|--------|
| Neural → Symbolic | VLM detections with 3D positions → geometric fact assertion | Grounds symbols in physical space |
| Neural → Symbolic | VLM `semantic_facts` output → rule engine fact base | Adds visual semantics geometry cannot compute |
| Symbolic → Neural | Derived fact base injected into VLM system prompt | Constrains and focuses VLM reasoning |
| Symbolic → Neural | `can_grasp(X)` / `reachable(X)` filters skill options | Prevents physically impossible skill selection |

---

## 2. Perception Pipeline

### 2.1 RGB-D Camera

The Xtion Pro Live provides synchronized RGB and registered depth at 30 Hz (640×480). Depth encoding is handled for both `32FC1` (metres, float) and `16UC1` (millimetres, uint16) formats. Valid depth range: 0.15 m – 3.5 m.

### 2.2 Open-Vocabulary Object Detection (YOLO Service)

A dedicated HTTP microservice (`yolo_service.py`, port 5001) runs on the host machine (Python 3.10) and serves detection results to the Docker-based agent (Python 3.6 / ROS Melodic). This split avoids Python version conflicts between ROS Melodic and modern ML libraries.

- **Model:** `yoloe-26s-seg` — open-vocabulary YOLO-E with segmentation
- **Class specification:** `X-Classes` HTTP header (comma-separated text prompts)
- **Output:** bounding boxes, confidence, segmentation mask centroids
- **Latency:** ~25 ms per frame on GPU, ~180 ms on CPU

The YOLO service is also used as ground-truth proxy in the evaluation suite for comparing against VLM bounding boxes.

### 2.3 VLM-Based Object Detection

For each agent step, the main perception call is `VLMReasoner.query_with_detection()`, a single API call that:
1. Encodes the current RGB frame as base64 JPEG (max 640 px, quality 85)
2. Sends frame + command + robot state + derived symbolic facts to Gemini 2.0 Flash
3. Receives in one response: detected objects with bounding boxes, selected skill and parameters, chain-of-thought rationale, and semantic facts

**Descriptive labelling:** When multiple instances of the same class are visible, the VLM is instructed to assign descriptive labels ("small bottle", "red cup", "left bottle") rather than generic ones. This enables downstream disambiguation in the grasping pipeline.

### 2.4 3D Position Extraction

For each detected bounding box, 3D position in `base_footprint` frame is extracted:

```
bbox → ROI pixels → depth values (valid range filter) →
back-project via camera intrinsics (fx, fy, cx, cy from /camera_info) →
3D points in camera frame →
transform via TF (xtion_rgb_optical_frame → base_footprint) →
median position [X, Y, Z] in base_footprint
```

TF is queried via `rosrun tf tf_echo` subprocess (compatible with both ROS 1 and Docker environments). The transform is cached for 500 ms to avoid excessive subprocess calls.

Minimum valid depth points required per object: 8. Falls back to 2D pixel heuristics if depth is unavailable.

### 2.5 Face Recognition

A separate HTTP microservice (port 5002) wraps the `face_recognition` library. The main agent:
1. Calls `/recognize` with the cropped person RGB patch
2. Matches against registered identities in `data/faces/known/<Name>/`
3. Returns name, confidence, and whether the person is known
4. Registers new persons via dialogue flow (ask name → `/register`)

---

## 3. Symbolic Reasoning Engine

### 3.1 Fact Base

The `SymbolicRuleEngine` maintains a typed fact base as a Python `set` of tuples:

```python
# Typed relational facts
('left_of',   'bottle', 'cup')
('reachable', 'bottle')
('can_grasp', 'bottle')
('holding',   'gripper', 'bottle')
# Affordance facts
('has_affordance', 'bottle', 'graspable')
('has_affordance', 'bottle', 'drinkable')
# Semantic facts (from VLM)
('bottle_is_open',)
('person_is_reaching',)
```

### 3.2 Affordance Ontology

Static per-class affordance assignments:

| Object class | Affordances |
|--------------|-------------|
| bottle | graspable, drinkable, rollable |
| cup | graspable, drinkable |
| remote | graspable, pressable |
| phone | graspable |
| bowl | graspable, fillable |
| person | can_receive, addressable |
| chair | sittable, pushable |
| table | surface, placeable_on |

### 3.3 Forward-Chaining Rules

Rules fire iteratively until fixpoint (max 20 iterations):

| Rule | Condition | Conclusion | Type |
|------|-----------|------------|------|
| Symmetry | `near(A,B)` | `near(B,A)` | Structural |
| Symmetry | `left_of(A,B)` | `right_of(B,A)` | Structural |
| Symmetry | `above(A,B)` | `below(B,A)` | Structural |
| Transitivity | `left_of(A,B) ∧ left_of(B,C)` | `left_of(A,C)` | Structural |
| Transitivity | `in_front_of(A,B) ∧ in_front_of(B,C)` | `in_front_of(A,C)` | Structural |
| Co-location | `on_surface(A,T) ∧ on_surface(B,T) ∧ A≠B` | `co_located(A,B)` | Semantic |
| Reachability | `distance_3d(obj) < 1.20 m ∧ |Y| < 0.65 m` | `reachable(obj)` | Physical |
| Graspability | `reachable(A) ∧ has_affordance(A, graspable)` | `can_grasp(A)` | Physical |
| Handover | `holding(gripper,X) ∧ near_3d(robot,person)` | `handover_possible(X,person)` | Social |
| Stacking | `above(A,B) ∧ near_xy(A,B)` | `stacked_on(A,B)` | Physical |

### 3.4 3D Spatial Relations

All spatial relations are computed in `base_footprint` frame (X=forward, Y=left, Z=up) with hysteresis thresholds to avoid oscillation:

| Relation | Condition | Threshold |
|----------|-----------|-----------|
| `left_of(A,B)` | `A.Y > B.Y + δ` | δ = 0.08 m |
| `right_of(A,B)` | `A.Y < B.Y − δ` | δ = 0.08 m |
| `in_front_of(A,B)` | `A.X < B.X − δ` | δ = 0.10 m |
| `above(A,B)` | `A.Z > B.Z + δ` | δ = 0.08 m |
| `near_3d(A,B)` | `‖A−B‖₃ < 0.30 m` | Euclidean |
| `on_table(A)` | `A.Z < 0.90 m ∧ A.X < 1.5 m` | Height heuristic |
| `reachable(A)` | `A.X < 1.20 m ∧ |A.Y| < 0.65 m` | Arm envelope |

Self-referential pairs (`A == B`) are explicitly filtered to prevent nonsensical facts such as `left_of(bottle, bottle)`.

---

## 4. VLM Integration

### 4.1 Model

**Gemini 2.0 Flash** via the Eden AI OpenAI-compatible endpoint (`https://api.edenai.run/v3/llm/chat/completions`). Parameters: temperature=0.1, max_tokens=2048, timeout=30 s. Conversation history is maintained for 20 turns.

### 4.2 System Prompt

The VLM system prompt injects the full robot state at each step:

```
You are an embodied robot agent controlling a TIAGo robot.

Robot state:
  Gripper: [open/closed/holding X]
  Torso height: 0.XX m
  Head: pan=X°, tilt=X°
  Location: [current location]

Detected objects: [class, bbox, confidence, 3D position, descriptive label]

Object positions (3D, base_footprint frame):
  bottle: [0.82, -0.04, 0.91]
  cup:    [0.95,  0.12, 0.89]

Symbolic facts (derived):
  can_grasp(bottle), reachable(bottle), left_of(bottle, cup), near_3d(bottle, cup)

Task history: [list of completed skills this session]

Available skills: [JSON list with name, description, parameters]

IMPORTANT: Only do what the user explicitly asks. Do not add steps not requested.
```

### 4.3 Single-Call Detection + Planning

`query_with_detection()` combines object detection and skill selection into one API call, returning:

```json
{
  "detections": [
    {"class_name": "small bottle", "bbox": [x,y,w,h], "confidence": 0.87},
    {"class_name": "big bottle",   "bbox": [x,y,w,h], "confidence": 0.91}
  ],
  "skill": "grab_object",
  "parameters": {"target_object": "small bottle"},
  "rationale": "User asked for small bottle. I see two bottles; the smaller one is on the left at 0.82m.",
  "semantic_facts": ["bottle_is_upright", "table_is_clear"]
}
```

### 4.4 Chain-of-Thought Propagation

The VLM rationale from the main agent call is propagated to the grasping subprocess via the `TARGET_DESCRIPTION` environment variable. The grasp pipeline uses this description in its own VLM disambiguation call (`_vlm_pick_best`), which draws numbered labels on all candidates and asks the VLM "which one is the small bottle?".

This chain ensures that the agent's high-level reasoning ("the small bottle, which is the one on the left") directly guides low-level execution without information loss.

---

## 5. Grasping Pipeline

The grasping pipeline runs as a subprocess (`reach_object_v5_torso_descent_working.py`) to isolate ROS message handling and prevent state corruption in the main agent loop.

### 5.1 Workflow

```
1. VLM detect_objects()
   → bounding boxes for target class (using TARGET_DESCRIPTION for disambiguation)
   → if multiple candidates: _vlm_pick_best() draws numbered labels, asks VLM to choose

2. Cache bbox (detection frame frozen for synchronized depth extraction)

3. 3D position extraction
   → Wait for 5 depth frames aligned to cached bbox
   → Back-project via camera intrinsics + TF (xtion_rgb_optical_frame → base_footprint)
   → RANSAC cylinder fitting for refined centroid

4. Grasp pose estimation
   → Apply calibrated offsets (grasp_offsets.yaml): dx, dy, dz
   → Fixed grasp orientation quaternion: (x=0.703, y=0.073, z=-0.034, w=0.706)

5. MoveIt planning
   → Add collision objects: table surface, obstacle objects
   → Plan arm trajectory (ompl/RRTConnect)
   → Torso descent strategy: lower torso to reach objects below default arm height

6. Execution
   → play_motion: pre_grasp → arm_IK_to_pose → close_gripper
   → Verify: capture after-image, ask VLM "is the object in the gripper?"
```

### 5.2 Cylinder Fitting

RANSAC cylinder fitting on the depth point cloud within the object bounding box provides a refined 3D centroid more robust than the raw depth median:

- **Iterations:** 300
- **Inlier threshold:** 15 mm
- **Radius range:** 10 mm – 250 mm
- **Min inliers:** 20 points

The fitted cylinder axis and centre are back-projected to image coordinates for debug visualization (`/tmp/tiago_cylinder.jpg`).

### 5.3 Grasp Calibration

Calibrated offsets are stored in `grasp_offsets.yaml` and can be refined using `calibrate_grasp_interactive.py`:

```yaml
grasp_dx: 0.22   # Forward (approach) offset in metres
grasp_dy: 0.03   # Lateral offset
grasp_dz: 0.04   # Vertical offset
```

Override at runtime via environment variables: `GRASP_DX`, `GRASP_DY`, `GRASP_DZ`.

---

## 6. Skill Library

Each skill inherits from `BaseSkill` (abstract) and implements:
- `check_affordance(params, state) → (bool, reason)`: pre-execution guard
- `execute(params) → bool`: executes via ROS action clients
- `get_description() → str`: natural-language description for VLM

| Skill | Key affordance checks | ROS interfaces used |
|-------|-----------------------|---------------------|
| `grab_object` | gripper empty, target graspable | subprocess → MoveIt, play_motion, gripper controller |
| `search_with_head` | none (always runnable) | head trajectory controller, VLM |
| `handover_to_person` | gripper holding object, person detected | MoveIt, play_motion |
| `push_aside_object` | target detected with 3D position | MoveIt (Cartesian push trajectory) |
| `navigate_to` | location name in config | move_base (NavigationClient) |
| `go_home` | none | play_motion |
| `wave` | none | play_motion |
| `open_hand` | none | play_motion |
| `close_hand` | none | play_motion |

---

## 7. Safety System

### 7.1 Consecutive Failure Safe Mode

The agent tracks consecutive skill failures. After **3 consecutive failures**, it enters safe mode:
1. Executes `go_home` (arm to rest pose)
2. Opens gripper (releases any held object)
3. Halts further skill execution
4. Reports to user and awaits new command

### 7.2 Graceful Shutdown

Signal handlers (`SIGTERM`, `SIGINT`) trigger a safe shutdown sequence:
1. Cancel any active MoveIt goal
2. Open gripper
3. Execute `go_home`
4. Shutdown ROS node

This prevents the robot from stopping mid-motion with the arm extended or gripper closed on an object.

### 7.3 Skill-Level Recovery

Each skill's `execute()` wraps its action client calls in try/except. On failure:
- Logs error with full traceback
- Returns `False` (failure signal to agent loop)
- Agent increments failure counter and may retry with a different skill

### 7.4 VLM Over-Reach Prevention

The follow-up query prompt explicitly constrains the VLM:

```
IMPORTANT: Only do what the original task explicitly asks for.
Do NOT add extra steps the user did not request (e.g. if the task
is 'grasp X', stop after grasping — do not handover unless asked).
If the original task is FULLY complete, respond with skill='done'.
```

This prevents the common LLM failure mode of continuing beyond task completion (e.g., attempting `handover_to_person` after a grasp-only command).

---

## 8. Multi-Step Task Execution

Tasks are executed in a loop (max 5 steps by default):

```python
for step in range(max_steps):
    image = perception.get_image()
    response = vlm.query_with_detection(command, image, state)
    # enrich detections with 3D positions, face recognition
    state.update(detections, semantic_facts)
    if response['skill'] == 'done':
        break
    skill = skills[response['skill']]
    ok, reason = skill.check_affordance(response['parameters'], state)
    if ok:
        success = skill.execute(response['parameters'])
    # follow-up query uses original command + completed skills list
```

The follow-up query (step ≥ 2) includes:
- Original user command (preserved throughout)
- List of skills already executed this step
- Current gripper state (open / closed / holding X)
- Explicit "do not add unrequested steps" instruction

---

## 9. Evaluation Methodology

### 9.1 Offline Evaluations (no robot required)

**Detection quality** (`eval_detection.py`): Compares YOLO (ground-truth proxy) with VLM bounding boxes on saved RGB images. Metrics: detection rate, confidence, IoU, area ratio.

**Depth refinement** (`eval_depth_refine.py`): Evaluates RANSAC cylinder fitting on saved RGBD pairs. Metrics: area reduction ratio (tight bbox vs loose VLM bbox), 3D centroid stability across frames, IoU improvement.

**Spatial reasoning** (`eval_statemanager.py`): Tests `StateManager` inference against hand-annotated scene graphs. 6 relations (left_of, right_of, above, below, near, on_table). Metrics: per-relation accuracy, precision, recall, F1.

**Face recognition** (`eval_face.py`): Tests FaceManager HTTP service at varying simulated distances and occlusion levels. Metrics: False Accept Rate (FAR), False Reject Rate (FRR), identification latency.

**Latency profiling** (`eval_latency.py`): Per-component latency with percentile distributions (p50, p90, p99). Components: VLM API, YOLO inference, depth refinement, face recognition, total task cycle time budget.

### 9.2 Live Evaluations (robot required)

**Grasp recorder** (`eval_grasp_recorder.py`): Runs N grasp trials on a specified object class, records success/failure, detection confidence, pose estimate, error type. Results saved to `eval/data/grasp_trials.json`.

**Search recorder** (`eval_search_recorder.py`): Runs N search trials, records positions scanned, first-found position index, total search time, YOLO confidence at detection.

### 9.3 Report Generation

`eval/report.py` aggregates all JSON result files into a summary table (CSV) and publication-quality figures (PDF).

---

## 10. Known Limitations

1. **Single-robot, single-arm:** The skill library and grasp pipeline are designed for TIAGo's single right arm. Adaptation to other robot platforms requires new MoveIt configuration and grasp offset calibration.

2. **Static scene assumption:** The symbolic rule engine rebuilds the full fact base each perception cycle. Fast-moving objects may cause inconsistent state between cycles.

3. **TF dependency:** 3D position extraction requires a live ROS TF tree. If TF is unavailable (e.g., robot head not publishing), the system falls back to 2D pixel heuristics with reduced accuracy.

4. **VLM latency:** Each Gemini API call takes ~1.5–3 s. Total task cycle time is 5–12 s, limiting responsiveness. On-device or locally-hosted VLMs would improve this.

5. **Grasp generalization:** The RANSAC cylinder fitting works well for bottles and cylindrical objects. Flat, deformable, or highly reflective objects require alternative shape priors.

6. **Map dependence for navigation:** Named location navigation requires a pre-built SLAM map of the environment. See [MAPPING_GUIDE.md](MAPPING_GUIDE.md) for the autonomous mapping workflow.

---

## 11. Future Work

- **On-device VLM:** Replace Eden AI with a locally-hosted model (Qwen-VL, LLaVA-Next) to reduce latency and eliminate API dependency
- **Learned grasp poses:** Replace hand-calibrated offsets with GraspNet-based pose estimation
- **Temporal memory:** Extend object memory to use robot odometry for persistent localization across sessions
- **Multi-robot:** Extend skill library and state manager for coordinated dual-arm manipulation
- **Active learning:** Use evaluation trial failures to trigger VLM prompt refinement

---

## 12. References

1. PAL Robotics. *TIAGo Robot Platform.* https://pal-robotics.com/robots/tiago/
2. Quigley et al. *ROS: an open-source Robot Operating System.* ICRA 2009.
3. Coleman et al. *Reducing the Barrier to Entry of Complex Robotic Software using the MoveIt! Motion Planning Framework.* Journal of Software Engineering for Robotics, 2014.
4. Wang et al. *EMPOWER: Embodied Multi-role Open-vocabulary Planning with Enhanced Whole-body Reasoning.* arXiv 2024.
5. Rana et al. *SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning.* CoRL 2023.
6. Jocher et al. *Ultralytics YOLO.* https://github.com/ultralytics/ultralytics
7. Google DeepMind. *Gemini 2.0 Flash.* https://deepmind.google/technologies/gemini/
8. Eden AI. *OpenAI-compatible LLM API gateway.* https://www.edenai.run
