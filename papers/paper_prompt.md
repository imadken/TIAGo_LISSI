# PAPER GENERATION PROMPT
# Use this prompt verbatim to generate the full LaTeX paper

---

## PROMPT

You are an expert academic writer specializing in robotics and AI, with deep knowledge of IEEE/ECAI paper style. Write a complete, submission-ready LaTeX paper based on everything below. Do not ask questions — generate the full paper end to end.

---

## CONTEXT: THE SYSTEM

The paper describes a real, working system built and tested on a **TIAGo mobile manipulator robot** at LISSI Laboratory, Université Paris-Est Créteil (UPEC), France. The system runs on ROS Melodic inside a Docker container, with the AI reasoning on a Python 3.10 host.

### Authors
- **[YOUR NAME]**, Ferhat Attal, Abdelghani Chibani, Yacine Amirat
- Affiliation: LISSI Laboratory, Université Paris-Est Créteil (UPEC), France

### What Was Built — Full System Description

**The complete pipeline:**
```
User Voice → STT → Natural Language Command
                        ↓
             VLMReasoner (Gemini 2.0 Flash)
          ← RGB image from Xtion camera
          ← Robot state + semantic context from StateManager
          → Outputs: skill selection, parameters, rationale, object detections [{label, bbox [x,y,w,h], confidence}]
                        ↓
              StateManager (semantic memory)
          - Tracks: detected objects, spatial relations ("bottle left of cup"),
            seen counts per object, gripper state (empty/holding), task history
          - Updated after every VLM call and every skill execution
          - Serialized as structured text injected into next VLM prompt
                        ↓
             Skill Library (executed on TIAGo)
          ┌─────────────────────────────────────────┐
          │ search_with_head │ grab_bottle │ handover│
          │ go_home │ wave │ open_hand │ close_hand  │
          └─────────────────────────────────────────┘
                        ↓
              TTS Response → Robot Speech (PAL TTS)
```

**Component 1: VLMReasoner**
- Uses Google Gemini 2.0 Flash via native API (google-generativeai SDK)
- Single model handles BOTH task planning AND open-vocabulary object detection
- Detection output format: `[{"box_2d": [y_min, x_min, y_max, x_max], "label": "name"}]` normalized 0-1000
- Task planning output: `{"skill": "...", "parameters": {...}, "rationale": "...", "detections": [...]}`
- Replaces: separate YOLO detector + separate LLM (as in Eladdachi et al. 2025)
- No GPU required — API call from CPU-only host
- `force_json=True` with `response_mime_type="application/json"` for reliable structured output

**Component 2: StateManager (Dynamic Semantic Memory)**
- Maintains per-session object memory: `{class: {seen_count, last_bbox, last_position, confidence_history}}`
- Computes spatial relations automatically: "bottle is left of box", "cup is near person", "object on table"
- Relations computed from detected bounding box centroids and relative positions
- Injected into VLM prompt as: "Spatial context: bottle left of box, cup near person | Seen 3 time(s) in this session"
- Enables: multi-instance disambiguation, session-persistent context, coherent multi-step task execution
- Analogous to RDF knowledge graph in Eladdachi et al. but dynamic and vision-grounded

**Component 3: FaceManager**
- Face encoding with dlib (128-dim embedding vector)
- Euclidean distance threshold 0.60 for identity matching
- Incremental identity memory: recognizes known persons across session
- Integrated into handover — TIAGo greets known persons by name

**Component 4: search_with_head skill**
- Systematically pans/tilts TIAGo's head to scan environment
- At each head position: capture RGB frame → VLM detect target → if found, stop and update StateManager
- Head positions: center, left, right, down-left, down-right
- Enables finding objects not in initial field of view

**Component 5: grab_bottle skill / reach_object_v5**
This is the most technically detailed component:

5a. VLM Open-Vocabulary Detection:
- `_detect_box()`: sends RGB frame to Gemini with target hint (class + description from agent rationale + StateManager spatial context)
- Prompt: "bottle. The operator selected this specific object: [rationale + spatial relations]. Return ONLY that one object."
- Single VLM call cached (`_cached_bbox`) — reused for all 5 depth frames to avoid rate limits
- Debug images saved to /tmp/ with bounding boxes drawn

5b. Depth-Based Bounding Box Refinement (`_refine_bbox_by_depth`):
- Problem: VLM produces semantically correct but spatially loose bboxes (~210×198 px average) vs YOLO's tight bboxes (~46×82 px) → point cloud contaminated with background/table
- Solution: within VLM bbox ROI, find nearest-depth cluster (10th percentile depth within 0.30–1.10m range + 12cm tolerance) → output tight refined bbox
- Result: YOLO-equivalent point cloud quality without any segmentation model (no SAM, no masks)
- Gives pixel-precise 3D localization from a semantically-driven loose bbox

5c. 3D Pose Estimation:
- `_depth_roi_to_base()`: project depth pixels inside refined bbox → 3D points in base_footprint frame using camera intrinsics (fx, fy, cx, cy from PinholeCameraModel) + TF (tf_echo base_footprint→xtion_rgb_optical_frame)
- Depth range: 0.15–3.5m
- `_fit_cylinder()`: RANSAC circle fit in XY plane (radius 0.01–0.25m, RANSAC threshold=0.015m)
  - 300 iterations, algebraic LS refinement on inliers
  - Requires inlier_ratio > 0.30
  - Centroid fallback when RANSAC fails (flat/box-shaped objects): median XY of front-face points
- `detect_bottle_pose_stable()`: collect 5 accepted poses, temporal average → stable XY position (max 30 attempts)
- Reachability filter: reject poses with cx > 1.05m or |cy| > 0.65m

5d. Grasping:
- Planning: MoveIt (move_group action server), group "arm_torso", 10 planning attempts, 10s timeout
- Grasp orientation: quaternion (x=0.703, y=0.073, z=-0.034, w=0.706) ≈ top-down gripper
- Pre-grasp: OMPL plans arm to 20cm above object (bottle as collision obstacle)
- Grasp: torso_lift_joint descends 20cm — guaranteed straight vertical line, no IK required
- Post-grasp: raise trunk to 0.30m, go home

**Component 6: handover_to_person skill**
- VLM detects 'person' in current RGB frame
- If detected: depth projects person bbox centroid → 3D position in base_footprint
- MoveIt plans arm extension at HANDOVER_HEIGHT = 0.85m toward person
- TTS: "Please take the [object] from my hand"
- Wait 4s → open gripper → go home
- FaceManager enriches person detection with identity if known

**Component 7: Execution Loop (EmbodiedAgent)**
- Continuous: listen → VLM plan → execute skill → VLM verify (before/after image) → update state → repeat
- Safety: consecutive failure counter (max 3) → safe mode
- Skill result: success/failure → rationale logged → StateManager updated

---

## PAPER TO CITE (Direct predecessor — must cite as [1] or [2])

**[Eladdachi et al. ECAI 2025]** "Integrating LLM, Semantic Perception and Spatial Reasoning for Improved Robot Action Control" — Oussama Eladdachi, Ferhat Attal, Abdelghani Chibani, Ilies Chibane, Imad-Eddine Kenai, Yacine Amirat. ECAI 2025, LISSI Lab, UPEC.
- Their system: Pepper robot, YOLOv8n (fixed classes) + Gemini 1.5 Flash text-only + static RDF/OWL knowledge graph + MiDaS monocular depth + basic IK grasping
- Your advancement: TIAGo, VLM replaces YOLO+LLM, dynamic StateManager replaces static RDF, RANSAC+RGBD replaces MiDaS+IK, full skill library

**Other references to include:**
- SayCan (Ahn et al. CoRL 2022)
- ViLa (Hu et al. arXiv 2311.17842)
- EMPOWER (Argenziano et al. arXiv 2408.17379) — same TIAGo robot
- SayPlan (Rana et al. CoRL 2023)
- Abdelkawy et al. RA-L 2021 — STJ-CNN + NKRL ontology (LISSI prior work)

---

## PAPER STYLE REQUIREMENTS

**Format:** ECAI IOS Press style (like Eladdachi et al.) OR IEEE two-column (like EMPOWER/RA-L)
- Prefer ECAI format since this continues LISSI's ECAI 2025 line of work
- Length: 6–8 pages
- Two-column layout

**Writing style based on LISSI papers:**
- Abstract: 150–200 words, dense technical, structure: problem → gap → proposal → key result
- Introduction: real-world motivation → limitations of prior work → numbered contribution list
- Architecture section: subsections per component, math equations where relevant (camera projection, coordinate transforms)
- Experiments: comparison table, ablation study, metrics (success rate, precision/recall)
- Conclusion: summary + explicit future work linking to other LISSI papers

**Tone:** precise, technical, no fluff. Every claim grounded in the system description above.

---

## FILLER RESULTS (realistic, make them look good — author will replace with real numbers)

Use these plausible results throughout:

**Main Results:**
- Overall task success rate: **78.3%** across 6 scenario types (N=60 trials)
- Grasp success rate: **81.7%** (49/60 trials across all object classes)
- Handover success rate: **86.7%** (26/30 trials)
- Active search success: **90.0%** (27/30 trials, object found within 5 head positions)
- VLM skill selection accuracy: **91.2%** (correct skill chosen for given command)

**Per-object grasp success:**
| Object class | Success rate |
|--|--|
| Bottle (cylindrical) | 90.0% (18/20) |
| Box (flat-faced) | 75.0% (15/20) |
| Mug (handle) | 70.0% (14/20) |
| Irregular objects | 65.0% (13/20) |
| **Overall** | **81.7%** |

**Ablation — depth bbox refinement:**
| Configuration | Grasp success | Avg. 3D error (cm) |
|--|--|--|
| VLM bbox only (no refinement) | 51.7% | 8.3 |
| + Depth refinement | 81.7% | 2.1 |
| Improvement | +30.0pp | −6.2cm |

**Ablation — StateManager context:**
| Configuration | Correct object selected (multi-instance) |
|--|--|
| No context (VLM only) | 53.3% (8/15) |
| + StateManager spatial relations | 86.7% (13/15) |

**Face recognition:**
| Scenario | Precision | Recall |
|--|--|--|
| Distance 0.5–1.5m | 94.2% | 91.8% |
| Distance 1.5–3.0m | 87.6% | 84.3% |
| Minor occlusion | 82.1% | 79.4% |

**Comparison with Eladdachi et al.:**
| Metric | Eladdachi et al. (Pepper) | Ours (TIAGo) |
|--|--|--|
| Object detection vocab | Fixed (80 COCO classes) | Open (any class) |
| GPU required | No | No |
| 3D localization error | ~9.2cm (MiDaS) | **~2.1cm** (RGBD+RANSAC) |
| Grasp success | 63.3% | **81.7%** |
| Multi-instance disambiguation | No | Yes (86.7%) |
| Skills available | 1 (grasp) | 7 |

---

## FIGURES TO DESCRIBE IN LATEX (use \includegraphics placeholders)

**Figure 1:** Full system architecture diagram — the pipeline from voice input to robot execution showing all 7 components connected with arrows. Caption: "Proposed architecture for VLM-driven semantic robot action control on TIAGo."

**Figure 2:** Depth bounding box refinement illustration — 4-panel: (a) RGB with loose VLM bbox, (b) depth image with bbox overlay, (c) refined tight bbox on depth nearest-cluster, (d) resulting clean point cloud. Caption: "Depth-based bbox refinement: VLM loose bbox (a,b) → nearest-depth cluster → tight refined bbox (c) → clean 3D point cloud (d)."

**Figure 3:** Grasp execution sequence — 4 frames: (1) VLM detection with bbox, (2) arm moving above object, (3) torso descent, (4) gripper closed with object. Caption: "Top-down grasp sequence: OMPL pre-grasp positioning followed by vertical torso descent."

**Figure 4:** Handover sequence — 3 frames: (1) VLM person detection, (2) arm extended toward person, (3) gripper open / person taking object. Caption: "Handover execution: VLM-detected person → arm extension → object transfer."

**Figure 5:** StateManager context enrichment — diagram showing how spatial relations flow from VLM detection → StateManager → enriched prompt → second VLM detection call. Caption: "Ontological context propagation: spatial relations from StateManager disambiguate multi-instance object selection."

---

## WHAT TO GENERATE

Write the complete LaTeX paper with:
1. `\documentclass` and all packages (ECAI IOS Press style or IEEE two-column)
2. Title, authors, affiliations, abstract
3. All sections: Introduction, Related Work, System Architecture (with all subsections), Experiments, Conclusion
4. Math equations: camera projection (intrinsic matrix K, 2D→3D, coordinate transforms), RANSAC cylinder fit criterion, Euclidean face distance
5. All tables with filler results as specified above
6. All figure environments with `\includegraphics[width=\columnwidth]{fig1}` placeholders and proper captions
7. BibTeX bibliography with all references
8. The paper must read as complete, polished, publication-ready — not a draft

Do not include explanatory comments about what you are doing. Just output the LaTeX.
