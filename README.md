# TIAGo Embodied AI Agent

**A neuro-symbolic, VLM-powered embodied AI system for the PAL Robotics TIAGo robot.**

The agent accepts natural-language commands, reasons about the 3D scene using a forward-chaining symbolic rule engine grounded by a Vision-Language Model, and executes multi-step manipulation tasks (grasp, search, handover, navigate, push aside) via MoveIt and PAL motion primitives.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TIAGo Embodied Agent                            │
│                                                                         │
│  User command ──► VLMReasoner (Gemini 2.0 Flash · Eden AI)             │
│                        │  detections + skill choice + rationale        │
│                        ▼                                                │
│               PerceptionManager ◄── /xtion/rgb/image_raw               │
│               (3D positions via depth + TF · face recognition)         │
│                        │                                                │
│                        ▼                                                │
│               StateManager + SymbolicRuleEngine                        │
│               forward-chaining rules → reachability, affordances,      │
│               spatial relations, handover, obstruction                  │
│                        │  derived facts injected into VLM prompt       │
│                        ▼                                                │
│               Skill Executor                                            │
│  ┌──────────────┬───────────┬───────────┬────────────┬─────────────┐  │
│  │ grab_object  │  search   │ handover  │ navigate   │ push_aside  │  │
│  └──────────────┴───────────┴───────────┴────────────┴─────────────┘  │
│                        │                                                │
│               reach_object_v5 subprocess                               │
│               VLM bbox → depth 3D → MoveIt plan → arm execute         │
└─────────────────────────────────────────────────────────────────────────┘

  YOLO service (host · port 5001)         Face recognition service (port 5002)
  open-vocabulary · segmentation          DeepFace / face_recognition library
```

---

## Features

- **Multimodal perception** — RGB-D camera (Xtion Pro Live) with TF-projected 3D object positions
- **Open-vocabulary detection** — YOLO-E with text-prompt class filtering, served as HTTP microservice
- **VLM reasoning** — Gemini 2.0 Flash via Eden AI; single call returns detections + skill selection + rationale
- **Neuro-symbolic scene graph** — forward-chaining rule engine derives spatial relations, reachability, affordances, obstruction from 3D positions
- **Descriptive object disambiguation** — VLM labels multiple same-class objects descriptively ("small bottle", "red cup"); downstream grasp pipeline uses chain-of-thought to pick the right one
- **MoveIt-based grasping** — RANSAC cylinder fitting, calibrated grasp offsets, collision-aware planning, torso descent strategy
- **Modular skill library** — 9 skills, each with affordance checking and clean base class
- **Face recognition** — identifies known persons, registers new ones via dialogue
- **Safety system** — consecutive-failure safe mode, graceful shutdown (open gripper → go home), watchdog
- **Evaluation suite** — 8 scripts covering detection IoU, depth accuracy, face recognition, latency, spatial reasoning, live grasp trials

---

## Prerequisites

### Hardware
- PAL Robotics **TIAGo** robot (Steel+ or above) with:
  - Xtion Pro Live RGB-D camera
  - PAL gripper
  - ROS Melodic or Noetic running on the robot
- Linux workstation (Ubuntu 20.04 or 22.04) on the same LAN as the robot

### Accounts / API Keys
| Service | Used for | Free tier |
|---------|---------|-----------|
| [Eden AI](https://www.edenai.run) | Gemini 2.0 Flash VLM | Yes |
| [Groq](https://console.groq.com) | LLaMA-3.3-70b LLM | Yes |

### Software (workstation)
- Python 3.10+
- Docker Engine
- Git

---

## Quick Start

### Environment (uv recommended)

```bash
# Install uv if needed: https://github.com/astral-sh/uv#installation
uv venv .venv              # create a local venv (or reuse an existing one)
source .venv/bin/activate
uv pip sync requirements.txt  # install from requirements.txt
```

> Prefer running modules with `python -m …` from the repo root, or export the
> project root onto `PYTHONPATH` to avoid import errors when invoking scripts:
> `export PYTHONPATH=$(pwd):${PYTHONPATH}`

### Minimal run steps

1. Install workstation prerequisites (Python 3.10+, Docker) and the ROS container following [docs/INSTALLATION_MANUAL.md](docs/INSTALLATION_MANUAL.md). Then install Python dependencies:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv pip sync requirements.txt
   ```
2. Copy `.env.example` to `.env`, then add your `EDENAI_API_KEY` and `GROQ_API_KEY`. Override network variables (`ROBOT_IP`, `HOST_IP`, `ROBOT_HOSTNAME`) as needed.
3. Start perception services when required:
   - YOLO open-vocabulary detector — see [docs/YOLO_WORLD_SETUP.md](docs/YOLO_WORLD_SETUP.md)
   - CLIP / re-ranking service — see [docs/CLIP_SETUP.md](docs/CLIP_SETUP.md)
4. Launch the agent (module path reflects new package layout):
   ```bash
   ./run_agent.sh
   ```

For the full ROS Docker setup required for live robot experiments, see [docs/USER_MANUAL_COMPLETE.md](docs/USER_MANUAL_COMPLETE.md).

---

## Configuration

All tunable parameters are controlled via environment variables or YAML files:

| Variable | Default | Description |
|----------|---------|-------------|
| `ROBOT_IP` | `10.68.0.1` | Robot ROS master IP |
| `HOST_IP` | auto-detect | This machine's IP (set as `ROS_IP`) |
| `ROBOT_HOSTNAME` | `tiago-161c` | Robot hostname (added to `/etc/hosts`) |
| `EDENAI_API_KEY` | — | Eden AI API key (required) |
| `GROQ_API_KEY` | — | Groq API key (required) |
| `DOCKER_CONTAINER` | `tiago_ros` | Docker container name |
| `ROS_SETUP` | `/opt/ros/melodic/setup.bash` | ROS setup script path |
| `PAL_SETUP` | `/workspace/pal_ws/devel/setup.bash` | PAL workspace setup script |

**YAML config files:**

| File | Purpose |
|------|---------|
| `grasp_offsets.yaml` | Calibrated gripper XYZ offsets (metres) |
| `config/locations.yaml` | Named navigation waypoints |
| `cyclonedds.xml` | CycloneDDS peer discovery (edit robot IP here) |

---

## Skill Library

| Skill name | Trigger phrase examples | Description |
|------------|------------------------|-------------|
| `grab_object` | "grab the bottle", "pick up the cup" | MoveIt-based grasp with depth 3D |
| `search_with_head` | "find the remote", "look for the phone" | Head scan + VLM detection at 12 positions |
| `handover_to_person` | "give it to Alice", "hand over the bottle" | Extend arm toward detected person |
| `push_aside_object` | "push the cup aside", "move the bottle" | Lateral push to clear path |
| `navigate_to` | "go to the table", "move to the shelf" | Base navigation to named location |
| `go_home` | "go home", "rest" | Arm to safe rest pose |
| `wave` | "wave", "say hello" | Greeting gesture |
| `open_hand` | "open your hand", "release" | Open gripper |
| `close_hand` | "close your hand", "grip" | Close gripper |

---

## Evaluation

The `eval/` directory contains 8 evaluation scripts:

| Script | Needs robot | What it measures |
|--------|-------------|-----------------|
| `eval_detection.py` | No | YOLO vs VLM bbox IoU, detection rate |
| `eval_depth_refine.py` | No | Depth-based bbox refinement, RANSAC accuracy |
| `eval_statemanager.py` | No | Spatial relation accuracy (6 relations) |
| `eval_face.py` | No | Face recognition FAR/FRR vs distance |
| `eval_latency.py` | No | Per-component latency profiling |
| `eval_grasp_recorder.py` | Yes | Live grasp success rate |
| `eval_search_recorder.py` | Yes | Head search efficiency |
| `report.py` | No | Compile all results → tables + PDF figures |

```bash
# Run all offline evals (no robot needed)
cd eval
python3 eval_detection.py --demo
python3 eval_depth_refine.py --demo
python3 eval_statemanager.py --demo
python3 eval_latency.py --skip_vlm --skip_face
python3 report.py
```

See [eval/README.md](eval/README.md) for full details.

---

## Project Structure

```
tiago-embodied-agent/
├── docs/                            # All setup guides, manuals, and reports
│   ├── README.md                    # Documentation index
│   ├── INSTALLATION_MANUAL.md       # Workstation + ROS container setup
│   ├── USER_MANUAL_COMPLETE.md      # Full live-robot workflow
│   ├── YOLO_WORLD_SETUP.md          # Open-vocabulary detector service
│   ├── CLIP_SETUP.md                # CLIP / re-ranking service
│   ├── MAP_INTEGRATION_GUIDE.md     # Integrating external maps
│   ├── MAPPING_GUIDE.md             # SLAM autonomous mapping
│   ├── SAFETY_FEATURES.md           # Safety system details
│   ├── GRASPING_IMPROVEMENTS.md     # Grasp pipeline technical notes
│   ├── IMPROVEMENTS_SUMMARY.md      # Release highlights and pointers
│   ├── PROMPTS.md                   # VLM prompt engineering
│   └── TECHNICAL_REPORT.md          # System architecture & methods
├── tiago_lissi/                     # Python package (code grouped by domain)
│   ├── agent/                       # Main agent loop + VLM reasoning + state
│   ├── perception/                  # RGB-D perception manager
│   ├── detection/                   # YOLO / CLIP detection viewers and tools
│   ├── navigation/                  # Base/torso navigation helpers
│   ├── manipulation/                # Grasping and calibration pipelines
│   ├── services/                    # HTTP services (YOLO, face recognition)
│   ├── visualization/               # Web/GUI visualization helpers
│   ├── tools/                       # Utility scripts (e.g., import checks)
│   └── skills/                      # Modular skill library
├── eval/                            # Evaluation scripts and metrics
├── tests/                           # Test and diagnostic scripts
├── config/                          # Navigation waypoints and robot config
├── docker/                          # Container and deployment assets
├── report/                          # LaTeX report sources and figures
├── .env.example                     # Template for API keys and network overrides
├── run_agent.sh                     # Agent launcher (configurable via env vars)
└── eval_setup.sh                    # Eval environment setup helper
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution process and expectations (new skills first) |
| [docs/README.md](docs/README.md) | Index of all documentation |
| [docs/INSTALLATION_MANUAL.md](docs/INSTALLATION_MANUAL.md) | Complete setup guide from zero |
| [docs/USER_MANUAL_COMPLETE.md](docs/USER_MANUAL_COMPLETE.md) | Full live-robot workflow and ROS Docker setup |
| [docs/YOLO_WORLD_SETUP.md](docs/YOLO_WORLD_SETUP.md) | YOLO open-vocabulary detection service |
| [docs/CLIP_SETUP.md](docs/CLIP_SETUP.md) | CLIP / re-ranking service setup |
| [docs/MAP_INTEGRATION_GUIDE.md](docs/MAP_INTEGRATION_GUIDE.md) | Using pre-built maps inside the agent |
| [docs/MAPPING_GUIDE.md](docs/MAPPING_GUIDE.md) | Autonomous SLAM mapping workflow |
| [docs/SAFETY_FEATURES.md](docs/SAFETY_FEATURES.md) | Safety system details |
| [docs/GRASPING_IMPROVEMENTS.md](docs/GRASPING_IMPROVEMENTS.md) | Grasping pipeline technical notes |
| [docs/IMPROVEMENTS_SUMMARY.md](docs/IMPROVEMENTS_SUMMARY.md) | Release highlights and links |
| [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) | Full system architecture and methods |
| [docs/PROMPTS.md](docs/PROMPTS.md) | VLM prompt engineering reference |
| [eval/README.md](eval/README.md) | Evaluation methodology and metrics |

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Citation

If you use this work in research, please cite:

```bibtex
@misc{tiago-embodied-agent-2026,
  title  = {TIAGo Embodied AI Agent: Neuro-Symbolic VLM-Powered Robot Control},
  year   = {2026},
  url    = {https://github.com/your-org/tiago-embodied-agent}
}
```
