# Installation Manual

**TIAGo Embodied AI Agent — Complete Setup Guide**

This guide takes you from zero to a running agent on a TIAGo robot. Follow every section in order.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone & Python Setup](#2-clone--python-setup)
3. [API Keys](#3-api-keys)
4. [Network Configuration](#4-network-configuration)
5. [ROS Melodic Docker Container](#5-ros-melodic-docker-container)
6. [Build the PAL Workspace](#6-build-the-pal-workspace)
7. [YOLO Detection Service](#7-yolo-detection-service)
8. [Face Recognition Service (optional)](#8-face-recognition-service-optional)
9. [Grasp Calibration](#9-grasp-calibration)
10. [Running the Agent](#10-running-the-agent)
11. [Running the Evaluation Suite](#11-running-the-evaluation-suite)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

### 1.1 Hardware

| Component | Minimum requirement |
|-----------|---------------------|
| Robot | PAL Robotics TIAGo (Steel+ or above) |
| Camera | Xtion Pro Live (built into TIAGo head) |
| Gripper | PAL parallel gripper |
| Robot OS | ROS Melodic (Ubuntu 18.04 on robot) |
| Workstation | x86-64 Linux (Ubuntu 20.04 or 22.04) |
| Network | Workstation and robot on the same LAN |

> **Without a TIAGo robot:** You can run all offline evaluation scripts and the YOLO service on any Linux machine. Only the live agent and live eval scripts require robot connectivity.

### 1.2 Software on Workstation

```bash
# Check Python version (must be 3.10 or newer)
python3 --version

# Install Docker Engine (if not already installed)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in for group change to take effect

# Install Git
sudo apt-get install -y git
```

### 1.3 API Accounts

| Service | URL | Used for | Notes |
|---------|-----|---------|-------|
| Eden AI | https://www.edenai.run | Gemini 2.0 Flash VLM | Free tier available |
| Groq | https://console.groq.com | LLaMA-3.3-70b LLM | Free tier available |

Create accounts and generate API keys before continuing.

---

## 2. Clone & Python Setup

```bash
# Clone the repository
git clone https://github.com/your-org/tiago-embodied-agent.git
cd tiago-embodied-agent

# Install Python dependencies (host-side: YOLO service + eval scripts)
pip install -r requirements.txt
```

If `requirements.txt` is missing, install the core dependencies manually:

```bash
pip install \
  ultralytics \
  opencv-python \
  numpy scipy \
  flask requests \
  pyyaml \
  python-dotenv \
  face-recognition \
  SpeechRecognition pyaudio
```

---

## 3. API Keys

```bash
# Copy the template
cp .env.example .env

# Edit with your keys
nano .env
```

The file should look like:
```bash
EDENAI_API_KEY=eyJhbGciOiJIUzI1NiIs...   # your Eden AI JWT
GROQ_API_KEY=gsk_...                       # your Groq key
```

**Never commit `.env` to Git.** It is already in `.gitignore`.

---

## 4. Network Configuration

### 4.1 Understand the Network Topology

```
┌─────────────────────┐       LAN (same subnet)      ┌──────────────────────┐
│  Workstation         │ ─────────────────────────── │  TIAGo Robot          │
│  Ubuntu 20.04/22.04  │                              │  ROS Melodic master   │
│  Python 3.10         │                              │  IP: 10.68.0.1        │
│  YOLO service :5001  │                              │  Hostname: tiago-*    │
│  Face service :5002  │                              │                       │
│                      │                              │  Topics:              │
│  Docker container    │                              │  /xtion/rgb/...       │
│  (ROS Melodic)       │ ─────────────────────────── │  /move_group          │
│  --network=host      │       ROS topics / TF        │  /play_motion         │
└─────────────────────┘                              └──────────────────────┘
```

Because the Docker container uses `--network=host`, it shares the workstation's IP address and can subscribe to robot ROS topics directly.

### 4.2 Find Your IPs

```bash
# Your workstation IP
hostname -I | awk '{print $1}'

# Robot IP (ask your lab technician, or check the robot's display/network settings)
# Default PAL TIAGo lab subnet: 10.68.0.1
```

### 4.3 Set Environment Variables

Add to your `~/.bashrc` (or set before each session):

```bash
export ROBOT_IP=10.68.0.1          # ← change to your robot's IP
export HOST_IP=10.68.0.128         # ← change to your workstation's IP
export ROBOT_HOSTNAME=tiago-161c   # ← change to your robot's hostname
```

### 4.4 Edit cyclonedds.xml

Open `cyclonedds.xml` and replace the peer address with your robot's IP:

```xml
<Peers>
    <Peer Address="YOUR_ROBOT_IP"/>   <!-- e.g. 10.68.0.1 -->
    <Peer Address="localhost"/>
</Peers>
```

Also check the network interface name (`wlo1` is WiFi; use `ip link show` to find yours):

```xml
<Interfaces>
    <NetworkInterface name="wlo1" priority="1"/>  <!-- change if needed -->
    <NetworkInterface name="lo" priority="0"/>
</Interfaces>
```

Export the config path:
```bash
export CYCLONEDDS_URI=file://$(pwd)/cyclonedds.xml
```

---

## 5. ROS Melodic Docker Container

The agent code runs inside a Docker container with ROS Melodic and the PAL TIAGo packages. This isolates the Python 3.6 ROS environment from the host's Python 3.10.

### Option A — Build the image yourself (recommended)

```bash
# Build the agent image (installs all Python extras on top of PAL base)
docker build -t tiago-agent -f docker/Dockerfile .

# Create and start the container
docker run -dit \
  --name tiago_ros \
  --network host \
  -v $(pwd):/workspace \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e ROBOT_IP=${ROBOT_IP:-10.68.0.1} \
  -e HOST_IP=${HOST_IP} \
  tiago-agent bash
```

### Option B — docker-compose (starts YOLO service + agent together)

```bash
export ROBOT_IP=10.68.0.1          # your robot's IP
export HOST_IP=$(hostname -I | awk '{print $1}')
docker-compose up -d               # builds images if needed, starts both services
docker-compose exec agent bash     # open a shell inside the agent container
```

### Option C — Pull PAL image directly (no build step)

```bash
docker pull palroboticssl/tiago_melodic_robot:latest

docker run -dit \
  --name tiago_ros \
  --network host \
  -v $(pwd):/workspace \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  palroboticssl/tiago_melodic_robot:latest bash

# Then install Python extras manually inside the container:
docker exec tiago_ros pip3 install -r /workspace/docker/requirements_ros.txt
```

> If the PAL image is unavailable, use `ros:melodic-robot` as fallback and install dependencies manually.

**Flags explained:**

- `--network host`: shares workstation network — container sees robot ROS topics directly
- `-v $(pwd):/workspace`: mounts your project directory into the container
- `-v /tmp/.X11-unix -e DISPLAY`: allows graphical tools (RViz, rqt) on your desktop

### 5.4 Verify the Container

```bash
docker exec -it tiago_ros bash

# Inside container:
source /opt/ros/melodic/setup.bash
rosversion -d                                  # should print: melodic

# Verify Python extras
python3 -c "import scipy, requests, cv2; print('OK')"

# Check robot connectivity (robot must be on)
export ROS_MASTER_URI=http://${ROBOT_IP}:11311
export ROS_IP=${HOST_IP}
rostopic list    # lists robot topics if reachable

exit
```

---

## 6. Build the PAL Workspace

The PAL workspace (`pal_ws/`) contains TIAGo-specific ROS packages needed for MoveIt configuration, motion primitives, and gripper control.

```bash
# Enter the container
docker exec -it tiago_ros bash

# Source ROS
source /opt/ros/melodic/setup.bash

# Build the workspace
cd /workspace/pal_ws
catkin_make -j4

# Source the built workspace
source devel/setup.bash

# Verify key packages are available
rospack find tiago_moveit_config   # should print a path
rospack find play_motion2          # should print a path
```

> **Build errors:** If you see missing package errors, the PAL workspace may need additional dependencies. Run `rosdep install --from-paths src --ignore-src -r -y` inside the container.

---

## 7. YOLO Detection Service

The YOLO service runs on the **host** (not inside Docker) because it requires Python 3.10+ and GPU-accelerated PyTorch.

### 7.1 Download YOLO Weights

```bash
# On the host (not in Docker)
python3 -c "from ultralytics import YOLO; YOLO('yoloe-26s-seg.pt')"
# Downloads ~26 MB to the current directory on first run
```

### 7.2 Start the Service

```bash
python3 yolo_service.py
```

Expected output:
```
[YOLO] Loading model yoloe-26s-seg...
[YOLO] Model loaded. Segmentation: True, Open vocab: True
[YOLO] Active classes: ['bottle', 'person', 'cup', 'chair', ...]
 * Running on http://0.0.0.0:5001
```

### 7.3 Verify

```bash
curl http://localhost:5001/health
# {"status": "ok", "model": "yoloe-26s-seg", "open_vocab": true, ...}
```

To change the port, edit `yolo_service.py` line ~40 (`PORT = 5001`) or set `YOLO_PORT` env var.

### 7.4 Run as Background Service (optional)

```bash
nohup python3 yolo_service.py > /tmp/yolo_service.log 2>&1 &
echo "YOLO service PID: $!"
```

---

## 8. Face Recognition Service (optional)

Face recognition enables the agent to greet known persons by name and register new ones.

### 8.1 Start the Service

```bash
# On the host
python3 face_recognition_service.py
# Runs on port 5002 by default
```

### 8.2 Add Known Faces

Create a directory per person under `data/faces/known/`:

```bash
mkdir -p data/faces/known/Alice
# Copy 3-6 clear face photos of Alice into this directory
cp /path/to/alice_*.jpg data/faces/known/Alice/
```

The service loads all known faces at startup. Restart it after adding new persons.

### 8.3 Verify

```bash
curl http://localhost:5002/health
# {"status": "ok", "known_persons": ["Alice", "Bob"]}
```

---

## 9. Grasp Calibration

Grasp offsets compensate for the difference between the object's 3D centroid (measured by depth) and the actual gripper contact point. They are robot-specific and must be calibrated once per setup.

### 9.1 Run the Calibration Script

```bash
# Inside Docker with ROS running on robot
docker exec -it tiago_ros bash -c "
  source /opt/ros/melodic/setup.bash &&
  source /workspace/pal_ws/devel/setup.bash &&
  export ROS_MASTER_URI=http://${ROBOT_IP}:11311 &&
  export ROS_IP=${HOST_IP} &&
  cd /workspace &&
  python3 calibrate_grasp_interactive.py
"
```

Follow the interactive prompts: the script moves the arm to a test pose and asks you whether the gripper is correctly aligned. Adjust values until consistent success.

### 9.2 Edit grasp_offsets.yaml

```yaml
# grasp_offsets.yaml
grasp_dx: 0.22   # Forward offset (metres) — increase if gripper stops short
grasp_dy: 0.03   # Lateral offset — adjust for left/right misalignment
grasp_dz: 0.04   # Vertical offset — increase if gripper is too high
```

### 9.3 Test a Single Grasp

```bash
docker exec -it tiago_ros bash -c "
  source /opt/ros/melodic/setup.bash &&
  source /workspace/pal_ws/devel/setup.bash &&
  export ROS_MASTER_URI=http://${ROBOT_IP}:11311 &&
  export ROS_IP=${HOST_IP} &&
  cd /workspace &&
  python3 reach_object_v5_torso_descent_working.py --target bottle
"
```

---

## 10. Running the Agent

### 10.1 Checklist Before Starting

- [ ] Robot is powered on and ROS master is running
- [ ] Workstation is on the same LAN as the robot
- [ ] YOLO service is running (`curl localhost:5001/health` returns OK)
- [ ] `.env` file exists with valid API keys
- [ ] `ROBOT_IP` and `HOST_IP` are set correctly

### 10.2 Launch

```bash
./run_agent.sh
```

Or manually:

```bash
export ROBOT_IP=10.68.0.1
export HOST_IP=$(hostname -I | awk '{print $1}')
export ROBOT_HOSTNAME=tiago-161c

# Load API keys
set -a; source .env; set +a

# Run inside Docker with full ROS environment
docker exec -it tiago_ros bash -c "
  source /opt/ros/melodic/setup.bash &&
  source /workspace/pal_ws/devel/setup.bash &&
  export ROS_MASTER_URI=http://${ROBOT_IP}:11311 &&
  export ROS_IP=${HOST_IP} &&
  export EDENAI_API_KEY=${EDENAI_API_KEY} &&
  export GROQ_API_KEY=${GROQ_API_KEY} &&
  cd /workspace &&
  python3 embodied_agent.py
"
```

### 10.3 Interacting with the Agent

Once running, the agent listens for text input (or voice if a microphone is available):

```
> grab the bottle
[Agent] Detected: bottle (0.87), cup (0.72)
[Agent] Executing: grab_object → target=bottle
[reach_object_v5] Grasping bottle at [0.82, -0.04, 0.91]...
[reach_object_v5] Grasp successful.
[Agent] Task complete.

> grab the small bottle and give it to Alice
[Agent] Step 1: grab_object → small bottle
[Agent] Step 2: handover_to_person → Alice
...
```

Type `exit` or press `Ctrl+C` to stop. The agent will open the gripper and return the arm home before shutting down.

---

## 11. Running the Evaluation Suite

### 11.1 Offline Evaluations (no robot, run on host)

```bash
cd eval

# Detection quality: YOLO vs VLM bbox comparison (demo mode uses synthetic images)
python3 eval_detection.py --demo

# Depth refinement quality
python3 eval_depth_refine.py --demo

# Spatial reasoning accuracy
python3 eval_statemanager.py --demo

# Latency profiling (skips VLM and face APIs to avoid cost)
python3 eval_latency.py --skip_vlm --skip_face

# Generate summary report
python3 report.py
```

Results are saved to `eval/data/` and `eval/results/`.

### 11.2 Live Evaluations (require robot + Docker)

First set up the eval environment:

```bash
source eval_setup.sh
# or: source eval_setup.sh --docker  (drops into Docker shell)
```

Then inside Docker:

```bash
docker exec -it tiago_ros bash -c "
  source /opt/ros/melodic/setup.bash &&
  source /workspace/pal_ws/devel/setup.bash &&
  export ROS_MASTER_URI=http://${ROBOT_IP}:11311 &&
  export ROS_IP=${HOST_IP} &&
  cd /workspace/eval &&

  # Grasp trials (15 attempts at grasping a bottle)
  python3 eval_grasp_recorder.py --object bottle --trials 15

  # Search trials (10 attempts to find a bottle via head scan)
  python3 eval_search_recorder.py --target bottle --trials 10
"
```

---

## 12. Troubleshooting

### `rospy not available — live camera feed disabled`

The Python process cannot find the `rospy` package. Cause: ROS setup not sourced before running.

```bash
# Fix: always source ROS setup before running agent code inside Docker
source /opt/ros/melodic/setup.bash
source /workspace/pal_ws/devel/setup.bash
```

### `rostopic list` times out or returns nothing

The workstation cannot reach the robot ROS master.

```bash
# Check connectivity
ping ${ROBOT_IP}

# Check ROS master URI
echo $ROS_MASTER_URI   # should be http://<ROBOT_IP>:11311

# Check ROS_IP matches your workstation IP
echo $ROS_IP
hostname -I             # should match
```

Also verify the robot is running: `ssh user@${ROBOT_IP}` → `rosnode list`.

### TF lookup timeout

The agent cannot get the camera → base_footprint transform.

```bash
# Check TF is publishing inside Docker
docker exec -it tiago_ros bash -c "
  source /opt/ros/melodic/setup.bash &&
  export ROS_MASTER_URI=http://${ROBOT_IP}:11311 &&
  export ROS_IP=${HOST_IP} &&
  rosrun tf tf_echo base_footprint xtion_rgb_optical_frame
"
# Should print a transform every 1-2 seconds
```

If it hangs, the robot head may not be publishing TF. Check that `tiago_bringup` is fully running on the robot.

### Depth image all zeros / NaN

```bash
# Check depth topic is publishing inside Docker
docker exec -it tiago_ros bash -c "
  source /opt/ros/melodic/setup.bash &&
  export ROS_MASTER_URI=http://${ROBOT_IP}:11311 &&
  export ROS_IP=${HOST_IP} &&
  rostopic hz /xtion/depth_registered/image_raw
"
# Should show ~30 Hz
```

If not publishing, check that the Xtion camera is powered and the `openni2_launch` driver is running on the robot.

### VLM picks the wrong object

The VLM may be selecting by class name rather than visual description. Try being more explicit:

```
> grab the small bottle on the left
```

You can also set `TARGET_DESCRIPTION` explicitly:

```bash
TARGET_DESCRIPTION="the small water bottle, not the large one" \
python3 reach_object_v5_torso_descent_working.py --target bottle
```

### MoveIt planning failure

```
[ERROR] Motion planning failed: NO_IK_SOLUTION
```

Possible causes:
1. Object is outside the arm's reachable workspace (>1.2 m forward, >0.65 m lateral)
2. Collision objects from a previous run are still in the planning scene
3. Torso is too low or too high

```bash
# Clear MoveIt planning scene
docker exec -it tiago_ros bash -c "
  source /opt/ros/melodic/setup.bash &&
  export ROS_MASTER_URI=http://${ROBOT_IP}:11311 &&
  rosservice call /clear_octomap
"
```

Increase torso height manually if the object is too low:
```
> raise torso
# or interact with /torso_controller directly
```

### YOLO service not reachable from Docker

Since the container uses `--network=host`, `localhost:5001` inside Docker **is** the host. Verify:

```bash
docker exec tiago_ros bash -c "python3 -c \"import urllib.request; print(urllib.request.urlopen('http://localhost:5001/health').read())\""
```

If this fails, the YOLO service is not running on the host. Start it with `python3 yolo_service.py`.

### `ModuleNotFoundError: No module named 'cv2'` (inside Docker)

OpenCV is not installed in the Docker container.

```bash
docker exec -it tiago_ros bash -c "pip3 install opencv-python-headless"
```

### Grasp consistently misses (too high / too low / too far)

Recalibrate using `calibrate_grasp_interactive.py` (see §9) or manually edit `grasp_offsets.yaml`:

```yaml
grasp_dx: 0.22   # Increase if arm stops short of object
grasp_dz: 0.04   # Decrease (or go negative) if gripper is too high
```

---

## Appendix: Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ROBOT_IP` | `10.68.0.1` | Robot ROS master IP |
| `HOST_IP` | auto-detect | Workstation IP (used as `ROS_IP`) |
| `ROBOT_HOSTNAME` | `tiago-161c` | Robot hostname |
| `DOCKER_CONTAINER` | `tiago_ros` | Docker container name |
| `ROS_SETUP` | `/opt/ros/melodic/setup.bash` | ROS distro setup script |
| `PAL_SETUP` | `/workspace/pal_ws/devel/setup.bash` | PAL workspace setup |
| `EDENAI_API_KEY` | — | Eden AI API key (required) |
| `GROQ_API_KEY` | — | Groq API key (required) |
| `YOLO_PORT` | `5001` | YOLO service HTTP port |
| `FACE_SERVICE_URL` | `http://localhost:5002` | Face recognition service URL |
| `GRASP_DX` | from YAML | Forward grasp offset (metres) |
| `GRASP_DY` | from YAML | Lateral grasp offset (metres) |
| `GRASP_DZ` | from YAML | Vertical grasp offset (metres) |
| `TARGET_DESCRIPTION` | — | VLM chain-of-thought hint for grasping |
| `OBSTACLE_OBJECTS` | — | JSON list of obstacles for MoveIt collision |
| `CYCLONEDDS_URI` | — | Path to CycloneDDS config XML |
