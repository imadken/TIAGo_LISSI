# TIAGo Embodied AI Agent — Complete User Manual for Absolute Beginners

**ROS 1 Melodic + Python 3.6 (Docker) + Python 3.10 (Host)**

> This manual takes you from ZERO to running the embodied AI agent on a real TIAGo robot.

---

## TABLE OF CONTENTS

1. [PART 0: SYSTEM OVERVIEW](#part-0-system-overview)
2. [PART 1: PREREQUISITES](#part-1-prerequisites)
3. [PART 2: NETWORK SETUP](#part-2-network-setup)
4. [PART 3: DOCKER & IMAGES](#part-3-docker--images)
5. [PART 4: STARTING SERVICES](#part-4-starting-services)
6. [PART 5: FIRST COMMANDS](#part-5-first-commands)
7. [PART 6: TROUBLESHOOTING](#part-6-troubleshooting)
8. [PART 7: REFERENCE](#part-7-reference)

---

# PART 0: SYSTEM OVERVIEW

## What You're Building

An AI system that:
- **Listens** to natural language commands
- **Sees** the world with RGB-D camera
- **Thinks** using a Vision-Language Model (Gemini 2.0 Flash)
- **Acts** by controlling a TIAGo humanoid robot

## Python Versions (Critical to Understand!)

This project uses **THREE different Python environments**:

```
┌─────────────────────────────────────────────────────────┐
│        Your Workstation (Ubuntu 20.04/22.04)           │
│                                                          │
│  Python 3.10+ (system Python)                           │
│  ├─ YOLO detection service (Flask HTTP server)          │
│  ├─ Face recognition service                           │
│  └─ Virtual environment: venv                          │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Docker Container (palrobotics/tiago_melodic)   │  │
│  │                                                  │  │
│  │  Python 3.6 (Ubuntu 18.04 default)              │  │
│  │  ├─ ROS 1 Melodic agent code                   │  │
│  │  ├─ embodied_agent.py (Python 3.6)            │  │
│  │  └─ Motion planning, vision processing          │  │
│  │                                                  │  │
│  │  Python 2.7 (ROS 1 Melodic base)                │  │
│  │  ├─ rospy (ROS 1 Python bindings)              │  │
│  │  └─ ROS core system                             │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
└──────────────┬──────────────────────────────────────────┘
               │ WiFi/Ethernet (LAN, same subnet)
               │ ROS 1 topics over network
               │
┌──────────────▼──────────────────────────────────────────┐
│        TIAGo Robot @ 10.68.0.1                         │
│                                                         │
│  ROS 1 Master (connects workstation to robot)          │
│  ├─ Camera: /xtion/rgb/image_raw, /xtion/depth/...   │
│  ├─ Motors: base, arm, gripper, head                  │
│  ├─ Planning: MoveIt (motion planning)                │
│  └─ Play-motion: prerecorded poses (home, wave, etc.) │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Why This Architecture?

- **Python 3.10 on host** = modern packages (YOLO, OpenCV, Flask work best on Python 3.10+)
- **Python 3.6 in Docker** = compatible with ROS 1 Melodic (Ubuntu 18.04)
- **Python 2.7 in Docker** = ROS 1 core system
- **Docker isolation** = keeps everything clean; no conflicts

### Key Concepts

**ROS 1 Melodic** is middleware that:
- Runs on the **TIAGo robot** as the **ROS Master** (at 10.68.0.1:11311)
- Lets your **workstation** (ROS client) subscribe to topics and send commands
- Communication happens over your local network (LAN)

```bash
export ROS_MASTER_URI=http://10.68.0.1:11311  # Where is ROS Master?
export ROS_IP=10.68.0.128                      # My workstation's IP
```

---

# PART 1: PREREQUISITES

## Step 1: Check Your Workstation

```bash
# 1. Verify Ubuntu 20.04 or 22.04
lsb_release -a
# Output should show: Release: 20.04  OR  Release: 22.04

# 2. Verify Python 3.10+ is available
python3 --version
# Output: Python 3.10.x  or higher

# 3. If missing, install it:
sudo apt update
sudo apt install -y python3.10 python3-pip
```

## Step 2: Install Docker Engine

Docker lets us run **Ubuntu 18.04 + ROS 1 Melodic + Python 3.6** without affecting your system Python 3.10.

```bash
# Official Docker installation
curl -fsSL https://get.docker.com | sh

# Add yourself to docker group (avoid typing sudo)
sudo usermod -aG docker $USER

# CRITICAL: Log out and log back in, OR run:
newgrp docker

# Verify
docker --version
docker run hello-world  # Should work WITHOUT sudo
```

## Step 3: Install Git

```bash
sudo apt install -y git
git --version
```

## Step 4: Clone Your Project

```bash
cd ~/Desktop  # or wherever
git clone https://github.com/Abderrahmeneben8/TIAGo_Benhamlaoui.git
cd tiago-embodied-agent

# Verify contents
ls -la | grep -E "docker|Dockerfile|embodied_agent|requirements"
# Should see: docker/, Dockerfile, embodied_agent.py, requirements.txt
```

## Step 5: Install Python 3.10 Dependencies (Host Only)

These packages run on YOUR workstation (not in Docker):

```bash
# Create virtual environment for host packages
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# If requirements.txt missing, manually install:
pip install ultralytics opencv-python numpy scipy flask \
    requests pyyaml python-dotenv face-recognition SpeechRecognition
```

**Verify:**
```bash
python3 -c "from ultralytics import YOLO; print('✓ YOLO loaded')"
python3 -c "import cv2; print('✓ OpenCV loaded')"
python3 -c "import flask; print('✓ Flask loaded')"
```

---

# PART 2: NETWORK SETUP

## Step 1: Find Robot IP

Ask your lab technician or check the robot's display. Default: **10.68.0.1**

```bash
# Test reaching robot
ping -c 3 10.68.0.1
# Should show: bytes from 10.68.0.1 (not "unreachable")
```

## Step 2: Find Your Workstation IP

```bash
# Quick method
hostname -I | awk '{print $1}'
# Output: something like 10.68.0.128

# Detailed view
ip a
# Look for your active interface (marked UP):
# wlo1: WiFi interface  OR  eth0/enp0s3: Ethernet interface
```

## Step 3: Create ROS Configuration File

```bash
# Create environment file
cat > ~/.bashrc_tiago << 'EOF'
# ROS 1 Melodic Environment — TIAGo Robot
export ROBOT_IP=10.68.0.1           # Robot IP (change if different)
export ROBOT_HOSTNAME=tiago-161c    # Robot hostname
export HOST_IP=$(hostname -I | awk '{print $1}')  # Your workstation IP
export ROS_MASTER_URI=http://${ROBOT_IP}:11311
export ROS_IP=${HOST_IP}

echo "[✓] TIAGo ROS environment loaded"
echo "    ROS_MASTER_URI=$ROS_MASTER_URI"
echo "    ROS_IP=$ROS_IP"
EOF

# Load for current session
source ~/.bashrc_tiago

# Make permanent (auto-load on new terminals)
echo "source ~/.bashrc_tiago" >> ~/.bashrc
```

## Step 4: Setup API Keys

```bash
# Copy template
cp .env.example .env

# Edit API keys
nano .env
```

Add these (get keys from Eden AI and Groq free tiers):

```bash
EDENAI_API_KEY=your_key_from_edenai
GROQ_API_KEY=your_key_from_groq
```

Save with `Ctrl+X, Y, Enter` (nano) or `:wq` (vim).

---

# PART 3: DOCKER & IMAGES

## Step 1: Build ROS 1 Melodic Agent Image

This image has:
- **Ubuntu 18.04** (Bionic)
- **ROS 1 Melodic**
- **Python 2.7** (ROS core)
- **Python 3.6** (agent code)
- MoveIt, PAL packages, cv_bridge, etc.

```bash
cd /path/to/tiago-embodied-agent

# Build (first time takes 10-15 minutes)
docker build -t tiago-agent -f docker/Dockerfile .
```

**Expected output:**
```
Sending build context to Docker daemon ...
Step 1/25 : FROM palroboticssl/tiago_melodic_robot:latest
...
Successfully tagged tiago-agent:latest
```

## Step 2: Build YOLO Service Image

```bash
# Build YOLO image (takes ~5 minutes)
docker build -t tiago-yolo-service -f docker/Dockerfile.yolo .
```

## Step 3: Verify Both Images Exist

```bash
docker images | grep tiago
# Output:
# REPOSITORY             TAG    IMAGE ID     CREATED
# tiago-agent            latest abc123...    5 min ago
# tiago-yolo-service     latest xyz789...    2 min ago
```

---

# PART 4: STARTING SERVICES

You need **3-4 terminal windows** open simultaneously.

### Terminal 1: Start YOLO Detection Service (Host Python 3.10)

```bash
# Activate Python 3.10 virtual environment
source venv/bin/activate

# Start YOLO HTTP server
python3 yolo_service.py
```

**Expected output (after ~30 seconds):**
```
[YOLO Service] Loading yoloe-26s-seg...
[YOLO Service] Warming up...
[YOLO Service] yoloe-26s-seg ready on port 5001
 * Running on http://0.0.0.0:5001
```

**Keep this terminal open!** If it crashes, restart.

### Terminal 2: Start Docker Container (Python 3.6 + ROS 1)

```bash
# Load ROS configuration
source ~/.bashrc_tiago

# Start Docker container with ROS 1 Melodic
docker run -dit \
  --name tiago_ros \
  --net host \
  -e DISPLAY=$DISPLAY \
  -e ROS_MASTER_URI=${ROS_MASTER_URI} \
  -e ROS_IP=${ROS_IP} \
  -v $(pwd):/workspace \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --env-file .env \
  tiago-agent bash

# Wait a moment
sleep 2

# Verify container started
docker ps | grep tiago_ros
# Should show container as "Up" (not "Exited")
```

### Terminal 3: Test ROS Connection to Robot

```bash
# Attach to Docker container
docker exec -it tiago_ros bash

# Inside container, verify ROS environment
echo $ROS_MASTER_URI
echo $ROS_IP
# Both should have values

# List topics from the robot
rostopic list | head -20
# You should see:
# /xtion/rgb/image_raw
# /xtion/depth/image_raw
# /joint_states
# /scan
# /cmd_vel
# ... (dozens more)
```

**If you see `/xtion/rgb/image_raw` → SUCCESS!** You're connected to the robot.

### Terminal 4 (Optional): Monitor YOLO Service

```bash
# Outside Docker, in a new terminal:
source venv/bin/activate

# Test YOLO service
curl -s http://localhost:5001/health | python3 -m json.tool
# Output: {"status": "ok", "model": "yoloe-26s-seg", ...}
```

---

# PART 5: FIRST COMMANDS

### Command 1: Move Robot Head

Inside Terminal 3 (inside Docker container):

```bash
# Move head down (look at table)
rostopic pub /head_controller/command trajectory_msgs/JointTrajectory \
  "joint_names: ['head_1_joint', 'head_2_joint']
points:
- positions: [0.0, -0.3]
  time_from_start: {secs: 2, nsecs: 0}" --once
```

**Robot's head should move down.**

### Command 2: Return to Home Pose

```bash
# Arms down, head forward
docker exec -it tiago_ros bash -c "source /opt/ros/melodic/setup.bash && \
rostopic pub /play_motion/goal play_motion_msgs/PlayMotionActionGoal \
\"goal: {motion_name: 'home', skip_planning: false}\" --once"
```

### Command 3: List All Nodes Running on Robot

```bash
# Inside Docker container
rosnode list
# Shows all active ROS nodes on the robot
```

### Command 4: Echo Camera Frames

```bash
# View a single image from camera
rostopic echo /xtion/rgb/image_raw --once
```

---

# PART 6: TROUBLESHOOTING

## "Cannot ping robot"

```
ping 10.68.0.1
ping: sendmsg: Network is unreachable
```

**Fixes:**

1. **Robot is off?** Turn it on
2. **Wrong IP?** Ask lab technician. Try SSH first: `ssh pal@10.68.0.1`
3. **Different network?** Join same WiFi as robot
4. **Firewall?** Try Ethernet cable instead of WiFi

---

## "rostopic list shows nothing"

```bash
docker exec -it tiago_ros bash
rostopic list
# (no output or only /rosout)
```

**Fix:**

```bash
# Check environment variables inside container
echo $ROS_MASTER_URI
echo $ROS_IP
# Both should have values

# If empty, restart container:
docker rm -f tiago_ros
# Then re-run Terminal 2 docker run command above

# Reset ROS daemon
rosgraph
```

---

## "Docker container exits"

```
docker ps
# Container shows as "Exited"
```

**Fix:**

```bash
# Check logs
docker logs tiago_ros

# Common issues:
# 1. Build failed
docker build -t tiago-agent -f docker/Dockerfile .

# 2. Permission error
sudo usermod -aG docker $USER
newgrp docker

# 3. Restart
docker rm -f tiago_ros
# Re-run Terminal 2 docker run command
```

---

## "YOLO service won't start"

```
python3 yolo_service.py
ModuleNotFoundError: No module named 'ultralytics'
```

**Fix:**

```bash
# Activate venv
source venv/bin/activate

# Reinstall
pip install --upgrade ultralytics opencv-python

# Try again
python3 yolo_service.py
```

---

## "Python version confusion"

**Remember:**

- **Host machine**: Python 3.10+ (where YOLO runs)
- **Docker container**: Python 3.6 (where agent code runs)
- **Both have** Python 2.7 (ROS 1 core system)

**Check what's running:**

```bash
# On your workstation
python3 --version  # Should be 3.10+

# Inside Docker container
python3 --version  # Will be 3.6
python2 --version  # Will be 2.7
```

---

# PART 7: REFERENCE

## Essential ROS 1 Commands

```bash
# Inside Docker container (where you can use ros commands):

# List topics
rostopic list

# View messages on a topic
rostopic echo /xtion/rgb/image_raw

# Publish to a topic
rostopic pub /topic_name MsgType "data: value" --once

# List nodes
rosnode list

# View ROS graph
rosgraph

# Launch a file
roslaunch package launch_file.launch

# View TF transforms
rosrun tf2_tools view_frames
```

## Key ROS 1 Topics for TIAGo

| Topic | Type | Purpose |
|-------|------|---------|
| `/xtion/rgb/image_raw` | sensor_msgs/Image | RGB camera frames |
| `/xtion/depth/image_raw` | sensor_msgs/Image | Depth camera (16-bit) |
| `/joint_states` | sensor_msgs/JointState | All joint angles |
| `/mobile_base_controller/cmd_vel_unstamped` | geometry_msgs/Twist | Move base (wheels) |
| `/head_controller/command` | trajectory_msgs/JointTrajectory | Move head |
| `/play_motion/goal` | play_motion_msgs/PlayMotionActionGoal | Prerecorded motions |
| `/cmd_vel` | geometry_msgs/Twist | Alternative velocity |
| `/gripper_controller/command` | std_msgs/Float64 | Gripper (open/close) |

## Docker Quick Reference

```bash
# Build image
docker build -t myimage -f Dockerfile .

# Run interactively
docker run -it myimage bash

# Run in background
docker run -dit myimage bash

# Attach to running container
docker exec -it container_name bash

# View logs
docker logs container_name

# List containers
docker ps              # Running only
docker ps -a           # All containers

# Stop/remove
docker stop container_name
docker rm container_name
docker rm -f container_name  # Force

# Important flags:
# --net host           ← Share workstation's IP (needed for ROS!)
# -e VAR=value         ← Set environment variable
# -v /path:/path       ← Mount directory
# --name myname        ← Container name
# -it                  ← Interactive+TTY
# -d                   ← Detached (background)
```

## Environment Variables (ROS 1)

```bash
export ROS_MASTER_URI=http://10.68.0.1:11311  # Where is ROS Master?
export ROS_IP=10.68.0.128                      # My IP address
export ROS_HOME=$HOME/.ros                     # ROS config dir
export ROS_LOG_DIR=$HOME/.ros/log              # Logs location
```

## Agent Architecture Overview

```
User speaks/types command
  ↓
Perception Manager (gets camera frames)
  ↓
YOLO Service (detects objects)
  ↓
VLM Reasoner (Gemini 2.0 Flash decides what to do)
  ↓
State Manager (builds scene graph)
  ↓
Skill Executor (chooses best skill)
  ↓
Skill publishes ROS commands
  ↓
Robot executes (grasps, moves, etc.)
  ↓
Result reported to user
```

## Common Workflow from Bash History

**Start from scratch:**

```bash
# Terminal 1: YOLO service (Python 3.10 on host)
source venv/bin/activate
python3 yolo_service.py

# Terminal 2: Docker container (Python 3.6 + ROS 1)
source ~/.bashrc_tiago
docker run -dit --name tiago_ros --net host \
  -e ROS_MASTER_URI=$ROS_MASTER_URI \
  -e ROS_IP=$ROS_IP \
  -v $(pwd):/workspace \
  --env-file .env \
  tiago-agent bash

# Terminal 3: Inside Docker
docker exec -it tiago_ros bash
source /opt/ros/melodic/setup.bash
python3 embodied_agent.py

# Terminal 4: Send commands to agent
docker exec -it tiago_ros bash -c "source /opt/ros/melodic/setup.bash && \
rostopic pub /head_controller/command trajectory_msgs/JointTrajectory \
\"joint_names: ['head_1_joint', 'head_2_joint']
points:
- positions: [0.0, -0.3]
  time_from_start: {secs: 2, nsecs: 0}\" --once"
```

## Project File Structure

```
tiago-embodied-agent/
├── embodied_agent.py           ← Main entry point (Python 3.6)
├── vlm_reasoner.py             ← LLM integration (Python 3.6)
├── perception_manager_v2.py    ← Vision (Python 3.6)
├── state_manager.py            ← Scene graph (Python 3.6)
├── face_manager.py             ← Face recognition (Python 3.6)
├── yolo_service.py             ← HTTP YOLO (Python 3.10 on host)
├── face_recognition_service.py ← HTTP faces (Python 3.10 on host)
├── skills/
│  ├── grab_bottle.py
│  ├── search_with_head.py
│  ├── handover.py
│  └── ...
├── docker/
│  ├── Dockerfile              ← ROS 1 Melodic base image
│  ├── Dockerfile.yolo         ← Python 3.10 YOLO image
│  └── requirements_ros.txt    ← Python 3.6 packages
├── requirements.txt            ← Python 3.10 host packages
├── .env                        ← API keys (secret!)
└── docker-compose.yml          ← Container orchestration
```

## Reset Everything

If things break:

```bash
# Stop all services
docker-compose down
pkill yolo_service
pkill python3

# Remove old containers/images
docker rm -f tiago_ros
docker rmi tiago-agent tiago-yolo-service

# Rebuild
docker build -t tiago-agent -f docker/Dockerfile .
docker build -t tiago-yolo-service -f docker/Dockerfile.yolo .

# Start fresh from Terminal 1, 2, 3 above
```

---

## Summary: The Three Pythons

| Where | Python | Purpose |
|-------|--------|---------|
| **Robot (ROS core)** | 2.7 | ROS 1 Melodic base system |
| **Docker (ROS + Agent)** | 3.6 | Agent code + older packages |
| **Your Workstation** | 3.10+ | YOLO, OpenCV, Flask, modern packages |

**Key Point:** Docker keeps them separate. No conflicts!

---

**End of Manual**

You now have everything needed to run the TIAGo embodied AI agent from scratch.

**Questions? Check PART 6 TROUBLESHOOTING first, then contact your lab supervisor.**
