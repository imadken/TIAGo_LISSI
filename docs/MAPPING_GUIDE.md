# Complete Autonomous Mapping Guide
## Docker + RViz + SLAM + Save Map

---

## 🚀 Quick Start (Full Workflow)

### Step 1: Start Docker with Display Support

```bash
# Allow X11 connections (run once per boot)
xhost +local:docker

# Start Docker container with display, ROS Melodic
docker run -it --net host \
  -e DISPLAY=$DISPLAY \
  -e ROS_MASTER_URI=http://10.68.0.1:11311 \
  -e ROS_IP=10.68.0.128 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/lissi/tiago_public_ws:/workspace \
  --name tiago_mapper \
  ros:melodic bash
```

### Step 2: Inside Docker - Install Dependencies

```bash
# Update and install required packages
apt update
apt install -y \
  ros-melodic-rviz \
  ros-melodic-map-server \
  ros-melodic-nav-msgs \
  ros-melodic-sensor-msgs \
  python3-pip

# Source ROS
source /opt/ros/melodic/setup.bash
```

### Step 3: Launch RViz

In the same Docker terminal:

```bash
rosrun rviz rviz
```

**Configure RViz:**
1. **Fixed Frame**: Change to `map` (dropdown at top)
2. **Add displays** (click "Add" button at bottom left):
   - **Map**:
     - Click "Add" → "By topic" → `/map` → OK
     - Color Scheme: map (or costmap)
   - **LaserScan**:
     - Click "Add" → "By topic" → `/scan` → OK
     - Size: 0.05
     - Color: Red or Yellow
   - **RobotModel**:
     - Click "Add" → "By display type" → "RobotModel" → OK
   - **TF** (optional - shows coordinate frames):
     - Click "Add" → "By display type" → "TF" → OK
3. **Save config** (optional): File → Save Config As → `/workspace/mapping.rviz`

---

## 🗺️ Running Autonomous Mapping

### Terminal 1: SSH to Robot - Start SLAM

```bash
ssh pal@10.68.0.1
# Password: pal

roslaunch tiago_2dnav mapping.launch
```

### Terminal 2: Docker - Run Autonomous Mapper

```bash
# Enter Docker (if you closed it)
docker exec -it tiago_mapper bash

# Set ROS environment
export ROS_MASTER_URI=http://10.68.0.1:11311
export ROS_IP=10.68.0.128
source /opt/ros/melodic/setup.bash
cd /workspace

# Run mapper (choose one mode)
python3 autonomous_mapper.py reactive 5    # 5 min reactive exploration
python3 autonomous_mapper.py spiral 10     # 10 min spiral pattern
python3 autonomous_mapper.py random 3      # 3 min random walk
```

**Watch in RViz** as the map builds in real-time!

---

## 💾 Saving the Map

### Option A: Save from Docker

```bash
# In Docker terminal
rosrun map_server map_saver -f /workspace/my_room_map

# Files saved:
# - /workspace/my_room_map.pgm  (image)
# - /workspace/my_room_map.yaml (metadata)
```

### Option B: Save from Robot

```bash
# SSH to robot
ssh pal@10.68.0.1
rosrun map_server map_saver -f ~/my_room_map

# Copy to your laptop
exit
scp pal@10.68.0.1:~/my_room_map.* ~/tiago_public_ws/
```

---

## 👀 Viewing the Saved Map

### In Image Viewer (on your laptop)

```bash
cd ~/tiago_public_ws
eog my_room_map.pgm
# or
xdg-open my_room_map.pgm
```

### In RViz (load saved map)

```bash
# In Docker
rosrun map_server map_server /workspace/my_room_map.yaml

# In RViz, the map will load from file instead of live SLAM
```

---

## 🔧 Troubleshooting

### Docker Issues

**Q: "Cannot connect to X server"**
```bash
# Run on host (laptop)
xhost +local:docker
echo $DISPLAY  # Should show :0 or :1
```

**Q: "Permission denied"**
```bash
# Add yourself to docker group
sudo usermod -aG docker $USER
# Then log out and back in
```

**Q: Container already exists**
```bash
# Remove old container
docker rm tiago_mapper

# Or enter existing container
docker exec -it tiago_mapper bash
```

### ROS Issues

**Q: "No topics visible in RViz"**
```bash
# Check ROS connection
rostopic list
rostopic echo /scan -n1

# Verify environment variables
echo $ROS_MASTER_URI  # Should be http://10.68.0.1:11311
echo $ROS_IP          # Should be your laptop IP (10.68.0.128)
```

**Q: "Map not appearing"**
```bash
# Check if SLAM is publishing
rostopic hz /map

# Check if topic exists
rostopic list | grep map
```

### Robot Stuck

**Q: Robot stuck in corner**
```bash
# Use turn and go script
cd /workspace
python3 turn_and_go.py 2.0
```

---

## 📊 Complete Example Session

```bash
# ========== TERMINAL 1: Host Laptop ==========
xhost +local:docker
docker run -it --net host \
  -e DISPLAY=$DISPLAY \
  -e ROS_MASTER_URI=http://10.68.0.1:11311 \
  -e ROS_IP=10.68.0.128 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/lissi/tiago_public_ws:/workspace \
  --name tiago_mapper \
  ros:melodic bash

# Inside Docker:
apt update && apt install -y ros-melodic-rviz ros-melodic-map-server
source /opt/ros/melodic/setup.bash
rosrun rviz rviz  # Configure as described above


# ========== TERMINAL 2: SSH to Robot ==========
ssh pal@10.68.0.1
roslaunch tiago_2dnav mapping.launch


# ========== TERMINAL 3: Docker (run mapper) ==========
docker exec -it tiago_mapper bash
export ROS_MASTER_URI=http://10.68.0.1:11311
export ROS_IP=10.68.0.128
source /opt/ros/melodic/setup.bash
cd /workspace

# Run for 5 minutes
python3 autonomous_mapper.py reactive 5

# When done, save map
rosrun map_server map_saver -f /workspace/my_office_map


# ========== TERMINAL 1: View map on laptop ==========
cd ~/tiago_public_ws
eog my_office_map.pgm
```

---

## 🎯 New Mapper Features

The enhanced mapper now has:

✅ **3 Exploration Modes**:
- `reactive` - Pure obstacle avoidance (recommended for cluttered spaces)
- `spiral` - Systematic spiral coverage (good for open rooms)
- `random` - Random walk (explores different areas)

✅ **Safety Features**:
- Zoned obstacle detection (front/left/right)
- Proportional speed control (slows near obstacles)
- Odometry-based rotation (precise turns)
- Watchdog timer (prevents runaway)
- Side collision protection

✅ **Statistics**:
- Distance traveled
- Obstacles avoided
- Exploration time

**Usage:**
```bash
python3 autonomous_mapper.py <mode> <minutes>

# Examples:
python3 autonomous_mapper.py reactive 5   # 5 min, avoid obstacles
python3 autonomous_mapper.py spiral 10    # 10 min, spiral pattern
python3 autonomous_mapper.py random 3     # 3 min, random exploration
```

---

## 🔄 Reusing Docker Container

Don't create a new container each time!

**Next session:**
```bash
# Start existing container
docker start tiago_mapper

# Enter it
docker exec -it tiago_mapper bash

# Set environment
export ROS_MASTER_URI=http://10.68.0.1:11311
export ROS_IP=10.68.0.128
source /opt/ros/melodic/setup.bash
cd /workspace
```

---

## 📝 Files Created

- `autonomous_mapper.py` - Enhanced mapper with 3 modes
- `turn_and_go.py` - Get unstuck from corners
- `navigate_home.py` - Navigate to (0,0,0)
- `go_to_pose.py` - Navigate to any position
- `check_robot_pose.py` - Check current position
- `check_navigation.py` - Check navigation status
- `MAPPING_GUIDE.md` - This guide

Happy Mapping! 🗺️🤖
