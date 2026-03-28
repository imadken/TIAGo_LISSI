#!/bin/bash
# Launcher script for TIAGo Embodied Agent
#
# Configure by setting env vars before running, e.g.:
#   ROBOT_IP=192.168.1.100 ROBOT_HOSTNAME=tiago ./run_agent.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ── Network configuration ──────────────────────────────────────────────────────
# Override these with your own values:
#   export ROBOT_IP=192.168.1.100
#   export HOST_IP=192.168.1.50
#   export ROBOT_HOSTNAME=tiago
ROBOT_IP="${ROBOT_IP:-10.68.0.1}"
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
ROBOT_HOSTNAME="${ROBOT_HOSTNAME:-tiago-161c}"

# ── Load API keys ──────────────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
    echo "[✓] API keys loaded from .env"
else
    echo "[✗] .env not found — copy .env.example to .env and fill in your keys"
    exit 1
fi

# ── Ensure robot hostname resolves ────────────────────────────────────────────
grep -q "$ROBOT_HOSTNAME" /etc/hosts || echo "$ROBOT_IP $ROBOT_HOSTNAME" | sudo tee -a /etc/hosts > /dev/null

# ── ROS environment ────────────────────────────────────────────────────────────
export ROS_MASTER_URI=http://${ROBOT_IP}:11311
export ROS_IP=${HOST_IP}

ROS_SETUP="${ROS_SETUP:-/opt/ros/melodic/setup.bash}"
PAL_SETUP="${PAL_SETUP:-/workspace/pal_ws/devel/setup.bash}"

[ -f "$ROS_SETUP" ] && source "$ROS_SETUP"
[ -f "$PAL_SETUP" ] && source "$PAL_SETUP"

echo "[✓] ROS_MASTER_URI=$ROS_MASTER_URI"
echo "[✓] ROS_IP=$ROS_IP"

# ── Run agent ──────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"
python3 embodied_agent.py
