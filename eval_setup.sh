#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# eval_setup.sh  —  Start the eval environment for TIAGo experiments
#
# What this does:
#   1. Loads API keys from .env
#   2. Starts the ROS Melodic docker container
#   3. Prints which eval scripts need docker vs run on host directly
#
# Usage:
#   source eval_setup.sh          # sets env vars in current shell
#   bash eval_setup.sh --docker   # also drops you into the docker container
#
# Configuration (override via env vars):
#   ROBOT_IP         Robot ROS master IP          (default: 10.68.0.1)
#   HOST_IP          This machine's IP             (default: auto-detect)
#   DOCKER_CONTAINER Docker container name         (default: tiago_ros)
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-tiago_ros}"
DOCKER_WORKSPACE="/workspace"

ROBOT_IP="${ROBOT_IP:-10.68.0.1}"
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"

# ── 1. Load API keys ──────────────────────────────────────────────────────────
ENV_FILE="$WORKSPACE/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "[✓] API keys loaded from .env"
    echo "    EDENAI_API_KEY = ${EDENAI_API_KEY:0:8}…"
    echo "    GROQ_API_KEY   = ${GROQ_API_KEY:0:8}…"
else
    echo "[✗] .env not found at $ENV_FILE"
    echo "    cp $WORKSPACE/.env.example $WORKSPACE/.env"
    echo "    # then fill in your API keys"
fi

# ── 2. Start / resume docker container ────────────────────────────────────────
if docker inspect "$DOCKER_CONTAINER" > /dev/null 2>&1; then
    STATUS=$(docker inspect -f '{{.State.Status}}' "$DOCKER_CONTAINER")
    if [ "$STATUS" = "running" ]; then
        echo "[✓] Docker '$DOCKER_CONTAINER' already running"
    else
        echo "[→] Starting docker container '$DOCKER_CONTAINER' …"
        docker start "$DOCKER_CONTAINER"
        sleep 1
        echo "[✓] Container started"
    fi
else
    echo "[✗] Container '$DOCKER_CONTAINER' not found"
    echo "    Create it with (see docs/INSTALLATION_MANUAL.md §4 for details):"
    echo ""
    echo "    docker run -dit --name $DOCKER_CONTAINER \\"
    echo "      --net=host \\"
    echo "      -v $WORKSPACE:$DOCKER_WORKSPACE \\"
    echo "      -v /tmp/.X11-unix:/tmp/.X11-unix \\"
    echo "      palroboticssl/tiago_melodic_robot:latest bash"
fi

# ── 3. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Eval scripts — run on HOST (no ROS needed):"
echo "   cd $WORKSPACE/eval"
echo "   python3 eval_latency.py --skip_vlm --skip_face"
echo "   python3 eval_detection.py --demo"
echo "   python3 eval_depth_refine.py --demo"
echo "   python3 eval_statemanager.py --demo"
echo "   python3 report.py"
echo ""
echo " Eval scripts — run INSIDE docker (need ROS + live robot):"
echo "   docker exec -it $DOCKER_CONTAINER bash"
echo "   # then inside docker:"
echo "   source /opt/ros/melodic/setup.bash"
echo "   source $DOCKER_WORKSPACE/pal_ws/devel/setup.bash"
echo "   export ROS_MASTER_URI=http://${ROBOT_IP}:11311"
echo "   export ROS_IP=${HOST_IP}"
echo "   cd $DOCKER_WORKSPACE/eval"
echo "   python3 eval_grasp_recorder.py --object bottle --trials 15"
echo "   python3 eval_search_recorder.py --target bottle --trials 10"
echo ""
echo " Face recognition service (run on host, port 5002):"
echo "   cd $WORKSPACE"
echo "   python3 -m tiago_lissi.services.face_manager &"
echo "   python3 eval/eval_face.py --demo"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 4. Optional: drop into docker ─────────────────────────────────────────────
if [[ "$1" == "--docker" ]]; then
    echo ""
    echo "[→] Opening shell inside docker container …"
    docker exec -it "$DOCKER_CONTAINER" bash -c "
        source /opt/ros/melodic/setup.bash 2>/dev/null
        source $DOCKER_WORKSPACE/pal_ws/devel/setup.bash 2>/dev/null
        export ROS_MASTER_URI=http://${ROBOT_IP}:11311
        export ROS_IP=${HOST_IP}
        export EDENAI_API_KEY='$EDENAI_API_KEY'
        export GROQ_API_KEY='$GROQ_API_KEY'
        cd $DOCKER_WORKSPACE
        echo '[✓] ROS Melodic sourced — workspace: $DOCKER_WORKSPACE'
        exec bash
    "
fi
