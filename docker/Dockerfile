# ─────────────────────────────────────────────────────────────────────────────
# TIAGo Embodied Agent — ROS Melodic Docker image
#
# Base: PAL Robotics TIAGo Melodic image (includes ROS Melodic, MoveIt,
#       PAL TTS, play_motion, gripper drivers, cv_bridge, image_geometry, etc.)
#
# Build:
#   docker build -t tiago-agent -f docker/Dockerfile .
#
# Run (standard):
#   docker run -dit --name tiago_ros --network host \
#     -v $(pwd):/workspace \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -e DISPLAY=$DISPLAY \
#     tiago-agent bash
#
# Or use docker-compose:
#   docker-compose up -d
# ─────────────────────────────────────────────────────────────────────────────

FROM palroboticssl/tiago_melodic_robot:latest

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-opencv \
    libsm6 libxext6 libxrender-dev \
    # Audio (optional — for microphone voice input)
    portaudio19-dev python3-pyaudio \
    && rm -rf /var/lib/apt/lists/*

# ── Python extras not in the PAL base image ──────────────────────────────────
COPY docker/requirements_ros.txt /tmp/requirements_ros.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements_ros.txt

# ── Workspace ─────────────────────────────────────────────────────────────────
WORKDIR /workspace

# Source ROS and PAL workspace on every bash session
RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc && \
    echo "[ -f /workspace/pal_ws/devel/setup.bash ] && source /workspace/pal_ws/devel/setup.bash" >> /root/.bashrc && \
    echo "export ROS_MASTER_URI=http://\${ROBOT_IP:-10.68.0.1}:11311" >> /root/.bashrc && \
    echo "export ROS_IP=\${HOST_IP:-\$(hostname -I | awk '{print \$1}')}" >> /root/.bashrc

CMD ["bash"]
