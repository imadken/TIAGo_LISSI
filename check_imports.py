#!/usr/bin/env python3
import subprocess

mods = [
    "rospy", "actionlib", "cv2", "speech_recognition",
    "image_geometry", "moveit_msgs", "moveit_commander",
    "sensor_msgs", "geometry_msgs", "std_msgs",
    "trajectory_msgs", "control_msgs", "tf",
    "play_motion_msgs", "pal_interaction_msgs",
    "google.generativeai", "requests", "yaml", "numpy", "PIL",
]

missing = []
for m in mods:
    r = subprocess.run(["python3", "-c", "import {}".format(m)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode == 0:
        print("OK      : {}".format(m))
    else:
        print("MISSING : {}".format(m))
        missing.append(m)

print("\n--- Summary ---")
if missing:
    print("Need to install: {}".format(", ".join(missing)))
else:
    print("All imports OK!")
