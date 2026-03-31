#!/usr/bin/env python3
"""
Interactive Grasp Calibration for TIAGo.

How it works:
  1. Robot attempts a grasp using current offsets
  2. You observe where the gripper lands vs the bottle
  3. You press arrow keys to adjust dx/dy/dz
  4. Repeat until grasp is accurate
  5. Offsets are saved to grasp_offsets.yaml

Usage:
  python3 calibrate_grasp_interactive.py
"""

import rospy
import yaml
import os
import sys
import subprocess

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFSETS_FILE = os.path.join(_REPO_ROOT, 'grasp_offsets.yaml')
STEP = 0.01  # 1cm per adjustment

# Load .env so EDENAI_API_KEY is available to subprocesses
_env_file = os.path.join(_REPO_ROOT, '.env')
if os.path.exists(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())


def load_offsets():
    if os.path.exists(OFFSETS_FILE):
        with open(OFFSETS_FILE, 'r') as f:
            d = yaml.safe_load(f)
            return d['grasp_dx'], d['grasp_dy'], d['grasp_dz']
    # Defaults from reach_object_v5
    return 0.16, -0.04, -0.04


def save_offsets(dx, dy, dz):
    with open(OFFSETS_FILE, 'w') as f:
        yaml.dump({'grasp_dx': dx, 'grasp_dy': dy, 'grasp_dz': dz}, f)
    print("\nSaved to {}".format(OFFSETS_FILE))
    print("  grasp_dx = {:.3f}  (forward/back)".format(dx))
    print("  grasp_dy = {:.3f}  (left/right)".format(dy))
    print("  grasp_dz = {:.3f}  (up/down)".format(dz))


def print_status(dx, dy, dz, attempt):
    print("\n" + "="*50)
    print("  ATTEMPT #{} — Current Offsets".format(attempt))
    print("="*50)
    print("  dx = {:.3f}  (+forward / -back)".format(dx))
    print("  dy = {:.3f}  (+left    / -right)".format(dy))
    print("  dz = {:.3f}  (+up      / -down)".format(dz))
    print("="*50)


def print_menu():
    print("""
After the grasp attempt, where did the gripper miss?

  [w] gripper too far back   → dx +{s}
  [s] gripper too far forward→ dx -{s}
  [a] gripper too far right  → dy +{s}
  [d] gripper too far left   → dy -{s}
  [u] gripper too low        → dz +{s}
  [j] gripper too high       → dz -{s}
  [W/S/A/D/U/J] = big step ({b})
  [r] retry same offsets
  [p] print & save offsets
  [q] quit and save
""".format(s=STEP, b=STEP*5))


def get_key():
    import tty, termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def run_grasp(dx, dy, dz, target='bottle', description=''):
    """Run reach_object_v5 with patched offsets as env vars."""
    env = os.environ.copy()
    env['GRASP_DX'] = str(dx)
    env['GRASP_DY'] = str(dy)
    env['GRASP_DZ'] = str(dz)

    script = os.path.join(_REPO_ROOT, 'tiago_lissi', 'manipulation',
                          'reach_object_v5_torso_descent_working.py')

    cmd = ['python3', script, '--target', target]
    if description:
        cmd += ['--description', description]

    print("\nRunning grasp with dx={:.3f} dy={:.3f} dz={:.3f} target={}...".format(
        dx, dy, dz, target))
    result = subprocess.run(cmd, env=env, timeout=120)
    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='bottle',
                        help='Object class to grasp (default: bottle)')
    parser.add_argument('--description', type=str, default='',
                        help='Optional VLM hint e.g. "red bottle on the left"')
    args = parser.parse_args()

    dx, dy, dz = load_offsets()
    attempt = 0

    print("\nTIAGo Grasp Calibration Tool  (VLM detection)")
    print("Target object : {}".format(args.target))
    if args.description:
        print("Description   : {}".format(args.description))
    print("Loaded offsets: dx={:.3f} dy={:.3f} dz={:.3f}".format(dx, dy, dz))
    print("Step size: {:.0f}mm  |  Big step: {:.0f}mm".format(STEP*1000, STEP*5*1000))
    print("\nPlace {} in front of robot (~60-80cm away).".format(args.target))
    input("Press Enter to start first grasp attempt...")

    while True:
        attempt += 1
        print_status(dx, dy, dz, attempt)

        success = run_grasp(dx, dy, dz, target=args.target, description=args.description)
        print("\nGrasp {}.".format("SUCCEEDED" if success else "FAILED/INCOMPLETE"))
        print_menu()

        key = get_key()

        if   key == 'w': dx += STEP;   print("+dx (forward)")
        elif key == 's': dx -= STEP;   print("-dx (back)")
        elif key == 'a': dy += STEP;   print("+dy (left)")
        elif key == 'd': dy -= STEP;   print("-dy (right)")
        elif key == 'u': dz += STEP;   print("+dz (up)")
        elif key == 'j': dz -= STEP;   print("-dz (down)")
        elif key == 'W': dx += STEP*5; print("+dx BIG")
        elif key == 'S': dx -= STEP*5; print("-dx BIG")
        elif key == 'A': dy += STEP*5; print("+dy BIG")
        elif key == 'D': dy -= STEP*5; print("-dy BIG")
        elif key == 'U': dz += STEP*5; print("+dz BIG")
        elif key == 'J': dz -= STEP*5; print("-dz BIG")
        elif key == 'r': print("Retrying same offsets...")
        elif key == 'p': save_offsets(dx, dy, dz)
        elif key == 'q':
            save_offsets(dx, dy, dz)
            print("\nUpdate reach_object_v5_torso_descent_working.py lines 113-115:")
            print("  self.grasp_dx = {:.3f}".format(dx))
            print("  self.grasp_dy = {:.3f}".format(dy))
            print("  self.grasp_dz = {:.3f}".format(dz))
            break

        input("\nReturn arm to home position, reposition bottle, then press Enter...")


if __name__ == '__main__':
    main()
