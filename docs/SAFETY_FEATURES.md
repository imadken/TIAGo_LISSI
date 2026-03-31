# TIAGo Embodied Agent - Safety Features

## Overview

The embodied agent now includes comprehensive safety features to protect the robot and prevent damage during operation. These features automatically handle errors, ensure safe poses, and provide graceful shutdown.

---

## Safety Features

### 1. **Automatic "Go Home" on Shutdown** ✅

**What it does:** When you shut down the agent (Ctrl+C, kill signal, or program exit), the robot automatically:
1. Stops any ongoing movements
2. Returns to home position (safe tucked pose)
3. Opens the gripper (releases any held objects)

**How it works:**
- Signal handlers catch SIGINT (Ctrl+C) and SIGTERM
- ROS shutdown hook executes `_safe_shutdown()` method
- Ensures robot is in safe state before powering off

**Example:**
```bash
# When you press Ctrl+C:
[SAFETY] INITIATING SAFE SHUTDOWN SEQUENCE
[SAFETY] Returning to home position...
[SAFETY] Opening gripper...
[SAFETY] Robot is now safe to power off
```

---

### 2. **Error Recovery and Safe Pose Management** ✅

**What it does:** When a skill fails, the robot automatically:
1. Opens the gripper (in case holding something)
2. Returns to home position (safe pose)
3. Logs the error for debugging

**How it works:**
- Every skill execution is wrapped in try-catch
- On failure: `_ensure_safe_pose()` is called
- Robot moves to known safe configuration

**Example:**
```bash
# If grab_bottle fails:
[Agent] Skill 'grab_bottle' failed
[SAFETY] Skill failed, ensuring safe pose...
[SAFETY] Opening gripper...
[SAFETY] Moving to home position...
[SAFETY] Robot in safe pose
```

---

### 3. **Safe Mode (Emergency Protection)** ✅

**What it does:** After 3 consecutive skill failures, robot enters "Safe Mode":
- Blocks all new commands
- Forces robot to safe pose
- Requires manual reset to resume

**How it works:**
- Tracks `consecutive_failures` counter
- When >= 3 failures: triggers `_enter_safe_mode()`
- User must send "reset" command to exit

**Example:**
```bash
# After 3 failures:
[SAFETY] ENTERING SAFE MODE
Reason: 3 consecutive failures
[SAFETY] Safe mode active. Send 'reset' command to exit.

# User prompt changes:
[SAFE MODE] >

# To resume:
[SAFE MODE] > reset
✓ Safe mode exited
```

---

### 4. **Consecutive Failure Tracking** ✅

**What it does:** Monitors skill success/failure rate to detect systematic problems.

**Thresholds:**
- 1 failure: Warning logged, safe pose restored
- 2 failures: Warning + safe pose, shows counter
- 3 failures: **Safe mode triggered**

**How it works:**
- `consecutive_failures` increments on each failure
- Resets to 0 on successful skill execution
- Prevents cascading failures

**Example:**
```bash
✗ Failed: Execution exception: No bottle detected
  ⚠ Consecutive failures: 2/3

# Next failure triggers safe mode
```

---

### 5. **Signal Handling (Graceful Shutdown)** ✅

**What it does:** Catches interrupt signals and ensures clean exit.

**Handled signals:**
- **SIGINT** (Ctrl+C) - User interrupt
- **SIGTERM** - System terminate request
- **ROS shutdown** - Node shutdown event

**How it works:**
- Signal handlers registered at startup
- All signals route to `_safe_shutdown()`
- Ensures robot safe before exit

---

## Safety Commands

### User Commands

| Command | Description |
|---------|-------------|
| `reset` | Exit safe mode and resume normal operation |
| `exit safe mode` | Alternative to "reset" |
| `resume` | Alternative to "reset" |

### Automatic Behaviors

| Trigger | Robot Response |
|---------|----------------|
| Skill failure | Go to safe pose (open gripper + home) |
| 3 consecutive failures | Enter safe mode |
| Ctrl+C / shutdown | Go home + open gripper + exit |
| Unexpected exception | Track failure, potentially enter safe mode |

---

## Configuration

### Adjustable Parameters

In `embodied_agent.py` `__init__()`:

```python
self.max_consecutive_failures = 3  # Change to adjust safe mode trigger
```

**Recommendations:**
- Testing/Development: Set to 5 (more lenient)
- Production/Demo: Keep at 3 (more protective)
- Critical operations: Set to 1 (maximum safety)

---

## Testing Safety Features

### Test 1: Safe Shutdown
```bash
# Start agent
./run_agent.sh

# Press Ctrl+C
# Expected: Robot goes home, opens gripper, exits cleanly
```

### Test 2: Single Failure Recovery
```bash
> grab the bottle
# (Make sure no bottle visible)
# Expected: Skill fails, robot goes to safe pose, ready for next command
```

### Test 3: Safe Mode Trigger
```bash
> grab the bottle  # Fail 1
> grab the bottle  # Fail 2
> grab the bottle  # Fail 3 → Safe mode!
[SAFE MODE] > reset  # Exit safe mode
```

### Test 4: Success After Failure
```bash
> grab the bottle  # Fail 1
> wave  # Success → failure counter resets
> grab the bottle  # Fail 1 again (not fail 2)
```

---

## Safety Logs

All safety events are logged with `[SAFETY]` prefix for easy filtering:

```bash
# View safety events only:
rosrun embodied_agent embodied_agent.py 2>&1 | grep "\[SAFETY\]"

# Example output:
[SAFETY] Safety shutdown handlers registered
[SAFETY] Skill failed, ensuring safe pose...
[SAFETY] Robot in safe pose
```

---

## How It Works Internally

### Initialization
```python
def __init__(self):
    # Safety state
    self.safe_mode = False
    self.consecutive_failures = 0
    self.max_consecutive_failures = 3

    # Register handlers
    self._register_shutdown_handlers()
```

### Execution Wrapper
```python
def execute_command(user_command):
    # 1. Check safe mode
    if self.safe_mode:
        return error "In safe mode"

    # 2. Execute skill
    try:
        success = skill.execute()
    except Exception as e:
        success = False

    # 3. Handle failure
    if not success:
        self.consecutive_failures += 1
        self._ensure_safe_pose()

        if self.consecutive_failures >= 3:
            self._enter_safe_mode()

    # 4. Handle success
    else:
        self.consecutive_failures = 0  # Reset!
```

### Shutdown Sequence
```python
def _safe_shutdown():
    # Called on Ctrl+C, kill, or ROS shutdown
    1. self.skills['go_home'].execute()
    2. self.skills['open_hand'].execute()
    3. Log "Safe to power off"
```

---

## Comparison: Before vs After

| Scenario | Before | After |
|----------|--------|-------|
| **Ctrl+C shutdown** | Robot stops mid-motion | Robot goes home, opens gripper, then exits |
| **Skill fails** | Log error, continue | Go to safe pose, track failure |
| **3 failures in row** | Keep trying | Enter safe mode, block commands |
| **Unexpected error** | Crash | Log, safe pose, track failure |
| **Power off** | Manual go_home needed | Automatic on shutdown |

---

## Best Practices

1. **Always use Ctrl+C to stop** - Don't kill -9, use normal interrupt
2. **Monitor failure counter** - If seeing 2/3, investigate before continuing
3. **Test after changes** - Verify safety features still work after code updates
4. **Use safe mode** - If robot behaving erratically, let it enter safe mode
5. **Reset after safe mode** - Understand why failures occurred before resuming

---

## Limitations

1. **Hardware emergencies** - Software safety can't prevent all hardware issues (e.g., collision mid-motion)
2. **ROS communication loss** - If ROS master dies, safety handlers may not execute
3. **Skill-specific safety** - Some skills (grab_bottle) have their own safety logic
4. **Motion interruption** - Can't interrupt play_motion mid-execution (ROS limitation)

**Mitigation:**
- Always supervise robot during operation
- Keep emergency stop button accessible
- Test in controlled environment first
- Use collision-aware navigation when available

---

## Future Enhancements

- [ ] Motion preemption (cancel ongoing skills)
- [ ] Watchdog timer (auto-safe-mode if no response)
- [ ] Collision detection via torque sensors
- [ ] Battery level monitoring (safe shutdown on low battery)
- [ ] Hardware e-stop integration
- [ ] Safe mode recovery suggestions (VLM diagnosis)

---

## Summary

✅ **Auto go-home on shutdown** - Robot always safe before power-off
✅ **Error recovery** - Failed skills trigger safe pose
✅ **Safe mode protection** - 3 failures = block further commands
✅ **Failure tracking** - Monitors consecutive failures
✅ **Signal handling** - Graceful Ctrl+C shutdown

**Result:** Robot is now significantly safer to operate, with automatic recovery from common failure modes and guaranteed safe state on shutdown.
