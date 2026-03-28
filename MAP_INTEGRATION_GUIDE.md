# Using Your Map in the Embodied AI Project

## Overview

The map you created enables:
- ✅ **Localization** - Robot knows where it is
- ✅ **Navigation** - Robot can go to specific places
- ✅ **Path planning** - Avoid obstacles automatically
- ✅ **Spatial understanding** - VLM can reason about locations

---

## 1. Launch Navigation with Your Map

**On the robot (every session):**
```bash
ssh pal@10.68.0.1
roslaunch tiago_2dnav tiago_2dnav.launch map:=$HOME/my_room_map.yaml
```

This starts:
- **AMCL** - Localization (tracks robot position on map)
- **move_base** - Navigation (path planning + obstacle avoidance)
- **Costmaps** - Dynamic obstacle detection

---

## 2. Set Initial Pose (Tell Robot Where It Is)

When navigation starts, the robot doesn't know where it is on the map.

**Option A: Via RViz**
1. Open RViz (shows the map)
2. Click "2D Pose Estimate" button (top toolbar)
3. Click on map where robot actually is
4. Drag to set orientation
5. Robot localizes within a few seconds

**Option B: Via command line** (if you know coordinates)
```bash
rostopic pub /initialpose geometry_msgs/PoseWithCovarianceStamped '{
  header: {frame_id: "map"},
  pose: {
    pose: {
      position: {x: 0.0, y: 0.0, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    }
  }
}' --once
```

---

## 3. Navigate to Specific Locations

### Using Scripts You Already Have

**Go to specific coordinates:**
```bash
python3 go_to_pose.py 2.0 1.0 0.0
# Goes to (x=2.0, y=1.0, theta=0.0)
```

**Return home:**
```bash
python3 navigate_home.py
```

### Finding Coordinates

**Method 1: Drive robot there, then check pose**
```bash
# Use teleop to drive robot to desired location
rosrun teleop_twist_keyboard teleop_twist_keyboard.py

# Check current position
python3 check_robot_pose.py
# Copy the coordinates shown
```

**Method 2: Use RViz**
1. Open RViz with map loaded
2. Click "Publish Point" tool
3. Click on desired location
4. Subscribe to `/clicked_point`:
   ```bash
   rostopic echo /clicked_point
   ```

---

## 4. Integration with Embodied Agent

### Update state_manager.py

Add map-based location tracking:

```python
# state_manager.py
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
import yaml

class StateManager:
    def __init__(self):
        self.current_location = 'unknown'
        self.pose = None
        
        # Load named locations
        with open('config/locations.yaml') as f:
            self.locations = yaml.safe_load(f)['locations']
        
        # Subscribe to robot pose
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, 
                         self.pose_callback)
    
    def pose_callback(self, msg):
        self.pose = msg.pose.pose
        self.update_current_location()
    
    def update_current_location(self):
        """Determine which named location robot is near."""
        if self.pose is None:
            return
        
        x = self.pose.position.x
        y = self.pose.position.y
        
        # Check if near any named location (within 0.5m)
        for name, loc in self.locations.items():
            dx = x - loc['x']
            dy = y - loc['y']
            dist = (dx*dx + dy*dy)**0.5
            
            if dist < 0.5:
                self.current_location = name
                return
        
        self.current_location = 'unknown'
    
    def get_state(self):
        return {
            'gripper': self.gripper_state,
            'location': self.current_location,
            'pose': self.pose,
            'detected_objects': self.detected_objects
        }
```

### Update VLM Prompts

Include map context in VLM reasoning:

```python
# vlm_reasoner.py
def query(self, user_command, image, detections, state):
    prompt = f"""
You are TIAGo, a mobile manipulator robot.

Current state:
- Location: {state['location']}
- Position: ({state['pose'].position.x:.1f}, {state['pose'].position.y:.1f})
- Gripper: {state['gripper']}
- Detected objects: {[d['class_name'] for d in detections]}

Available locations:
- home (0.0, 0.0)
- table (2.0, 1.0)
- shelf (3.5, -1.0)

User command: "{user_command}"

Available skills:
- navigate_to(location) - Go to named location
- grab_bottle() - Grasp bottle in view
- search_for(object) - Search for object by moving head
- handover_to_person() - Hand object to person

Respond with JSON: {{"skill": "...", "parameters": {{...}}, "rationale": "..."}}
"""
    # Send to Gemini VLM with image...
```

---

## 5. Create navigate_to Skill

```python
# skills/navigate_to.py
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import yaml
import math

class NavigateToSkill(BaseSkill):
    def __init__(self):
        self.name = "navigate_to"
        
        # Load locations
        with open('config/locations.yaml') as f:
            self.locations = yaml.safe_load(f)['locations']
        
        # Navigation client
        self.client = actionlib.SimpleActionClient(
            'move_base', MoveBaseAction)
        self.client.wait_for_server()
    
    def check_affordance(self, params, state):
        """Check if location exists."""
        location = params.get('location')
        if location not in self.locations:
            return False, f"Unknown location: {location}"
        return True, f"Can navigate to {location}"
    
    def execute(self, params):
        """Navigate to named location."""
        location_name = params['location']
        loc = self.locations[location_name]
        
        # Create goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        
        goal.target_pose.pose.position.x = loc['x']
        goal.target_pose.pose.position.y = loc['y']
        
        # Convert theta to quaternion
        theta = loc['theta']
        goal.target_pose.pose.orientation.z = math.sin(theta / 2.0)
        goal.target_pose.pose.orientation.w = math.cos(theta / 2.0)
        
        rospy.loginfo(f"[NavigateTo] Going to {location_name}")
        
        self.client.send_goal(goal)
        success = self.client.wait_for_result(rospy.Duration(60))
        
        return success
```

---

## 6. Example Usage in Embodied Agent

```python
# embodied_agent.py
from skills.navigate_to import NavigateToSkill

class EmbodiedAgent:
    def __init__(self):
        # ... existing init ...
        
        # Load navigation skill
        self.skills['navigate_to'] = NavigateToSkill()
    
    def main_loop(self):
        # User: "Go to the table and grab the bottle"
        
        # VLM plans:
        # 1. navigate_to(table)
        # 2. grab_bottle()
        
        for skill_call in plan:
            skill = self.skills[skill_call['skill']]
            success = skill.execute(skill_call['params'])
```

---

## 7. Testing Navigation

```bash
# Terminal 1: Robot - Start navigation with map
ssh pal@10.68.0.1
roslaunch tiago_2dnav tiago_2dnav.launch map:=$HOME/my_room_map.yaml

# Terminal 2: Docker - Set initial pose
python3 check_robot_pose.py  # See current position
# Use RViz to set initial pose if needed

# Terminal 3: Docker - Test navigation
python3 go_to_pose.py 1.0 0.5 0.0  # Test coordinates
python3 navigate_home.py           # Return to start
```

---

## 8. Update Locations with Real Coordinates

After testing, update `config/locations.yaml` with actual coordinates from your environment:

1. Drive robot to important locations using teleop
2. At each spot, run: `python3 check_robot_pose.py`
3. Copy coordinates to `config/locations.yaml`
4. Test navigation to each location

---

## Summary: Map Integration Workflow

```
┌─────────────────┐
│  Create Map     │  ← You did this!
│  (SLAM)         │
└────────┬────────┘
         │
┌────────▼────────┐
│  Define Locs    │  ← Add table, shelf, etc. coordinates
│  (YAML)         │
└────────┬────────┘
         │
┌────────▼────────┐
│  Launch Nav     │  ← roslaunch tiago_2dnav with map
│  (AMCL+move_base)│
└────────┬────────┘
         │
┌────────▼────────┐
│  Set Init Pose  │  ← Tell robot where it starts
│  (RViz)         │
└────────┬────────┘
         │
┌────────▼────────┐
│  Navigate       │  ← go_to_pose.py or VLM commands
│  (Skills)       │
└─────────────────┘
```

---

## Next Steps

1. ✅ Map created and saved
2. ⏭️ Find coordinates of key locations (table, shelf, etc.)
3. ⏭️ Update `config/locations.yaml`
4. ⏭️ Test `navigate_to` skill
5. ⏭️ Integrate with VLM in `embodied_agent.py`
6. ⏭️ Run full demo: "Go to table, grab bottle, bring to me"

The map is now the foundation for spatial reasoning in your embodied AI system!
