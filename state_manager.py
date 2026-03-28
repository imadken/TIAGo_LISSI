#!/usr/bin/env python3
"""
State Manager for TIAGo Embodied AI
Tracks robot state and scene state for VLM context
"""

import time
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from typing import Dict, List, Any, Optional, Set, Tuple


# ── Affordance ontology ────────────────────────────────────────────────────────
AFFORDANCES: Dict[str, List[str]] = {
    'bottle':  ['graspable', 'drinkable', 'rollable'],
    'cup':     ['graspable', 'drinkable'],
    'remote':  ['graspable', 'pressable'],
    'phone':   ['graspable'],
    'bowl':    ['graspable', 'fillable'],
    'book':    ['graspable'],
    'person':  ['can_receive', 'addressable'],
    'chair':   ['sittable', 'pushable'],
    'table':   ['surface', 'placeable_on'],
}

# ── 3D spatial thresholds (base_footprint frame, metres) ──────────────────────
# base_footprint: X=forward, Y=left, Z=up
_TH_LR    = 0.08   # left/right hysteresis
_TH_FB    = 0.10   # front/back hysteresis
_TH_UD    = 0.08   # up/down hysteresis
_TH_NEAR  = 0.30   # near threshold (Euclidean 3D)
_TH_REACH_X = 1.20  # arm reach (forward)
_TH_REACH_Y = 0.65  # arm reach (lateral)
_TH_TABLE_Z = 0.90  # max Z for table-surface objects


class SymbolicRuleEngine:
    """
    Forward-chaining symbolic rule engine for scene graph reasoning.

    Facts are stored as tuples, e.g.:
        ('left_of', 'bottle', 'cup')
        ('reachable', 'bottle')
        ('can_grasp', 'bottle')
        ('has_affordance', 'bottle', 'graspable')

    Raw string facts from VLM are stored as single-element tuples:
        ('bottle_is_open',)
    """

    MAX_ITER = 20   # max forward-chain iterations

    def __init__(self):
        self.facts: Set[tuple] = set()

    # ── Fact management ────────────────────────────────────────────────────────

    def clear(self):
        self.facts.clear()

    def assert_fact(self, *args):
        self.facts.add(tuple(args))

    def assert_raw(self, fact_str: str):
        """Assert a VLM-returned string fact (e.g. 'bottle_is_open')."""
        self.facts.add((fact_str.strip(),))

    def retract_fact(self, *args):
        """Remove a fact if present (used for VLM spatial corrections)."""
        self.facts.discard(tuple(args))

    def has(self, *args) -> bool:
        return tuple(args) in self.facts

    def facts_of_type(self, relation: str) -> List[tuple]:
        return [f for f in self.facts if f[0] == relation]

    # ── Forward chaining ──────────────────────────────────────────────────────

    def forward_chain(self):
        """Fire rules until fixpoint (no new facts derived)."""
        for _ in range(self.MAX_ITER):
            new = set()
            new |= self._rule_symmetry()
            new |= self._rule_transitivity()
            new |= self._rule_surface_colocation()
            new |= self._rule_reachability()
            new |= self._rule_can_grasp()
            new |= self._rule_handover()
            new |= self._rule_stacking()
            added = new - self.facts
            if not added:
                break
            self.facts |= added

    # ── Rules ──────────────────────────────────────────────────────────────────

    def _rule_symmetry(self) -> set:
        new = set()
        symmetric = [
            ('near_3d', 'near_3d'),
            ('co_located', 'co_located'),
            ('left_of', 'right_of'),
            ('right_of', 'left_of'),
            ('above', 'below'),
            ('below', 'above'),
            ('in_front_of', 'behind'),
            ('behind', 'in_front_of'),
        ]
        for rel_in, rel_out in symmetric:
            for f in self.facts_of_type(rel_in):
                if len(f) == 3:
                    new.add((rel_out, f[2], f[1]))
        return new

    def _rule_transitivity(self) -> set:
        new = set()
        transitive = ['left_of', 'right_of', 'in_front_of', 'behind', 'above', 'below']
        for rel in transitive:
            pairs = {(f[1], f[2]) for f in self.facts_of_type(rel)}
            for a, b in pairs:
                for _, b2, c in [(f[0], f[1], f[2])
                                 for f in self.facts_of_type(rel) if f[1] == b]:
                    if a != c:
                        new.add((rel, a, c))
        return new

    def _rule_surface_colocation(self) -> set:
        """Objects on same surface → co_located."""
        new = set()
        surfaces = {f[2] for f in self.facts_of_type('on_surface')}
        for surf in surfaces:
            objs = [f[1] for f in self.facts_of_type('on_surface') if f[2] == surf]
            for i, a in enumerate(objs):
                for b in objs[i+1:]:
                    new.add(('co_located', a, b))
                    new.add(('co_located', b, a))
        return new

    def _rule_reachability(self) -> set:
        new = set()
        for f in self.facts_of_type('dist_forward'):
            obj, dist = f[1], f[2]
            lat_facts = [g for g in self.facts_of_type('dist_lateral') if g[1] == obj]
            lat = lat_facts[0][2] if lat_facts else 999.0
            if dist < _TH_REACH_X and lat < _TH_REACH_Y:
                new.add(('reachable', obj))
        return new

    def _rule_can_grasp(self) -> set:
        new = set()
        for f in self.facts_of_type('reachable'):
            obj = f[1]
            if self.has('has_affordance', obj, 'graspable'):
                new.add(('can_grasp', obj))
        return new

    def _rule_handover(self) -> set:
        new = set()
        for f in self.facts_of_type('holding'):
            obj = f[2]
            if self.has('near_3d', 'robot', 'person') or self.has('reachable', 'person'):
                new.add(('handover_possible', obj, 'person'))
        return new

    def _rule_stacking(self) -> set:
        new = set()
        for f in self.facts_of_type('above'):
            a, b = f[1], f[2]
            if self.has('near_3d', a, b):
                new.add(('stacked_on', a, b))
        return new

    # ── Serialisation for VLM prompt ─────────────────────────────────────────

    def derived_facts_str(self, max_facts: int = 20) -> str:
        """Human-readable derived facts for VLM prompt injection."""
        lines = []
        priority = ['can_grasp', 'reachable', 'handover_possible', 'stacked_on',
                    'co_located', 'near_3d', 'left_of', 'right_of',
                    'in_front_of', 'behind', 'above', 'below', 'on_surface']
        seen = set()
        for rel in priority:
            for f in sorted(self.facts_of_type(rel)):
                s = '{}({})'.format(f[0], ', '.join(str(x) for x in f[1:]))
                if s not in seen:
                    lines.append(s)
                    seen.add(s)
                if len(lines) >= max_facts:
                    break
            if len(lines) >= max_facts:
                break
        # Append raw VLM string facts
        for f in self.facts:
            if len(f) == 1 and f[0] not in seen:
                lines.append(f[0])
        return ', '.join(lines) if lines else 'none'


class StateManager:
    def __init__(self):
        """
        Initialize state manager.
        Subscribes to joint states and odometry for state tracking.
        """
        # Robot state
        self.gripper_status = 'empty'  # 'empty' | 'holding:object_name'
        self.base_location = 'unknown'  # Named location or 'unknown'
        self.torso_height = 0.0  # Current torso lift height (0.0-0.35m)
        self.gripper_position = 0.0  # Gripper joint position (0.0=open, 0.04=closed)

        # Scene state
        self.detected_objects = []  # List of detected object descriptions
        self.task_history = []  # List of recently executed skills
        self.max_history_length = 10

        # Scene graph - spatial relationships between objects
        # Format: {(obj1, obj2): relation}
        self.scene_graph = {}
        self.object_positions = {}   # {object_name: (x_px, y_px, 0)}  2D fallback
        self.object_positions_3d = {}  # {object_name: [X, Y, Z]} in base_footprint

        # Symbolic rule engine
        self._rule_engine = SymbolicRuleEngine()
        self.semantic_facts: List[str] = []   # VLM-returned string facts

        # Face recognition state
        self.recognized_person = None   # Name of the last recognised person in scene

        # ── Object memory ontology ────────────────────────────────────────────
        # Persistent memory of every object ever seen, keyed by class_name.
        # Entry format:
        #   {
        #     'class_name': str,
        #     'last_seen': float,          # time.time() timestamp
        #     'last_bbox': [x,y,w,h],     # image bbox at time of last sighting
        #     'robot_pose': (x,y,theta),   # robot odometry pose at last sighting
        #     'head_pan': float,           # head_1_joint angle when last seen (rad)
        #     'head_tilt': float,          # head_2_joint angle when last seen (rad)
        #     'seen_count': int,           # total detection count
        #     'confidence': float,         # last confidence score
        #     'relations': List[str],      # spatial relations at last sighting
        #     'location_label': str,       # named location label (e.g. 'table_area')
        #   }
        self.object_memory = {}   # type: Dict[str, Dict]

        # Head joint angles (updated by joint state callback)
        self.head_pan  = 0.0   # head_1_joint (positive = left)
        self.head_tilt = 0.0   # head_2_joint (positive = down)

        # Base pose (from odometry)
        self.base_x = 0.0
        self.base_y = 0.0
        self.base_theta = 0.0

        # Subscribers
        rospy.Subscriber('/joint_states', JointState, self._joint_state_callback)
        rospy.Subscriber('/mobile_base_controller/odom', Odometry, self._odom_callback, queue_size=1)

        rospy.loginfo("[State] State manager initialized")

    def _joint_state_callback(self, msg: JointState):
        """
        Update joint states (torso height, gripper position).

        Args:
            msg: JointState message
        """
        try:
            # Update torso height
            if 'torso_lift_joint' in msg.name:
                idx = msg.name.index('torso_lift_joint')
                self.torso_height = float(msg.position[idx])

            # Update head pan/tilt
            if 'head_1_joint' in msg.name:
                self.head_pan  = float(msg.position[msg.name.index('head_1_joint')])
            if 'head_2_joint' in msg.name:
                self.head_tilt = float(msg.position[msg.name.index('head_2_joint')])

            # Update gripper position (check for PAL gripper joints)
            # PAL parallel gripper typically has gripper_left_finger_joint and gripper_right_finger_joint
            # When gripper is closed, these joints are at non-zero positions
            gripper_joints = [
                'gripper_left_finger_joint',
                'gripper_right_finger_joint',
                'gripper_finger_joint'  # Some configs use this
            ]

            for joint_name in gripper_joints:
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    self.gripper_position = abs(float(msg.position[idx]))
                    break

            # Infer gripper status from position
            # If gripper joints are at near-zero position → open (empty)
            # If gripper joints are at significant position → closed (possibly holding)
            if self.gripper_position < 0.005:  # Threshold for "open"
                if self.gripper_status.startswith('holding'):
                    # Gripper opened → object released
                    self.gripper_status = 'empty'
            elif self.gripper_position > 0.015:  # Threshold for "closed"
                # Only update to holding if we're not already empty
                # (This prevents false positives - actual holding is set by skills)
                pass

        except Exception as e:
            rospy.logwarn_throttle(10, f"[State] Joint state processing error: {e}")

    def _odom_callback(self, msg: Odometry):
        """
        Update base pose from odometry.

        Args:
            msg: Odometry message
        """
        try:
            self.base_x = msg.pose.pose.position.x
            self.base_y = msg.pose.pose.position.y

            # Extract yaw from quaternion
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            self.base_theta = 2.0 * np.arctan2(qz, qw)

        except Exception as e:
            rospy.logwarn_throttle(10, f"[State] Odometry processing error: {e}")

    def update_gripper_status(self, status: str):
        """
        Manually update gripper status (called by skills after grasping/placing).

        Args:
            status: 'empty' or 'holding:object_name'
        """
        self.gripper_status = status
        rospy.loginfo(f"[State] Gripper status updated: {status}")

    def update_base_location(self, location: str):
        """
        Update named base location.

        Args:
            location: Named location ('table', 'shelf', etc.)
        """
        self.base_location = location
        rospy.loginfo(f"[State] Base location updated: {location}")

    def update_detected_objects(self, detections: List[Dict]):
        """
        Update detected objects from perception.

        Args:
            detections: List of {class_name, bbox, confidence[, position_3d, position_frame]}
        """
        # Human-readable descriptions
        def _fmt(d):
            p = d.get('position_3d')
            if p:
                return '{} at ({:.2f}m fwd, {:.2f}m lat)'.format(
                    d['class_name'], p[0], p[1])
            return '{} at px({}, {})'.format(
                d['class_name'], d['bbox'][0], d['bbox'][1])
        self.detected_objects = [_fmt(d) for d in detections]

        # Update 2D and 3D positions
        for det in detections:
            name = det['class_name']
            bbox = det['bbox']
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            self.object_positions[name] = (cx, cy, 0)
            p3 = det.get('position_3d')
            if p3 and det.get('position_frame') == 'base_footprint':
                self.object_positions_3d[name] = p3

        # Rebuild rule engine facts
        self._rebuild_symbolic_facts(detections)

        # Update persistent object memory
        now = time.time()
        current_relations = self.get_spatial_relations()
        robot_pose = (self.base_x, self.base_y, self.base_theta)
        for det in detections:
            name = det['class_name']
            entry = self.object_memory.get(name, {
                'class_name': name, 'seen_count': 0, 'relations': [],
            })
            entry['last_seen']      = now
            entry['last_bbox']      = det['bbox']
            entry['last_3d']        = det.get('position_3d')
            entry['robot_pose']     = robot_pose
            entry['head_pan']       = self.head_pan
            entry['head_tilt']      = self.head_tilt
            entry['seen_count']     = entry.get('seen_count', 0) + 1
            entry['confidence']     = det.get('confidence', 0.9)
            entry['relations']      = [r for r in current_relations if name in r]
            entry['location_label'] = self.base_location
            self.object_memory[name] = entry

    def add_spatial_relation(self, obj1: str, relation: str, obj2: str):
        """
        Add spatial relationship to scene graph.

        Args:
            obj1: First object (subject)
            relation: Spatial relation ('on', 'near', 'in', 'above', 'below', 'left_of', 'right_of')
            obj2: Second object (reference)
        """
        self.scene_graph[(obj1, obj2)] = relation
        rospy.loginfo("[State] Added relation: {} {} {}".format(obj1, relation, obj2))

    def get_spatial_relations(self, obj_name: Optional[str] = None) -> List[str]:
        """
        Get spatial relations involving an object, or all relations.

        Args:
            obj_name: Object name to filter by, or None for all relations

        Returns:
            List of relation strings like "bottle on table"
        """
        relations = []
        for (obj1, obj2), relation in self.scene_graph.items():
            if obj_name is None or obj_name in [obj1, obj2]:
                relations.append("{} {} {}".format(obj1, relation, obj2))
        return relations

    def _rebuild_symbolic_facts(self, detections: List[Dict]):
        """
        Rebuild the symbolic fact base from current detections.
        Uses 3D positions (base_footprint) when available, falls back to 2D pixels.
        Then runs the forward-chaining rule engine.
        """
        engine = self._rule_engine
        engine.clear()

        # Re-assert gripper state
        if self.gripper_status.startswith('holding:'):
            held = self.gripper_status.split(':', 1)[1]
            engine.assert_fact('holding', 'gripper', held)

        # Assert affordances
        for det in detections:
            name = det['class_name']
            for aff in AFFORDANCES.get(name, []):
                engine.assert_fact('has_affordance', name, aff)

        # Assert spatial relations (3D preferred, 2D fallback)
        has_3d = [d for d in detections if d.get('position_3d') and
                  d.get('position_frame') == 'base_footprint']
        has_2d = [d for d in detections if d not in has_3d]

        if has_3d:
            self._infer_3d_relations(has_3d, engine)
            rospy.loginfo_throttle(5, "[State] Using 3D spatial relations ({} objects)".format(len(has_3d)))
        if has_2d:
            self._infer_2d_relations_fallback(has_2d, engine)
            if has_2d:
                rospy.logwarn_throttle(10, "[State] Fallback to 2D for: {}".format(
                    [d['class_name'] for d in has_2d]))

        # Re-assert VLM semantic facts from last cycle
        for fact_str in self.semantic_facts:
            engine.assert_raw(fact_str)

        # Forward chain to derive new facts
        engine.forward_chain()

        # Mirror derived facts back to scene_graph for backwards-compat
        current_objects = {d['class_name'] for d in detections}
        self.scene_graph = {}
        for f in engine.facts:
            if len(f) == 3 and f[1] in current_objects:
                self.scene_graph[(f[1], f[2])] = f[0]

    def _infer_3d_relations(self, detections: List[Dict], engine: SymbolicRuleEngine):
        """
        Assert geometric spatial facts from 3D positions in base_footprint frame.
        base_footprint: X=forward (away from robot), Y=left, Z=up
        """
        for i, d1 in enumerate(detections):
            n1 = d1['class_name']
            p1 = d1['position_3d']   # [X, Y, Z]

            # Reachability distance facts (used by rule engine)
            engine.assert_fact('dist_forward', n1, abs(p1[0]))
            engine.assert_fact('dist_lateral', n1, abs(p1[1]))

            # Surface (table-height) inference
            if p1[2] < _TH_TABLE_Z and p1[0] < 1.5:
                engine.assert_fact('on_surface', n1, 'table')

            for j, d2 in enumerate(detections):
                if i >= j:
                    continue
                n2 = d2['class_name']
                if n1 == n2:
                    continue  # skip same-class pairs — can't meaningfully name the relation
                p2 = d2['position_3d']

                dx = p1[0] - p2[0]   # forward diff
                dy = p1[1] - p2[1]   # lateral diff
                dz = p1[2] - p2[2]   # height diff
                dist3d = float(np.linalg.norm([dx, dy, dz]))

                # Near
                if dist3d < _TH_NEAR:
                    engine.assert_fact('near_3d', n1, n2)

                # Left / right  (Y axis: positive = left)
                if dy > _TH_LR:
                    engine.assert_fact('left_of', n1, n2)
                elif dy < -_TH_LR:
                    engine.assert_fact('right_of', n1, n2)

                # In front / behind  (X axis: smaller X = closer to robot = in front)
                if dx < -_TH_FB:
                    engine.assert_fact('in_front_of', n1, n2)
                elif dx > _TH_FB:
                    engine.assert_fact('behind', n1, n2)

                # Above / below  (Z axis: positive = up)
                if dz > _TH_UD:
                    engine.assert_fact('above', n1, n2)
                elif dz < -_TH_UD:
                    engine.assert_fact('below', n1, n2)

    def _infer_2d_relations_fallback(self, detections: List[Dict],
                                     engine: SymbolicRuleEngine):
        """
        Legacy 2D pixel-based spatial inference for objects without depth.
        Kept as fallback when depth/TF unavailable.
        """
        NEAR_PX = 150
        for i, d1 in enumerate(detections):
            n1 = d1['class_name']
            b1 = d1['bbox']
            cx1 = b1[0] + b1[2] / 2
            cy1 = b1[1] + b1[3] / 2
            # Surface heuristic: object bottom in lower 65% of image
            if cy1 + b1[3] / 2 > 480 * 0.60:
                engine.assert_fact('on_surface', n1, 'table')
            # Use pixel depth as proxy for forward distance
            engine.assert_fact('dist_forward', n1, 0.8)   # assume ~0.8m
            engine.assert_fact('dist_lateral', n1, abs(cx1 - 320) / 554.25 * 0.8)
            for j, d2 in enumerate(detections):
                if i >= j:
                    continue
                n2 = d2['class_name']
                if n1 == n2:
                    continue  # skip same-class pairs
                b2 = d2['bbox']
                cx2 = b2[0] + b2[2] / 2
                cy2 = b2[1] + b2[3] / 2
                dist = float(np.hypot(cx1 - cx2, cy1 - cy2))
                if dist < NEAR_PX:
                    engine.assert_fact('near_3d', n1, n2)
                if cx1 < cx2 - 20:
                    engine.assert_fact('left_of', n1, n2)
                elif cx1 > cx2 + 20:
                    engine.assert_fact('right_of', n1, n2)
                if cy1 < cy2 - 20:
                    engine.assert_fact('above', n1, n2)
                elif cy1 > cy2 + 20:
                    engine.assert_fact('below', n1, n2)

    def merge_semantic_facts(self, facts: List[str]):
        """
        Merge VLM-returned semantic facts into the symbolic fact base and re-chain.
        Call this after each VLM response that includes 'semantic_facts'.

        Args:
            facts: List of string facts, e.g. ['bottle_is_open', 'person_is_reaching']
        """
        self.semantic_facts = facts
        for f in facts:
            self._rule_engine.assert_raw(f)
        self._rule_engine.forward_chain()
        rospy.loginfo("[State] Merged {} semantic facts: {}".format(len(facts), facts))

    def get_object_memory(self, class_name: str) -> Optional[Dict]:
        """Return memory entry for a specific object, or None if never seen."""
        return self.object_memory.get(class_name)

    def get_recently_seen(self, max_age_sec: float = 120.0) -> List[Dict]:
        """Return objects seen within the last max_age_sec seconds."""
        now = time.time()
        return [e for e in self.object_memory.values()
                if now - e.get('last_seen', 0) <= max_age_sec]

    def memory_summary(self, max_age_sec: float = 300.0) -> str:
        """
        Human-readable summary of object memory for VLM context.
        Only includes objects not currently detected (to avoid duplication).
        """
        now = time.time()
        lines = []
        for name, entry in sorted(self.object_memory.items(),
                                   key=lambda x: x[1].get('last_seen', 0),
                                   reverse=True):
            age = now - entry.get('last_seen', 0)
            if age > max_age_sec:
                continue
            mins = int(age // 60)
            secs = int(age % 60)
            age_str = "{}m{}s ago".format(mins, secs) if mins else "{}s ago".format(secs)
            rel_str = (", ".join(entry['relations'][:2])
                       if entry.get('relations') else "no relations")
            head_str = ""
            if 'head_pan' in entry:
                head_str = ", head=({:.2f}rad pan, {:.2f}rad tilt)".format(
                    entry['head_pan'], entry['head_tilt'])
            lines.append("- {} (last seen: {}, {}, seen {} time(s){})".format(
                name, age_str, rel_str, entry.get('seen_count', 1), head_str))
        return "\n".join(lines) if lines else "No objects in memory."

    def add_task(self, skill_name: str, success: bool = True):
        """
        Add executed skill to task history.

        Args:
            skill_name: Name of executed skill
            success: Whether skill succeeded
        """
        status_str = "✓" if success else "✗"
        task_entry = f"{status_str} {skill_name}"
        self.task_history.append(task_entry)

        # Trim history
        if len(self.task_history) > self.max_history_length:
            self.task_history = self.task_history[-self.max_history_length:]

        rospy.loginfo(f"[State] Task added: {task_entry}")

    def set_recognized_person(self, name):
        # type: (str) -> None
        """Update the name of the person currently recognised in the scene."""
        self.recognized_person = name
        rospy.loginfo("[State] Recognised person in scene: {}".format(name))

    def get_state(self) -> Dict[str, Any]:
        """
        Get complete robot state for VLM context.

        Returns dict with:
            gripper, base_location, detected_objects, task_history,
            spatial_relations, object_memory, object_positions_3d,
            derived_facts, semantic_facts, affordances, recognized_person
        """
        state = {
            'gripper':           self.gripper_status,
            'base_location':     self.base_location,
            'detected_objects':  self.detected_objects[:5],
            'task_history':      self.task_history[-3:],
            'spatial_relations': self.get_spatial_relations()[:10],
            'object_memory':     self.memory_summary(),
            # ── 3D + symbolic additions ──────────────────────────────────
            'object_positions_3d': {
                k: [round(v, 3) for v in pos]
                for k, pos in self.object_positions_3d.items()
            },
            'derived_facts':   self._rule_engine.derived_facts_str(),
            'semantic_facts':  self.semantic_facts,
            'affordances': {
                name: AFFORDANCES.get(name, [])
                for name in self.object_positions_3d
            },
            'relation_quality': '3d' if self.object_positions_3d else '2d',
        }
        if self.recognized_person:
            state['recognized_person'] = self.recognized_person
        return state

    def get_base_pose(self) -> tuple:
        """
        Get current base pose from odometry.

        Returns:
            Tuple (x, y, theta)
        """
        return (self.base_x, self.base_y, self.base_theta)

    def reset(self):
        """
        Reset state (useful for testing).
        """
        self.gripper_status = 'empty'
        self.base_location = 'unknown'
        self.detected_objects = []
        self.task_history = []
        self.scene_graph = {}
        self.object_positions = {}
        self.recognized_person = None
        rospy.loginfo("[State] State reset")


if __name__ == '__main__':
    # Test state manager
    rospy.init_node('state_test', anonymous=True)

    state_mgr = StateManager()

    rospy.loginfo("State manager initialized. Monitoring state...")

    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        state = state_mgr.get_state()
        rospy.loginfo(f"Current state: {state}")
        rospy.loginfo(f"Base pose: {state_mgr.get_base_pose()}")
        rospy.loginfo(f"Torso height: {state_mgr.torso_height:.3f}m")
        rospy.loginfo(f"Gripper position: {state_mgr.gripper_position:.4f}")
        rospy.loginfo("---")
        rate.sleep()
