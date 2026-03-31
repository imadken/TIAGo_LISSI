#!/usr/bin/env python3
"""
Base Skill Class for TIAGo Embodied AI
Abstract class defining skill interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import rospy


class BaseSkill(ABC):
    """
    Abstract base class for robot skills.

    Each skill must implement:
    - check_affordance: Can this skill be executed right now?
    - execute: Execute the skill
    - get_description: Human-readable description for VLM
    """

    def __init__(self, name: str):
        """
        Initialize base skill.

        Args:
            name: Skill name (e.g., 'grab_bottle', 'place_at')
        """
        self.name = name
        self.last_execution_time = None
        self.execution_count = 0

    @abstractmethod
    def check_affordance(self, params: Dict[str, Any], state: Dict[str, Any]) -> tuple:
        """
        Check if skill can be executed in current state.

        Args:
            params: Skill parameters from VLM
            state: Current robot state (from StateManager)

        Returns:
            Tuple (can_execute: bool, reason: str)
            - can_execute: True if skill can run, False otherwise
            - reason: Human-readable explanation
        """
        pass

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> bool:
        """
        Execute the skill.

        Args:
            params: Skill parameters from VLM

        Returns:
            True if execution succeeded, False otherwise
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get human-readable description of this skill for VLM prompt.

        Returns:
            String like "grab_bottle() - Grasp a bottle in front of you"
        """
        pass

    def get_expected_outcome(self, params: Dict[str, Any]) -> str:
        """
        Get expected outcome for verification.
        Override in subclasses for skill-specific outcomes.

        Args:
            params: Skill parameters

        Returns:
            Human-readable expected outcome
        """
        return f"{self.name} should complete successfully"

    def on_success(self):
        """
        Called when skill execution succeeds.
        Override for skill-specific success handling.
        """
        self.execution_count += 1
        self.last_execution_time = rospy.Time.now()
        rospy.loginfo(f"[Skill:{self.name}] Execution #{self.execution_count} succeeded")

    def on_failure(self, error: str):
        """
        Called when skill execution fails.
        Override for skill-specific failure handling.

        Args:
            error: Error description
        """
        rospy.logwarn(f"[Skill:{self.name}] Execution failed: {error}")

    def __str__(self):
        return self.get_description()

    def __repr__(self):
        return f"<Skill:{self.name} executions={self.execution_count}>"
