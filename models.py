"""
Pydantic models for the Customer Support Ticket Routing Environment
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class Observation(BaseModel):
    """
    Observation returned by the environment
    Represents a customer support ticket that needs classification
    """
    ticket_text: str = Field(description="The text content of the support ticket")
    ticket_id: int = Field(description="Unique identifier for this ticket")
    step: int = Field(description="Current step number in the episode")
    remaining_tickets: int = Field(description="Number of tickets left to classify")


class Action(BaseModel):
    """
    Action taken by the agent
    Represents the classification decision
    """
    category: str = Field(
        description="The category to route this ticket to: 'billing', 'technical', or 'general'"
    )


class Reward(BaseModel):
    """
    Reward returned by the environment
    """
    value: float = Field(description="Reward value between 0.0 and 1.0")
    reason: str = Field(description="Explanation for the reward")


class TaskConfig(BaseModel):
    """
    Configuration for a task
    """
    name: str = Field(description="Task identifier (easy, medium, hard)")
    description: str = Field(description="Description of the task")
    difficulty: str = Field(description="Difficulty level")
    max_steps: int = Field(description="Maximum number of steps allowed")