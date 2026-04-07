"""
Customer Support Ticket Routing Environment
A real-world simulation of support ticket classification and routing
"""
from typing import Dict, Any, Optional, List, Tuple
import random
from models import Observation, Action, Reward, TaskConfig


class TicketRoutingEnv:
    """
    Environment simulating customer support ticket routing.
    Agent must classify incoming tickets into correct categories.
    """
    
    def __init__(self, task: str = "easy"):
        self.task = task
        self.current_step = 0
        self.max_steps = 10
        self.current_ticket_idx = 0
        self.tickets: List[Dict[str, str]] = []
        self.correct_classifications = 0
        self.total_tickets = 0
        self.episode_rewards: List[float] = []
        
        # Load task-specific tickets
        self._load_tickets()
        
    def _load_tickets(self):
        """Load tickets based on task difficulty"""
        if self.task == "easy":
            self.tickets = self._get_easy_tickets()
            self.max_steps = 5
        elif self.task == "medium":
            self.tickets = self._get_medium_tickets()
            self.max_steps = 8
        elif self.task == "hard":
            self.tickets = self._get_hard_tickets()
            self.max_steps = 10
        else:
            raise ValueError(f"Unknown task: {self.task}")
            
    def _get_easy_tickets(self) -> List[Dict[str, str]]:
        """Simple, clear-cut tickets"""
        return [
            {"text": "I can't log into my account. My password isn't working.", "category": "technical"},
            {"text": "I was charged twice for my subscription this month.", "category": "billing"},
            {"text": "What are your business hours?", "category": "general"},
            {"text": "My credit card was declined but I have sufficient funds.", "category": "billing"},
            {"text": "The app keeps crashing when I try to upload files.", "category": "technical"},
        ]
    
    def _get_medium_tickets(self) -> List[Dict[str, str]]:
        """Tickets with multiple aspects, requires careful classification"""
        return [
            {"text": "I was charged for a premium feature but I can't access it. Is this a bug?", "category": "billing"},
            {"text": "Can I get a refund? The service doesn't work as advertised.", "category": "billing"},
            {"text": "My API key expired and I need to generate a new one. Where do I find this?", "category": "technical"},
            {"text": "I'm interested in the enterprise plan. What's included?", "category": "general"},
            {"text": "Error 500 when accessing dashboard after payment.", "category": "technical"},
            {"text": "Do you offer student discounts?", "category": "general"},
            {"text": "I accidentally deleted my project. Can you recover it?", "category": "technical"},
            {"text": "Invoice shows wrong amount compared to the pricing page.", "category": "billing"},
        ]
    
    def _get_hard_tickets(self) -> List[Dict[str, str]]:
        """Ambiguous tickets requiring nuanced understanding"""
        return [
            {"text": "I paid for annual plan but it says I'm on free tier.", "category": "billing"},
            {"text": "Getting timeout errors only when processing large files over 100MB.", "category": "technical"},
            {"text": "Is there a way to export my data before canceling?", "category": "general"},
            {"text": "Charged after cancellation. System still shows active subscription.", "category": "billing"},
            {"text": "Integration with Slack stopped working after your update yesterday.", "category": "technical"},
            {"text": "Need GDPR data deletion but the form doesn't work.", "category": "technical"},
            {"text": "Can I transfer my subscription to a different email?", "category": "general"},
            {"text": "Duplicate charges from last 3 months, need full audit.", "category": "billing"},
            {"text": "SSO configuration fails with cryptic error message.", "category": "technical"},
            {"text": "Upgrade to team plan but can't add members.", "category": "technical"},
        ]
    
    def reset(self) -> Observation:
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_ticket_idx = 0
        self.correct_classifications = 0
        self.total_tickets = 0
        self.episode_rewards = []
        
        # Shuffle tickets for variety
        random.shuffle(self.tickets)
        
        return self._get_observation()
    
    def _get_observation(self) -> Observation:
        """Get current observation"""
        if self.current_ticket_idx >= len(self.tickets):
            return Observation(
                ticket_text="",
                ticket_id=self.current_ticket_idx,
                step=self.current_step,
                remaining_tickets=0
            )
        
        ticket = self.tickets[self.current_ticket_idx]
        return Observation(
            ticket_text=ticket["text"],
            ticket_id=self.current_ticket_idx,
            step=self.current_step,
            remaining_tickets=len(self.tickets) - self.current_ticket_idx
        )
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Returns:
            observation: Current state
            reward: Reward for the action
            done: Whether episode is complete
            info: Additional information
        """
        if self.current_ticket_idx >= len(self.tickets):
            # Episode already done
            return self._get_observation(), Reward(value=0.0, reason="Episode complete"), True, {}
        
        # Get current ticket
        ticket = self.tickets[self.current_ticket_idx]
        correct_category = ticket["category"]
        predicted_category = action.category.lower().strip()
        
        # Calculate reward
        reward_value = 0.0
        reason = ""
        
        if predicted_category == correct_category:
            reward_value = 1.0
            reason = "Correct classification"
            self.correct_classifications += 1
        elif self._is_partially_correct(predicted_category, correct_category):
            reward_value = 0.3
            reason = "Partially correct (related category)"
        else:
            reward_value = 0.0
            reason = f"Incorrect (expected {correct_category}, got {predicted_category})"
        
        reward = Reward(value=reward_value, reason=reason)
        self.episode_rewards.append(reward_value)
        
        # Update state
        self.current_step += 1
        self.total_tickets += 1
        self.current_ticket_idx += 1
        
        # Check if done
        done = (self.current_ticket_idx >= len(self.tickets)) or (self.current_step >= self.max_steps)
        
        # Get next observation
        observation = self._get_observation()
        
        # Info dict
        info = {
            "correct_category": correct_category,
            "predicted_category": predicted_category,
            "accuracy": self.correct_classifications / self.total_tickets if self.total_tickets > 0 else 0.0,
            "total_reward": sum(self.episode_rewards)
        }
        
        return observation, reward, done, info
    
    def _is_partially_correct(self, predicted: str, correct: str) -> bool:
        """Check if prediction is in a related category"""
        # For this simple environment, we don't have partial credit beyond exact match
        # But we could extend this logic for more nuanced scoring
        return False
    
    def state(self) -> Dict[str, Any]:
        """Return current environment state"""
        return {
            "task": self.task,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "current_ticket_idx": self.current_ticket_idx,
            "total_tickets": len(self.tickets),
            "correct_classifications": self.correct_classifications,
            "total_classifications": self.total_tickets,
            "accuracy": self.correct_classifications / self.total_tickets if self.total_tickets > 0 else 0.0,
            "episode_rewards": self.episode_rewards,
            "total_reward": sum(self.episode_rewards)
        }