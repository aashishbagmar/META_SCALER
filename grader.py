"""
Grading logic for evaluating agent performance on tasks
"""
from typing import Dict, Any, List
from env import TicketRoutingEnv
from models import Action


class TaskGrader:
    """
    Grades agent performance on ticket routing tasks
    Returns a score between 0.0 and 1.0
    """
    
    def __init__(self, task: str):
        self.task = task
        self.env = TicketRoutingEnv(task=task)
        
    def grade(self, actions: List[Action]) -> float:
        """
        Grade a sequence of actions
        
        Args:
            actions: List of actions taken by the agent
            
        Returns:
            score: Float between 0.0 and 1.0
        """
        self.env.reset()
        total_reward = 0.0
        steps = 0
        
        for action in actions:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward.value
            steps += 1
            
            if done:
                break
        
        # Normalize score by maximum possible reward
        max_possible_reward = float(len(self.env.tickets))
        
        if max_possible_reward == 0:
            return 0.0
        
        # Calculate final score
        score = total_reward / max_possible_reward
        
        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))
        
        return score
    
    def grade_episode(self, episode_rewards: List[float]) -> float:
        """
        Grade based on episode rewards
        
        Args:
            episode_rewards: List of rewards from an episode
            
        Returns:
            score: Float between 0.0 and 1.0
        """
        if not episode_rewards:
            return 0.0
        
        # Get max possible based on task
        if self.task == "easy":
            max_possible = 5.0
        elif self.task == "medium":
            max_possible = 8.0
        elif self.task == "hard":
            max_possible = 10.0
        else:
            max_possible = len(episode_rewards)
        
        total_reward = sum(episode_rewards)
        score = total_reward / max_possible
        
        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))
        
        return score


def grade_task(task: str, episode_data: Dict[str, Any]) -> float:
    """
    Main grading function called by evaluation harness
    
    Args:
        task: Task name (easy, medium, hard)
        episode_data: Dictionary containing episode information
        
    Returns:
        score: Float between 0.0 and 1.0
    """
    grader = TaskGrader(task=task)
    
    # Extract rewards from episode data
    if "rewards" in episode_data:
        rewards = episode_data["rewards"]
    elif "episode_rewards" in episode_data:
        rewards = episode_data["episode_rewards"]
    else:
        # Fallback: compute from state
        rewards = episode_data.get("total_reward", 0.0)
        if isinstance(rewards, float):
            # Single reward value, convert to list
            return min(1.0, max(0.0, rewards / grader.env.max_steps))
    
    return grader.grade_episode(rewards)