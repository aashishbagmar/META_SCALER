"""
Inference script for running baseline agent on all tasks
Must print exact log format for automated validation
"""
import os
import sys
from typing import List, Dict, Any
from openai import OpenAI
from env import TicketRoutingEnv
from models import Action, Observation
from grader import grade_task
from dotenv import load_dotenv
load_dotenv()


# Environment variables with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-72b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable is required", file=sys.stderr)
    sys.exit(1)


class BaselineAgent:
    """Simple baseline agent using LLM for ticket classification"""
    
    def __init__(self, model_name: str, api_base_url: str):
        self.client = OpenAI(
            api_key=HF_TOKEN,
            base_url=api_base_url
        )
        self.model_name = model_name
    
    def classify_ticket(self, observation: Observation) -> Action:
        """
        Classify a ticket using LLM
        
        Args:
            observation: Current observation with ticket text
            
        Returns:
            Action with predicted category
        """
        if not observation.ticket_text:
            return Action(category="general")
        
        prompt = f"""You are a STRICT classifier.

                Classify the support ticket into ONLY ONE category:
                - billing
                - technical
                - general

                Rules:
                - Output EXACTLY one word
                - No explanation
                - No sentence
                - No extra text

                Examples:
                Ticket: I was charged twice
                Answer: billing

                Ticket: App is crashing
                Answer: technical

                Ticket: What are your pricing plans?
                Answer: general

                Now classify:

                Ticket: {observation.ticket_text}
                Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a ticket classification system. Respond only with: billing, technical, or general."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.0
            )
            
            category = response.choices[0].message.content.strip().lower()

            # FORCE VALID OUTPUT
            if "billing" in category:
                category = "billing"
            elif "technical" in category:
                category = "technical"
            else:
                category = "general"
            
            # Validate category
            if category not in ["billing", "technical", "general"]:
                # Default to general if invalid
                category = "general"
            
            return Action(category=category)
        
        except Exception as e:
            # Fallback to general on error
            print("ERROR:", e)   # ADD THIS LINE

            return Action(category="general")


def run_task(env_name: str, task: str, agent: BaselineAgent) -> Dict[str, Any]:
    """
    Run a single task
    
    Args:
        env_name: Environment name
        task: Task name
        agent: Agent instance
        
    Returns:
        Results dictionary
    """
    env = TicketRoutingEnv(task=task)
    observation = env.reset()
    
    step_num = 0
    rewards: List[float] = []
    success = True
    
    # Print start log
    print(f"[START] task={task} env={env_name} model={agent.model_name}")
    
    while True:
        # Get action from agent
        try:
            action = agent.classify_ticket(observation)
            error_msg = "null"
        except Exception as e:
            error_msg = str(e)
            success = False
            action = Action(category="general")
        
        # Take step
        observation, reward, done, info = env.step(action)
        
        step_num += 1
        rewards.append(reward.value)
        
        # Print step log with exact format
        done_str = "true" if done else "false"
        print(f"[STEP] step={step_num} action={action.category} reward={reward.value:.2f} done={done_str} error={error_msg}")
        
        if done:
            break
    
    # Print end log
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    success_str = "true" if success else "false"
    score = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"[END] success={success_str} steps={step_num} score={score:.2f} rewards={rewards_str}")
    
    # Calculate score
    state = env.state()
    score = grade_task(task, {"episode_rewards": rewards})
    
    return {
        "task": task,
        "success": success,
        "steps": step_num,
        "rewards": rewards,
        "total_reward": sum(rewards),
        "score": score,
        "accuracy": state["accuracy"]
    }


def main():
    """Main inference loop"""
    env_name = "ticket-routing"
    tasks = ["easy", "medium", "hard"]
    
    # Initialize agent
    agent = BaselineAgent(
        model_name=MODEL_NAME,
        api_base_url=API_BASE_URL
    )
    
    # Run all tasks
    results = []
    for task in tasks:
        try:
            result = run_task(env_name, task, agent)
            results.append(result)
        except Exception as e:
            print(f"ERROR running task {task}: {e}", file=sys.stderr)
            # Continue with next task
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for result in results:
        print(f"Task: {result['task']}")
        print(f"  Score: {result['score']:.2f}")
        print(f"  Accuracy: {result['accuracy']:.2f}")
        print(f"  Total Reward: {result['total_reward']:.2f}")
        print()


if __name__ == "__main__":
    main()