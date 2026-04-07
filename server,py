from fastapi import FastAPI
from pydantic import BaseModel
from env import TicketRoutingEnv

app = FastAPI()

env = None

# ---- Models ----
class StepRequest(BaseModel):
    category: str


# ---- RESET ----
@app.post("/reset")
def reset():
    global env
    env = TicketRoutingEnv(task="easy")
    obs = env.reset()
    return {
        "observation": obs.__dict__ if hasattr(obs, "__dict__") else obs
    }


# ---- STEP ----
@app.post("/step")
def step(req: StepRequest):
    global env
    obs, reward, done, info = env.step(req)

    return {
        "observation": obs.__dict__ if hasattr(obs, "__dict__") else obs,
        "reward": reward.value if hasattr(reward, "value") else reward,
        "done": done,
        "info": info
    }


# ---- STATE ----
@app.get("/state")
def state():
    global env
    return env.state()