"""
FastAPI server exposing the OpenEnv HTTP API.
Endpoints required by the OpenEnv validator:
  POST /reset        -> observation
  POST /step         -> {observation, reward, done, info}
  GET  /state        -> state dict
  GET  /             -> 200 health check
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from env import TicketRoutingEnv
from models import Action

app = FastAPI(title="Ticket Routing OpenEnv", version="1.0.0")

# One env instance per server process (sufficient for sequential evaluation)
_env: Optional[TicketRoutingEnv] = None


# ── Request schemas ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"   # easy | medium | hard


class StepRequest(BaseModel):
    category: str                  # billing | technical | general


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "env": "ticket-routing"}


# ── reset() ──────────────────────────────────────────────────────────────────

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    global _env
    task = "easy"  # default
    if req and req.task:
        task = req.task
    
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")

    _env = TicketRoutingEnv(task=task)
    obs = _env.reset()
    return {"observation": obs.model_dump()}


# ── step() ───────────────────────────────────────────────────────────────────

@app.post("/step")
def step(req: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    category = req.category.lower().strip()
    if category not in ("billing", "technical", "general"):
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

    action = Action(category=category)
    obs, reward, done, info = _env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward.value,
        "done": done,
        "info": info,
    }


# ── state() ──────────────────────────────────────────────────────────────────

@app.get("/state")
def state():
    global _env
    if _env is None:
        return {"status": "not_initialized"}
    return _env.state()