from fastapi import FastAPI, HTTPException

from .subscription import validate_plan_limits, update_user_usage

app = FastAPI()

PLAN_LIMITS = {
    "free": {"projects": 1, "generations": 3},
    "pro": {"projects": 5, "generations": 10},
}

def enforce_limits(user_id: str, plan: str = "free"):
    limits = PLAN_LIMITS.get(plan, {})
    if not validate_plan_limits(user_id, limits):
        raise HTTPException(status_code=403, detail="Plan limit reached")


@app.post("/projects/create")
def create_project(user_id: str, plan: str = "free"):
    enforce_limits(user_id, plan)
    update_user_usage(user_id, projects=1)
    return {"status": "project created"}


@app.post("/video/generate")
def generate_video(user_id: str, plan: str = "free"):
    enforce_limits(user_id, plan)
    update_user_usage(user_id, generations=1)
    return {"status": "video generated"}


@app.post("/tts")
def tts(user_id: str, plan: str = "free"):
    enforce_limits(user_id, plan)
    update_user_usage(user_id, generations=1)
    return {"status": "tts generated"}
