from typing import Dict

# Simple in-memory storage for usage data
_user_usage: Dict[str, Dict[str, int]] = {}


def get_user_usage(user_id: str) -> Dict[str, int]:
    """Return current usage for the given user."""
    return _user_usage.get(user_id, {"projects": 0, "generations": 0})


def update_user_usage(user_id: str, *, projects: int = 0, generations: int = 0) -> None:
    """Update usage counters for a user."""
    usage = get_user_usage(user_id)
    usage["projects"] += projects
    usage["generations"] += generations
    _user_usage[user_id] = usage


def validate_plan_limits(user_id: str, limits: Dict[str, int]) -> bool:
    """Check if the user is within their plan limits."""
    usage = get_user_usage(user_id)
    proj_limit = limits.get("projects")
    gen_limit = limits.get("generations")

    if proj_limit is not None and usage["projects"] >= proj_limit:
        return False
    if gen_limit is not None and usage["generations"] >= gen_limit:
        return False
    return True
