"""Controller tools that call the environment API."""

from controller import mcp, http_client
from hud.tools.types import EvaluationResult

@mcp.tool
async def act() -> str:
    """Perform one action step in the environment (increment the counter)."""
    if not http_client:
        raise RuntimeError("HTTP client not initialized")
    resp = await http_client.post("/act")
    data = resp.json()
    return f"Action #{data.get('count', 0)} performed. Current c count: {data.get('count', 0)}"


@mcp.tool
async def setup() -> str:
    """Initialize or reset the environment to its starting state."""
    if not http_client:
        raise RuntimeError("HTTP client not initialized")
    await http_client.post("/reset")
    return "Counter reset to 0"


@mcp.tool
async def evaluate(target: int = 10) -> EvaluationResult:
    """Evaluate progress toward the target count and return a reward and done flag."""
    if not http_client:
        raise RuntimeError("HTTP client not initialized")
    resp = await http_client.get("/state")
    current_count = resp.json().get("count", 0)
    delta = target - current_count
    reward = max(1 - abs(delta) / target, 0.0) if target > 0 else current_count 
    done = current_count >= target
    return EvaluationResult(reward=reward, done=done, content=f"Counter at {current_count}/{target}")


