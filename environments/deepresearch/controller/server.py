"""Controller bridges MCP tools to the DeepResearch environment HTTP API."""

from typing import List, Dict

from controller import mcp, http_client
from hud.tools.types import EvaluationResult


@mcp.initialize
async def init(_: dict):
    # Ensure environment server is reachable
    await http_client.get("/health")


@mcp.shutdown
async def cleanup():
    await http_client.aclose()


@mcp.tool()
async def setup() -> str:
    await http_client.post("/setup")
    return ""


@mcp.tool()
async def search(query: str) -> List[Dict[str, str]]:
    resp = await http_client.post("/search", json={"query": query})
    return resp.json()


@mcp.tool()
async def fetch(url: str) -> str:
    resp = await http_client.post("/fetch", json={"url": url})
    data = resp.json()
    return data.get("content", "")


@mcp.tool()
async def answer(final_answer: str) -> str:
    await http_client.post("/answer", json={"final_answer": final_answer})
    return f"Answer submitted: {final_answer}"


@mcp.tool()
async def evaluate(expected_answer: str) -> EvaluationResult:
    resp = await http_client.post("/evaluate", json={"expected_answer": expected_answer})
    return EvaluationResult(**resp.json())


if __name__ == "__main__":
    mcp.run()
