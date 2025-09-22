"""
FastAPI server for DeepResearch environment.
Holds EXA API key on the server side and exposes simple HTTP endpoints
that the controller calls. Mirrors the browser/blank environment pattern.
"""

import os
import socket
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class _EnvState:
    """In-memory environment state for tracking usage and agent answer."""

    def __init__(self) -> None:
        self.search_count: int = 0
        self.fetch_count: int = 0
        self.submitted_answer: Optional[str] = None

    def reset(self) -> None:
        self.search_count = 0
        self.fetch_count = 0
        self.submitted_answer = None


state = _EnvState()


class SearchRequest(BaseModel):
    query: str


class FetchRequest(BaseModel):
    url: str


class AnswerRequest(BaseModel):
    final_answer: str


class EvaluateRequest(BaseModel):
    expected_answer: str


app = FastAPI(title="DeepResearch Environment API", version="0.1.0")


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "healthy"}


async def _is_port_open(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.15)
    try:
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


@app.post("/setup")
async def setup() -> Dict[str, Any]:
    state.reset()
    return {"ok": True}


@app.post("/search")
async def search(req: SearchRequest) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    max_results: int = 1

    exa_api_key: Optional[str] = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        raise HTTPException(status_code=400, detail="EXA_API_KEY not set on environment")

    try:
        search_url = "https://api.exa.ai/search"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                search_url,
                headers={"x-api-key": exa_api_key, "Content-Type": "application/json"},
                json={
                    "query": req.query,
                    "numResults": max_results,
                    "type": "keyword",
                    "userLocation": "us",
                    "contents": {"text": {"maxCharacters": 1000}},
                },
            )
            response.raise_for_status()
            data = response.json()
            for item in data.get("results", []):
                title = item.get("title", "")
                url = item.get("url", "")
                if title and url:
                    results.append({"title": title, "url": url})

            if not results:
                autoprompt = data.get("autopromptString", req.query)
                return [{"message": "No results found", "query": req.query, "autopromptString": autoprompt}]

    except httpx.HTTPStatusError as e:  # pragma: no cover - network errors are environment dependent
        status_code = e.response.status_code
        if status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid EXA_API_KEY")
        if status_code == 429:
            raise HTTPException(status_code=429, detail="Exa API rate limit exceeded")
        raise HTTPException(status_code=502, detail=f"Exa API error: {status_code}")
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Search failed: {type(e).__name__}: {e}")

    state.search_count += 1
    return results


@app.post("/fetch")
async def fetch(req: FetchRequest) -> Dict[str, str]:
    from urllib.parse import urlparse

    max_length: int = 2500
    parsed = urlparse(req.url)
    if not parsed.scheme or not parsed.netloc:
        raise HTTPException(status_code=400, detail=f"Invalid URL: {req.url}")

    exa_api_key: Optional[str] = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        raise HTTPException(status_code=400, detail="EXA_API_KEY not set on environment")

    try:
        contents_url = "https://api.exa.ai/contents"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                contents_url,
                headers={"x-api-key": exa_api_key, "Content-Type": "application/json"},
                json={
                    "urls": [req.url],
                    "text": {"maxCharacters": max_length, "includeHtmlTags": False},
                    "highlights": {"numSentences": 5, "highlightsPerUrl": 3},
                    "summary": {"query": "main takeaways"},
                    "livecrawl": "fallback",
                },
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if results:
                result = results[0]
                text = result.get("text", "")
                summary = result.get("summary", "")
                highlights = result.get("highlights", [])

                parts: List[str] = []
                if summary:
                    parts.append("=== SUMMARY (Main Takeaways) ===")
                    parts.append(summary)
                    parts.append("")
                if highlights:
                    parts.append("=== KEY HIGHLIGHTS ===")
                    for idx, hl in enumerate(highlights[:3], 1):
                        parts.append(f"\nHighlight {idx}:")
                        parts.append(str(hl))
                    parts.append("")
                if text:
                    parts.append("=== FULL CONTENT ===")
                    if len(text) > max_length:
                        text = text[:max_length] + "...[truncated]"
                    parts.append(text)

                content = "\n".join(parts) if parts else "No content available"
            else:
                content = "No content available for this URL"

    except httpx.HTTPStatusError as e:  # pragma: no cover
        status_code = e.response.status_code
        if status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid EXA_API_KEY")
        if status_code == 429:
            raise HTTPException(status_code=429, detail="Exa API rate limit exceeded")
        raise HTTPException(status_code=502, detail=f"Exa API error: {status_code}")
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Fetch failed: {type(e).__name__}: {e}")

    state.fetch_count += 1
    return {"content": content}


@app.post("/answer")
async def answer(req: AnswerRequest) -> Dict[str, Any]:
    state.submitted_answer = req.final_answer
    return {"ok": True, "message": "Answer submitted"}


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    submitted = state.submitted_answer
    if submitted is None:
        return {
            "reward": 0.0,
            "content": f"No answer submitted. Searches: {state.search_count}, Fetches: {state.fetch_count}",
            "done": False,
        }

    submitted_clean = submitted.strip().lower()
    expected_clean = req.expected_answer.strip().lower()
    is_correct = expected_clean in submitted_clean if expected_clean else False

    result_msg = (
        ("Correct! " if is_correct else "Incorrect. ")
        + f"Submitted: '{submitted}', Expected: '{req.expected_answer}'. "
        + f"Stats: {state.search_count} searches, {state.fetch_count} fetches, {state.search_count + state.fetch_count} total operations."
    )

    return {"reward": 1.0 if is_correct else 0.0, "content": result_msg, "done": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


