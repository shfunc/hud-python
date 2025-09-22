"""Minimal FastAPI environment server (HTTP-based)."""

from fastapi import FastAPI

app = FastAPI(title="Blank Environment API")

_count = 0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/act")
def act():
    global _count
    _count += 1
    return {"count": _count}

@app.post("/reset")
def reset():
    global _count
    _count = 0
    return {"ok": True}

@app.get("/state")
def state():
    return {"count": _count}
