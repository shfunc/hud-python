# Todo App

Simple todo list application with Next.js frontend and FastAPI backend.

## Local Development

```bash
# Backend (terminal 1)
cd backend
uv run uvicorn main:app --reload

# Frontend (terminal 2)  
cd frontend
npm install
npm run dev
```

## Launching via MCP

This app is designed to be launched dynamically:

```python
await client.call_tool("setup_websites", {
    "app_name": "todo",
    "frontend_port": 3001,
    "backend_port": 5001
})
```

## Tech Stack

- **Frontend**: Next.js, TypeScript, Tailwind CSS
- **Backend**: FastAPI, SQLite, uv for dependency management 