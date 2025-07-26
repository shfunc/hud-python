# Apps Directory

Each subdirectory here is a launchable web application.

## Structure

Each app must have:
- `launch.py` - Script that accepts `--frontend-port` and `--backend-port` arguments
- `frontend/` - Frontend code (optional)
- `backend/` - Backend code (optional)

## Creating a New App

1. Create a new directory: `apps/myapp/`
2. Add a `launch.py` that starts your services
3. Use the `setup_websites` MCP tool to launch it

## Example

See `todo/` for a complete example with Next.js frontend and FastAPI backend. 