# Deployment Guide for HUD Browser Environment

This guide explains how to build and deploy the simple browser environment for use with the HUD MCP server.

## Quick Start

To use this environment with the HUD MCP server, configure your MCP client:

```python
config = {
    "mcpServers": {
        "browser": {
            "url": "https://mcp.hud.so/api/v3/mcp",
            "headers": {
                "Authorization": f"Bearer {HUD_API_KEY}",
                "Mcp-Image": "your-username/hud-browser:latest"  # Your published image
            }
        }
    }
}
```

## What's in the Image

The `hud-browser` image contains:
- **Chromium Browser** with Playwright for automation
- **MCP Server** running in stdio mode with computer tools
- **VNC Server** for remote viewing (port 8080)
- **Pre-built Web Apps** (e.g., todo app with Next.js frontend and FastAPI backend)

## Building and Publishing the Image

### Option 1: Docker Hub (Public/Private)

```bash
# Build the image
docker build -t your-username/hud-browser:latest environments/simple_browser/

# Push to Docker Hub
docker login
docker push your-username/hud-browser:latest

# Use in MCP client
"Mcp-Image": "your-username/hud-browser:latest"
```

### Option 2: GitHub Container Registry

```bash
# Build and tag
docker build -t ghcr.io/your-org/hud-browser:latest environments/simple_browser/

# Login and push
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
docker push ghcr.io/your-org/hud-browser:latest

# Use in MCP client
"Mcp-Image": "ghcr.io/your-org/hud-browser:latest"
```

### Option 3: Google Container Registry (GCR)

```bash
# Build and tag
docker build -t gcr.io/your-project/hud-browser:latest environments/simple_browser/

# Configure auth and push
gcloud auth configure-docker
docker push gcr.io/your-project/hud-browser:latest

# Use in MCP client
"Mcp-Image": "gcr.io/your-project/hud-browser:latest"
```

### Option 4: AWS Elastic Container Registry (ECR)

```bash
# Build the image
docker build -t hud-browser:latest environments/simple_browser/

# Tag for ECR
docker tag hud-browser:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/hud-browser:latest

# Login and push
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/hud-browser:latest

# Use in MCP client
"Mcp-Image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/hud-browser:latest"
```

### Option 5: Private Registry

```bash
# Build and tag for your registry
docker build -t registry.company.com/hud-browser:latest environments/simple_browser/

# Push to private registry
docker push registry.company.com/hud-browser:latest

# Use in MCP client
"Mcp-Image": "registry.company.com/hud-browser:latest"
```

## Configuring the Environment

Environment variables are passed through headers with the `Env-` prefix. The header name is transformed to create the environment variable:
- The `Env-` prefix is removed
- Dashes are converted to underscores
- The name is converted to uppercase

For example: `Env-launch-apps` → `LAUNCH_APPS`

### Available Environment Variables

- `LAUNCH_APPS` - Comma-separated list of apps to launch (default: none)
- `BROWSER_URL` - URL to navigate browser to (default: https://google.com)
- `WAIT_FOR_APPS` - Wait for apps before browser navigation (default: true)
- `FRONTEND_PORT_START` - Starting port for frontend services (default: 3000)
- `BACKEND_PORT_START` - Starting port for backend services (default: 5000)
- `MCP_TRANSPORT` - MCP server transport mode: stdio or streamable-http (default: stdio)

### Example: Launch with Todo App

```python
config = {
    "mcpServers": {
        "browser": {
            "url": "https://mcp.hud.so/api/v3/mcp",
            "headers": {
                "Authorization": f"Bearer {HUD_API_KEY}",
                "Mcp-Image": "your-username/hud-browser:latest",
                "Env-launch-apps": "todo",
                "Env-browser-url": "http://localhost:3000"
            }
        }
    }
}
```

### Example: Multiple Apps with Custom Ports

```python
config = {
    "mcpServers": {
        "browser": {
            "url": "https://mcp.hud.so/api/v3/mcp",
            "headers": {
                "Authorization": f"Bearer {HUD_API_KEY}",
                "Mcp-Image": "your-username/hud-browser:latest",
                "Env-launch-apps": "todo,calendar",
                "Env-frontend-port-start": "3000",
                "Env-backend-port-start": "5000",
                "Env-wait-for-apps": "true"
            }
        }
    }
}
```
