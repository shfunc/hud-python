#!/bin/bash
# Test MCP initialization without BROWSER_PROVIDER

echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {"roots": {"listChanged": true}}, "clientInfo": {"name": "Test", "version": "1.0"}}}' | docker run --rm -i hud-remote-browser:latest