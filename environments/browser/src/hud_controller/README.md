# HUD Controller

MCP server implementation for the browser automation environment with comprehensive evaluation system.

## Architecture

The controller manages:
- X11 display server (Xvfb)
- VNC server (x11vnc + websockify) 
- Browser automation (via HudComputerTool and PlaywrightTool)
- Dynamic app launching
- **Evaluation System**: Setup tools, evaluators, and class-based problems
- **MCP Resources**: Dynamic registry discovery and schema introspection

## Key Files

### Core Files
- **server.py** - FastMCP server with tools, resources, and initialization
- **runtime.py** - Implementation of setup and evaluate tools
- **services.py** - Service management (X11, VNC, browser, apps)

### Evaluation System
- **evaluators/** - `@evaluator` decorated classes with `EvaluatorRegistry`
- **setup/** - `@setup` decorated classes with `SetupRegistry`
- **problems/** - `@problem` decorated classes with `ProblemRegistry`

## Evaluation System Architecture

### Class-Based Problems with Inheritance

```python
# Base classes for reusability
class BaseSeedProblem:
    def get_setup(self): 
        return {"function": "todo_seed", "args": {"num_items": 5}}

# Concrete problems using multiple inheritance
@problem("todo_basic_usage", app="todo", difficulty="easy")
class TodoBasicUsageProblem(BaseSeedProblem, BaseCompletionProblem):
    pass  # Inherits setup and evaluation
```

### Registry Pattern

```python
# 1. Global registry + decorator
@evaluator("todo_completed", app="todo", description="Check completion count")
class TodoCompletedEvaluator:
    async def __call__(self, context, expected_count: int):
        # Implementation here

# 2. Registry class with factory methods
class EvaluatorRegistry:
    @classmethod
    def create_evaluator(cls, spec, context): pass
```

### BrowserEnvironmentContext

Unified interface for environment interactions:
- `call_app_api(app, endpoint, method, data)` - Call app backend API
- `execute_setup(setup_spec)` - Execute setup tool
- `execute_evaluation(eval_spec)` - Execute evaluator
- `get_page_content()` - Get current page content
- `get_app_port(app)` - Get app port number

## Adding New Components

### Evaluators
```python
@evaluator("my_evaluator", app="todo", description="Custom evaluator")
class MyEvaluator:
    async def __call__(self, context, param: str):
        data = await context.call_app_api("todo", "/api/eval/stats")
        return {"reward": 1.0, "done": True, "info": {...}}
```

### Setup Tools
```python
@setup("my_setup", app="todo", description="Custom setup")
class MySetup:
    async def __call__(self, context, param: str):
        result = await context.call_app_api("todo", "/api/eval/reset", method="DELETE")
        return {"status": "success", "setup": "my_setup"}
```

### Problems
```python
@problem("my_problem", app="todo", difficulty="medium")
class MyProblem(BaseSeedProblem):  # Inherit common setup
    def get_evaluation(self):
        return {"function": "my_evaluator", "args": {"param": "value"}}
```

## MCP Resources

Registries exposed as MCP resources:
- `evaluators://registry` - All evaluators
- `evaluators://{env}` - Environment-specific evaluators
- `setup://registry` - All setup tools
- `problems://registry` - All problems
- `schema://evaluator/{name}` - Detailed schemas
- `telemetry://live` - VNC URL and service status

## Runtime Tools

Core evaluation tools in `runtime.py`:
- **setup** - `{"name": "problem_name"}` or `{"function": "setup_name", "args": {...}}`
- **evaluate** - `{"name": "problem_name"}` or `{"function": "evaluator_name", "args": {...}}`

## Development Workflow

1. **Add evaluators** in `evaluators/{app}.py` with `@evaluator`
2. **Add setup tools** in `setup/{app}.py` with `@setup`
3. **Define problems** in `problems/{app}.py` with `@problem`, using inheritance
4. **Extend app backends** with `/api/eval/*` endpoints
5. **Test with MCP tools** and resources 