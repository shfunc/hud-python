"""HUD MCP Agent that collects ART trajectories for training."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import art
import hud
from hud.agent import MCPAgent
from hud.types import AgentResponse, MCPToolCall, MCPToolResult
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

if TYPE_CHECKING:
    from hud.clients.base import AgentMCPClient
    from hud.datasets import TaskConfig

logger = logging.getLogger(__name__)


class ARTTrainingAgent(MCPAgent):
    """
    HUD MCP Agent that collects ART trajectories during execution.
    
    This agent bridges HUD's MCP infrastructure with ART's training system by:
    1. Using HUD's MCPClient for all MCP communication
    2. Collecting messages in ART's Trajectory format during execution
    3. Using ART's model for inference
    4. Returning trajectories that can be used with ART's training pipeline
    """
    
    def __init__(
        self,
        art_model: art.Model,
        mcp_client: AgentMCPClient,
        max_turns: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ART training agent.
        
        Args:
            art_model: ART model to use for inference and training
            mcp_client: HUD MCP client for server communication
            max_turns: Maximum number of turns for agent execution
            **kwargs: Additional arguments passed to MCPAgent
        """
        super().__init__(mcp_client=mcp_client, **kwargs)
        
        self.art_model = art_model
        self.max_turns = max_turns
        self.current_trajectory: art.Trajectory | None = None
        
        # Override model name for telemetry
        self.model_name = f"art-{art_model.name}"
        
    async def run_with_trajectory(
        self,
        task: TaskConfig | str,
        initial_screenshot: bool = False,
    ) -> art.Trajectory:
        """
        Run the agent and return an ART trajectory.
        
        Args:
            task: Either a TaskConfig with setup/evaluate or a string prompt
            initial_screenshot: Whether to capture initial screenshot
            
        Returns:
            ART Trajectory containing the execution history and rewards
        """
        from hud.datasets import TaskConfig
        
        # Initialize the trajectory
        self.current_trajectory = art.Trajectory(
            messages_and_choices=[],
            reward=0.0,
            metadata={},
            metrics={
                "task_completed": False,
                "num_turns": 0,
                "tool_calls": 0,
                "tool_errors": 0,
            },
        )
        
        # Store task metadata
        if isinstance(task, TaskConfig):
            self.current_trajectory.metadata["task_id"] = task.id
            self.current_trajectory.metadata["task_prompt"] = task.prompt
        else:
            self.current_trajectory.metadata["task_prompt"] = task
            
        try:
            # Run the task using HUD's infrastructure
            trace = await self.run(task, max_steps=self.max_turns)
            
            # Extract reward from trace
            self.current_trajectory.reward = trace.reward
            
            # Add completion metrics
            self.current_trajectory.metrics["task_completed"] = trace.done
            self.current_trajectory.metrics["is_error"] = trace.isError
            
            # Add any additional info from trace
            if trace.info:
                self.current_trajectory.metadata.update(trace.info)
                
        except Exception as e:
            logger.error(f"Error during trajectory collection: {e}")
            self.current_trajectory.reward = 0.0
            self.current_trajectory.metrics["error"] = str(e)
            
        return self.current_trajectory.finish()
    
    async def create_initial_messages(
        self, prompt: str, screenshot: str | None = None
    ) -> list[Any]:
        """
        Create initial messages for the conversation.
        
        Overrides MCPAgent to add messages to ART trajectory.
        """
        # Get available tools for the trajectory
        tool_schemas = self.get_tool_schemas()
        
        # Convert to OpenAI format for ART
        openai_tools = []
        for schema in tool_schemas:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema.get("description", ""),
                    "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
                }
            }
            openai_tools.append(openai_tool)
            
        # Store tools in trajectory
        if self.current_trajectory:
            self.current_trajectory.tools = openai_tools
            
            # Add system message
            system_prompt = self.get_system_prompt()
            self.current_trajectory.messages_and_choices.append({
                "role": "system",
                "content": system_prompt,
            })
            
            # Add user message
            user_content = prompt
            if screenshot:
                # Handle screenshot if needed
                user_content = f"{prompt}\n[Screenshot captured]"
                
            self.current_trajectory.messages_and_choices.append({
                "role": "user", 
                "content": user_content,
            })
            
        # Return messages for HUD's agent loop
        return [{"prompt": prompt, "screenshot": screenshot}]
    
    @hud.instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_model_response(self, messages: list[Any]) -> AgentResponse:
        """
        Get response from ART model.
        
        Overrides MCPAgent to use ART's model and collect trajectory.
        """
        if not self.current_trajectory:
            raise ValueError("No active trajectory")
            
        # Get OpenAI client from ART model
        client = self.art_model.openai_client()
        
        # Get tool schemas
        tool_schemas = []
        if self.current_trajectory.tools:
            tool_schemas = self.current_trajectory.tools
            
        try:
            # Call ART model
            response = await client.chat.completions.create(
                model=self.art_model.get_inference_name(),
                messages=self.current_trajectory.messages(),
                tools=tool_schemas if tool_schemas else None,
                max_completion_tokens=4000,
                temperature=1.0,  # Important for training
            )
            
            # Store the choice in trajectory
            choice = response.choices[0]
            self.current_trajectory.messages_and_choices.append(choice)
            
            # Convert to HUD's AgentResponse
            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append(
                        MCPToolCall(
                            name=tc.function.name,
                            arguments=json.loads(tc.function.arguments) if tc.function.arguments else {},
                        )
                    )
                    
            # Check if agent is done
            done = False
            content = choice.message.content or ""
            
            # Simple heuristic: if no tool calls and has content, might be done
            if not tool_calls and content:
                done = True
                
            return AgentResponse(
                content=content,
                tool_calls=tool_calls,
                done=done,
            )
            
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            return AgentResponse(
                content=f"Error: {str(e)}",
                tool_calls=[],
                done=True,
                isError=True,
            )
    
    async def format_tool_results(
        self,
        tool_calls: list[MCPToolCall],
        tool_results: list[MCPToolResult],
    ) -> list[Any]:
        """
        Format tool results for inclusion in conversation.
        
        Overrides to add tool results to ART trajectory.
        """
        formatted = []
        
        for call, result in zip(tool_calls, tool_results):
            # Extract text content from result
            content = ""
            if result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        content += item.text
                    else:
                        content += str(item)
                        
            # Add to trajectory
            if self.current_trajectory:
                # Find the tool_call_id from the last assistant message
                last_msg = self.current_trajectory.messages_and_choices[-1]
                tool_call_id = None
                
                if hasattr(last_msg, 'message') and hasattr(last_msg.message, 'tool_calls'):
                    for tc in last_msg.message.tool_calls:
                        if tc.function.name == call.name:
                            tool_call_id = tc.id
                            break
                            
                if tool_call_id:
                    self.current_trajectory.messages_and_choices.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,
                    })
                    
                # Update metrics
                self.current_trajectory.metrics["tool_calls"] += 1
                if result.isError:
                    self.current_trajectory.metrics["tool_errors"] += 1
                    
            # Format for HUD's agent loop
            formatted.append({
                "tool_name": call.name,
                "result": content,
                "is_error": result.isError,
            })
            
        return formatted