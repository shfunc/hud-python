from __future__ import annotations

import enum
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel


class CustomGym(BaseModel):
    """
    Public environment specification with a dockerfile and controller.

    If the location is remote, the env will be created on the server.
    If the location is local, the env will be created locally via docker.

    The dockerfile can be specified directly or automatically found in the controller_source_dir.
    If neither is provided, an error will be raised during validation.
    """

    type: Literal["public"] = "public"
    location: Literal["local", "remote"]
    # A. If path, then it is a docker build context on the local computer.
    #    If the location is local, docker build will be used to create the image.
    #    If the location is remote, we will build the image remotely.
    #    The controller will be automatically installed and kept in sync with local changes
    #    as long as a pyproject.toml is present at the root of the folder.
    # B. If string, then it is the uri of the docker image to use.
    #    The controller must already be installed in the image.
    image_or_build_context: str | Path
    # host_config will be passed to the docker client when creating the environment.
    # refer to official docker api documentation for available configs.
    host_config: dict[str, Any] | None = None


class MCPConfig(BaseModel):
    """
    MCP config for the environment.
    """

    type: Literal["mcp"] = "mcp"
    config: dict[str, Any]


class EnvironmentStatus(str, enum.Enum):
    """
    Status of the environment.

    Attributes:
        INITIALIZING: The environment is initializing
        RUNNING: The environment is running
        COMPLETED: The environment is completed
        ERROR: The environment is in an error state
    """

    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


# Available HUD gyms
ServerGym: TypeAlias = Literal["qa", "hud-browser", "OSWorld-Ubuntu", "docker"]

# Gyms can be either custom or server-side
Gym: TypeAlias = CustomGym | MCPConfig | ServerGym


# Metadata keys for the environment.
# partial: Whether the environment evaluator should give partial grades.
# eval_model: The model to use for evaluation when running a VLM. Wraps langchain.
# agent_name: The name of the agent that was used for running this task.
ServerMetadataKeys: TypeAlias = Literal["partial", "eval_model", "agent_name"]
MetadataKeys: TypeAlias = str | ServerMetadataKeys


# Dictionary of sensitive data (only supported for hud-browser environments)
# key: website name or page identifier
# value: Dictionary of credentials for the sensitive data
# Example:
# {
#     "google.com": {
#         "google_username": "my_username",
#         "google_password": "my_password"
#     }
# }
# The agent only has access to the key of the credential, not the value. (i.e. google_username)
# The value is only available to the environment. (i.e. my_username)
SensitiveData: TypeAlias = dict[str, dict[str, str]]
