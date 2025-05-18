from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hud.env.environment import Environment
from hud.env.local_docker_client import LocalDockerClient
from hud.env.remote_client import RemoteClient
from hud.env.remote_docker_client import RemoteDockerClient
from hud.exceptions import GymMakeException
from hud.telemetry.context import get_current_task_run_id
from hud.telemetry.trace import trace
from hud.types import CustomGym, Gym
from hud.utils.common import get_gym_id

if TYPE_CHECKING:
    from hud.job import Job
    from hud.task import Task

logger = logging.getLogger("hud.gym")


async def make(
    env_src: Gym | Task,
    *,
    job: Job | None = None,
    job_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    skip_trace: bool = False,
) -> Environment:
    """
    Create an environment from an environment ID or a Task object.

    Args:
        env_src: Environment ID or Task object
        job: Job object to associate with this environment
        job_id: ID of job to associate with this environment (deprecated, use job instead)
        metadata: Additional metadata for the environment
        skip_trace: If True, don't automatically create a trace for this environment
    """
    # Check if we're already in a trace context
    in_trace = get_current_task_run_id() is not None
    use_trace = not skip_trace and not in_trace
    
    # Extract task info for tracing if needed
    task = None
    if isinstance(env_src, str | CustomGym):
        gym = env_src
    else:
        gym = env_src.gym
        task = env_src
    
    # Handle job parameter
    effective_job_id = None
    if job is not None:
        effective_job_id = job.id
    elif job_id is not None:
        effective_job_id = job_id
    else:
        # Try to get an active job from the decorator context
        try:
            import hud.job

            active_job = hud.job.get_active_job()
            if active_job:
                effective_job_id = active_job.id
        except ImportError:
            pass  # Module not available, skip
    
    # Prepare trace attributes if we're going to use tracing
    trace_attrs = {}
    if use_trace:
        # Generate a client-side task_run_id
        task_run_id = f"env-{uuid.uuid4()}"
        
        # Add relevant attributes
        if task:
            trace_attrs["task_id"] = task.id
            trace_attrs["task_prompt"] = task.prompt
        
        if effective_job_id:
            trace_attrs["job_id"] = effective_job_id
        
        trace_attrs["gym"] = str(gym)
    
    # Function to create the environment
    async def create_env() -> Environment:
        # data that is generated as we create the environment
        # we want to attach this to the exception if the environment creation fails
        build_data = {}

        try:
            if metadata is None:
                metadata_copy = {}
            else:
                metadata_copy = metadata.copy()

            # If we're in a trace context, add the task_run_id to metadata
            current_task_run_id = get_current_task_run_id() 
            if current_task_run_id:
                metadata_copy["task_run_id"] = current_task_run_id

            if isinstance(gym, CustomGym):
                if isinstance(gym.image_or_build_context, str):
                    uri = gym.image_or_build_context
                elif isinstance(gym.image_or_build_context, Path):
                    # need to build the image
                    if gym.location == "local":
                        uri, build_data = await LocalDockerClient.build_image(
                            gym.image_or_build_context
                        )
                    elif gym.location == "remote":
                        uri, build_data = await RemoteDockerClient.build_image(
                            gym.image_or_build_context
                        )
                    else:
                        raise ValueError(f"Invalid environment location: {gym.location}")
                else:
                    raise ValueError(f"Invalid image or build context: {gym.image_or_build_context}")

                if gym.location == "local":
                    logger.info("Creating local environment")
                    client = await LocalDockerClient.create(uri)
                elif gym.location == "remote":
                    logger.info("Creating remote environment")
                    client = await RemoteDockerClient.create(
                        image_uri=uri,
                        job_id=effective_job_id,
                        task_id=task.id if task else None,
                        metadata=metadata_copy,
                    )
                else:
                    raise ValueError(f"Invalid environment location: {gym.location}")

                # Set up the environment with a source path
                if isinstance(gym.image_or_build_context, Path):
                    logger.info("Setting source path")
                    client.set_source_path(gym.image_or_build_context)
            elif isinstance(gym, str):
                logger.info("Creating private environment")
                # Note: the gym_name_or_id is a unique identifier, but it is not a true
                # gym_id for the purposes of building the environment
                # we therefore fetch the gym_id from the HUD API here
                true_gym_id = await get_gym_id(gym)

                # Create the environment
                client, build_data = await RemoteClient.create(
                    gym_id=true_gym_id,
                    job_id=effective_job_id,
                    task_id=task.id if task else None,
                    metadata=metadata_copy,
                )
            else:
                raise ValueError(f"Invalid gym source: {gym}")

            # Create the environment itself
            environment = Environment(
                client=client, metadata=metadata_copy, task=task, build_data=build_data
            )

            if task:
                await environment._setup()
            return environment
        except Exception as e:
            build_data["exception"] = str(e)
            raise GymMakeException("Failed to create environment", build_data) from e
    
    # If we're using a trace context, create one
    if use_trace:
        with trace(task_run_id=task_run_id, **trace_attrs) as trace_id:
            # Will implicitly pass trace_id via contextvars
            return await create_env()
    else:
        # No trace context needed (or we're already in one)
        return await create_env()
