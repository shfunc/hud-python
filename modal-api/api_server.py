#!/usr/bin/env python3
"""
Modal API Server for HUD RL Training

A model-centric API that manages vLLM servers and training runs through Modal.
"""
from __future__ import annotations

import asyncio
import json
import os
from enum import Enum

# Import our Modal apps
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
import logging
import modal
from supabase import Client as SupabaseClient
from hud.settings import settings
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import our dependencies
from .dependencies import AuthContext, get_auth_context
from .supabase import get_supabase_client

sys.path.append(str(Path(__file__).parent.parent))
# Modal imports moved inside functions to avoid serialization issues
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="HUD RL Training API",
    description="Model-centric API for managing vLLM servers and training runs",
    version="1.0.0"
)

class ModelStatus(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    READY = "ready"
    TRAINING = "training"
    ERROR = "error"
    TERMINATED = "terminated"

# Pydantic models
class ModelCreateRequest(BaseModel):
    """Request to create a new model"""
    name: str = Field(..., description="Unique name for the model within your team")
    base_model: str = Field("Qwen/Qwen2.5-VL-3B-Instruct", description="Base model to use")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class ModelDeployRequest(BaseModel):
    """Request to deploy a vLLM server for a model"""
    gpu_type: str = "A100"
    gpu_count: int = 1


class TrainingLaunchRequest(BaseModel):
    """Request to launch a training run"""
    config: dict[str, Any] = Field(..., description="Training configuration")
    tasks: list[dict[str, Any]] = Field(..., description="Training tasks")
    gpu_type: str = "A100"
    gpu_count: int = 1


class ModelInfo(BaseModel):
    """Information about a model from database"""
    id: str
    name: str
    base_model: str
    status: ModelStatus
    vllm_url: str | None
    metadata: dict[str, Any] | None
    trainer_name: str | None  # Modal app name for current training
    created_at: datetime
    terminated_at: datetime | None
    time_alive: timedelta | None = None


# API Routes
@app.get("/")
async def root() -> dict[str, Any]:
    """API root endpoint"""
    return {
        "service": "HUD RL Training API",
        "version": "1.0.0",
        "endpoints": {
            "models": "/models",
            "docs": "/docs"
        }
    }

async def health_check_vllm(url: str) -> bool:
    """Health check the vLLM server for a model"""
    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            response = await client.get(url + "/health", headers={"Authorization": "Bearer token-abc123"})
            response.raise_for_status()
            return response.status_code == 200
        except Exception:
            return False

async def check_training_health(trainer_name: str) -> bool:
    """Check the health of a Modal training app by querying Modal API"""
    if not trainer_name:
        return False
    
    try:
        import subprocess
        from hud.settings import settings
        
        # Check if Modal credentials are available
        if not settings.modal_token_id or not settings.modal_token_secret:
            logger.warning("Modal credentials not configured - cannot check app status")
            return False
        
        # Query Modal for app status
        env = {
            **os.environ,
            "MODAL_TOKEN_ID": settings.modal_token_id,
            "MODAL_TOKEN_SECRET": settings.modal_token_secret
        }
        
        # Use modal app list to check if the app exists and is running
        result = subprocess.run(
            ["modal", "app", "list"],
            capture_output=True,
            text=True,
            env=env
        )
        
        if result.returncode == 0:
            # Check if our trainer_name appears in the list of running apps
            return trainer_name in result.stdout
        else:
            logger.warning(f"Failed to query Modal apps: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking Modal app health: {e}")
        return False


async def check_update_status(supabase: SupabaseClient, model: ModelInfo) -> ModelInfo:
    """Check and update the status of a model"""
    # Check vLLM health for deployed models
    vllm_health = False
    if model.vllm_url and model.status in [ModelStatus.DEPLOYING, ModelStatus.READY, ModelStatus.TRAINING]:
        vllm_health = await health_check_vllm(model.vllm_url)
    
    # Check training health for training models
    training_health = False
    if model.trainer_name and model.status == ModelStatus.TRAINING and model.time_alive and model.time_alive.total_seconds() > 600:
        training_health = await check_training_health(model.trainer_name)
    
    # Update status based on health checks
    if model.status == ModelStatus.DEPLOYING and vllm_health:
        model.status = ModelStatus.READY
        await (
            supabase.table("models")
            .update({
                "status": ModelStatus.READY
            })
            .eq("id", model.id)
            .execute()
        )
    elif model.status == ModelStatus.TRAINING:
        # For training models, check if training is still running
        if not training_health:
            model.status = ModelStatus.READY
            await (
                supabase.table("models")
                .update({
                    "status": ModelStatus.READY,
                    "trainer_name": None  # Clear trainer name since it's not running
                })
                .eq("id", model.id)
                .execute()
            )
    if (model.status == ModelStatus.READY or model.status == ModelStatus.TRAINING) and not vllm_health:
        model.status = ModelStatus.ERROR
        await (
            supabase.table("models")
            .update({
                "status": ModelStatus.ERROR,
                "terminated_at": datetime.now(timezone.utc).isoformat()
            })
            .eq("id", model.id)
            .execute()
        )
    return model
        

@app.get("/models", response_model=list[ModelInfo])
async def list_models(
    auth_context: AuthContext = Depends(get_auth_context),
) -> list[ModelInfo]:
    """List all models for the authenticated team"""
    supabase = await get_supabase_client()
    
    # Query models for this team
    result = await (
        supabase.table("models")
        .select("*")
        .eq("user_team_membership_id", auth_context.membership["id"])
        .execute()
    )

    time_alive = datetime.now(timezone.utc) - datetime.fromisoformat(result.data[0]["created_at"])
    
    models = [await check_update_status(supabase, ModelInfo(**row, time_alive=time_alive)) for row in result.data if row]
    
    return models


@app.post("/models")
async def create_model(
    request: ModelCreateRequest,
    auth_context: AuthContext = Depends(get_auth_context),
) -> ModelInfo:
    """Create a new model"""
    supabase = await get_supabase_client()
    
    # Check if model name already exists for this team
    existing = await (
        supabase.table("models")
        .select("id")
        .eq("user_team_membership_id", auth_context.membership["id"])
        .eq("name", request.name)
        .execute()
    )
    
    if existing.data:
        raise HTTPException(status_code=409, detail=f"Model '{request.name}' already exists")
    
    # Create the model
    result = await (
        supabase.table("models")
        .insert({
            "name": request.name,
            "base_model": request.base_model,
            "metadata": request.metadata or {},
            "user_team_membership_id": auth_context.membership["id"],
            "status": ModelStatus.PENDING
        })
        .execute()
    )
    
    return ModelInfo(**result.data[0])


@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model(
    model_name: str,
    auth_context: AuthContext = Depends(get_auth_context),
) -> ModelInfo:
    """Get information about a specific model"""
    supabase = await get_supabase_client()
    
    result = await (
        supabase.table("models")
        .select("*")
        .eq("user_team_membership_id", auth_context.membership["id"])
        .eq("name", model_name)
        .maybe_single()
        .execute()
    )
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    row = result.data
    
    return await check_update_status(supabase, ModelInfo(**row))


@app.post("/models/{model_name}/deploy")
async def deploy_model(
    model_name: str,
    request: ModelDeployRequest,
    auth_context: AuthContext = Depends(get_auth_context),
) -> ModelInfo:
    """Deploy a vLLM server for a model"""
    supabase = await get_supabase_client()
    
    # Get the model
    result = await (
        supabase.table("models")
        .select("*")
        .eq("user_team_membership_id", auth_context.membership["id"])
        .eq("name", model_name)
        .maybe_single()
        .execute()
    )
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model = result.data

    model = await check_update_status(supabase, ModelInfo(**model))
    
    # Check if already deployed
    if model.status == ModelStatus.READY:
        return model
    
    # Deploy vLLM server using Modal
    try:
        logger.info(f"Starting deployment for model {model_name}")
        
        # Import and use the deployer directly
        from .modal_vllm_deployer import deploy_vllm_for_model
        
        # Deploy vLLM for this specific model
        # Run in thread pool to avoid blocking the event loop
        import asyncio
        vllm_url = await asyncio.get_event_loop().run_in_executor(
            None,
            deploy_vllm_for_model,
            model.id,
            model_name,
            model.base_model,
            request.gpu_type
        )
        
        logger.info(f"Deployment successful, URL: {vllm_url}")
        
        # Update model with vLLM URL
        await (
            supabase.table("models")
            .update({
                "vllm_url": vllm_url,
                "status": ModelStatus.DEPLOYING,
                "metadata": {
                    **(model.metadata or {}),
                    "gpu_type": request.gpu_type,
                    "gpu_count": request.gpu_count
                }
            })
            .eq("id", model.id)
            .execute()
        )
        
        return await check_update_status(supabase, model)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to deploy vLLM: {e!s}")


@app.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    auth_context: AuthContext = Depends(get_auth_context),
) -> ModelInfo:
    """Delete a model and its vLLM server"""
    supabase = await get_supabase_client()
    
    # Get the model
    result = await (
        supabase.table("models")
        .select("*")
        .eq("user_team_membership_id", auth_context.membership["id"])
        .eq("name", model_name)
        .maybe_single()
        .execute()
    )
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model = result.data
    
    # TODO: Actually stop the vLLM server on Modal
    
    # Delete from database
    await (
        supabase.table("models")
        .update({
            "status": ModelStatus.TERMINATED,
            "terminated_at": datetime.now(timezone.utc).isoformat()
        })
        .eq("id", model["id"])
        .execute()
    )
    
    return ModelInfo(**model)


# vLLM Passthrough
@app.api_route("/models/{model_name}/vllm/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def vllm_passthrough(
    model_name: str,
    path: str,
    request: Request,
    auth_context: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """Proxy requests to the vLLM server for a specific model"""
    supabase = await get_supabase_client()
    
    # Get the model
    result = await (
        supabase.table("models")
        .select("*")
        .eq("user_team_membership_id", auth_context.membership["id"])
        .eq("name", model_name)
        .maybe_single()
        .execute()
    )
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model = await check_update_status(supabase, ModelInfo(**result.data))
    
    vllm_url = model.vllm_url
    if not vllm_url:
        raise HTTPException(status_code=503, detail=f"vLLM server not deployed for model '{model_name}'")
    
    # Build target URL
    target_url = f"{vllm_url}/v1/{path}"
    
    # Forward the request
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Get request body
            body = await request.body()
            
            # Forward headers (excluding host and authorization)
            headers = dict(request.headers)
            headers.pop("host", None)
            headers.pop("authorization", None)
            
            # Add vLLM API key
            headers["Authorization"] = "Bearer token-abc123"
            
            # Make the request
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                params=dict(request.query_params)
            )
            
            # Return the response
            return response.json()
            
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="vLLM server timeout")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"vLLM server error: {e!s}")


# Training Management
@app.post("/models/{model_name}/training/launch")
async def launch_training(
    model_name: str,
    request: TrainingLaunchRequest,
    auth_context: AuthContext = Depends(get_auth_context),
) -> ModelInfo:
    """Launch a training run for a model"""
    supabase = await get_supabase_client()
    
    # Get the model
    result = await (
        supabase.table("models")
        .select("*")
        .eq("user_team_membership_id", auth_context.membership["id"])
        .eq("name", model_name)
        .maybe_single()
        .execute()
    )
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model = result.data
    model = await check_update_status(supabase, ModelInfo(**model))
    # Check if already has active training
    if model.trainer_name:
        # TODO: Check if training is actually still running
        raise HTTPException(status_code=409, detail="Model already has active training")
    
    # Check vLLM URL
    vllm_url = model.vllm_url
    if not vllm_url:
        raise HTTPException(
            status_code=400,
            detail=f"vLLM server not deployed for model '{model_name}'. Deploy it first with POST /models/{model_name}/deploy"
        )
    
    # Create a job entry in runs table
    job_name = request.config.get("job_name", f"Training {model_name}")
    job_result = await (
        supabase.table("runs")
        .insert({
            "name": job_name,
            "status": "initializing",
            "user_id": auth_context.user["id"],
            "user_team_membership_id": auth_context.membership["id"]
        })
        .execute()
    )
    
    if not job_result.data or len(job_result.data) == 0:
        raise HTTPException(status_code=500, detail="Failed to create job entry")
    
    job_id = job_result.data[0]["id"]
    
    # Create mapping in model_jobs table
    model_job_result = await (
        supabase.table("model_jobs")
        .insert({
            "model_id": model.id,
            "job_id": job_id
        })
        .execute()
    )
    
    if not model_job_result.data:
        raise HTTPException(status_code=500, detail="Failed to create model-job mapping")
    
    # Add job_id to config and ensure correct vLLM URL
    config_with_updates = {
        **request.config,
        "job_id": job_id,
    }
    
    # Update vLLM URL in actor config to use the deployed server
    if "actor" in config_with_updates:
        config_with_updates["actor"]["vllm_base_url"] = f"{vllm_url}/v1"
    
    # Prepare config
    config_json = json.dumps(config_with_updates)
    tasks_json = json.dumps(request.tasks)
    
    # Launch training based on GPU type
    try:
        import asyncio

        from .modal_training_deployer import launch_training_for_model
        app_name = await asyncio.get_event_loop().run_in_executor(
            None,
            launch_training_for_model,
            model.id,
            model_name,
            model.base_model,
            request.gpu_type,
            request.gpu_count,
            config_json,
            tasks_json,
            f"{vllm_url}/v1",
            f"/checkpoints/{model.id}",
            auth_context.api_key
        )
        
        # Update model with trainer_name
        result = await (
            supabase.table("models")
            .update({
                "status": ModelStatus.TRAINING,
                "trainer_name": app_name,
            })
            .eq("id", model.id)
            .execute()
        )
        
        return ModelInfo(**result.data[0])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to launch training: {e!s}") from e


@app.get("/models/{model_name}/training/logs")
async def get_training_logs(
    model_name: str,
    auth_context: AuthContext = Depends(get_auth_context),
) -> StreamingResponse:
    """Stream logs from the current training run"""
    supabase = await get_supabase_client()
    
    # Get the model
    result = await (
        supabase.table("models")
        .select("*")
        .eq("user_team_membership_id", auth_context.membership["id"])
        .eq("name", model_name)
        .maybe_single()
        .execute()
    )
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model = await check_update_status(supabase, ModelInfo(**result.data))
    
    if not model.trainer_name:
        raise HTTPException(status_code=404, detail="No active training for this model")
    
    app_name = model.trainer_name
    
    # Stream logs from Modal app
    async def log_generator() -> AsyncGenerator[str, None]:
        yield f"=== Fetching logs for training app: {app_name} ===\n\n"
        
        try:
            process = await asyncio.create_subprocess_exec(
                "modal", "app", "logs", app_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    "MODAL_TOKEN_ID": settings.modal_token_id
                }
            )
            
            # Stream stdout
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                yield line.decode('utf-8')
            
            # Check for any errors
            _, stderr = await process.communicate()
            if stderr:
                yield f"\n=== Error fetching logs ===\n{stderr.decode('utf-8')}\n"
                
        except Exception as e:
            yield f"\n=== Error streaming logs: {str(e)} ===\n"
    
    return StreamingResponse(log_generator(), media_type="text/plain")


@app.delete("/models/{model_name}/training")
async def cancel_training(
    model_name: str,
    auth_context: AuthContext = Depends(get_auth_context),
) -> ModelInfo:
    """Cancel the current training run"""
    supabase = await get_supabase_client()
    
    # Get the model
    result = await (
        supabase.table("models")
        .select("*")
        .eq("user_team_membership_id", auth_context.membership["id"])
        .eq("name", model_name)
        .maybe_single()
        .execute()
    )
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model = await check_update_status(supabase, ModelInfo(**result.data))
    
    if not model.trainer_name:
        if model.status == ModelStatus.TRAINING:
            await (
                supabase.table("models")
                .update({
                    "status": ModelStatus.READY,
                })
                .eq("id", model.id)
                .execute()
            )
            return ModelInfo(**model.model_dump())
        else:
            raise HTTPException(status_code=404, detail="Model is not training")
    
    # Get the job_id from model_jobs table
    job_result = await (
        supabase.table("model_jobs")
        .select("job_id")
        .eq("model_id", model.id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    
    job_id = job_result.data[0]["job_id"] if job_result.data else None
    
    # Cancel the Modal app
    try:
        import subprocess
        from hud.settings import settings
        app_name = model.trainer_name
        
        # Check if Modal credentials are available
        if not settings.modal_token_id or not settings.modal_token_secret:
            logger.error("Modal credentials not configured - cannot stop Modal app")
            raise HTTPException(
                status_code=500, 
                detail="Modal credentials not configured. Please set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET."
            )
        
        # Set up environment with Modal credentials
        env = {
            **os.environ,
            "MODAL_TOKEN_ID": settings.modal_token_id,
            "MODAL_TOKEN_SECRET": settings.modal_token_secret
        }
        
        # Use Modal CLI to stop the app
        result = subprocess.run(
            ["modal", "app", "stop", app_name],
            capture_output=True,
            text=True,
            env=env
        )
        if result.returncode != 0:
            logger.error(f"Error stopping Modal app: {result.stderr}")
            # Try to provide more context if the app doesn't exist
            if "Could not find a deployed app" in result.stderr:
                logger.warning(f"Modal app '{app_name}' may have already been stopped or completed")
                # Don't raise an error - the app is already gone which is what we wanted
            else:
                # For other errors, raise an exception
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to stop Modal app: {result.stderr}"
                )
        else:
            logger.info(f"Successfully stopped Modal app: {app_name}")
            
            # Clean up log file
            log_file = f"/tmp/modal-training-{model.id[:8]}.log"
            try:
                if os.path.exists(log_file):
                    os.remove(log_file)
                    logger.info(f"Cleaned up log file: {log_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up log file: {e}")
            
            # Clean up Modal volume files
            try:
                import modal
                info_volume = modal.Volume.from_name("hud-rl-info", create_if_missing=False)
                if info_volume:
                    # Remove the entire model directory from the volume
                    model_dir = f"/training/{model.id}"
                    # Modal volumes don't have a direct delete method, so we'll use Modal CLI
                    volume_cleanup_result = subprocess.run(
                        ["modal", "volume", "rm", "-r", "hud-rl-info", model_dir],
                        capture_output=True,
                        text=True,
                        env=env
                    )
                    if volume_cleanup_result.returncode == 0:
                        logger.info(f"Cleaned up Modal volume directory: {model_dir}")
                    else:
                        logger.warning(f"Failed to clean up Modal volume: {volume_cleanup_result.stderr}")
            except Exception as e:
                logger.warning(f"Failed to clean up Modal volume: {e}")
                
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error stopping Modal app: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping Modal app: {e!s}") from e
    
    # Update job status if we have a job_id
    if job_id:
        await (
            supabase.table("runs")
            .update({
                "status": "cancelled",
                "terminated_at": datetime.now(timezone.utc).isoformat()
            })
            .eq("id", job_id)
            .execute()
        )
    
    # Clear trainer_name and update model status
    await (
        supabase.table("models")
        .update({
            "trainer_name": None,
            "status": ModelStatus.READY
        })
        .eq("id", model.id)
        .execute()
    )
    
    return ModelInfo(**model.model_dump())


@app.get("/models/{model_name}/checkpoints")
async def list_checkpoints(
    model_name: str,
    auth_context: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """List available checkpoints for a model"""
    supabase = await get_supabase_client()
    
    # Get the model
    result = await (
        supabase.table("models")
        .select("*")
        .eq("user_team_membership_id", auth_context.membership["id"])
        .eq("name", model_name)
        .single()
        .execute()
    )
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model = result.data
    model_id = model["id"]
    
    # TODO: This would need to query the Modal Volume to list actual checkpoints
    # For now, return metadata stored in the model
    checkpoints_info = {
        "model_id": model_id,
        "checkpoint_path": f"/checkpoints/{model_id}",
        "latest_checkpoint": model.get("metadata", {}).get("latest_checkpoint"),
        "checkpoints": model.get("metadata", {}).get("checkpoints", []),
        "message": "To list actual checkpoints from Modal Volume, integrate Modal SDK or use vLLM endpoint"
    }
    
    return checkpoints_info


@app.post("/models/{model_name}/checkpoints/{checkpoint_name}/load")
async def load_checkpoint(
    model_name: str,
    checkpoint_name: str,
    auth_context: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """Load a specific checkpoint into the vLLM server"""
    supabase = await get_supabase_client()
    
    # Get the model
    result = await (
        supabase.table("models")
        .select("*")
        .eq("user_team_membership_id", auth_context.membership["id"])
        .eq("name", model_name)
        .single()
        .execute()
    )
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model = result.data
    model = await check_update_status(supabase, ModelInfo(**model))
    vllm_url = model.vllm_url
    
    if not vllm_url:
        raise HTTPException(
            status_code=400,
            detail=f"vLLM server not deployed for model '{model_name}'"
        )
    
    # Construct the checkpoint path
    checkpoint_path = f"/checkpoints/{model.id}/{checkpoint_name}"
    
    # Call vLLM to load the adapter
    # This would use the vLLM API to dynamically load the LoRA adapter
    # For now, return a placeholder response
    # TODO: Implement this
    return {
        "status": ModelStatus.READY,
        "model_name": model_name,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": checkpoint_path,
        "vllm_url": vllm_url,
        "message": "In production, this would call vLLM's adapter loading API"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
