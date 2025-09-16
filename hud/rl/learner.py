"""GRPO learner for vision-language models."""
from __future__ import annotations

import logging
import os
from typing import Any

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False

from hud.rl.distributed import (
    get_local_rank,
    get_world_size,
    is_main_process,
)
from hud.rl.utils import get_gpu_utilization, get_memory_usage, prepare_inputs, entropy_from_logits, batch_training_samples
from hud.utils.hud_console import HUDConsole
from contextlib import nullcontext
from .config import Config
from .types import TrainingMetrics, TrainingSample

logger = logging.getLogger(__name__)
hud_console = HUDConsole(logger)

class GRPOLearner:
    """GRPO learning algorithm for VLMs."""
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.local_rank = get_local_rank()
        self.world_size = get_world_size()
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        
        # Load models and processor
        self.processor, self.policy, self.ref, self.optimizer = self._load_models()
        self.metrics: list[TrainingMetrics] = []
    
    def _load_models(self) -> tuple[Any, Any, Any, Any]:
        """Load policy, reference models and optimizer."""
        model_cfg = self.config.model
        
        # Apply Liger kernel optimizations if available and enabled
        if model_cfg.use_liger and LIGER_AVAILABLE:
            hud_console.info_log("Applying Liger kernel optimizations to Qwen2.5-VL")
            apply_liger_kernel_to_qwen2_5_vl(
                rope=True,  # Optimized RoPE
                rms_norm=True,  # Optimized RMSNorm
                swiglu=True,  # Optimized SwiGLU
                fused_linear_cross_entropy=True  # Fused Linear+CrossEntropy for memory efficiency
            )
        elif model_cfg.use_liger and not LIGER_AVAILABLE:
            hud_console.warning_log("Liger kernel requested but not installed. Install with: pip install liger-kernel")
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_cfg.base_model,
            min_pixels=model_cfg.min_pixels,
            max_pixels=model_cfg.max_pixels
        )
        
        # Load policy model with LoRA
        # Use attention implementation from config
        attn_implementation = model_cfg.attn_implementation
        try:
            policy = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_cfg.base_model,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
            )
            hud_console.info_log(f"Using {attn_implementation} for attention")
        except (ImportError, ValueError) as e:
            # Only fallback if explicitly using flash_attention_2 and it's not available
            if attn_implementation == "flash_attention_2":
                hud_console.info_log(f"Flash Attention 2 not available ({e}), using eager attention")
                policy = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_cfg.base_model,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                )
            else:
                raise  # Re-raise if it's a different error
        
        # Move model to device
        policy = policy.to(self.device)
        # Enable gradient checkpointing for memory efficiency
        if model_cfg.gradient_checkpointing:
            policy.gradient_checkpointing_enable()
            hud_console.info_log("Gradient checkpointing enabled for memory efficiency")
        
        # Add LoRA adapters
        lora_config = LoraConfig(
            r=model_cfg.lora_r,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            task_type="CAUSAL_LM",
            bias="none",
            target_modules=list(model_cfg.target_modules)
        )
        policy = get_peft_model(policy, lora_config)
        
        # Wrap with DDP if in distributed mode
        if self.world_size > 1:
            policy = DDP(policy, device_ids=[self.local_rank], output_device=self.local_rank)
            hud_console.info_log(f"[DDP] Wrapped model on rank {self.local_rank}")
        
        # Create optimizer - need to access underlying model if DDP
        base_model = policy.module if hasattr(policy, "module") else policy
        trainable_params = [p for _, p in base_model.named_parameters() if p.requires_grad]
        
        # Use 8-bit optimizer if configured
        if self.config.training.use_8bit_optimizer:
            hud_console.info("Using 8-bit AdamW optimizer from bitsandbytes")
            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=self.config.training.lr,
                betas=self.config.training.adam_betas,
                eps=self.config.training.adam_eps
            )
        else:
            hud_console.info("Using standard FP32 AdamW optimizer")
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.training.lr,
                betas=self.config.training.adam_betas,
                eps=self.config.training.adam_eps
            )
        
        # Log optimizer info
        hud_console.info_log(f"Optimizer: {type(optimizer).__name__}")
        num_params = sum(p.numel() for p in trainable_params)
        hud_console.info_log(f"Number of trainable parameters: {num_params:,}")
        
        return processor, policy, None, optimizer

    def compute_logprobs(self, model: Any, inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute masked per-token log probabilities via the model.

        Uses assistant_masks to select tokens to score and returns a 1D tensor
        of log-probs for those positions.
        """
        out = model(**inputs)

        logits = out.logits / self.config.actor.temperature  # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)

        entropy = entropy_from_logits(logits)

        return log_probs, entropy

    def prepare_groups(self, samples: list[TrainingSample],) -> list[list[TrainingSample]]:
        """Prepare groups of samples for training."""
        # Prepare inputs with messages
        batch = []
        for sample in samples:
            inputs = prepare_inputs(sample, self.processor)
            new_sample = TrainingSample(**sample.model_dump())
            new_sample.inputs = inputs
            batch.append(new_sample)

        hud_console.info_log(f"[update] Processing batch of {len(batch)} traces")

        # Precompute logprobs
        with torch.no_grad():
            for sample in batch:
                sample = sample.to_device(self.device)
                sample.old_logprobs = self.compute_logprobs(
                    self.policy,
                    sample.inputs,
                )
            policy_module = self.policy.module if hasattr(self.policy, "module") else self.policy
            with policy_module.disable_adapter():
                for sample in batch:
                    sample.ref_logprobs = self.compute_logprobs(
                        self.policy,
                        sample.inputs,
                    )
                    sample.to_device(torch.device("cpu"))
        
        # Find minibatches and group them via batch_training_samples
        # Minibatches control the size of the forward pass to the model
        mb_size = self.config.training.mini_batch_size
        samples_minibatched = [batch_training_samples(batch[i:i+mb_size]) for i in range(0, len(batch), mb_size)]

        # Convert to grouped batches (if updating the model after each task group)
        if self.config.training.update_after_group:
            return [samples_minibatched[i:i+self.config.training.group_size] for i in range(0, len(samples_minibatched), self.config.training.group_size)]
        else:
            return [samples_minibatched]
            
    def update(self, samples: list[TrainingSample]) -> TrainingMetrics:
        """Perform a gradient update on a batch."""
        import time
        training_start_time = time.time()

        # Always create metrics for synchronization
        self.metrics.append(TrainingMetrics())
        metrics = self.metrics[-1]
        
        # Prepare groups for GRPO training
        groups = self.prepare_groups(samples)
        
        # Update over mini batch size
        with hud_console.progress("Gradient update...") as progress:
            for epoch in range(self.config.training.epochs):
                for group_idx, group in enumerate(groups):
                    progress.update(f"Training epoch {epoch+1}/{self.config.training.epochs}")
                        
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated_loss = 0.0
                    
                    grad_accum_steps = len(group)
                    update_after_minibatch = self.config.training.update_after_minibatch and len(group) > 1
                    for s_idx, sample_minibatch in enumerate(group):
                        is_last = s_idx == len(group) - 1 and not update_after_minibatch
                        ddp_ctx = nullcontext() if (is_last or self.world_size == 1) else self.policy.no_sync()

                        local_has = (sample_minibatch.advantage.abs().sum() > 0)
                        flag = torch.tensor(int(local_has), device=self.device)
                        if torch.distributed.is_initialized():
                            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.SUM)
                        global_has = flag.item() > 0

                        if update_after_minibatch:
                            self.optimizer.zero_grad(set_to_none=True)

                        with ddp_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            if global_has:
                                if local_has:
                                    loss = self.compute_loss(sample_minibatch) / grad_accum_steps
                                    loss.backward()
                                else:
                                    # Dummy backward that touches all params, produces zero grads, triggers hooks
                                    dummy = sum(p.sum() for p in self.policy.parameters()) * 0.0
                                    dummy.backward()
                            else:
                                # Everyone does a synchronized zero backward; no step after accumulation
                                dummy = sum(p.sum() for p in self.policy.parameters()) * 0.0
                                dummy.backward()

                            metrics.update({
                                "gpu_util": get_gpu_utilization(),  # Track peak utilization
                                "gpu_memory": get_memory_usage(),  # Track memory usage
                            })
                            progress.update(f"GPU Util: {get_gpu_utilization():.1f}% | Memory: {get_memory_usage():.2f} GB")

                        if update_after_minibatch:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.training.grad_clip, error_if_nonfinite=True)
                            self.optimizer.step()

                    if not global_has:
                        # Skip step if no updates are needed
                        continue

                    hud_console.info_log(f"Gradient update completed: {group_idx} group, final loss: {accumulated_loss:.4f}")
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.training.grad_clip, error_if_nonfinite=True)
                    self.optimizer.step()

                    metrics.update({
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                    })
                
                hud_console.info_log(f"Gradient update completed: {group_idx} group, final loss: {accumulated_loss:.4f}")
                
        
        # Log summary after progress completes
        hud_console.info_log(f"Gradient update completed: {group_idx} group, final loss: {accumulated_loss:.4f}")
        
        # Calculate training time and throughput
        training_time = time.time() - training_start_time
        total_samples = len(groups) * self.config.training.group_size * self.config.training.mini_batch_size
        samples_per_second = total_samples / training_time if training_time > 0 else 0.0
        
        metrics.update({
            "training_time": training_time,
            "samples_per_second": samples_per_second,
        })
        
        return metrics
    
    def compute_loss(self, sample: TrainingSample) -> torch.Tensor:
        """Compute GRPO loss for a batch of samples."""
        training_cfg = self.config.training
        metrics = self.metrics[-1] if len(self.metrics) > 0 else TrainingMetrics()
                    
        sample.to_device(self.device)

        pol_logp, pol_entropy = self.compute_logprobs(
            self.policy,
            sample.inputs,
        )
        old_logp = sample.old_logprobs
        ref_logp = sample.ref_logprobs

        # Aggregate per trace or per token
        if self.config.training.ppo_mode == "per_trace":
            pol_logp = pol_logp.mean(dim=1)
            pol_entropy = pol_entropy.mean(dim=1)
            old_logp = old_logp.mean(dim=1)
            ref_logp = ref_logp.mean(dim=1)
        
        # Clip log probability differences
        log_ratio = pol_logp - old_logp
        ratio_tok = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))

        unclipped = ratio_tok * sample.advantage
        clipped = torch.clamp(
            ratio_tok,
            1 - training_cfg.top_eps,
            1 + training_cfg.bottom_eps
        ) * sample.advantage
        
        policy_term = -torch.minimum(unclipped, clipped)
        
        # Clip log probability differences in KL
        log_rho = pol_logp - ref_logp
        rho_tok = torch.exp(log_rho.clamp(min=-20.0, max=20.0))
        kl_approx = (rho_tok - torch.log(rho_tok) - 1)

        total_loss = policy_term + training_cfg.kl_beta * kl_approx + training_cfg.entropy_beta * pol_entropy

        # Aggregate via normalizing by the number of tokens or allowing sum
        # This is a no-op for per_trace mode
        total_loss = total_loss.mean() if training_cfg.token_agg == "mean" else total_loss.sum()

        metrics.update({
            "policy_ratio": ratio_tok.detach().mean().item(),
            "kl": kl_approx.mean().item(),
            "entropy": pol_entropy.mean().item(),
            "tokens": sample.inputs["input_ids"].numel(),
            "loss": total_loss.item(),
        })

        sample.to_device(torch.device("cpu"))
        
        return total_loss
    
    def save(self, path: str) -> None:
        """Save the current policy checkpoint (only on rank 0)."""
        if is_main_process():
            os.makedirs(path, exist_ok=True)
            # Unwrap DDP model if needed
            model_to_save = self.policy.module if hasattr(self.policy, "module") else self.policy
            model_to_save.save_pretrained(path)
            hud_console.info(f"[Learner] Saved checkpoint to {path}")
    
    def load(self, path: str) -> None:
        """Load a policy checkpoint."""
        # Would need to reload LoRA weights
        logger.info(f"[Learner] Loading checkpoint from {path}")
        # Implementation depends on PEFT version
