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
from hud.rl.utils import get_gpu_utilization, get_memory_usage, prepare_inputs
from hud.utils.hud_console import HUDConsole

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
        
        # Apply vision-specific optimizations
        if hasattr(policy, "model") and hasattr(policy.model, "vision_tower"):
            vision_tower = policy.model.vision_tower
            
            # Freeze vision tower if configured
            if model_cfg.freeze_vision_tower:
                for param in vision_tower.parameters():
                    param.requires_grad = False
                hud_console.info_log("Vision tower frozen to save memory")
            
            # Enable gradient checkpointing for vision tower
            if model_cfg.vision_gradient_checkpointing and hasattr(vision_tower, "gradient_checkpointing_enable"):
                vision_tower.gradient_checkpointing_enable()
                hud_console.info_log("Vision tower gradient checkpointing enabled")
            
            # Set vision compute dtype
            if model_cfg.vision_compute_dtype == "float16":
                vision_tower = vision_tower.half()
            elif model_cfg.vision_compute_dtype == "bfloat16":
                vision_tower = vision_tower.to(torch.bfloat16)
            
            # Apply vision feature selection strategy
            if model_cfg.vision_feature_select_strategy == "cls_only":
                # This would need model-specific implementation
                hud_console.info_log("Vision feature selection: CLS token only (saves memory)")
        
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

    def compute_logprobs(self, model: Any, inputs: Any) -> torch.Tensor:
        """Compute per-token log probabilities via the model."""
        out = model(**inputs)

        logits = out.logits / self.config.actor.temperature
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs of actual generated tokens
        target_ids = inputs["input_ids"][:, inputs["logits_to_keep"] + 1]
        gathered = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        
        return gathered
    
    def update(self, samples: list[TrainingSample]) -> TrainingMetrics:
        """Perform a gradient update on a batch."""
        import time
        training_start_time = time.time()
        
        # Always create metrics for synchronization
        self.metrics.append(TrainingMetrics())
        metrics = self.metrics[-1]
        
        # Handle empty batch - still need to call backward for DDP sync
        if not samples or len(samples) == 0:
            if self.world_size > 1:
                hud_console.warning_log("Empty batch, performing dummy backward for DDP sync")
                # Perform a dummy forward/backward with the model to maintain DDP synchronization
                dummy_input_ids = torch.zeros((1, 1), dtype=torch.long, device=self.device)
                dummy_inputs = {"input_ids": dummy_input_ids}
                dummy_output = self.policy(**dummy_inputs)
                dummy_loss = dummy_output.logits.sum() * 0  # Zero loss but uses model
                dummy_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            return metrics
        
        # Prepare inputs with messages
        batch = []
        for sample in samples:
            inputs = prepare_inputs(sample, self.processor, self.policy, self.config)
            for inp in inputs:
                new_sample = TrainingSample(**sample.model_dump())
                new_sample.inputs = inp
                batch.append(new_sample)

        hud_console.info_log(f"[update] Processing batch of {len(batch)} traces")
        group_size = self.config.training.group_size
        if len(batch) % group_size != 0:
            hud_console.warning_log(f"Group size {group_size} does not divide batch size {len(batch)}")
            # Continue anyway with partial group
        
        groups = [batch[i:i+group_size] for i in range(0, len(batch), group_size)]
        
        # Initialize accumulated_loss to handle empty groups
        accumulated_loss = 0.0
        
        with hud_console.progress("Gradient update...") as progress:
            for i, samples in enumerate(groups):
                progress.update(f"Computing logprobs for {len(samples)} samples")
                hud_console.debug_log(f"Computing old and reference logprobs for {len(samples)} samples")
                with torch.no_grad():
                    for sample in samples:
                        sample.old_logprobs = self.compute_logprobs(
                            self.policy,
                            sample.inputs,
                        ).cpu()  # Move to CPU immediately
                    policy_module = self.policy.module if hasattr(self.policy, "module") else self.policy
                    with policy_module.disable_adapter():
                        for sample in samples:
                            sample.ref_logprobs = self.compute_logprobs(
                                self.policy,
                                sample.inputs,
                            ).cpu()  # Move to CPU immediately

                progress.update(f"Training group {i+1}/{len(groups)} for {self.config.training.epochs} epochs")
                for epoch in range(self.config.training.epochs):
                    mini_batch_size = min(self.config.training.mini_batch_size, len(samples))
                    grad_accum_steps = max(1, len(samples) // mini_batch_size)
                    
                    self.optimizer.zero_grad()
                    accumulated_loss = 0.0
                    
                    for accum_step in range(grad_accum_steps):
                        # Get samples for this accumulation step
                        start_idx = accum_step * mini_batch_size
                        end_idx = min(start_idx + mini_batch_size, len(samples))
                        if start_idx >= len(samples):
                            break
                            
                        step_samples = samples[start_idx:end_idx]
                        
                        # Track GPU utilization during compute
                        gpu_util_before = get_gpu_utilization()
                        
                        loss = self.compute_loss(step_samples) / grad_accum_steps
                        accumulated_loss += loss.item() * grad_accum_steps

                        gpu_util_after = get_gpu_utilization()
                        gpu_mem = get_memory_usage()
                        progress.update(f"GPU Util: {gpu_util_after:.1f}% | Memory: {gpu_mem:.2f} GB")
                        metrics.update({
                            "gpu_util": gpu_util_after,  # Track peak utilization
                            "gpu_memory": gpu_mem,  # Track memory usage
                        })
                        
                        loss.backward()
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.training.grad_clip)
                    self.optimizer.step()

                    metrics.update({
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                    })
                    
                    progress.update(f"Step {i}, Epoch {epoch}, Loss: {accumulated_loss:.4f}, GradNorm: {grad_norm:.4f} (grad_accum={grad_accum_steps})")
        
        # Log summary after progress completes
        hud_console.info_log(f"Gradient update completed: {len(groups)} groups, final loss: {accumulated_loss:.4f}")
        
        # Calculate training time and throughput
        training_time = time.time() - training_start_time
        total_samples = len(samples)
        samples_per_second = total_samples / training_time if training_time > 0 else 0.0
        
        metrics.update({
            "training_time": training_time,
            "samples_per_second": samples_per_second,
        })
        
        return metrics
    
    def compute_loss(self, samples: list[TrainingSample]) -> torch.Tensor:
        """Compute GRPO loss for a batch of samples."""
        training_cfg = self.config.training
        metrics = self.metrics[-1]

        policy_terms = []
        kl_terms = []
        
        for sample in samples:
            pol_logp = self.compute_logprobs(
                self.policy,
                sample.inputs,
            )
            
            # Clip log probability differences to prevent numerical explosion
            log_ratio = pol_logp - sample.old_logprobs.to(self.device)
            log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
            ratio_tok = torch.exp(log_ratio)
            ratio = ratio_tok.mean() if training_cfg.token_agg == "mean" else ratio_tok.sum()

            unclipped = ratio * sample.advantage
            clipped = torch.clamp(
                ratio,
                1 - training_cfg.top_eps,
                1 + training_cfg.bottom_eps
            ) * sample.advantage
            
            policy_term = torch.minimum(unclipped, clipped)
            policy_terms.append(policy_term)
            
            # Clip log probability differences to prevent numerical explosion in KL
            log_rho = pol_logp - sample.ref_logprobs.to(self.device)
            log_rho = torch.clamp(log_rho, min=-20.0, max=20.0)
            rho_tok = torch.exp(log_rho)
            kl_approx = (rho_tok - torch.log(rho_tok) - 1).mean()
            kl_terms.append(kl_approx)

            metrics.update({
                "policy_ratio": ratio.detach().mean().item(),
                "kl": kl_approx.item(),
                "tokens": sample.inputs["input_ids"].numel(),
            })
        
        # Combine losses
        policy_loss = torch.stack(policy_terms).mean()
        kl_loss = torch.stack(kl_terms).mean()
        
        total_loss = -policy_loss + training_cfg.kl_beta * kl_loss
        
        metrics.update({
            "loss": total_loss.item(),
        })
        
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
