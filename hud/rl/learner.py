"""GRPO learner for vision-language and text models."""
from __future__ import annotations

import logging
import os
from typing import Any

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoProcessor, 
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
)

try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl, apply_liger_kernel_to_qwen2_5
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
    """GRPO learning algorithm for Vision-Language Models (VLMs) and Text Models."""
    
    def __init__(self, config: Config) -> None:
        self.config = config
        self.local_rank = get_local_rank()
        self.world_size = get_world_size()
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        
        # Detect model type
        self.is_vl_model = "VL" in config.model.base_model
        
        # Load models and processor
        self.processor, self.policy, self.ref, self.optimizer = self._load_models()
        self.metrics: list[TrainingMetrics] = []
    
    def _load_models(self) -> tuple[Any, Any, Any, Any]:
        """Load policy, reference models and optimizer."""
        model_cfg = self.config.model
        
        # Detect if this is a VL model or standard text model
        is_vl_model = "VL" in model_cfg.base_model
        model_type = "Vision-Language" if is_vl_model else "Text"
        hud_console.info_log(f"Loading {model_type} model: {model_cfg.base_model}")
        
        # Apply Liger kernel optimizations if available and enabled
        if model_cfg.use_liger and LIGER_AVAILABLE:
            if is_vl_model:
                hud_console.info_log("Applying Liger kernel optimizations to Qwen2.5-VL")
                apply_liger_kernel_to_qwen2_5_vl(
                    rope=True,  # Optimized RoPE
                    rms_norm=True,  # Optimized RMSNorm
                    swiglu=True,  # Optimized SwiGLU
                    fused_linear_cross_entropy=True  # Fused Linear+CrossEntropy for memory efficiency
                )
            else:
                hud_console.info_log("Applying Liger kernel optimizations to Qwen2.5")
                apply_liger_kernel_to_qwen2_5(
                    rope=True,  # Optimized RoPE
                    rms_norm=True,  # Optimized RMSNorm
                    swiglu=True,  # Optimized SwiGLU
                    fused_linear_cross_entropy=True  # Fused Linear+CrossEntropy for memory efficiency
                )
        elif model_cfg.use_liger and not LIGER_AVAILABLE:
            hud_console.warning_log("Liger kernel requested but not installed. Install with: pip install liger-kernel")
        
        # Load processor/tokenizer based on model type
        if is_vl_model:
            processor = AutoProcessor.from_pretrained(
                model_cfg.base_model,
                min_pixels=model_cfg.min_pixels,
                max_pixels=model_cfg.max_pixels
            )
        else:
            processor = AutoTokenizer.from_pretrained(model_cfg.base_model)
        
        # Load policy model with LoRA
        # Use attention implementation from config
        attn_implementation = model_cfg.attn_implementation
        
        # Choose the appropriate model class
        model_class = Qwen2_5_VLForConditionalGeneration if is_vl_model else AutoModelForCausalLM
        
        try:
            policy = model_class.from_pretrained(
                model_cfg.base_model,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
            )
            hud_console.info_log(f"Using {attn_implementation} for attention")
        except (ImportError, ValueError) as e:
            # Only fallback if explicitly using flash_attention_2 and it's not available
            if attn_implementation == "flash_attention_2":
                hud_console.info_log(f"Flash Attention 2 not available ({e}), using eager attention")
                policy = model_class.from_pretrained(
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

    def prepare_groups(self, samples: list[TrainingSample],) -> list[list[TrainingSample]]:
        """Prepare groups of samples for training."""
        # Prepare inputs with messages
        batch = []
        for sample in samples:
            inputs = prepare_inputs(sample, self.processor)
            # Create new sample, preserving tensor fields that don't serialize well
            new_sample = TrainingSample(**sample.model_dump())
            new_sample.inputs = inputs
            new_sample.advantage = sample.advantage
            batch.append(new_sample)

        hud_console.info_log(f"[update] Processing batch of {len(batch)} traces")

        # Precompute logprobs
        with torch.no_grad():
            for sample in batch:
                sample = sample.to_device(self.device)
                sample.old_logprobs, _ = self.compute_logprobs(
                    self.policy,
                    sample.inputs,
                )
                sample.to_device(torch.device("cpu"))
                
            policy_module = self.policy.module if hasattr(self.policy, "module") else self.policy
            with policy_module.disable_adapter():
                for sample in batch:
                    sample = sample.to_device(self.device)
                    sample.ref_logprobs, _ = self.compute_logprobs(
                        self.policy,
                        sample.inputs,
                    )
                    sample.to_device(torch.device("cpu"))
        
        # Find minibatches and group them via batch_training_samples
        # Minibatches control the batch size of the forward pass to the model
        mb_size = self.config.training.mini_batch_size
        adj_group_size = self.config.training.group_size // mb_size
        samples_minibatched = []
        for i in range(0, len(batch), mb_size):
            samples_minibatched.extend(batch_training_samples(batch[i:i+mb_size]))

        for sample in samples_minibatched:
            sample.to_device(torch.device("cpu"))

        # Convert to grouped batches (if updating the model after each task group)
        if self.config.training.update_after_group:
            return [samples_minibatched[i:i+adj_group_size] for i in range(0, len(samples_minibatched), adj_group_size)]
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
        hud_console.info_log(f"Updating over {len(groups)} groups")
        
        # Update over mini batch size
        with hud_console.progress("Gradient update...") as progress:
            for epoch in range(self.config.training.epochs):
                progress.update(f"Training epoch {epoch+1}/{self.config.training.epochs}")
                for group_idx, group in enumerate(groups):
                        
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated_loss = 0.0
                    
                    grad_accum_steps = len(group)
                    update_after_minibatch = self.config.training.update_after_minibatch and len(group) > 1
                    skip_this_group = False
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
                            try:
                                if global_has:
                                    if local_has:
                                        loss = self.compute_loss(sample_minibatch) / grad_accum_steps
                                        loss.backward()
                                    else:
                                        # Dummy backward that touches all params, produces zero grads, triggers hooks
                                        hud_console.info_log(f"Dummy backward for {sample_minibatch.inputs['input_ids'].numel()} tokens")
                                        dummy = sum(p.sum() for p in self.policy.parameters()) * 0.0
                                        dummy.backward()
                                else:
                                    # Everyone does a synchronized zero backward; no step after accumulation
                                    hud_console.info_log(f"Dummy backward for {sample_minibatch.inputs['input_ids'].numel()} tokens")
                                    dummy = sum(p.sum() for p in self.policy.parameters()) * 0.0
                                    dummy.backward()
                                hud_console.info_log(f"GPU Backward: {get_gpu_utilization():.1f}% | Memory: {get_memory_usage():.2f} GB")
                            except torch.cuda.OutOfMemoryError:
                                hud_console.warning_log("CUDA OOM on rank %s; skipping minibatch", str(self.local_rank))
                                torch.cuda.empty_cache()
                                skip_flag = torch.tensor(1, device=self.device)
                                if torch.distributed.is_initialized():
                                    torch.distributed.all_reduce(skip_flag, op=torch.distributed.ReduceOp.MAX)
                                # Dummy backward to keep DDP happy
                                dummy = sum(p.sum() for p in self.policy.parameters()) * 0.0
                                dummy.backward()
                                skip_this_group = True
                                continue

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        if update_after_minibatch and not skip_this_group:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.training.grad_clip, error_if_nonfinite=True)
                            self.optimizer.step()

                    if not global_has:
                        # Skip step if no updates are needed
                        continue

                    if skip_this_group:
                        continue
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.training.grad_clip, error_if_nonfinite=True)
                    self.optimizer.step()

                    metrics.update({
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                    })
                
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

        sanity_check(sample, pol_logp, sample.old_logprobs, sample.ref_logprobs)

        metrics.update({
            "gpu_util": get_gpu_utilization(),  # Track peak utilization
            "gpu_memory": get_memory_usage(),  # Track memory usage
        })
        hud_console.info_log(f"GPU Util: {get_gpu_utilization():.1f}% | Memory: {get_memory_usage():.2f} GB")

        old_logp = sample.old_logprobs
        ref_logp = sample.ref_logprobs

        # Use assistant mask to remove non-assistant tokens
        m = sample.inputs["assistant_mask"]

        # Aggregate per trace or per token
        if training_cfg.ppo_mode == "per_trace":
            counts = m.sum(dim=1).clamp_min(1.0)
            pol_logp = (pol_logp * m.float()).sum(dim=1) / counts
            pol_entropy = (pol_entropy * m.float()).sum(dim=1) / counts
            old_logp = (old_logp * m.float()).sum(dim=1) / counts
            ref_logp = (ref_logp * m.float()).sum(dim=1) / counts
        
        # Clip log probability differences
        log_ratio = torch.where(m, pol_logp - old_logp, torch.zeros_like(pol_logp))
        ratio_tok = torch.exp(log_ratio.clamp(-20.0, 20.0))

        # Ensure advantage shape matches ratio_tok for broadcasting
        advantage = sample.advantage.view(-1, 1) if ratio_tok.dim() == 2 else sample.advantage.squeeze(-1)
        
        unclipped = ratio_tok * advantage
        clipped = torch.clamp(
            ratio_tok,
            1 - training_cfg.top_eps,
            1 + training_cfg.bottom_eps
        ) * advantage
        
        policy_term = -torch.minimum(unclipped, clipped)
        
        # Clip log probability differences in KL
        log_rho = torch.where(m, pol_logp - ref_logp, torch.zeros_like(pol_logp))
        rho_tok = torch.exp(log_rho.clamp(-20.0, 20.0))
        kl_approx = (rho_tok - torch.log(rho_tok) - 1)

        total_loss = policy_term + training_cfg.kl_beta * kl_approx + training_cfg.entropy_beta * pol_entropy

        # Aggregate loss
        if training_cfg.ppo_mode == "per_trace":
            total_loss = total_loss.mean() if training_cfg.token_agg == "mean" else total_loss.sum()
        else:
            if training_cfg.token_agg == "mean":
                total_loss = (total_loss * m).sum() / m.sum().clamp_min(1.0)
            else:
                total_loss = (total_loss * m).sum()

        metrics.update({
            "policy_ratio": ratio_tok.detach().mean().item(),
            "kl": kl_approx.mean().item(),
            "entropy": pol_entropy.mean().item(),
            "tokens": sample.inputs["input_ids"].numel(),
            "loss": total_loss.item(),
        })

        sample.to_device(torch.device("cpu"))
        
        return total_loss

    def compute_logprobs(self, model: Any, inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute masked per-token log probabilities via the model.

        Returns log probabilities for the actual next tokens.
        """
        model_inputs = {k: v for k, v in inputs.items() if k != "assistant_mask"}
        out = model(**model_inputs)

        logits = out.logits / self.config.actor.temperature 
        log_probs = F.log_softmax(logits, dim=-1)

        targets = inputs["input_ids"][:, 1:]
        token_log_probs = log_probs[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        
        # Compute entropy only for assistant tokens to save memory
        assistant_mask = inputs["assistant_mask"]
        entropy = torch.zeros_like(token_log_probs)
        entropy[assistant_mask] = entropy_from_logits(logits[:, :-1][assistant_mask])

        return token_log_probs, entropy
    
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


def sanity_check(sample: TrainingSample, pol_logp: torch.Tensor, old_logp: torch.Tensor, ref_logp: torch.Tensor) -> None:
    m = sample.inputs["assistant_mask"]
    with torch.no_grad():
        B, K = pol_logp.shape
        assert old_logp.shape == (B, K), "old_logp shape mismatch"
        assert ref_logp.shape == (B, K), "ref_logp shape mismatch"
        assert m.shape == (B, K), "assistant_mask shape mismatch"

        # Check mask is subset of attention_mask[:, 1:]
        att = sample.inputs.get("attention_mask", None)
        if att is not None and att.dim() == 2:
            att_shift = att[:, 1:].bool()
            bad = (m & ~att_shift).sum().item()
            if bad > 0:
                hud_console.warning_log("assistant_mask overlaps padding: %s tokens", str(bad))

        # Finiteness on masked entries only
        def _stats(name: str, t: torch.Tensor) -> None:
            sel = t[m]
            if sel.numel() == 0:
                hud_console.info_log("%s empty under mask", name)
                return
            finite = torch.isfinite(sel)
            if finite.sum() < sel.numel():
                hud_console.warning_log("%s non-finite: %s/%s", name, str((~finite).sum().item()), str(sel.numel()))
            sel = sel[finite].float()

        _stats("pol_logp", pol_logp)
        _stats("old_logp", old_logp)
        _stats("ref_logp", ref_logp)

        # Log-probabilities should be <= 0 (log-softmax)
        if (pol_logp[m] > 1e-6).any():
            hud_console.warning_log("pol_logp has positive values under mask")

        # Precompute masked deltas and ratios for diagnostics (before exp)
        masked_log_ratio = torch.zeros_like(pol_logp)
        masked_log_ratio[m] = (pol_logp - old_logp)[m]
        masked_log_rho = torch.zeros_like(pol_logp)
        masked_log_rho[m] = (pol_logp - ref_logp)[m]

        _stats("log_ratio(masked)", masked_log_ratio)
        _stats("log_rho(masked)", masked_log_rho)

        # Ratios after clamp (diagnostic only)
        ratio_diag = torch.zeros_like(pol_logp)
        rho_diag = torch.zeros_like(pol_logp)
        ratio_diag[m] = torch.exp(masked_log_ratio[m].clamp(-20.0, 20.0))
        rho_diag[m] = torch.exp(masked_log_rho[m].clamp(-20.0, 20.0))
        _stats("ratio_tok(masked)", ratio_diag)
        _stats("rho_tok(masked)", rho_diag)