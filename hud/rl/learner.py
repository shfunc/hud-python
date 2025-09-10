"""GRPO learner for vision-language models."""

import os
import logging
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

from .config import Config
from .types import TrainingSample, Batch

logger = logging.getLogger(__name__)

def log_memory_usage(name: str, prev_mem: float | None = None):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem = torch.cuda.memory_allocated() / 1024**3
        if prev_mem is not None:
            logger.debug(f"{name}: {mem:.2f} GB (+{mem - prev_mem:.2f} GB)")
        else:
            logger.debug(f"{name}: {mem:.2f} GB")

class GRPOLearner:
    """GRPO learning algorithm for VLMs."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models and processor
        self.processor, self.policy, self.ref, self.optimizer = self._load_models()
        self.step = 0
        self.last_metrics = {}
        self.last_loss = 0.0
    
    def _load_models(self):
        """Load policy, reference models and optimizer."""
        model_cfg = self.config.model
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_cfg.base_model,
            min_pixels=model_cfg.min_pixels,
            max_pixels=model_cfg.max_pixels
        )
        
        # Load policy model with LoRA
        # Try to use Flash Attention 2 if available
        attn_implementation = "flash_attention_2"
        try:
            policy = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_cfg.base_model,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
            )
            logger.info(f"Using {attn_implementation} for attention")
        except (ImportError, ValueError) as e:
            # Fallback to default attention if Flash Attention is not available
            logger.info(f"Flash Attention 2 not available ({e}), using default attention")
            policy = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_cfg.base_model,
                torch_dtype=torch.bfloat16,
            )
        
        # Move model to device
        policy = policy.to(self.device)
        # Enable gradient checkpointing for memory efficiency
        policy.gradient_checkpointing_enable()
        
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
        
        # Create optimizer
        trainable_params = [p for _, p in policy.named_parameters() if p.requires_grad]
        
        if self.config.training.use_8bit_optimizer and HAS_BITSANDBYTES:
            logger.info("Using 8-bit AdamW optimizer from bitsandbytes")
            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=self.config.training.lr,
                betas=self.config.training.adam_betas,
                eps=self.config.training.adam_eps
            )
        else:
            if self.config.training.use_8bit_optimizer and not HAS_BITSANDBYTES:
                logger.warning("8-bit optimizer requested but bitsandbytes not available, using regular AdamW")
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.training.lr,
                betas=self.config.training.adam_betas,
                eps=self.config.training.adam_eps
            )
        
        # Log optimizer info
        logger.info(f"Optimizer: {type(optimizer).__name__}")
        num_params = sum(p.numel() for p in trainable_params)
        logger.info(f"Number of trainable parameters: {num_params:,}")
        
        return processor, policy, None, optimizer
    
    def compute_logprobs(self, model, inputs):
        """Compute per-token log probabilities via the model."""
        prev_mem = log_memory_usage("GPU Memory before forward")
        
        out = model(**inputs)
        
        log_memory_usage("GPU Memory after forward", prev_mem)

        logits = out.logits / self.config.actor.temperature
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs of actual generated tokens
        target_ids = inputs["input_ids"][:, inputs["logits_to_keep"] + 1]
        gathered = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        
        return gathered
    
    def compute_loss(self, samples: List[TrainingSample]) -> torch.Tensor:
        """Compute GRPO loss for a batch of samples."""
        training_cfg = self.config.training
        
        policy_terms = []
        kl_terms = []
        ratios = []
        advantages = []
        clipped_fractions = []
        
        for sample in samples:
            # Get current policy log probs
            pol_logp = self.compute_logprobs(
                self.policy,
                sample.inputs,
            )
            
            # Compute ratio
            ratio_tok = torch.exp(pol_logp - sample.old_logprobs)
            
            # Token aggregation
            if training_cfg.token_agg == "mean":
                ratio = ratio_tok.mean()
            else:
                ratio = ratio_tok.sum()
            
            # Track metrics
            ratios.append(ratio.detach())
            logger.debug(f"Ratios: {ratios[-1]}")
            advantages.append(sample.advantage.detach())
            logger.debug(f"Advantages: {advantages[-1]}")
            
            # Clipped objective
            unclipped = ratio * sample.advantage
            clipped = torch.clamp(
                ratio,
                1 - training_cfg.clip_eps,
                1 + training_cfg.clip_eps
            ) * sample.advantage
            
            # Track if clipping occurred
            clipped_fractions.append((ratio < 1 - training_cfg.clip_eps) | (ratio > 1 + training_cfg.clip_eps))
            logger.debug(f"Clipped: {clipped_fractions[-1]}")
            
            policy_term = -torch.minimum(unclipped, clipped)
            policy_terms.append(policy_term)
            
            # KL penalty vs reference
            rho_tok = torch.exp(pol_logp - sample.ref_logprobs)
            kl_approx = (rho_tok - torch.log(rho_tok) - 1).mean()
            logger.debug(f"KL: {kl_approx}")
            kl_terms.append(kl_approx)
        
        # Combine losses
        policy_loss = torch.stack(policy_terms).mean()
        kl_loss = torch.stack(kl_terms).mean()
        
        # Store metrics for logging
        self.last_metrics = {
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": (policy_loss + training_cfg.kl_beta * kl_loss).item(),
            "ratios": torch.stack(ratios).float().cpu().numpy(),  # Convert to float32 for numpy
            "advantages": torch.tensor(advantages).float().cpu().numpy(),  # Convert to float32 for numpy
            "clipped_fraction": torch.stack(clipped_fractions).float().mean().item(),
        }
        
        return policy_loss + training_cfg.kl_beta * kl_loss
    
    def update(self, batch: Batch):
        """Perform a gradient update on a batch."""
        if not batch.samples:
            return
        
        # Compute advantages
        rewards = torch.tensor(batch.rewards, device=self.device)
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        
        if std_reward < 1e-6:
            logger.warning("Standard deviation of rewards is too small, skipping update")
            return
        
        # Normalize advantages
        for sample, reward in zip(batch.samples, rewards):
            sample.advantage = ((reward - mean_reward) / (std_reward + 1e-6)) * sample.weight

        with torch.no_grad():
            for sample in batch.samples:
                sample.old_logprobs = self.compute_logprobs(
                    self.policy,
                    sample.inputs,
                )
            with self.policy.disable_adapter():
                for sample in batch.samples:
                    sample.ref_logprobs = self.compute_logprobs(
                    self.policy,
                        sample.inputs,
                    )

        # Training epochs
        for epoch in range(self.config.training.epochs):
            # Split samples for gradient accumulation
            mini_batch_size = self.config.training.mini_batch_size
            if mini_batch_size == 0:
                mini_batch_size = len(batch.samples)
                grad_accum_steps = 1
            else:
                grad_accum_steps = len(batch.samples) // mini_batch_size
            # Zero gradients at start of epoch
            self.optimizer.zero_grad()
            accumulated_loss = 0.0
            
            for accum_step in range(grad_accum_steps):
                # Get samples for this accumulation step
                start_idx = accum_step * mini_batch_size
                end_idx = min(start_idx + mini_batch_size, len(batch.samples))
                if start_idx >= len(batch.samples):
                    break
                    
                step_samples = batch.samples[start_idx:end_idx]
                
                prev_mem = log_memory_usage("GPU Memory before compute")
                    
                # Compute loss (scaled by accumulation steps)
                loss = self.compute_loss(step_samples) / grad_accum_steps
                accumulated_loss += loss.item() * grad_accum_steps

                log_memory_usage("GPU Memory after compute", prev_mem)
                
                # Backward pass
                loss.backward()
            
            # Gradient clipping and optimizer step after all accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            
            logger.info(f"[Learner] Step {self.step}, Epoch {epoch}, Loss: {accumulated_loss:.4f} (grad_accum={grad_accum_steps})")
            
            # Store loss and grad norm for external logging
            self.last_loss = accumulated_loss
            self.last_metrics['grad_norm'] = float(grad_norm)
        
        self.step += 1
    
    def save(self, path: str):
        """Save the current policy checkpoint."""
        os.makedirs(path, exist_ok=True)
        self.policy.save_pretrained(path)
        logger.info(f"[Learner] Saved checkpoint to {path}")
    
    def load(self, path: str):
        """Load a policy checkpoint."""
        # Would need to reload LoRA weights
        logger.info(f"[Learner] Loading checkpoint from {path}")
        # Implementation depends on PEFT version