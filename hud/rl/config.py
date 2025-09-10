"""Configuration for RL training."""

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class ModelConfig:
    """Model and LoRA configuration."""
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj", "vision_language_merger.*"
    )
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 256 * 28 * 28


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # GRPO parameters
    group_size: int = 6
    mini_batch_size: int = 2
    clip_eps: float = 0.2
    kl_beta: float = 1e-3
    lr: float = 1e-4
    epochs: int = 2
    token_agg: str = "mean"
    use_8bit_optimizer: bool = True
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    
    # Turn weighting
    turn_weighting: str = "last"  # "last" | "last_k" | "all_discounted"
    last_k: int = 3
    gamma: float = 0.9
    
    # Reward shaping
    step_penalty: float = 0.0
    format_penalty: float = 0.0
    
    # Training schedule
    episodes_per_batch: int = 16
    save_every_batches: int = 1
    max_training_steps: int = 1000
    max_total_episodes: int = 10000


@dataclass
class ActorConfig:
    """Actor/episode collection configuration."""
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "token-abc123"
    max_steps_per_episode: int = 100
    parallel_episodes: int = 16  # Increased from 4
    temperature: float = 0.7
    max_tokens: int = 2048  # Increased to allow complete tool calls
    system_prompt: str = "You are an expert agent. Complete the task efficiently."
    tasks_file: str = "browser_2048_tasks.jsonl"


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    
    # Paths
    out_dir: str = "./checkpoints"
    adapter_prefix: str = "cua-grpo-step"
    
    # Misc
    seed: int = 1234
    debug: bool = True
    
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Create config from dictionary."""
        model = ModelConfig(**d.get("model", {}))
        training = TrainingConfig(**d.get("training", {}))
        actor = ActorConfig(**d.get("actor", {}))
        
        return cls(
            model=model,
            training=training,
            actor=actor,
            out_dir=d.get("out_dir", cls.out_dir),
            adapter_prefix=d.get("adapter_prefix", cls.adapter_prefix),
            seed=d.get("seed", cls.seed),
            debug=d.get("debug", cls.debug),
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "actor": self.actor.__dict__,
            "out_dir": self.out_dir,
            "adapter_prefix": self.adapter_prefix,
            "seed": self.seed,
            "debug": self.debug,
        }