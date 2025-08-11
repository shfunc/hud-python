import verifiers as vf
from hud_vf_gym import load_environment

vf_env = load_environment(
    taskset="hud-evals/2048-taskset",  # HuggingFace dataset
    config_path="./configs/2048.yaml",
    num_tasks=2
)

# Model configuration
model_name = "Qwen/Qwen2.5-3B"
run_name = "2048-grpo_" + model_name.split("/")[-1].lower()

# Load model and tokenizer
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Get default GRPO training arguments
training_args = vf.grpo_defaults(run_name=run_name)

# Adjust hyperparameters for 2048 game
training_args.per_device_train_batch_size = 2  # Both tasks in one batch
training_args.num_generations = 4  # 4 rollouts each = 8 total
training_args.gradient_accumulation_steps = 4
training_args.max_tokens = 512
training_args.max_seq_len = 512
training_args.learning_rate = 1e-6

# Training schedule
training_args.max_steps = 500
training_args.save_strategy = "steps"
training_args.save_steps = 50
training_args.logging_steps = 10

# Create GRPO trainer with LoRA
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
    peft_config=vf.lora_defaults(),
)

# Start training
print(f"Starting training for {run_name}")
print(f"Dataset size: {len(vf_env.dataset)}")
trainer.train()