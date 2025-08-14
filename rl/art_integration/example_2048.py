#!/usr/bin/env python
"""Example: Train a 2048 agent using HUD + ART."""

import asyncio
import logging
import os

from train_hud_art import train_hud_with_art

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def main():
    """Train a 2048 agent with ART."""
    
    print("=" * 60)
    print("HUD + ART: Training a 2048 Agent")
    print("=" * 60)
    
    # Train on the 2048 environment
    await train_hud_with_art(
        # Environment settings
        environment="2048",
        
        # Model settings
        base_model="Qwen/Qwen2.5-3B-Instruct",  # Small model for quick training
        model_name="2048-agent-demo",
        project_name="hud-art-demo",
        
        # Training data
        num_training_scenarios=8,  # Reduced for demo
        num_val_scenarios=4,
        
        # Training settings
        groups_per_step=2,  # Process 2 scenarios per step
        rollouts_per_group=3,  # 3 trajectories per scenario for comparison
        num_epochs=1,
        learning_rate=1e-5,
        
        # Agent settings
        max_turns=20,  # Max moves per game
        
        # Output
        output_dir="./.art-demo",
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nYour trained model is saved in: ./.art-demo/hud-art-demo/models/2048-agent-demo/")
    print("\nTo use the model:")
    print("1. Load with vLLM for inference")
    print("2. Continue training with more scenarios")
    print("3. Export to HuggingFace format")
    
    # Optional: Test the trained model
    print("\n" + "=" * 60)
    print("Testing Trained Model")
    print("=" * 60)
    
    from datasets import load_dataset
    from hud.datasets import to_taskconfigs
    import art
    
    # Load a test scenario from HuggingFace
    dataset = load_dataset("hud-evals/2048-taskset", split="train")
    test_scenarios = to_taskconfigs(dataset)
    test_scenario = test_scenarios[0]  # First task
    
    # Initialize model (it will load the latest checkpoint)
    trained_model = art.Model(
        name="2048-agent-demo",
        project="hud-art-demo",
        inference_base_url="http://localhost:8000/v1",  # Assuming vLLM server
        inference_api_key="dummy",
    )
    
    # Note: In practice, you'd need to start a vLLM server with the trained model
    # For now, we'll just show the setup
    
    print(f"\nTest scenario: {test_scenario.prompt}")
    print("(To actually test, start a vLLM server with the trained model)")


if __name__ == "__main__":
    # Run the training
    asyncio.run(main())