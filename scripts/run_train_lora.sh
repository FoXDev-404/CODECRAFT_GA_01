#!/bin/bash

# Script to run GPT-2 fine-tuning with LoRA/PEFT

# Ensure the virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source .venv/Scripts/activate || { echo "Failed to activate virtual environment. Please ensure .venv exists and is correctly set up."; exit 1; }
fi

# Define paths
CONFIG_PATH="configs/lora.yaml"

# Check if the config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: LoRA config file not found at $CONFIG_PATH"
    exit 1
fi

echo "Starting LoRA fine-tuning with config: $CONFIG_PATH"

# Run the training script with the LoRA config
# Using `accelerate launch` for distributed training support
accelerate launch -m src.train_gpt2 --config "$CONFIG_PATH"

# Check the exit status of the accelerate command
if [ $? -eq 0 ]; then
    echo "LoRA fine-tuning completed successfully."
else
    echo "LoRA fine-tuning failed. Please check the logs for errors."
fi
