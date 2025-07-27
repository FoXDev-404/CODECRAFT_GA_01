#!/bin/bash

echo "Starting GPT-2 training..."

python -m src.train_gpt2 --config configs/default.yaml

echo "Training finished."