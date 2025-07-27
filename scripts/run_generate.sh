#!/bin/bash

echo "Starting text generation..."

# Default values (can be overridden by command line arguments)
MODEL_PATH="output/gpt2_finetuned/final_model"
PROMPT="The quick brown fox"
MAX_LENGTH=100
NUM_RETURN_SEQUENCES=3

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    key="$1"
    case $key in
        --model_path)
        MODEL_PATH="$2"
        shift # past argument
        shift # past value
        ;;
        --prompt)
        PROMPT="$2"
        shift # past argument
        shift # past value
        ;;
        --max_length)
        MAX_LENGTH="$2"
        shift # past argument
        shift # past value
        ;;
        --num_return_sequences)
        NUM_RETURN_SEQUENCES="$2"
        shift # past argument
        shift # past value
        ;;
        *)
        echo "Unknown option: $key"
        exit 1
        ;;
    esac
done

echo "${PROMPT}" | python src/generate_text.py     --model_path "${MODEL_PATH}"     --max_length ${MAX_LENGTH}     --num_return_sequences ${NUM_RETURN_SEQUENCES}

echo "Text generation finished."