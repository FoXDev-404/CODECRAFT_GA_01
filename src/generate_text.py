import torch
import argparse
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_text(
    model_path,
    prompt,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
):
    """
    Generates text using a fine-tuned GPT-2 model.

    Args:
        model_path (str): Path to the fine-tuned model directory.
        prompt (str): The initial text prompt for generation.
        max_length (int): The maximum length of the generated text.
        num_return_sequences (int): The number of sequences to generate.
        temperature (float): The value used to modulate the next token
                             probabilities.
        top_k (int): The number of highest probability vocabulary tokens to
                     keep for top-k-filtering.
        top_p (float): If set to float < 1, only the most probable tokens with
                       probabilities that add up to `top_p` are kept for
                       generation.
    """
    print(f"Loading model and tokenizer from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print(
            "Please ensure the model path is correct and contains "
            "tokenizer_config.json, vocab.json, merges.txt, and "
        )
        print("pytorch_model.bin.")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    print(f"Generating text with prompt: '{prompt}'...")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    print("\nGenerated Text(s):")
    for i, sample_output in enumerate(output):
        decoded_text = tokenizer.decode(
            sample_output, skip_special_tokens=True
        )
        print(f"--- Generated Sequence {i+1}:")
        print(decoded_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using a fine-tuned GPT-2 model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Path to the directory containing the fine-tuned "
            "model (e.g., output/gpt2_finetuned/final_model)"
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        help=(
            "The initial text prompt for generation. If not "
            "provided, will read from stdin."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum length of the generated text.",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help=("The value used to modulate the next token " "probabilities."),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help=(
            "The number of highest probability vocabulary "
            "tokens to keep for top-k-filtering."
        ),
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help=(
            "If set to float < 1, only the most probable "
            "tokens with probabilities that add up to `top_p` "
            "are kept for generation."
        ),
    )

    args = parser.parse_args()

    if args.prompt is None:
        print("Enter prompt (Ctrl+D to finish):")
        args.prompt = sys.stdin.read().strip()
        if not args.prompt:
            print("Error: No prompt provided.")
            sys.exit(1)

    generate_text(
        model_path=args.model_path,
        prompt=args.prompt,
        max_length=args.max_length,
        num_return_sequences=args.num_return_sequences,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
