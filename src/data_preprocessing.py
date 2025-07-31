import os
from datasets import load_dataset
from transformers import AutoTokenizer


def load_and_preprocess_data(file_path, tokenizer_name="gpt2", block_size=128):
    """
    Loads a text dataset, tokenizes it, and prepares it for GPT-2 training.

    Args:
        file_path (str): Path to the text file.
        tokenizer_name (str): Name of the tokenizer to use (e.g., "gpt2").
        block_size (int): The maximum sequence length after tokenization.

    Returns:
        datasets.Dataset: Processed dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    print(f"Loading dataset from {file_path}...")
    raw_datasets = load_dataset("text", data_files={"train": file_path})

    print(f"Loading tokenizer {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=block_size
        )

    print("Tokenizing dataset...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    def group_texts(examples):
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [
                t[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print("Grouping texts into blocks...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    print(
        f"Dataset preprocessing complete. Number of samples: "
        f"{len(lm_datasets['train'])}"
    )
    return lm_datasets["train"], tokenizer


if __name__ == "__main__":
    # Create a dummy corpus for testing
    dummy_corpus_path = "data/dummy_corpus.txt"
    os.makedirs(os.path.dirname(dummy_corpus_path), exist_ok=True)
    with open(dummy_corpus_path, "w") as f:
        f.write(
            "This is a test sentence. This is another test sentence. "
            "And one more for good measure."
        )

    try:
        dataset, tokenizer = load_and_preprocess_data(dummy_corpus_path)
        print(dataset[0])
        print(tokenizer.decode(dataset[0]["input_ids"]))
    except FileNotFoundError as e:
        print(e)
    finally:
        if os.path.exists(dummy_corpus_path):
            os.remove(dummy_corpus_path)
