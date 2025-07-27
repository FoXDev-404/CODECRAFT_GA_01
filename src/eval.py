import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def calculate_perplexity(model, tokenizer, text, device):
    encodings = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True
    ).to(device)
    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask

    with torch.no_grad():
        outputs = model(
            input_ids, attention_mask=attention_mask, labels=input_ids
        )
        loss = outputs.loss

    perplexity = torch.exp(loss)
    return perplexity.item()


def calculate_ngram_diversity(text, n=2):
    tokens = text.split()
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i : i + n]))

    if not ngrams:
        return 0.0

    return len(set(ngrams)) / len(ngrams)


def evaluate_model(model_path, tokenizer_path, eval_text_path, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    with open(eval_text_path, "r", encoding="utf-8") as f:
        eval_text = f.read()

    print(f"Perplexity: {len(eval_text.split())} tokens...")
    perplexity = calculate_perplexity(model, tokenizer, eval_text, device)
    print(f"Perplexity: {perplexity:.2f}")

    print("Calculating 2-gram diversity...")
    diversity = calculate_ngram_diversity(eval_text, n=2)
    print(f"2-gram Diversity: {diversity:.4f}")

    return {"perplexity": perplexity, "2-gram_diversity": diversity}


if __name__ == "__main__":
    # Example usage:
    # model_path = "path/to/your/fine_tuned_model"
    # tokenizer_path = "path/to/your/fine_tuned_tokenizer"
    # eval_text_path = "path/to/your/evaluation_corpus.txt"
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # results = evaluate_model(model_path, tokenizer_path, eval_text_path, device)
    # print(results)
    pass
