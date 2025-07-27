# CODECRAFT_GA_01 - Text Generation with GPT-2

## Project Overview

CODECRAFT_GA_01 is a machine learning project designed to fine-tune GPT-2 on a custom dataset for coherent and contextually relevant text generation. The project is structured to be beginner-friendly yet production-ready, featuring training scripts, text generation tools, and evaluation metrics.

## Key Features

- Fine-tuning GPT-2 (small) with Hugging Face Transformers.
- Support for **full fine-tuning** and **LoRA/PEFT** for low-VRAM environments (e.g., NVIDIA 940MX).
- Ready-to-run **Google Colab notebook** (`notebooks/Colab_Train_GPT2.ipynb`) for free GPU training.
- Modular, clean code with logging (rich + tqdm progress bars).
- Inference scripts supporting temperature, top_k, top_p sampling.
- Lightweight evaluation (perplexity + n-gram diversity).
- Compatible with **CPU and CUDA** (auto-detection).

## Repository Structure

```
CODECRAFT_GA_01/
├── README.md
├── requirements.txt
├── setup.sh
├── configs/
│   ├── default.yaml
│   └── lora.yaml
├── data/
│   └── sample_corpus.txt
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── data_preprocessing.py
│   ├── train_gpt2.py
│   ├── generate_text.py
│   └── eval.py
├── notebooks/
│   └── Colab_Train_GPT2.ipynb
└── scripts/
    ├── run_train.sh
    ├── run_train_lora.sh
    ├── run_eval.sh
    └── run_generate.sh
```

## Tech Stack

- **Language:** Python 3.9+
- **Libraries:** torch, transformers, datasets, peft, accelerate, tqdm, rich, pyyaml, evaluate
- **Environment:** Google Colab (primary training), local inference support.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/FoXDev-404/CODECRAFT_GA_01.git
    cd CODECRAFT_GA_01
    ```

2.  **Set up the Python environment:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Dataset Setup:**
    Place your custom text corpus in the `data/` directory. By default, the project uses `data/sample_corpus.txt`. Ensure your dataset is a plain text file.

## Training

The project supports full fine-tuning and LoRA/PEFT for efficient training.

### Full Fine-tuning

To perform full fine-tuning of GPT-2, use the `run_train.sh` script with the `default.yaml` configuration.

```bash
./scripts/run_train.sh
```

### LoRA/PEFT Fine-tuning

For low-VRAM environments, LoRA (Low-Rank Adaptation) is supported. Use the `run_train_lora.sh` script with the `lora.yaml` configuration.

```bash
./scripts/run_train_lora.sh
```

## Inference

After training, you can generate text using the `generate_text.py` script.

```bash
python src/generate_text.py --model_path "path/to/your/fine_tuned_model" --tokenizer_path "path/to/your/fine_tuned_tokenizer" --prompt "Your starting prompt here." --max_length 100 --temperature 0.7 --top_k 50 --top_p 0.95
```

You can also use the `run_generate.sh` script for a quick test:

```bash
./scripts/run_generate.sh
```

## Evaluation

Evaluate your fine-tuned model using `eval.py`. This script calculates perplexity and n-gram diversity.

```bash
python src/eval.py --model_path "path/to/your/fine_tuned_model" --tokenizer_path "path/to/your/fine_tuned_tokenizer" --eval_text_path "data/sample_corpus.txt"
```

You can also use the `run_eval.sh` script:

```bash
./scripts/run_eval.sh
```

## Google Colab Usage

A ready-to-run Google Colab notebook (`notebooks/Colab_Train_GPT2.ipynb`) is provided for free GPU training.

1.  **Open the notebook:** Navigate to `notebooks/Colab_Train_GPT2.ipynb` in Google Colab.
2.  **Upload your dataset:**
    ```python
    from google.colab import files
    uploaded = files.upload()
    # This will prompt you to upload your dataset file (e.g., your_corpus.txt)
    ```
    Move the uploaded file to the `data/` directory within the Colab environment.
3.  **Run the cells:** Follow the instructions in the notebook to install dependencies, prepare data, and train your model.

## Sample Outputs

*(Placeholder: After fine-tuning, you can add examples of generated text here to showcase the model's capabilities.)*

```
# Example of generated text:
# Prompt: "The quick brown fox"
# Output: "The quick brown fox jumps over the lazy dog, and then proceeds to..."
```
