import torch
import yaml
import argparse
import os
from transformers import AutoModelForCausalLM
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from src.data_preprocessing import load_and_preprocess_data
from src.utils import print_trainable_parameters


def train_gpt2(config_path):
    """
    Fine-tunes a GPT-2 model based on the provided configuration.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    print(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    dataset_path = config["dataset_path"]
    output_dir = config["output_dir"]
    training_args_config = config["training_args"]
    peft_config_data = config.get("peft_config")

    # 1. Load and preprocess data
    print(f"Loading and preprocessing data from {dataset_path}")
    train_dataset, tokenizer = load_and_preprocess_data(
        dataset_path, tokenizer_name=model_name
    )

    # 2. Load model
    print(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Check for GPU availability
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        model.to("cuda")
    else:

        print("CUDA is not available. Using CPU.")

    # 3. Apply LoRA if configured
    if peft_config_data:
        print("Applying LoRA configuration...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=peft_config_data["r"],
            lora_alpha=peft_config_data["lora_alpha"],
            lora_dropout=peft_config_data["lora_dropout"],
            bias=peft_config_data["bias"],
        )
        model = get_peft_model(model, peft_config)
        print_trainable_parameters(model)
    else:
        print("Running full fine-tuning.")

    # 4. Set up TrainingArguments
    print("Setting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_args_config["num_train_epochs"],
        per_device_train_batch_size=training_args_config[
            "per_device_train_batch_size"
        ],
        gradient_accumulation_steps=training_args_config[
            "gradient_accumulation_steps"
        ],
        learning_rate=float(training_args_config["learning_rate"]),
        weight_decay=training_args_config["weight_decay"],
        save_steps=training_args_config["save_steps"],
        logging_steps=training_args_config["logging_steps"],
        save_total_limit=training_args_config["save_total_limit"],
        fp16=training_args_config["fp16"],
        push_to_hub=training_args_config["push_to_hub"],
        report_to="none",
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # 5. Initialize Trainer and train
    print("Initializing Trainer and starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")
    print("Attempting to save model...")
    # 6. Save the fine-tuned model
    final_model_output_path = os.path.join(output_dir, "final_model")
    print(f"Saving fine-tuned model to {final_model_output_path}")
    trainer.save_model(final_model_output_path)
    tokenizer.save_pretrained(final_model_output_path)
    print("Training complete and model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help=(
            "Path to the configuration YAML file (e.g., "
            "configs/default.yaml or configs/lora.yaml)"
        ),
    )
    args = parser.parse_args()
    train_gpt2(args.config)
