import os
import pytest
import yaml
from unittest.mock import patch, MagicMock
from src.train_gpt2 import train_gpt2

@pytest.fixture(scope="module")
def dummy_config_path(tmp_path_factory):
    """Creates a dummy configuration file for testing."""
    config_data = {
        "model_name": "gpt2",
        "dataset_path": "data/sample_corpus.txt",
        "output_dir": "output/test_model",
        "training_args": {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "save_steps": 500,
            "logging_steps": 50,
            "save_total_limit": 1,
            "fp16": False,
            "push_to_hub": False,
        },
        "peft_config": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",
        },
    }
    path = tmp_path_factory.mktemp("configs").joinpath("test_config.yaml")
    with open(path, "w") as f:
        yaml.dump(config_data, f)
    return str(path)

@patch('src.train_gpt2.load_and_preprocess_data')
@patch('src.train_gpt2.AutoModelForCausalLM.from_pretrained')
@patch('src.train_gpt2.Trainer')
@patch('src.train_gpt2.print_trainable_parameters')
def test_train_gpt2_lora(mock_print_trainable_parameters, mock_trainer, mock_from_pretrained, mock_load_and_preprocess_data, dummy_config_path):
    """Tests the train_gpt2 function with LoRA configuration."""
    # Mock return values
    mock_load_and_preprocess_data.return_value = (MagicMock(), MagicMock())
    mock_from_pretrained.return_value = MagicMock()
    mock_trainer.return_value = MagicMock()

    train_gpt2(dummy_config_path)

    # Assertions
    mock_load_and_preprocess_data.assert_called_once()
    mock_from_pretrained.assert_called_once()
    mock_print_trainable_parameters.assert_called_once()
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()
    mock_trainer.return_value.save_model.assert_called_once()

@patch('src.train_gpt2.load_and_preprocess_data')
@patch('src.train_gpt2.AutoModelForCausalLM.from_pretrained')
@patch('src.train_gpt2.Trainer')
def test_train_gpt2_full_finetuning(mock_trainer, mock_from_pretrained, mock_load_and_preprocess_data, tmp_path_factory):
    """Tests the train_gpt2 function with full fine-tuning (no LoRA)."""
    config_data = {
        "model_name": "gpt2",
        "dataset_path": "data/sample_corpus.txt",
        "output_dir": "output/test_model",
        "training_args": {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "save_steps": 500,
            "logging_steps": 50,
            "save_total_limit": 1,
            "fp16": False,
            "push_to_hub": False,
        },
    }
    path = tmp_path_factory.mktemp("configs").joinpath("test_config_full.yaml")
    with open(path, "w") as f:
        yaml.dump(config_data, f)

    # Mock return values
    mock_load_and_preprocess_data.return_value = (MagicMock(), MagicMock())
    mock_from_pretrained.return_value = MagicMock()
    mock_trainer.return_value = MagicMock()

    train_gpt2(str(path))

    # Assertions
    mock_load_and_preprocess_data.assert_called_once()
    mock_from_pretrained.assert_called_once()
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()
    mock_trainer.return_value.save_model.assert_called_once()
