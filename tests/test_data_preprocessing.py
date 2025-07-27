import os
import pytest
from src.data_preprocessing import load_and_preprocess_data
from transformers import AutoTokenizer, PreTrainedTokenizerFast

@pytest.fixture(scope="module")
def dummy_corpus_path(tmp_path_factory):
    """Creates a dummy text file for testing."""
    path = tmp_path_factory.mktemp("data").joinpath("dummy_corpus.txt")
    with open(path, "w") as f:
        f.write(
            "This is a test sentence. This is another test sentence. "
            "And one more for good measure."
        )
    return str(path)

def test_load_and_preprocess_data(dummy_corpus_path):
    """Tests the load_and_preprocess_data function."""
    dataset, tokenizer = load_and_preprocess_data(dummy_corpus_path, block_size=16)

    assert dataset is not None
    assert isinstance(tokenizer, (AutoTokenizer, PreTrainedTokenizerFast))
    assert "input_ids" in dataset.column_names
    assert "attention_mask" in dataset.column_names
    assert "labels" in dataset.column_names
    assert len(dataset) > 0

def test_file_not_found():
    """Tests that FileNotFoundError is raised for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_and_preprocess_data("non_existent_file.txt")
