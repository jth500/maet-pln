import pytest
from pathlib import Path


from data_handler.data_handler import (
    T5DatasetHandler,
    GPT2DatasetHandler,
    BARTDatasetHandler,
)
from tokenization import (
    T5TokenizationHandler,
    GPT2TokenizationHandler,
    BARTTokenizationHandler,
)
import pytest


@pytest.fixture(scope="module")
def tokenizer():
    return T5TokenizationHandler(model_id="t5-small").create_tokenizer()


DATASET_NAME = "EdinburghNLP/xsum"


def test_gpt2_dataset_handler():
    tk = GPT2TokenizationHandler()
    tokenizer = tk.create_tokenizer()
    data_handler = GPT2DatasetHandler(
        DATASET_NAME, tokenizer, data_size=100, push_to_hub=False
    )
    data_handler.data_to_json()
    data = data_handler.process_data()
    assert len(data) == 2
    assert len(data[0]) == 79
    assert list(data[0].features.keys()) == [
        "input",
        "output",
        "input_ids",
        "attention_mask",
        "labels",
    ]


def test_gpt2_dataset_handler_no_tokenizer():
    data_handler = GPT2DatasetHandler(DATASET_NAME, data_size=100, push_to_hub=False)
    data_handler.data_to_json()
    data = data_handler.process_data()
    assert len(data) == 2
    assert len(data[0]) == 79
    assert list(data[0].features.keys()) == [
        "input",
        "output",
        "input_ids",
        "attention_mask",
        "labels",
    ]


def test_gpt2_dataset_handler_rlaif():
    tk = GPT2TokenizationHandler()
    tokenizer = tk.create_tokenizer()
    data_handler = GPT2DatasetHandler(
        DATASET_NAME, tokenizer, rlaif=True, data_size=100, push_to_hub=False
    )
    data_handler.data_to_json()
    data = data_handler.process_data()
    assert len(data) == 3
    assert len(data[0]) == 17
    assert list(data[0].features.keys()) == [
        "input",
        "output",
        "input_ids",
        "attention_mask",
        "labels",
    ]


def test_t5_dataset_handler():
    tk = T5TokenizationHandler()
    tokenizer = tk.create_tokenizer()
    data_handler = T5DatasetHandler(
        DATASET_NAME, tokenizer, data_size=100, push_to_hub=False
    )
    data_handler.data_to_json()
    data = data_handler.process_data()
    assert len(data[0]) == 53
    assert list(data[0].features.keys()) == [
        "input",
        "output",
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "labels",
    ]


def test_t5_dataset_handler_no_tokenizer():
    data_handler = T5DatasetHandler(DATASET_NAME, data_size=100, push_to_hub=False)
    data_handler.data_to_json()
    data = data_handler.process_data()
    assert len(data[0]) == 53
    assert list(data[0].features.keys()) == [
        "input",
        "output",
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "labels",
    ]


def test_BART_dataset_handler():
    tk = BARTTokenizationHandler()
    tokenizer = tk.create_tokenizer()
    data_handler = BARTDatasetHandler(
        DATASET_NAME, tokenizer, data_size=100, push_to_hub=False
    )
    data_handler.data_to_json()
    data = data_handler.process_data()
    assert len(data) == 2
    assert len(data[0]) == 80
    assert list(data[0].features.keys()) == [
        "input",
        "output",
        "input_ids",
        "attention_mask",
        "labels",
    ]


def test_BART_dataset_handler_no_tokenizer():
    data_handler = BARTDatasetHandler(DATASET_NAME, data_size=100, push_to_hub=False)
    data_handler.data_to_json()
    data = data_handler.process_data()
    assert len(data) == 2
    assert len(data[0]) == 80
    assert list(data[0].features.keys()) == [
        "input",
        "output",
        "input_ids",
        "attention_mask",
        "labels",
    ]


# def test_t5_dataset_handler(tokenizer):
#     data_handler = T5DatasetHandler(DATASET_NAME, tokenizer)
#     assert data_handler is not None
#     assert data_handler.data_dir == "data_json"


# def test_dataset_handler_raises_error_no_tokenizer():
#     with pytest.raises(TypeError):
#         T5DatasetHandler(DATASET_NAME)


# def test_gen_tokenize_prompt(tokenizer):
#     data_handler = T5DatasetHandler(DATASET_NAME, tokenizer)
#     prompt = "Hello there"
#     tokenized_prompt = data_handler.generate_and_tokenize_prompt(
#         {"input": prompt, "output": prompt}
#     )
#     assert tokenized_prompt == {
#         "input_ids": [
#             148,
#             33,
#             46,
#             2205,
#             16,
#             1499,
#             4505,
#             1635,
#             1707,
#             5,
#             148,
#             33,
#             787,
#             8,
#             423,
#             1499,
#             5,
#             21425,
#             613,
#             19,
#             12,
#             4505,
#             17289,
#             7,
#             15,
#             8,
#             1499,
#             38,
#             22874,
#             120,
#             11,
#             12700,
#             38,
#             487,
#             5,
#             1713,
#             30345,
#             86,
#             2562,
#             10,
#             8774,
#             132,
#             1713,
#             30345,
#             16361,
#             10,
#             8774,
#             132,
#             1,
#         ],
#         "attention_mask": [
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#         ],
#     }
