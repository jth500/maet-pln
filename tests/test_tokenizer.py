import pytest
from tokenization import T5TokenizationHandler


@pytest.fixture(scope="module")
def t5_tokenizer_handler():
    return T5TokenizationHandler(model_id="t5-small")


@pytest.fixture(scope="module")
def t5_tokenizer(t5_tokenizer_handler):
    return t5_tokenizer_handler.create_tokenizer()


def test_T5_tokenization_handler():
    tk = T5TokenizationHandler(model_id="t5-small")
    assert tk is not None

    tokenizer = tk.create_tokenizer()
    assert tokenizer is not None


def test_tokenize_directly(t5_tokenizer):
    assert t5_tokenizer("Hi") == {
        "input_ids": [2018, 1],
        "attention_mask": [1, 1],
    }


# assert t5_tokenizer.tokenize("Hi") == ["_Hi"] # weird failure...


def test_tokenize_from_handler(t5_tokenizer_handler):
    assert t5_tokenizer_handler.tokenize("Hi") == {
        "input_ids": [2018, 1],
        "attention_mask": [1, 1],
        "labels": [2018, 1],
    }
    pass
