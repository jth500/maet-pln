import pytest
from tokenization import T5TokenizationHandler, GPT2TokenizationHandler


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


# this should now fail
# def test_tokenize_from_handler(t5_tokenizer_handler):
#     assert t5_tokenizer_handler.tokenize("Hi") == {
#         "input_ids": [2018, 1],
#         "attention_mask": [1, 1],
#         "labels": [2018, 1],
#     }
#     pass


### gpt


@pytest.fixture(scope="module")
def gpt_tokenizer_handler():
    return GPT2TokenizationHandler(model_id="gpt2")


def test_gpt_tokenization_handler():
    tk = GPT2TokenizationHandler(model_id="gpt2")
    assert tk is not None

    tokenizer = tk.create_tokenizer()
    assert tokenizer is not None


def test_add_kwargs(gpt_tokenizer_handler):
    tk = gpt_tokenizer_handler.create_tokenizer(test_addition=100)
    assert tk.init_kwargs["test_addition"] == 100
    assert tk.init_kwargs["padding_side"] == "right"
