import pytest
from tokenization import (
    GPT2TokenizationHandler,
    T5TokenizationHandler,
    BARTTokenizationHandler,
)
import transformers


def test_initialise_gpt_tokenization():
    tk = GPT2TokenizationHandler()
    tokenizer = tk.create_tokenizer()
    assert isinstance(
        tokenizer, transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast
    )
    assert tokenizer.eos_token == "<|endoftext|>"
    assert tokenizer.bos_token == "<|startoftext|>"


def test_initialise_BART_tokenization():
    tk = BARTTokenizationHandler()
    tokenizer = tk.create_tokenizer()
    assert isinstance(
        tokenizer, transformers.models.bart.tokenization_bart_fast.BartTokenizerFast
    )
    assert tokenizer.eos_token == "</s>"
    assert tokenizer.bos_token == "<s>"


def test_initialise_t5_tokenization():
    tk = T5TokenizationHandler()
    tokenizer = tk.create_tokenizer()
    assert isinstance(
        tokenizer, transformers.models.t5.tokenization_t5_fast.T5TokenizerFast
    )
