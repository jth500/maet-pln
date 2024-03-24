from transformers import AutoTokenizer
from abc import ABC, abstractmethod
import logging
from utils import update_kwargs

logger = logging.getLogger(__name__)


class TokenizationHandler:
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise ValueError(
                "Tokenizer has not been created. Run TokenizationHandler.create_tokenizer()"
            )
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tk):
        self._tokenizer = tk

    def _create_tokenizer(self, **kwargs):
        defaults = dict(
            eos_token="</s>",
            bos_token="<s>",
            pad_token="<pad>",
            model_max_length=1024,
            )
        kwargs = update_kwargs(kwargs, defaults)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        return self.tokenizer
    
    def create_tokenizer(self, **kwargs):
        try:
            tokenizer = self.tokenizer
        except ValueError:
            tokenizer = self._create_tokenizer(**kwargs)
        tokenizer.padding_side = "right" # BART is a model with absolute position embeddings; pad the inputs on the right
        self.tokenizer = tokenizer
        return tokenizer


class EncoderDecoderTokenizationHandler(TokenizationHandler):
    def __init__(self):
        super().__init__(model_id="google-t5/t5-base")

    def _create_tokenizer(self, **kwargs):
        defaults = dict(
            eos_token="</s>",
            bos_token="<s>",
            pad_token="<pad>",
            model_max_length=1024,
            )
        kwargs = update_kwargs(kwargs, defaults)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        return self.tokenizer


class DecoderTokenizationHandler(TokenizationHandler):
    def __init__(self):
        super().__init__(model_id="gpt2-medium")

    def _create_tokenizer(self, **kwargs):
        defaults = dict(
            eos_token="<|endoftext|>",
            bos_token="<|startoftext|>",
            pad_token="<|pad|>",
            model_max_length=1024
            )
        kwargs = update_kwargs(kwargs, defaults)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        return self.tokenizer
    
    def create_tokenizer(self, **kwargs):
        try:
            tokenizer = self.tokenizer
        except ValueError:
            tokenizer = self._create_tokenizer(**kwargs)
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        return tokenizer
    

# if __name__ == "__main__":
