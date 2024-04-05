from transformers import AutoTokenizer
from abc import ABC, abstractmethod
import logging
from utils import update_kwargs

logger = logging.getLogger(__name__)


class TokenizationHandler(ABC):
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

    @property
    @abstractmethod
    def defaults(self) -> dict:
        # add default values for each model
        pass

    def create_tokenizer(self, **kwargs):
        kwargs = update_kwargs(kwargs, self.defaults)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        tokenizer.padding_side = "right"  # BART is a model with absolute position embeddings; pad the inputs on the right
        self.tokenizer = tokenizer
        return tokenizer


class GPT2TokenizationHandler(TokenizationHandler):
    def __init__(self, model_id="gpt2-medium"):
        super().__init__(model_id)

    @property
    def defaults(self):
        return dict(
            eos_token="<|endoftext|>",
            bos_token="<|startoftext|>",
            pad_token="<|pad|>",
            model_max_length=1024,
        )


class BARTTokenizationHandler(TokenizationHandler):
    def __init__(self, model_id="facebook/bart-large"):
        super().__init__(model_id)

    @property
    def defaults(self):
        return dict(
            eos_token="</s>",
            bos_token="<s>",
            pad_token="<pad>",
            model_max_length=1024,
        )


class T5TokenizationHandler(TokenizationHandler):
    def __init__(self, model_id="t5-base"):
        super().__init__(model_id)

    @property
    def defaults(self):
        return dict(truncation=True)
