from transformers import AutoTokenizer
from abc import ABC, abstractmethod
import logging
from utils import update_kwargs

logger = logging.getLogger(__name__)


class TokenizationHandler(ABC):
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = None

    def _init_tokenizer(self, defaults={}, **kwargs):
        """
        Creates a tokenizer for the model.

        Args:
            max_length (int, optional): The maximum length of the input sequences. Defaults to 512.
            truncation (bool, optional): Whether to truncate the input sequences to `max_length`. Defaults to True.
            padding (bool, optional): Whether to pad the input sequences to `max_length`. Defaults to True.

        Returns:
            tokenizer: The created tokenizer.
        """
        kwargs = update_kwargs(kwargs, defaults)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        return self.tokenizer

    def _create_tokenizer(self, **kwargs):
        defaults = {}
        return self._init_tokenizer(defaults, **kwargs)

    def create_tokenizer(self, **kwargs):
        tokenizer = self._create_tokenizer(**kwargs)
        self.tokenizer = tokenizer
        return tokenizer


class T5TokenizationHandler(TokenizationHandler):
    # def __init__(self, model_id="t5-base"):
    def __init__(self, model_id="t5-small"):
        super().__init__(model_id)


class GPT2TokenizationHandler(TokenizationHandler):
    def __init__(self, model_id="gpt2-medium"):
        super().__init__(model_id)

    def _create_tokenizer(self, **kwargs):
        defaults = dict(
            eos_token="<|endoftext|>",
            bos_token="<|startoftext|>",
            pad_token="<|pad|>",
            model_max_length=2096,
            padding_side="right"
        )
        return self._init_tokenizer(defaults, **kwargs)

class BartTokenizationHandler(TokenizationHandler):
    def __init__(self, model_id="facebook/bart-base"):
        super().__init__(model_id)

    def _create_tokenizer(self, **kwargs):
        defaults = dict(
            model_max_length=2096,
            padding_side="right")
        return self._init_tokenizer(defaults, **kwargs)




if __name__ == "__main__":
    # tk = T5TokenizationHandler(model_id="t5-small")
    # tokenizer = tk.create_tokenizer()
    # out = tk.tokenize("Hi")
    # out

    tk = GPT2TokenizationHandler(model_id="gpt2")
    tk = tk.create_tokenizer(test_addition=100)
    pass

