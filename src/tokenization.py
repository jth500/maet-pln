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

    def _create_tokenizer(self, **kwargs):
        """
        Creates a tokenizer for the model.

        Args:
            max_length (int, optional): The maximum length of the input sequences. Defaults to 512.
            truncation (bool, optional): Whether to truncate the input sequences to `max_length`. Defaults to True.
            padding (bool, optional): Whether to pad the input sequences to `max_length`. Defaults to True.

        Returns:
            tokenizer: The created tokenizer.
        """
        defaults = dict(model_max_length=512, truncation=True, padding=True)
        kwargs = update_kwargs(kwargs, defaults)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        return self.tokenizer

    def create_tokenizer(self, **kwargs):
        # TODO: This stuff will probably be model specific
        # try:
        #     tokenizer = self.tokenizer
        # except ValueError:
        tokenizer = self._create_tokenizer(**kwargs)

        # config stuff
        # NB: This was originally after Data Cell
        tokenizer.pad_token = tokenizer.eos_token  # end-of-sequence (eos) padding
        # self.base_model.resize_token_embeddings(
        #     len(tokenizer)
        # )  # resize to embeddings to match updated tokenizer
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )  # set id of padding token to be id of eos token
        # self.base_model.config.end_token_id = tokenizer.eos_token_id
        # self.base_model.config.pad_token_id = self.base_model.config.eos_token_id
        self.tokenizer = tokenizer
        return tokenizer

    def tokenize(self, prompt, **kwargs):
        """
        Tokenizes the given prompt using the tokenizer.

        Args:
            prompt (str): The prompt to be tokenized.

        Returns:
            dict: A dictionary containing the tokenized prompt and labels.
        """
        defaults = {
            "truncation": True,
            "max_length": 512,
            "padding": False,
            "return_tensors": None,
        }
        kwargs = update_kwargs(kwargs, defaults)
        result = self.tokenizer(prompt, **kwargs)
        result["labels"] = result["input_ids"].copy()
        return result


class T5TokenizationHandler(TokenizationHandler):
    def __init__(self, model_id="t5-base"):
        super().__init__(model_id)
        pass


if __name__ == "__main__":
    tk = T5TokenizationHandler(model_id="t5-small")
    tokenizer = tk.create_tokenizer()
    out = tk.tokenize("Hi")
    out
