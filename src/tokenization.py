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
    tk = T5TokenizationHandler(model_id="t5-small") #small ???
    tokenizer = tk.create_tokenizer()
    out = tk.tokenize("Hi")
    out

class BartTokenizationHandler(TokenizationHandler):
    def __init__(self, model_id="facebook/bart-large"):
        super().__init__(model_id)
    def tokenize(self, prompt, text_target=None, **kwargs):
        """
        Tokenizes the given text and optional target using the tokenizer.

        Args:
            prompt (str): The text to be tokenized.
            text_target (str, optional): The target text for given prompt i.e. summary. Used at finetuning step but not during inference.

        Returns:
            dict: A dictionary containing the tokenized text and labels.
        """
        defaults = {
            "truncation": True,
            "max_length": 512,
            "padding": False,
            "return_tensors": None,
        }
        kwargs = update_kwargs(kwargs, defaults)

        return self.tokenizer(prompt, text_target = text_target, **kwargs)