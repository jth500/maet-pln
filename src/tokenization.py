from transformers import AutoTokenizer
from abc import ABC, abstractmethod
import logging
from utils import update_kwargs

logger = logging.getLogger(__name__)


class TokenizationHandler(ABC):
    """
    A base class for tokenization handlers.

    Attributes:
        model_id (str): The ID of the model.
        tokenizer (AutoTokenizer): The tokenizer object.

    Methods:
        create_tokenizer: Creates and returns a tokenizer object.
    """

    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = None

    @property
    def tokenizer(self):
        """
        Get the tokenizer object.

        Raises:
            ValueError: If the tokenizer has not been created.

        Returns:
            AutoTokenizer: The tokenizer object.
        """
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
        """
        Get the default values for each model.

        Returns:
            dict: A dictionary of default values.
        """
        # add default values for each model
        pass

    def create_tokenizer(self, **kwargs):
        """
        Create and return a tokenizer object.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            AutoTokenizer: The created tokenizer object.
        """
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
