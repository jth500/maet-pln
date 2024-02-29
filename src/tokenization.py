from transformers import AutoTokenizer
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class TokenizationHandler(ABC):
    def __init__(self, model_id):
        self.model_id = model_id
        self.tokenizer = None

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
        return AutoTokenizer.from_pretrained(self.model_id, **kwargs)

    def create_tokenizer(self, **kwargs):
        # TODO: This stuff will probably be model specific
        tokenizer = self._create_tokenizer(**kwargs)

        # config stuff
        # NB: This was originally after Data Cell
        tokenizer.pad_token = tokenizer.eos_token  # end-of-sequence (eos) padding
        self.base_model.resize_token_embeddings(
            len(tokenizer)
        )  # resize to embeddings to match updated tokenizer
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )  # set id of padding token to be id of eos token
        self.base_model.config.end_token_id = tokenizer.eos_token_id
        self.base_model.config.pad_token_id = self.base_model.config.eos_token_id
        self.tokenizer = tokenizer
        return tokenizer

    def tokenize(self, prompt):
        """
        Tokenizes the given prompt using the tokenizer.

        Args:
            prompt (str): The prompt to be tokenized.

        Returns:
            dict: A dictionary containing the tokenized prompt and labels.
        """
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result


class T5TokenizationHandler(TokenizationHandler):
    def __init__(self):
        super().__init__(model_id="t5-large")
        pass
