from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

import logging
import json
from abc import ABC, abstractmethod
from utils import update_kwargs

logger = logging.getLogger(__name__)


class ModelBuilder(ABC):
    """
    A class used to build and manage a machine learning model and its tokenizer.

    ...

    Attributes
    ----------
    model_id : str
        a formatted string to print out the model id
    model_type : transformers.PreTrainedModel
        the type of the model
    tokenizer : transformers.PreTrainedTokenizer
        the tokenizer of the model
    base_model : transformers.PreTrainedModel
        the base model

    Methods
    -------
    init(self, model_id, model_type)
        Initializes the ModelBuilder with a model id and model type.
    create_base_model(self)
        Creates and returns the base model.
    """

    def __init__(self, model_id, model_type):
        self.model_id = model_id
        self.model_type = model_type

        self.tokenizer = None
        self.base_model = None

    @property
    def base_model(self):
        if self._base_model is None:
            self._base_model = self.create_base_model()
        return self._base_model

    @base_model.setter
    def base_model(self, bm):
        self._base_model = bm

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise ValueError("Tokenizer has not been created")
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tk):
        self._tokenizer = tk

    def create_base_model(self, **kwargs):
        """
        Creates and returns the base model.

        Returns:
            The base model.
        """
        defaults = dict(
            use_cache=False,
            # load_in_8bit=True,
            device_map="auto",
        )
        kwargs = update_kwargs(kwargs, defaults)
        logger.info("Creating base model")
        base_model = self.model_type.from_pretrained(self.model_id, **kwargs)
        return base_model


class T5ModelBuilder(ModelBuilder):
    def __init__(self):
        super().__init__(model_id="t5-large", model_type=AutoModelForSeq2SeqLM)
