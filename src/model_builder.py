from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, BartForConditionalGeneration

import logging
import json
from abc import ABC, abstractmethod
from utils import update_kwargs
from typing import Optional

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

    def __init__(self, model_id, model_type, tokenizer=None):
        self.model_id = model_id
        self.model_type = model_type

        self.tokenizer = tokenizer
        self.base_model = None

    @property
    def base_model(self):
        if self._base_model is None:
            self._base_model = self.create_base_model()
        return self._base_model

    @base_model.setter
    def base_model(self, bm):
        self._base_model = bm

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
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        base_model.resize_token_embeddings(len(self.tokenizer))
        return base_model
    
    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.base_model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


class T5ModelBuilder(ModelBuilder):
    def __init__(self, model_id, tokenizer):
        super().__init__(model_id, model_type=AutoModelForSeq2SeqLM, tokenizer=tokenizer)


class BARTModelBuilder(ModelBuilder):
    def __init__(self, model_id, tokenizer):
        super().__init__(model_id, model_type=AutoModelForSeq2SeqLM, tokenizer=tokenizer)


class GPT2ModelBuilder(ModelBuilder):
    def __init__(self, model_id, tokenizer):
        super().__init__(model_id, model_type=AutoModelForCausalLM, tokenizer=tokenizer)
