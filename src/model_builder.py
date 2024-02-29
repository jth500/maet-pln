from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

import logging
import json

logger = logging.getLogger("model_builder")


class ModelBuilder:
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
    create_tokenizer(self, max_length=512, truncation=True, padding=True)
        Creates a tokenizer for the model.
    _tokenize(self, prompt)
        Tokenizes a given prompt.
    generate_and_tokenize_prompt(self, data_point, tokenizer)
        Generates and tokenizes a prompt from a given data point.
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
            self._tokenizer = self.create_tokenizer()
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tk):
        self._tokenizer = tk
    
    def create_base_model(self):
        """
        Creates and returns the base model.

        Returns:
            The base model.
        """
        logger.info("Creating base model")
        base_model = self.model_type.from_pretrained(
            self.model_id,
            use_cache=False,
            # load_in_8bit=True,
            device_map="auto",
        )
        return base_model

    def create_tokenizer(self, max_length=512, truncation=True, padding=True):
        """
        Creates a tokenizer for the model.

        Args:
            max_length (int, optional): The maximum length of the input sequences. Defaults to 512.
            truncation (bool, optional): Whether to truncate the input sequences to `max_length`. Defaults to True.
            padding (bool, optional): Whether to pad the input sequences to `max_length`. Defaults to True.

        Returns:
            tokenizer: The created tokenizer.
        """
        tk = AutoTokenizer.from_pretrained(
            self.model_id,
            model_max_length=max_length,
            truncation=truncation,
            padding=padding,
        )
        
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
