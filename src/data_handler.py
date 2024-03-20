import json
import abc
from datasets import load_dataset
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

from utils import update_kwargs

class DatasetHandler(ABC):
    """
    A class used to handle the dataset used for training the model.

    ...

    Attributes
    ----------
    dataset_name : str
        a formatted string to print out the dataset name
    data_dir : str
        the directory where the data is stored
    tokenizer : transformers.PreTrainedTokenizer
        the tokenizer of the model

    Methods
    -------
    __init__(self, dataset_name, tokenizer, data_dir: str = "data_json")
        Initializes the DatasetHandler with a dataset name, tokenizer, and data directory.
    data_to_json(self)
        Converts the dataset to JSON format and writes it to a file.
    generate_prompt(input, output="")
        Generates a prompt from a given input and output.
    process_data(self, dataset_name)
        Processes the data by loading the dataset, converting it to JSON format, splitting it into training and validation sets, and tokenizing the prompts.
    """

    def __init__(self, dataset_name, tokenizer, data_dir: str = "data_json"):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    @property
    @abstractmethod
    def template(self):
        # implement this in the subclass
        # e..g return "You are an expert summary... {input} {output}"
        pass

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tk):
        assert tk is not None
        self._tokenizer = tk

    def _data_to_json(self, input_label="prompt", target_label="summary"):
        """
        Converts the dataset to JSON format and saves it to a file named 'data_json'.

        This method loads the dataset, iterates over the items in the dataset, and converts each item
        to a JSON object with 'input' and 'output' keys. The JSON objects are then written to a file
        named 'data_json' in the current directory.

        Note: The dataset is limited to the first 1000 examples from the 'train' split.

        Returns:
            None
        """
        dataset = load_dataset(self.dataset_name, split="train[:10000]")
        dataset_splits = {"train": dataset}

        for key, ds in dataset_splits.items():
            # overwriting instead of appending - is this what we want?
            with open("data_json", "w") as f:
                for item in ds:
                    newitem = {
                        "input": item[input_label],
                        "output": item[target_label],
                    }
                    f.write(json.dumps(newitem) + "\n")

    def generate_prompt(self, input, output=""):
        return self.template.format(input=input, output=output)
    
    def tokenize(self, prompt, **kwargs):
        """
        Tokenizes the given prompt using the tokenizer.

        Args:
            prompt (str): The prompt to be tokenized.

        Returns:
            dict: A dictionary containing the tokenized prompt and labels.
        """
        defaults = dict(
            truncation=True,
            max_length=1024, # gpt 2 specific; should we keep constant for comparatives?
            padding=False,
            return_tensors=None
            )
        kwargs = update_kwargs(kwargs, defaults)
        result = self.tokenizer(prompt, **kwargs)
        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self, data_point, output=True):
        """
        Generates a full prompt using the input and output from the given data point,
        and then tokenizes the full prompt using the provided tokenizer.

        Args:
            data_point (dict): A dictionary containing the input and output data.
            tokenizer: The tokenizer object used for tokenization.

        Returns:
            tokenized_full_prompt: The tokenized version of the full prompt.
        """
        if output == False:
            full_prompt = self.generate_prompt(
                data_point["input"],
            )
        else:
            full_prompt = self.generate_prompt(
                data_point["input"],
                data_point["output"]
            )
        tokenized_full_prompt = self.tokenize(full_prompt)

        return tokenized_full_prompt

    def process_data(self, input_label="prompt", target_label="summary"):
        """
        Process the data for training and validation.

        Returns:
            train_data (Dataset): Processed training data.
            val_data (Dataset): Processed validation data.
        """
        self._data_to_json(input_label, target_label)

        data = load_dataset("json", data_files="data_json")
        train_val = data["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
        sft_rlaif = train_val["train"].train_test_split(test_size=0.2, shuffle=True, seed=42) # split for sft and rlaif; rlaif does not need outputs

        sft_train_data = (
            sft_rlaif["test"]
            .shuffle(seed=42)
            .map(lambda x: self.generate_and_tokenize_prompt(x))
        )
        rlaif_train_data = (
            sft_rlaif["train"]
            .shuffle(seed=42)
            .map(lambda x: self.generate_and_tokenize_prompt(x, output=False))
        )
        val_data = (
            train_val["test"]
            .shuffle(seed=42)
            .map(lambda x: self.generate_and_tokenize_prompt(x, output=False))
        )

        # only allow inputs with token length less than models max length
        sft_train_data = sft_train_data.filter(
            lambda x: len(x["input_ids"]) < self.tokenizer.model_max_length
        )
        rlaif_train_data = rlaif_train_data.filter(
            lambda x: len(x["input_ids"]) < self.tokenizer.model_max_length - 50
        )
        val_data = val_data.filter(
            lambda x: len(x["input_ids"]) < self.tokenizer.model_max_length - 50
        )

        return sft_train_data, rlaif_train_data, val_data


class T5DatasetHandler(DatasetHandler):
    def __init__(self, dataset_name, tokenizer, data_dir: str = "data_json"):
        super().__init__(dataset_name, tokenizer, data_dir)

    @property
    def template(self):
        return (
            """You are an expert in text summarization. You are given the full text."""
            """Your job is to summarise the text as concisely and accurately as possible.\n\n"""
            """### Input:\n{input}\n\n### Response:\n{output}"""
        )


class GPT2DatasetHandler(DatasetHandler):
    def __init__(self, dataset_name, tokenizer, data_dir: str = "data_json"):
        super().__init__(dataset_name, tokenizer, data_dir)
        pass

    @property
    def template(self):
        return (
            """You are an expert in text summarization. You are given the full text."""
            """Your job is to summarise the text in a single sentence and accurately as possible.\n\n"""
            """### Input:\n{input}\n\n### Response:\n{output}"""
        )
    
    def generate_prompt(self, input, output=""):
        if output:
            output = output + self.tokenizer.eos_token
        return self.tokenizer.bos_token + self.template.format(input=input, output=output)
    

if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path

    CWD = Path(os.path.dirname(os.path.realpath(__file__)))
    SRC = CWD.parent / "src"
    sys.path.append(str(CWD))

    from tokenization import T5TokenizationHandler

    tokenizer = T5TokenizationHandler(model_id="t5-small").create_tokenizer()
    DATASET_NAME = "CarperAI/openai_summarize_tldr"
    data_handler = T5DatasetHandler(DATASET_NAME, tokenizer)
    prompt = "Hello there"
    tokenized_prompt = data_handler.generate_and_tokenize_prompt(
        {"input": prompt, "output": prompt}
    )
    pass
