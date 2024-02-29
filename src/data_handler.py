import json
import abc
from datasets import load_dataset
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# TODO: Load dataset is called a couple of times
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
        if tk is None:
            raise Exception("Must pass in a tokenizer at initialisation")
        else:
            self._tokenizer = tk

    def _data_to_json(self):
        """
        Converts the dataset to JSON format and saves it to a file named 'data_json'.

        This method loads the dataset, iterates over the items in the dataset, and converts each item
        to a JSON object with 'input' and 'output' keys. The JSON objects are then written to a file
        named 'data_json' in the current directory.

        Note: The dataset is limited to the first 1000 examples from the 'train' split.

        Returns:
            None
        """
        dataset = load_dataset(self.dataset_name, split="train[:1000]")
        dataset_splits = {"train": dataset}

        for key, ds in dataset_splits.items():
            with open(
                "data_json", "w"
            ) as f:  # overwriting instead of appending - is this what we want?
                for item in ds:
                    newitem = {
                        "input": item["prompt"],
                        "output": item["label"],
                    }
                    f.write(json.dumps(newitem) + "\n")

    def generate_prompt(self, input, output=""):
        return self.template.format(input=input, output=output)

    def generate_and_tokenize_prompt(self, data_point):
        """
        Generates a full prompt using the input and output from the given data point,
        and then tokenizes the full prompt using the provided tokenizer.

        Args:
            data_point (dict): A dictionary containing the input and output data.
            tokenizer: The tokenizer object used for tokenization.

        Returns:
            tokenized_full_prompt: The tokenized version of the full prompt.
        """
        full_prompt = self.generate_prompt(
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenizer.tokenize(full_prompt)
        return tokenized_full_prompt

    def process_data(self):
        """
        Process the data for training and validation.

        Returns:
            train_data (Dataset): Processed training data.
            val_data (Dataset): Processed validation data.
        """
        # dataset = load_dataset(self.dataset_name, split="train")  # why called twice?# it's not used?
        self._data_to_json()

        data = load_dataset("json", data_files="data_json")
        train_val = data["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)

        f = lambda x: self.generate_and_tokenize_prompt(x)
        data = [train_val[slice].shuffle().map(f) for slice in ["train", "test"]]
        return data[0], data[1]


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


# class GPTDatasetHandler(DatasetHandler):
#     def __init__(self, dataset_name, tokenizer, data_dir: str = "data_json"):
#         super().__init__(dataset_name, tokenizer, data_dir)
#         pass

#     def generate_prompt(input, output="", template=template):
#         return (
#             """You are an expert in text summarization. You are given the full text."""
#             """Your job is to summarise the text as concisely and accurately as possible.\n\n"""
#             f"""### Input:\n{input}\n\n### Response:\n{output}"""
#         )
