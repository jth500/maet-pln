import json
from datasets import load_dataset


# TODO: Load dataset is called a couple of times
class DatasetHandler:
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
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tk):
        if tk is None:
            raise Exception("Must pass in a tokenizer at initialisation")
        else:
            self._tokenizer = tk

    def _tokenize(self, prompt):
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

    @staticmethod
    def generate_prompt(input, output=""):
        return (
            """You are an expert in text summarization. You are given the full text."""
            """Your job is to summarise the text as concisely and accurately as possible.\n\n"""
            f"""### Input:\n{input}\n\n### Response:\n{output}"""
        )

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
        tokenized_full_prompt = self._tokenize(full_prompt)
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

        train_data = (
            train_val["train"]
            .shuffle()
            .map(lambda x: self.generate_and_tokenize_prompt(x))
        )
        val_data = (
            train_val["test"]
            .shuffle()
            .map(lambda x: self.generate_and_tokenize_prompt(x))
        )
        return train_data, val_data
