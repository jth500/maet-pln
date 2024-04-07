import json
from datasets import load_dataset
import logging
from abc import ABC, abstractmethod
from data_handler.prompt_handler import (
    GPT2PromptHandler,
    T5PromptHandler,
    BARTPromptHandler,
)
from tokenization import (
    T5TokenizationHandler,
    GPT2TokenizationHandler,
    BARTTokenizationHandler,
)
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
import os

logger = logging.getLogger(__name__)
load_dotenv()


class DatasetHandler(ABC):
    """
    A class for handling datasets.

    Args:
        dataset_name (str): The name of the dataset.
        default_tokenizer (Callable): The default tokenizer function.
        tokenizer (Tokenizer): The tokenizer object.
        prompt_handler (PromptHandler): The prompt handler object.
        rlaif (bool, optional): Whether to use the "rlaif" model. Defaults to False.
        input_label (str, optional): The label for the input data. Defaults to "document".
        target_label (str, optional): The label for the target data. Defaults to "summary".
        data_size (int, optional): The size of the data. Defaults to 10000.
        data_dir (str, optional): The directory for the data. Defaults to "data_json".
        save_locally (bool, optional): Whether to save the dataset locally. Defaults to None.
        push_to_hub (bool, optional): Whether to push the dataset to the Hugging Face Hub. Defaults to True.

    Attributes:
        dataset_name (str): The name of the dataset.
        input_label (str): The label for the input data.
        target_label (str): The label for the target data.
        default_tokenizer (Callable): The default tokenizer function.
        tokenizer (Tokenizer): The tokenizer object.
        prompt_handler (PromptHandler): The prompt handler object.
        rlaif (bool): Whether to use the "rlaif" model.
        data_size (int): The size of the data.
        data_dir (str): The directory for the data.
        _hf_login_attempted (bool): Whether the Hugging Face login has been attempted.
        push_to_hub (bool): Whether to push the dataset to the Hugging Face Hub.
        save_locally (bool): Whether to save the dataset locally.
        datasets (dict): The processed datasets.

    """

    def __init__(
        self,
        dataset_name,
        default_tokenizer,
        tokenizer,
        prompt_handler,
        rlaif=False,
        input_label="document",
        target_label="summary",
        data_size: int = 10000,
        data_dir: str = "data_json",
        save_locally=None,
        push_to_hub: bool = True,
    ):
        self.dataset_name = dataset_name
        self.input_label = input_label
        self.target_label = target_label
        self.default_tokenizer = default_tokenizer
        self.tokenizer = tokenizer
        self.prompt_handler = prompt_handler
        self.rlaif = rlaif
        self.data_size = data_size
        self.data_dir = data_dir
        self._hf_login_attempted = False
        self.push_to_hub = push_to_hub
        self.save_locally = save_locally is True or (
            save_locally is None and data_size < 10**6
        )
        self.datasets = None

    DATA_DIR = Path(__file__).parent.parent.parent / "data"

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tk):
        if tk is None:
            tk_handler = self.default_tokenizer()
            self._tokenizer = tk_handler.create_tokenizer()
        else:
            self._tokenizer = tk

    @property
    def push_to_hub(self):
        return self._push_to_hub

    @push_to_hub.setter
    def push_to_hub(self, val):
        if val and not self._hf_login_attempted:
            self._hf_login_attempted = True
            login(os.environ["HF_TOKEN"], add_to_git_credential=True)
        self._push_to_hub = val

    @property
    @abstractmethod
    def EXPECTED_COLUMNS(self):
        """
        The expected columns in the dataset.

        Returns:
            list: A list of column names.
        """
        pass

    @property
    @abstractmethod
    def MAX_LENGTHS(self):
        """
        The maximum lengths for different models.

        Returns:
            dict: A dictionary mapping model names to their maximum lengths.
        """
        pass

    @property
    def max_lengths(self):
        """
        The maximum lengths for different models.

        Returns:
            dict: A dictionary mapping model names to their maximum lengths.
        """
        return {
            "sft": self.tokenizer.model_max_length,
            "rlaif": self.tokenizer.model_max_length - self.MAX_LENGTHS["rlaif"],
            "val": self.tokenizer.model_max_length - self.MAX_LENGTHS["val"],
        }

    @property
    def prompt_handler(self):
        """
        The prompt handler object.

        Returns:
            PromptHandler: The prompt handler object.
        """
        # instantiate the thing
        if isinstance(self._prompt_handler, type):
            self._prompt_handler = self._prompt_handler(self.tokenizer)
        return self._prompt_handler

    @prompt_handler.setter
    def prompt_handler(self, val):
        self._prompt_handler = val

    def data_to_json(self):
        """
        Converts the dataset to JSON format and saves it to a file.

        This method loads the dataset specified by `dataset_name` and selects a subset of the data based on `data_size`.
        It then converts each item in the dataset to a JSON object with "input" and "output" fields, and writes the JSON objects
        to a file named "data_json" in the "raw" directory of the `DATA_DIR`.

        Returns:
            None
        """
        dataset = load_dataset(self.dataset_name, split=f"train[:{self.data_size}]")
        dataset_splits = {"train": dataset}

        for key, ds in dataset_splits.items():
            # overwriting instead of appending - is this what we want?
            with open(self.DATA_DIR / "raw/data_json", "w") as f:
                for item in ds:
                    newitem = {
                        "input": item[self.input_label],
                        "output": item[self.target_label],
                    }
                    f.write(json.dumps(newitem) + "\n")

    def train_val_split(self, data) -> dict:
        """
        Splits the data into training and validation sets.

        Args:
            data (dict): The input data dictionary containing the "train" dataset.

        Returns:
            dict: A dictionary containing the training and validation sets.
                If `rlaif` is True, the dictionary will have the following keys:
                    - "sft": The training set for the "sft" model.
                    - "rlaif": The training set for the "rlaif" model.
                    - "val": The validation set.
                If `rlaif` is False, the dictionary will have the following keys:
                    - "sft": The training set for the "sft" model.
                    - "val": The validation set.
        """
        data = data["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
        if self.rlaif:
            s = data["train"].train_test_split(test_size=0.2, shuffle=True, seed=42)
            return {"sft": s["test"], "rlaif": s["train"], "val": data["test"]}
        else:
            return {"sft": data["train"], "val": data["test"]}

    def save_dataset(self, d, k):
        """
        Saves the dataset `d` to the specified location.

        Args:
            d (Dataset): The dataset to be saved.
            k (str): The identifier for the dataset.

        Raises:
            Exception: If there is an error while saving the dataset locally or pushing it to the Hugging Face Hub.

        """
        if self.save_locally:
            try:
                d.save_to_disk(str(self.DATA_DIR / f"processed/{self.ID}_{k}"))
            except Exception as e:
                msg = "Couldn't save locally"
                logger.exception(msg, exc_info=e)
                pass
        if self.push_to_hub:
            try:
                d.push_to_hub(f"{os.environ['HF_UN']}/{self.ID}_{k}")
            except Exception as e:
                msg = "Couldn't push the data to HF. Is HF_UN  set in .env?"
                logger.exception(msg, exc_info=e)
                pass

    def process_data(self) -> list:
        """
        Process the data by loading it, splitting it into train and validation sets,
        adding prompts, formatting the datasets, and saving them.

        Returns:
            A list of processed datasets.
        """
        try:
            data = load_dataset(
                "json", data_files=str(self.DATA_DIR / f"raw/{self.data_dir}")
            )
            logger.debug("Loading data from JSON")
            assert set(["input", "output"]) <= set(list(data["train"].features.keys()))
        except FileNotFoundError:
            logger.warning("No data JSON found, loading directly from HF.")
            dataset = load_dataset(self.dataset_name, split=f"train[:{self.data_size}]")
            data = dataset.rename_columns(
                {self.input_label: "input", self.target_label: "output"}
            )
        datasets = self.train_val_split(data)

        # Add prompts and format datasets
        for k in datasets.keys():
            tk = lambda x: self.prompt_handler.tokenize_prompt(x, output=k == "sft")
            truncator = lambda row: len(row["input_ids"]) < self.max_lengths[k]
            datasets[k] = datasets[k].map(tk)
            datasets[k] = datasets[k].filter(truncator)
            datasets[k].set_format(type="torch", columns=self.EXPECTED_COLUMNS)
            self.save_dataset(datasets[k], k)
        self.datasets = datasets
        return list(datasets.values())


class GPT2DatasetHandler(DatasetHandler):
    # Decoder only architecture

    EXPECTED_COLUMNS = ["input", "output", "input_ids", "attention_mask", "labels"]
    MAX_LENGTHS = {"sft": 0, "rlaif": 150, "val": 150}
    ID = "GPT"

    def __init__(
        self,
        dataset_name,
        tokenizer=None,
        rlaif=False,
        input_label="document",
        target_label="summary",
        data_size=100,
        data_dir: str = "data_json",
        save_locally=None,
        push_to_hub=True,
    ):
        super().__init__(
            dataset_name=dataset_name,
            default_tokenizer=GPT2TokenizationHandler,
            tokenizer=tokenizer,
            prompt_handler=GPT2PromptHandler,
            rlaif=rlaif,
            input_label=input_label,
            target_label=target_label,
            data_size=data_size,
            data_dir=data_dir,
            save_locally=save_locally,
            push_to_hub=push_to_hub,
        )


class T5DatasetHandler(DatasetHandler):

    EXPECTED_COLUMNS = [
        "input",
        "output",
        "input_ids",
        "decoder_input_ids",
        "attention_mask",
        "decoder_attention_mask",
        "labels",
    ]
    MAX_LENGTHS = {"sft": 0, "rlaif": 50, "val": 50}
    ID = "T5"

    # Encoder-decoder architecture
    def __init__(
        self,
        dataset_name,
        tokenizer=None,
        rlaif=False,
        input_label="document",
        target_label="summary",
        data_size=100,
        data_dir: str = "data_json",
        save_locally=None,
        push_to_hub=True,
    ):
        super().__init__(
            dataset_name=dataset_name,
            default_tokenizer=T5TokenizationHandler,
            tokenizer=tokenizer,
            prompt_handler=T5PromptHandler,
            rlaif=rlaif,
            input_label=input_label,
            target_label=target_label,
            data_size=data_size,
            data_dir=data_dir,
            save_locally=save_locally,
            push_to_hub=push_to_hub,
        )


class BARTDatasetHandler(DatasetHandler):

    EXPECTED_COLUMNS = ["input", "output", "input_ids", "attention_mask", "labels"]
    MAX_LENGTHS = {"sft": 0, "rlaif": 50, "val": 50}
    ID = "BART"

    # Encoder-decoder architecture
    def __init__(
        self,
        dataset_name,
        tokenizer=None,
        rlaif=False,
        input_label="document",
        target_label="summary",
        data_size=100,
        data_dir: str = "data_json",
        save_locally=None,
        push_to_hub=True,
    ):
        super().__init__(
            dataset_name=dataset_name,
            default_tokenizer=BARTTokenizationHandler,
            tokenizer=tokenizer,
            prompt_handler=BARTPromptHandler,
            rlaif=rlaif,
            input_label=input_label,
            target_label=target_label,
            data_size=data_size,
            data_dir=data_dir,
            save_locally=save_locally,
            push_to_hub=push_to_hub,
        )
