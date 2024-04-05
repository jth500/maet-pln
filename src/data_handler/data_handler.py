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
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
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
        pass

    @property
    @abstractmethod
    def MAX_LENGTHS(self):
        pass

    @property
    def max_lengths(self):
        return {
            "sft": self.tokenizer.model_max_length,
            "rlaif": self.tokenizer.model_max_length - self.MAX_LENGTHS["rlaif"],
            "val": self.tokenizer.model_max_length - self.MAX_LENGTHS["val"],
        }

    @property
    def prompt_handler(self):
        # instantiate the thing
        if isinstance(self._prompt_handler, type):
            self._prompt_handler = self._prompt_handler(self.tokenizer)
        return self._prompt_handler

    @prompt_handler.setter
    def prompt_handler(self, val):
        self._prompt_handler = val

    def data_to_json(self):
        """
        Converts the dataset to JSON format and saves it to a file named 'data_json'.

        This method loads the dataset, iterates over the items in the dataset, and converts each item
        to a JSON object with 'input' and 'output' keys. The JSON objects are then written to a file
        named 'data_json' in the current directory.

        Note: The dataset is limited to the first x examples from the 'train' split.

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
        data = data["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
        if self.rlaif:
            s = data["train"].train_test_split(test_size=0.2, shuffle=True, seed=42)
            return {"sft": s["test"], "rlaif": s["train"], "val": data["test"]}
        else:
            return {"sft": data["train"], "val": data["test"]}

    def save_dataset(self, d, k):
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
