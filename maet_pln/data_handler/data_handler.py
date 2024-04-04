import json
from datasets import load_dataset
import logging
from abc import ABC, abstractmethod
from data_handler.prompt_handler import GPT2PromptHandler, T5PromptHandler
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
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tk):
        self._tokenizer = tk

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
        val_data = data["test"]
        if self.rlaif:
            sft_rlaif = data["train"].train_test_split(
                test_size=0.2, shuffle=True, seed=42
            )
            sft_train_data = sft_rlaif["test"]
            rlaif_train_data = sft_rlaif["train"]
            d = {"sft": sft_train_data, "rlaif": rlaif_train_data, "val": val_data}
        else:
            sft_train_data = data["train"]
            rlaif_train_data = None
            d = {"sft": sft_train_data, "val": val_data}
        return d

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
            data = load_dataset("json", data_files=str(self.DATA_DIR / "raw/data_json"))
        except FileNotFoundError:
            raise FileNotFoundError("Have you run the thing first?")

        assert set(["input", "output"]) <= set(list(data["train"].features.keys()))
        datasets = self.train_val_split(data)

        # Add prompts and format datasets
        for k in datasets.keys():
            tk = lambda x: self.prompt_handler.tokenize_prompt(x, output=k == "sft")
            truncator = lambda row: len(row["input_ids"]) < self.max_lengths[k]
            datasets[k] = datasets[k].map(tk)
            datasets[k] = datasets[k].filter(truncator)
            datasets[k].set_format(type="torch", columns=self.EXPECTED_COLUMNS)
            self.save_dataset(datasets[k], k)
        return list(datasets.values())


class GPT2DatasetHandler(DatasetHandler):
    # Decoder only architecture

    EXPECTED_COLUMNS = ["input", "output", "input_ids", "attention_mask", "labels"]
    MAX_LENGTHS = {"sft": 0, "rlaif": 150, "val": 150}
    ID = "GPT"

    def __init__(
        self,
        dataset_name,
        tokenizer,
        prompt_handler=GPT2PromptHandler,
        rlaif=False,
        input_label="document",
        target_label="summary",
        data_size=100,
        data_dir: str = "data_json",
        save_locally=None,
        push_to_hub=True,
    ):
        super().__init__(
            dataset_name,
            tokenizer,
            prompt_handler,
            rlaif,
            input_label,
            target_label,
            data_size,
            data_dir,
            save_locally,
            push_to_hub,
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
        tokenizer,
        prompt_handler=T5PromptHandler,
        rlaif=False,
        input_label="document",
        target_label="summary",
        data_size=100,
        data_dir: str = "data_json",
        save_locally=None,
        push_to_hub=True,
    ):
        super().__init__(
            dataset_name,
            tokenizer,
            prompt_handler,
            rlaif,
            input_label,
            target_label,
            data_size,
            data_dir,
            save_locally,
            push_to_hub,
        )


# class BARTDatasetHandler(DatasetHandler):
#     # Denoising autoencoder architecture
#     def __init__(self, dataset_name, tokenizer, data_dir: str = "data_json"):
#         super().__init__(dataset_name, tokenizer, data_dir)

#     @property
#     def template(self):
#         return (
#             """<s>You are an expert in text summarization. You are given the full text."""
#             """Your job is to summarise the text as concisely and accurately as possible in a single sentence.\n\n"""
#             """### TEXT:\n{input}\n\n"""
#             """### SUMMARY:</s>"""
#         )

#     def generate_prompt(self, data_point):
#         data_point["input"] = self.template.format(input=data_point["input"])
#         data_point["output"] = "<s>{output}</s>".format(output=data_point["output"])
#         return data_point

#     def generate_and_tokenize_prompt(self, data_point):
#         """
#         Generates a full prompt using the input and output from the given data point,
#         and then tokenizes the full prompt using the provided tokenizer.

#         Args:
#             data_point (dict): A dictionary containing the input and output data.
#             tokenizer: The tokenizer object used for tokenization.

#         Returns:
#             tokenized_full_prompt: The tokenized version of the full prompt.
#         """
#         defaults = dict(
#             truncation=True,
#             padding=False,
#             return_tensors=None,
#         )
#         input_tokens = self.tokenizer(
#             data_point["input"], add_special_tokens=False, max_length=1024, **defaults
#         )
#         target_tokens = self.tokenizer(
#             data_point["output"], add_special_tokens=False, max_length=1024, **defaults
#         )
#         tokenized_full_prompt = {
#             "input_ids": input_tokens["input_ids"],
#             "attention_mask": input_tokens["attention_mask"],
#             "labels": target_tokens["input_ids"],
#         }
#         return tokenized_full_prompt

#     def process_data(self, input_label="prompt", target_label="summary", rlaif=False):
#         """
#         Process the data for training and validation.

#         Returns:
#             train_data (Dataset): Processed training data.
#             val_data (Dataset): Processed validation data.
#         """
#         self._data_to_json(input_label, target_label)
#         data = load_dataset("json", data_files="data_json")
#         train_val = data["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)

#         if rlaif:
#             sft_rlaif = train_val["train"].train_test_split(
#                 test_size=0.2, shuffle=True, seed=42
#             )  # split for sft and rlaif; rlaif does not need outputs
#             sft_train_data = (
#                 sft_rlaif["test"]
#                 .shuffle(seed=42)
#                 .map(lambda x: self.generate_prompt(x))
#                 .map(lambda x: self.generate_and_tokenize_prompt(x))
#             )
#             rlaif_train_data = (
#                 sft_rlaif["train"]
#                 .shuffle(seed=42)
#                 .map(lambda x: self.generate_prompt(x))
#                 .map(lambda x: self.generate_and_tokenize_prompt(x))
#             )

#         else:
#             sft_train_data = (
#                 train_val["train"]
#                 .shuffle(seed=42)
#                 .map(lambda x: self.generate_prompt(x))
#                 .map(lambda x: self.generate_and_tokenize_prompt(x))
#             )
#             rlaif_train_data = None
#         val_data = (
#             train_val["test"]
#             .shuffle(seed=42)
#             .map(lambda x: self.generate_prompt(x))
#             .map(lambda x: self.generate_and_tokenize_prompt(x), batched=True)
#         )
#         # only allow inputs with token length less than models max length
#         sft_train_data = sft_train_data.filter(
#             lambda x: len(x["input_ids"]) < self.tokenizer.model_max_length
#         )
#         if rlaif_train_data:
#             rlaif_train_data = rlaif_train_data.filter(
#                 lambda x: len(x["input_ids"]) < self.tokenizer.model_max_length - 50
#             )
#         val_data = val_data.filter(
#             lambda x: len(x["input_ids"]) < self.tokenizer.model_max_length - 50
#         )

#         columns = ["input", "output", "input_ids", "attention_mask", "labels"]
#         sft_train_data.set_format(type="torch", columns=columns)
#         if rlaif_train_data:
#             rlaif_train_data.set_format(type="torch", columns=columns)
#         val_data.set_format(type="torch", columns=columns)
#         return sft_train_data, rlaif_train_data, val_data


# # if __name__ == "__main__":
