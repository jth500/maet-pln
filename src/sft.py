from tqdm import tqdm
from transformers import (TrainingArguments,
                          Trainer)

from utils import update_kwargs


class SFT():

    def __init__(self, base_model, tokenizer, save_dir):
        self._save_dir = save_dir
        self.tokenizer = tokenizer
        self.base_model = base_model
        self._training_config = None
        self._trainer = None
        
    def _collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])
    
    @property
    def training_config(self, **kwargs):
        if self._training_config is None:
            defaults = {
                "output_dir": self._save_dir,
                "overwrite_output_dir": True,
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "optim": "adamw_bnb_8bit",
                "evaluation_strategy": "no",
                "save_strategy": "epoch",
                "learning_rate": 1e-5,
                "num_train_epochs": 3,
                "warmup_steps": 0.1,
                "weight_decay": 0.01,
                "seed": 42,
                "save_total_limit": 1,
                "disable_tqdm": False,
                "remove_unused_columns": True,
                "report_to": "none",
                "torch_compile": True
            }
            kwargs = update_kwargs(kwargs, defaults)
            self._training_config = TrainingArguments(**kwargs)
        return self._training_config

    @property
    def trainer(self):
        if self._trainer is None:
            self._trainer = self.create_trainer()
        return self._trainer

    def create_trainer(self, train_dataset):
        """
        Creates a trainer for the model.
        
        Args:
            train_dataset (Dataset): The training dataset.
            
        Returns:
            PPOTrainer: The created trainer.
        """
        self._trainer = Trainer(
            model=self.base_model,
            args=self.training_config,
            train_dataset=train_dataset,
            data_collator=self._collator,
            push_to_hub=True
            )
        return self._trainer

    def train(self):
        """
        Trains the model.
        
        Returns:
            None
        """
        self._trainer.train()

    def push_to_hub(self):
        """
        Pushes the model to the Hugging Face Hub for access anywhere.

        Returns:
            None
        """
        self._trainer.push_to_hub()
    