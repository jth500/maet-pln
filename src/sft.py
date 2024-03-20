from tqdm import tqdm
import torch

from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

from utils import update_kwargs

class SFT:

    def __init__(self, base_model, tokenizer, save_dir, train_dataset):
        self._save_dir = save_dir
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.train_dataset = train_dataset
        self.trainer = None
        self.training_config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def _collator(data):
    #     return dict((key, [d[key] for d in data]) for key in data[0])

    @property
    def training_config(self):
        self._training_config = self.create_training_config()
        return self._training_config
    
    @training_config.setter
    def training_config(self, value):
        self._training_config = value

    def create_training_config(self, **kwargs):
        defaults = dict(
            output_dir = "/tmp",
            push_to_hub = True,
            push_to_hub_model_id = self._save_dir,
            warmup_steps = 0.1,
            num_train_epochs=4,
            # max_steps = 200,
            per_device_train_batch_size = 1,
            # per_device_eval_batch_size = 1,
            gradient_accumulation_steps = 8,
            learning_rate = 2e-5,
            logging_steps = 0.25,
            evaluation_strategy = "no",
            save_strategy = "no",
            # eval_steps=0.25,
            # save_steps=0.25,
            optim = 'adamw_torch',
            load_best_model_at_end = False,
            group_by_length = True,
            # seed = 42
        )
        kwargs = update_kwargs(kwargs, defaults)
        training_config = TrainingArguments(**kwargs)
        return training_config

    @property
    def trainer(self):
        self._trainer = self.create_trainer()
        return self._trainer
    
    @trainer.setter
    def trainer(self, value):
        self._trainer = value

    def create_trainer(self):
        trainer = Trainer(
            model=self.base_model,
            args=self.training_config,
            train_dataset=self.train_dataset,  # Set your actual train_dataset here
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer, 
                pad_to_multiple_of=4, 
                return_tensors="pt",
                padding=True
            )
        )
        return trainer

    def train_model(self):
        self.trainer.train()

    def push_model_to_hub(self):
        """
        Pushes the model to the Hugging Face Hub for access anywhere.

        Returns:
            None
        """
        self.base_model.push_to_hub(self._save_dir)
