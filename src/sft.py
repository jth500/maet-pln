from tqdm import tqdm
import torch

from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

from utils import update_kwargs

class SFT:

    def __init__(self, base_model, tokenizer, save_dir, train_dataset, train_epochs=4):
        self._save_dir = save_dir
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.train_dataset = train_dataset
        self.trainer = None
        self.training_config = None
        self.train_epochs = train_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            num_train_epochs=self.train_epochs,
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


if __name__ == "__main__":

    from huggingface_hub import login

    login("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")

    # CWD = Path(os.path.dirname(os.path.realpath(__file__)))
    # SRC = CWD.parent / "src"
    # sys.path.append(str(CWD))

    from tokenization import TokenizationHandler
    from model_builder import T5ModelBuilder
    from data_handler import T5DatasetHandler

    model_id = "t5-base"
    tokenizer = TokenizationHandler().create_tokenizer()
    model = T5ModelBuilder(model_id, tokenizer).base_model

    dataset_name = "EdinburghNLP/xsum"
    data_handler = T5DatasetHandler(dataset_name, tokenizer)
    sft_train_data, rlaif_train_data, val_data = data_handler.process_data(input_label="document", target_label="summary")

    sft = SFT(model, tokenizer, "sft_model", sft_train_data)
    sft.train_model()
