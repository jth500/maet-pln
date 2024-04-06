from tqdm import tqdm
import torch
from abc import ABC

import os

# from dotenv import load_dotenv

# load_dotenv()
cohere_api_key = os.getenv("COHERE_PROD_API_KEY")

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    create_reference_model,
)
from peft import LoraConfig, TaskType
from torch.optim import Adam
import re
import cohere

from utils import update_kwargs


class RLAIF:

    def __init__(self, base_dir, tokenizer, save_dir, train_dataset, ppo_epochs=4):
        self.base_dir = base_dir  # The base model is the SFT model
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.base_model = None
        self.ref_model = None
        self.ppo_config = None
        self.ppo_trainer = None
        self.train_dataset = train_dataset
        self.ppo_epochs = ppo_epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def base_model(self):
        if self._base_model is None:
            self._base_model = self.load_base_model()
        return self._base_model

    @base_model.setter
    def base_model(self, base):
        self._base_model = base

    def load_base_model(self):
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM" if "gpt2" in self.base_dir else TaskType.SEQ_2_SEQ_LM,
        )
        if "gpt2" in self.base_dir:
            base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.base_dir, peft_config=lora_config, device_map="auto"
            )
        else:
            base_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
                self.base_dir, peft_config=lora_config
            )
        self.base_model = base_model
        return base_model

    @property
    def ref_model(self):
        self._ref_model = create_reference_model(self.base_model)
        return self._ref_model

    @ref_model.setter
    def ref_model(self, ref):
        self._ref_model = ref

    @staticmethod
    def _collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    @property
    def ppo_config(self, **kwargs):
        if self._ppo_config is None:
            defaults = dict(
                model_name=self.base_dir,
                learning_rate=2e-5,
                ppo_epochs=self.ppo_epochs,
                batch_size=1,
                mini_batch_size=1,
                seed=42,
            )
            kwargs = update_kwargs(kwargs, defaults)
            self._ppo_config = PPOConfig(**kwargs)
        return self._ppo_config

    @ppo_config.setter
    def ppo_config(self, ppo_config):
        self._ppo_config = ppo_config

    @property
    def ppo_trainer(self):
        if self._ppo_trainer is None:
            self._ppo_trainer = self.create_ppo_trainer()
        return self._ppo_trainer

    @ppo_trainer.setter
    def ppo_trainer(self, ppo_trainer):
        self._ppo_trainer = ppo_trainer

    def create_ppo_trainer(self):
        """
        Creates a PPO trainer for the model.

        Args:
            train_dataset (Dataset): The training dataset.

        Returns:
            PPOTrainer: The created PPO trainer.
        """
        optimizer = Adam(
            filter(lambda p: p.requires_grad, self.base_model.parameters()), lr=2e-5
        )
        self._ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.base_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=self.train_dataset,
            data_collator=self._collator,
            optimizer=optimizer,
        )
        return self._ppo_trainer

    @staticmethod
    def score_summaries(full_text, summarized_text):
        """
        Generates a score for the summarized text using the cohere API.

        Args:
            full_text (str): The full text.
            summarized_text (str): The summarized text.

        Returns:
            float: The score of the summarized text.
        """

        co = cohere.Client(cohere_api_key)

        score = 0

        try:
            format = f"""### FULL TEXT:\n {full_text} \n
            ### SUMMARIZED TEXT: \n {summarized_text}"""

            response = (
                co.generate(
                    model="command-nightly",
                    prompt=f"""You are an expert in text summarization.

Your role is to rate the provided summarization with scores ranging from 0.01 to 0.99, where: 0.01 is the lowest score, 0.99 is the highest score.
0.01 represents a summary that is completely irrelevant, factually incorrect, or incoherent. 
0.99 represents a perfect summary that is highly relevant, factually correct, and coherent.
The summary should be a single sentence, accurate, and capture the main points of the full text.

Use the following chain-of-thought reasoning to evaluate the summary:
1. Does the summary accurately represent the full text?
2. Is the summary factually correct?
3. Is the summary coherent and easy to understand? 

Your response should only be a double precision number that represents the score.

Below, you are given the full text and its summarization.

{format}

### Score:
""",
                    max_tokens=5,
                    temperature=0.2,
                )
                .generations[0]
                .text
            )
            score = float(re.findall(r"[-+]?(?:\d*\.*\d+)", response)[0])
        except:
            score = 0.5

        return score

    def train_model(self, max_ppo_steps=float("inf")):
        """
        Trains the model using PPO.

        Args:
            max_ppo_steps (int, optional): The maximum number of PPO steps. Defaults to inf (i.e. length of epoch).

        Returns:
            list: The KL divergence between the new and old policies.
            list: The mean of the returns.
            list: The mean of the advantages.
        """

        # Min and max length of the summaries
        # output_min_length = 1
        # output_max_length = 50
        # output_length_sampler = LengthSampler(output_min_length, output_max_length)

        # Generation kwargs
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.base_model.config.pad_token_id,
            "max_new_tokens": 150,
        }

        ref_ppo_delta = []
        returns_mean = []
        kl = []
        loss = []

        # PPO training loop
        for step, batch in enumerate(tqdm(self.ppo_trainer.dataloader)):
            if step >= max_ppo_steps:
                break

            # Decode the input_ids to get the prompts
            prompts = [self.tokenizer.decode(input) for input in batch["input_ids"]]
            prompt_tensors = batch["input_ids"]

            # Generate summary tensors
            summary_tensors = []
            for prompt_tensor in prompt_tensors:
                # max_new_tokens = output_length_sampler()
                # generation_kwargs["max_new_tokens"] = max_new_tokens
                max_new_tokens = generation_kwargs["max_new_tokens"]
                prompt_tensor = torch.tensor(prompt_tensor).to(self.device)
                generation_output = self._ppo_trainer.generate(
                    prompt_tensor, **generation_kwargs
                )
                output = generation_output[0]
                summary_text = self.tokenizer.decode(output, skip_special_tokens=True)
                if "### TL;DR:" in summary_text:
                    summary_text = summary_text.split("### TL;DR:")[1].strip()
                # print(summary_text)
                summary = self.tokenizer.encode(summary_text)
                summary_tensor = torch.tensor(summary)
                summary_tensors.append(
                    summary_tensor.squeeze()[-max_new_tokens:]
                )  # truncate to max_new_tokens if necessary

            # Decode the summary tensors to get the summaries
            batch["response"] = [
                self.tokenizer.decode(r.squeeze()) for r in summary_tensors
            ]
            response = batch["response"]

            # Compute the rewards
            reward_tensors = []
            for prompt, summary in zip(prompts, response):
                score = self.score_summaries(prompt, response)  # utilises cohere API
                reward_tensors.append(torch.tensor(score).float())

            # Convert the prompts to tensors
            prompt_tensors = [torch.tensor(tensor).long() for tensor in prompt_tensors]
            summary_tensors = [
                torch.tensor(tensor).long() for tensor in summary_tensors
            ]
            reward_tensors = [torch.tensor(tensor) for tensor in reward_tensors]

            # Step the PPO trainer
            stats = self.ppo_trainer.step(
                prompt_tensors, summary_tensors, reward_tensors
            )
            self.ppo_trainer.log_stats(stats, batch, reward_tensors)

            # Log the stats
            ref_ppo_delta.append(stats["ppo/mean_non_score_reward"])
            returns_mean.append(stats["ppo/returns/mean"])
            kl.append(stats["objective/kl"])
            loss.append(stats["ppo/loss/value"])

        return ref_ppo_delta, returns_mean, kl, loss

    def push_model_to_hub(self):
        """
        Pushes the model to the Hugging Face Hub for access anywhere.

        Returns:
            None
        """
        self.base_model.push_to_hub(self.save_dir)
