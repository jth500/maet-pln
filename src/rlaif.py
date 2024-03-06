from tqdm import tqdm
import torch

from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model
    )
from trl.core import LengthSampler

from utils import update_kwargs


class RLAIF():

    def __init__(self, base_dir, tokenizer, save_dir):
        self.base_dir = base_dir # The base model is the SFT model
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.base_model = None
        self.ref_model = None
        self._ppo_config = None
        self._ppo_trainer = None
        
    
    @property
    def base_model(self):
        if self._base_model is None:
            self._base_model = self.load_base_model()
        return self._base_model
    
    @base_model.setter
    def model(self, base):
        self._base_model = base

    def load_base_model(self):
        """
        Loads the base model. Different from create_base_model in SFT class since need Value Head.

        Returns:
            The base model.
        """
        self.base_model =  AutoModelForCausalLMWithValueHead.from_pretrained(self.base_dir)
        return self.base_model

    @property
    def reference_model(self):
        self._ref_model = create_reference_model(self.sft_model)
        return self._ref_model

    def _collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])
    
    @property
    def ppo_config(self, **kwargs):
        if self._ppo_config is None:
            defaults = {
                "model_name": self.base_dir,
                "learning_rate": 1e-5,
                "ppo_epochs": 1,
                "mini_batch_size": 1,
                "batch_size": 2}
            kwargs = update_kwargs(kwargs, defaults)
            self._ppo_config = PPOConfig(**kwargs)
        return self._ppo_config

    @property
    def ppo_trainer(self):
        if self._ppo_trainer is None:
            self._ppo_trainer = self.create_ppo_trainer()
        return self._ppo_trainer

    def create_ppo_trainer(self, train_dataset):
        """
        Creates a PPO trainer for the model.
        
        Args:
            train_dataset (Dataset): The training dataset.
            
        Returns:
            PPOTrainer: The created PPO trainer.
        """
        self._ppo_trainer = PPOTrainer(
            config=self._ppo_config,
            model=self.base_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=train_dataset,
            data_collator=self._collator,
            push_to_hub=True
            )

    def score_summaries(full_text, summarized_text):
        """
        Generates a score for the summarized text using the cohere API.

        Args:
            full_text (str): The full text.
            summarized_text (str): The summarized text.

        Returns:
            float: The score of the summarized text.
        """

        import re
        import cohere
        co = cohere.Client('ZXPdIn0oozFbK6YtZ3FI0aBH9NIH2gw0MStEXGWz')

        score = 0

        try:
            format = f"""### FULL TEXT:\n {full_text} \n
            ### SUMMARIZED TEXT: \n {summarized_text}"""

            response = co.generate(
                model='command-nightly',
                # Can this prompt be tailored to cohere's command-nightly model?
                prompt=f"""You are an expert in text summarization. Below, you are given the full text and its summarization.

                {format}

                Your role is to rate the provided summarization with scores ranging from 0 to 1, where: 0 is the lowest score, 1 is the highest score.
                Your response should only be a double precision number that represents the scoring rate.
                """,
                max_tokens=5,
                temperature=0.9
                ).generations[0].text
            score = float(re.findall(r"[-+]?(?:\d*\.*\d+)", response)[0])
        except:
            score = 0.5

        return score

    def train(self, max_ppo_steps=100):
        """
        Trains the model using PPO.

        Args:
            max_ppo_steps (int, optional): The maximum number of PPO steps. Defaults to 100.

        Returns:
            list: The KL divergence between the new and old policies.
            list: The mean of the returns.
            list: The mean of the advantages.
        """

        # Min and max length of the summaries
        output_min_length = 10
        output_max_length = 150
        output_length_sampler = LengthSampler(output_min_length, output_max_length)

        # Generation kwargs
        generation_kwargs = {
            "temperature": 0.7,
            "min_length": 5,
            "top_p": 0.3,
            "do_sample": True
        }

        objective_kl    = [] # KL divergence between the new and old policies
        returns_mean    = [] # Mean of the returns
        advantages_mean = [] # Mean of the advantages

        # PPO training loop
        for step, batch in tqdm(enumerate(self._ppo_trainer.dataloader)):
            if step >= max_ppo_steps:
                break

            # Decode the input_ids to get the prompts
            prompts = [self._tokenizer.decode(input) for input in batch['input_ids']]
            prompt_tensors = batch["input_ids"]

            # Generate summary tensors
            summary_tensors = []
            for prompt_tensor in prompt_tensors:
                max_new_tokens = output_length_sampler()
                generation_kwargs["max_new_tokens"] = max_new_tokens
                prompt_tensor = torch.tensor(prompt_tensor)
                summary = self._ppo_trainer.generate(prompt_tensor, **generation_kwargs)
                summary_tensors.append(summary.squeeze()[-max_new_tokens:])

            # Decode the summary tensors to get the summaries
            batch["response"] = [self._tokenizer.decode(r.squeeze()) for r in summary_tensors]
            response = batch["response"]

            # Compute the rewards
            reward_tensors = []
            for prompt, summary in zip(prompts, response):
                score = self.score_summaries(prompt, response) # utilises cohere API
                reward_tensors.append(torch.tensor(score))

            # Convert the lists to tensors
            prompt_tensors = [torch.tensor(tensor) for tensor in prompt_tensors]
            summary_tensors = [torch.tensor(tensor) for tensor in summary_tensors]
            reward_tensors = [torch.tensor(tensor) for tensor in reward_tensors]

            # Step the PPO trainer
            stats = self._ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
            self._ppo_trainer.log_stats(stats, batch, reward_tensors)

            # Log the stats
            objective_kl.append(stats["objective/kl"])
            returns_mean.append(stats["ppo/returns/mean"])
            advantages_mean.append(stats["ppo/policy/advantages_mean"])

        return objective_kl, returns_mean, advantages_mean
    

    def push_to_hub(self):
        """
        Pushes the model to the Hugging Face Hub for access anywhere.

        Returns:
            None
        """
        self._ppo_trainer.push_to_hub()
    





