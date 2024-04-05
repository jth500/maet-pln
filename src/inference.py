from transformers import GenerationConfig
from tqdm import tqdm
import torch
from abc import ABC
import numpy as np
import os
import re
import cohere

from dotenv import load_dotenv

load_dotenv()


class Inference:

    def __init__(self, model, tokenizer, val_data):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.val_data = val_data

    def sample_inference(self, sample_size=None):

        if sample_size is None:
            sample_size = len(self.val_data)

        posts = []
        model_summaries = []
        true_summaries = []

        for i in tqdm(range(sample_size)):
            input_ids = self.val_data["input_ids"][i].unsqueeze(0).to(self.device)
            attention_mask = (
                self.val_data["attention_mask"][i].unsqueeze(0).to(self.device)
            )
            generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.3,  # too high and will possibly stray too far from the training data
                top_p=1.0,  # chooses from the smallest possible set of words whose cumulative probability exceeds top_p
                num_beams=3,
                max_new_tokens=150,
                pad_token_id=self.model.config.pad_token_id,
            )
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=self.model.config.pad_token_id,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            s = generation_output.sequences[0]
            output = self.tokenizer.decode(s, skip_special_tokens=True)
            if "### TL;DR:" in output:
                response_text = output.split("### TL;DR:")[1].strip()
            else:
                response_text = output

            posts.append(self.val_data[i]["input"])
            model_summaries.append(response_text)
            true_summaries.append(self.val_data[i]["output"])

        return posts, model_summaries, true_summaries


def _ai_rank_summaries(full, true, pred):

    co = cohere.Client(os.environ["COHERE_PROD_API_KEY"])

    format = f"""### FULL TEXT:\n {full}
    ### SUMMARY 1:\n{true}\n\n
    ### SUMMARY 2:\n{pred}"""

    response = (
        co.generate(
            model="command-nightly",
            # Can this prompt be tailored to cohere's command-nightly model?
            prompt=f"""You are an expert in text summarization. Below, you are given two summaries.

        {format}

        Your role is to determine the best of the two summaries provided.
        Return 1 if you think summary 1 is the best.
        Return 2 if you think summary 2 is the best.

        Use the following chain-of-thought reasoning to evaluate each summary:
        1. Does the summary accurately represent the full text?
        2. Is the summary factually correct?
        3. Is the summary coherent and easy to understand?

        Ideally, you must chose one of the two summaries. 
        However, if you think both summaries are equally good and you cannot make a decision, return 0.

        You must only return 0, 1, or 2. Do not return any text or additional information.
        """,
            max_tokens=5,
            temperature=0.2,
            seed=42,
        )
        .generations[0]
        .text
    )

    return response


def win_rate(full_texts, summaries_1, summaries_2, n_samples=100):
    results = []
    i = 0
    for full, sum_1, sum_2 in zip(full_texts, summaries_1, summaries_1):
        # randomly permute the summaries; avoid bias
        p = np.random.permutation([1, 0])
        sum_1, sum_2 = [sum_1, sum_2][p[0]], [sum_1, sum_2][p[1]]
        if i >= n_samples:
            break
        result = _ai_rank_summaries(full, sum_1, sum_2)
        result = int(result)
        if p[0] == 1 and result == 1:
            results.append(2)
        elif p[0] == 1 and result == 2:
            results.append(1)
        else:
            results.append(result)
        i += 1
    win_rate_1 = results.count(1) / len(results)
    win_rate_2 = results.count(2) / len(results)
    if 0 in results:
        draw_rate = results.count(0) / len(results)
    else:
        draw_rate = 0
    return results, win_rate_1, win_rate_2, draw_rate
