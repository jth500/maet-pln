from transformers import GenerationConfig
from tqdm import tqdm
import torch
from abc import ABC

class Inference():

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
            attention_mask = self.val_data["attention_mask"][i].unsqueeze(0).to(self.device)
            generation_config = GenerationConfig(
                do_sample=True,
                temperature=1.0, # too high, possibly stray too far from the training data
                top_p=0.7, # chooses from the smallest possible set of words whose cumulative probability exceeds 0.3
                num_beams=3,
                max_new_tokens=150,
                min_new_tokens=10
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