from transformers import GenerationConfig
from tqdm import tqdm
import torch

from data_handler import GPT2DatasetHandler

class Inference:

    def __init__(self, model, tokenizer, val_data):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.val_data = val_data

    @property
    def template(self):
        return (
            """You are an expert in text summarization. You are given the full text."""
            """Your job is to summarise the text in a single sentence and accurately as possible.\n\n"""
            """### Input:\n{input}\n\n### Response:\n{output}"""
        )
    
    def generate_prompt(self, input, output=""):
        if output:
            output = output + self.tokenizer.eos_token
        return self.tokenizer.bos_token + self.template.format(input=input, output=output)

    def sample_inference(self, sample_size=None):

        if sample_size is None:
            sample_size = len(self.val_data)

        posts = []
        model_summaries = []
        true_summaries = []

        for i in tqdm(range(sample_size)):
            prompt = self.generate_prompt(self.val_data[i]["input"])
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.8,
                top_p=0.3,
                num_beams=1,
                max_new_tokens=50,
            )
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    pad_token_id=self.model.config.pad_token_id,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            s = generation_output.sequences[0]
            output = self.tokenizer.decode(s, skip_special_tokens=True)
            response_text = output.split("### Response:")[1].strip()

            posts.append(self.val_data[i]["input"])
            model_summaries.append(response_text)
            true_summaries.append(self.val_data[i]["output"])

        return posts, model_summaries, true_summaries    