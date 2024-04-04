from abc import ABC, abstractmethod


class PromptHandler(ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @property
    @abstractmethod
    def TEMPLATE(self):
        pass

    def generate_prompt(self, row):
        # take one or two strings and return one or two strings
        # gpt2 takes one input and optional output
        # t5 takes input and output, although output not always needed for RL
        pass

    def tokenize_prompt(self, row, output=True):
        # to be mapped over the dataset, so it takes a row
        # gpt2 takes input and output columns from that row
        pass


class GPT2PromptHandler(PromptHandler):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    @property
    def TEMPLATE(self):
        return (
            """You are an expert in text summarization. You are given the full text. Your job is to summarise the text as accurately as possible in a single sentence. Refer only to the text provided. Do not not make anything up or infer any information."""
            """\n\n"""
            """### TEXT:\n{input}"""
            """\n\n"""
            """### TL;DR:\n{output}"""
        )

    def generate_prompt(self, row):
        output = row["output"] + self.tokenizer.eos_token
        return self.tokenizer.bos_token + self.TEMPLATE.format(
            input=row["input"], output=output
        )

    def tokenize_prompt(self, row, output=True):
        full_prompt = self.generate_prompt(row)
        if not output:
            # trim off the output
            p = "TL;DR:\n"
            i = full_prompt.index(p) + len(p)
            full_prompt = full_prompt[:i]
        tokenized_full_prompt = self.tokenizer(full_prompt, add_special_tokens=False)
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()
        return tokenized_full_prompt


class T5PromptHandler(PromptHandler):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    @property
    def TEMPLATE(self):
        return """summarize: {input}"""

    def generate_prompt(self, row):
        row["input"] = self.TEMPLATE.format(input=row["input"])
        row["output"] = "<s>{output}</s>".format(output=row["output"])
        return row

    def tokenize_prompt(self, row, output=True):
        """
        Generates a full prompt using the input and output from the given data point,
        and then tokenizes the full prompt using the provided tokenizer.

        Args:
            data_point (dict): A dictionary containing the input and output data.
            tokenizer: The tokenizer object used for tokenization.

        Returns:
            tokenized_full_prompt: The tokenized version of the full prompt.
        """
        row = self.generate_prompt(row)  # add the prompt template
        input_tokens = self.tokenizer(row["input"])
        target_tokens = self.tokenizer(row["output"])

        tokenized_full_prompt = {
            "input_ids": input_tokens["input_ids"],
            "attention_mask": input_tokens["attention_mask"],
            "decoder_input_ids": target_tokens["input_ids"],
            "decoder_attention_mask": target_tokens["attention_mask"],
            "labels": target_tokens["input_ids"],
        }
        return tokenized_full_prompt
