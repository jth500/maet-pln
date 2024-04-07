from abc import ABC, abstractmethod


class PromptHandler(ABC):
    """
    Abstract base class for prompt handlers.
    """

    def __init__(self, tokenizer):
        """
        Initializes a new instance of the PromptHandler class.

        Args:
            tokenizer: The tokenizer to be used for tokenization.
        """
        self.tokenizer = tokenizer

    @property
    @abstractmethod
    def TEMPLATE(self):
        """
        Abstract property representing the template for the prompt.
        """
        pass

    def generate_prompt(self, row):
        """
        Generates a prompt based on the given row.

        Args:
            row: The input row.

        Returns:
            The generated prompt.
        """
        # take one or two strings and return one or two strings
        # gpt2 takes one input and optional output
        # t5 takes input and output, although output not always needed for RL
        pass

    def tokenize_prompt(self, row, output=True):
        """
        Tokenizes the prompt based on the given row.

        Args:
            row: The input row.
            output: Whether to include the output in the tokenization process. Default is True.

        Returns:
            The tokenized prompt.
        """
        # to be mapped over the dataset, so it takes a row
        # gpt2 takes input and output columns from that row
        pass


class GPT2PromptHandler(PromptHandler):
    """
    A class that handles prompts for GPT-2 model.

    Args:
        tokenizer (Tokenizer): The tokenizer used for tokenizing the prompts.

    Attributes:
        TEMPLATE (str): The template for generating the prompt.

    Methods:
        generate_prompt: Generates the prompt for a given row of data.
        tokenize_prompt: Tokenizes the prompt for a given row of data.

    """

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
        """
        Generates the prompt for a given row of data.

        Args:
            row (dict): The row of data containing the input and output.

        Returns:
            str: The generated prompt.

        """
        output = row["output"] + self.tokenizer.eos_token
        return self.tokenizer.bos_token + self.TEMPLATE.format(
            input=row["input"], output=output
        )

    def tokenize_prompt(self, row, output=True):
        """
        Tokenizes the prompt for a given row of data.

        Args:
            row (dict): The row of data containing the input and output.
            output (bool): Whether to include the output in the tokenized prompt.

        Returns:
            dict: The tokenized prompt.

        """
        full_prompt = self.generate_prompt(row)
        if not output:
            # trim off the output
            p = "TL;DR:\n"
            i = full_prompt.index(p) + len(p)
            full_prompt = full_prompt[:i]
        tokenized_full_prompt = self.tokenizer(full_prompt, add_special_tokens=False)
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()
        return tokenized_full_prompt


class EncoderDecoderPromptHandler(PromptHandler):
    """
    A class that handles the prompt generation and tokenization for encoder-decoder models.

    Args:
        tokenizer: The tokenizer object used for tokenization.

    Attributes:
        tokenizer: The tokenizer object used for tokenization.

    Methods:
        map_tokens: Abstract method to be implemented by subclasses for mapping input and target tokens.
        generate_prompt: Generates a full prompt by formatting the input and output data.
        tokenize_prompt: Generates a full prompt and tokenizes it using the provided tokenizer.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    @staticmethod
    @abstractmethod
    def map_tokens(input_tokens, target_tokens):
        pass

    def generate_prompt(self, row):
        """
        Generates a full prompt by formatting the input and output data.

        Args:
            row (dict): A dictionary containing the input and output data.

        Returns:
            row (dict): The updated dictionary with the formatted input and output data.
        """
        row["input"] = self.TEMPLATE.format(input=row["input"])
        row["output"] = "<s>{output}</s>".format(output=row["output"])
        return row

    def tokenize_prompt(self, row, output=True):
        """
        Generates a full prompt using the input and output from the given data point,
        and then tokenizes the full prompt using the provided tokenizer.

        Args:
            row (dict): A dictionary containing the input and output data.
            output (bool): Whether to include the output in the prompt. Default is True.

        Returns:
            tokenized_full_prompt: The tokenized version of the full prompt.
        """
        row = self.generate_prompt(row)  # add the prompt template
        input_tokens = self.tokenizer(row["input"])
        target_tokens = self.tokenizer(row["output"])

        tokenized_full_prompt = self.map_tokens(input_tokens, target_tokens)
        return tokenized_full_prompt


class T5PromptHandler(EncoderDecoderPromptHandler):
    """
    A class that handles prompts for T5 models.

    Args:
        tokenizer (Tokenizer): The tokenizer used for tokenization.

    Attributes:
        TEMPLATE (str): The template string used for generating prompts.

    Methods:
        map_tokens(input_tokens, target_tokens): Maps input and target tokens to the required format.

    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    @property
    def TEMPLATE(self):
        return """summarize: {input}"""

    @staticmethod
    def map_tokens(input_tokens, target_tokens):
        """
        Maps input and target tokens to the required format.

        Args:
            input_tokens (dict): A dictionary containing input tokens.
            target_tokens (dict): A dictionary containing target tokens.

        Returns:
            dict: A dictionary containing the mapped tokens.

        """
        return {
            "input_ids": input_tokens["input_ids"],
            "attention_mask": input_tokens["attention_mask"],
            "decoder_input_ids": target_tokens["input_ids"],
            "decoder_attention_mask": target_tokens["attention_mask"],
            "labels": target_tokens["input_ids"],
        }


class BARTPromptHandler(EncoderDecoderPromptHandler):
    """
    A class that handles prompts for the BART model.

    Inherits from the EncoderDecoderPromptHandler class.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    @property
    def TEMPLATE(self):
        return """Summarize the following document. {input}\nSummary:"""

    def generate_prompt(self, row):
        """
        Generates a prompt for the BART model.

        Args:
            row (dict): A dictionary containing the input and output data.

        Returns:
            dict: A dictionary containing the generated prompt.
        """
        row["input"] = self.TEMPLATE.format(input=row["input"])
        row["output"] = "{output}".format(output=row["output"])
        return row

    @staticmethod
    def map_tokens(input_tokens, target_tokens):
        """
        Maps the input and target tokens for the BART model.

        Args:
            input_tokens (dict): A dictionary containing the input tokens.
            target_tokens (dict): A dictionary containing the target tokens.

        Returns:
            dict: A dictionary containing the mapped tokens.
        """
        return {
            "input_ids": input_tokens["input_ids"],
            "attention_mask": input_tokens["attention_mask"],
            "labels": target_tokens["input_ids"],
        }
