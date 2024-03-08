from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer
import tqdm
import numpy as np

class ModelEvaluator:
    # Class to evaluate models. Includes Rouge, Bleu and Bert score.
    # Input is a pair of lists of the generated summaries and the reference summaries.
    def __init__(self, y_true: list, y_pred: list, batch_size: int = 30):
        self.y_true = y_true
        self.y_pred = y_pred
        assert len(self.y_true) == len(self.y_pred), "y_true and y_pred must have the same length"

    def rouge(self):
        # TODO: Implement rouge evaluation
        pass

    def BERTscore(self):
        scores = {"f1": [], "p": [], "r": []}
        scorer = BERTScorer(model_type='bert-base-uncased')

        for i in range(0, len(self.y_pred), self.batch_size):
            pred_batches = self.y_pred[i:i+self.batch_size]
            true_batches = self.y_true[i:i+self.batch_size]
            P, R, F1 = scorer.score(true_batches, pred_batches)

            scores["f1"].append(F1.numpy())
            scores["p"].append(P.numpy())
            scores["r"].append(R.numpy())

        scores_mean = {key: np.mean(value) for key, value in scores.items()}
        scores_var = {key: np.var(value) for key, value in scores.items()}

        return scores_mean, scores_var, np.array(scores)

    def blue(self):
        # TODO: Implement blue evaluation
        pass
    def eval_loss(self, loss_metric="rouge"):
        LOSS_FUNCS = {"rouge": self.rouge, "bert": self.BERTscore}
        return [LOSS_FUNCS[loss_metric](self.y_true[i], self.y_pred[i]) for i in range(len(self.y_true))]

# use like this 
    #evaluator = ModelEvaluator(y_true, y_pred)
#losses = evaluator.eval_loss("rouge")
    


def rouge(y_true: str, y_pred: str) -> float:
    """Rouge loss function to be applied element wise.

    Args:
        y_true (str): True value.
        y_pred (str): Predicted value.

    Returns:
        float: Loss value.
    """
    # TODO: Implement rouge loss
    # Not sure if inputs should be strings or tokenised lists?? Haven't looked at this at all
    # presumably there's a library for this?
    return None


def other_loss_metric(y_true: str, y_pred: str) -> float:
    # TODO
    pass


def eval_loss(y_true: list, y_pred: list, loss_metric="rouge") -> list:
    """Apply chosen loss metric to each element in the batch

    Args:
        y_true (list): List of true values.
        y_pred (list): List of predicted values.

    Returns:
        list: List of losses for each element in the batch.
    """
    # TODO: Vectorize this (if possible)
    LOSS_FUNCS = {"rouge": rouge, "other": other_loss_metric}
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    return [LOSS_FUNCS[loss_metric](y_true[i], y_pred[i]) for i in range(len(y_true))]


def test_rouge():
    """
    Test the rouge function.

    This function tests the rouge function by comparing the output of the function
    with expected values. It asserts that the calculated rouge score matches the
    expected score for different input strings.

    Returns:
        None
    """
    y_true = "This is a test"
    y_pred = "This is a test"
    assert rouge(y_true, y_pred) == 1.0  # guessing this is the expected value?
    y_pred = "This is not a test"
    assert rouge(y_true, y_pred) == 0.0
    y_pred = ""

    y_list = ["This is a test", "This is a test", "This is a test"]
    assert eval_loss(y_list, y_list) == [1.0, 1.0, 1.0]


if __name__ == "__main__":
    test_rouge()
