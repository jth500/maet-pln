!pip install rouge-score
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                  use_stemmer=True)
def rouge(y_true: str, y_pred: str) -> float:
    """Rouge loss function to be applied element wise.

    Args:
        y_true (str): True value.
        y_pred (str): Predicted value.

    Returns:
        float: Loss value.
    """
    scores = scorer.score(y_true, y_pred)

    precision = [scores['rouge1'].precision, scores['rouge2'].precision, scores['rougeL'].precision]
    recall = [scores['rouge1'].recall, scores['rouge2'].recall, scores['rougeL'].recall]
    f1_score = [scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure]

    return precision, recall, f1_score
#currently has three version of rouge evaluation metrics
#would probably use rougeL

y_true = "my name is Louise "
y_pred = "is my name Louise"

out = rouge(y_true, y_pred)
print(out)

!pip install transformers
!pip install bert-score

from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def BERTScore(y_true: str, y_pred: str) -> float:

  # prepare texts for BERT
  scorer = BERTScorer(model_type='bert-base-uncased')
  P, R, F1 = scorer.score([y_true], [y_pred])

  precision = round(P.mean().item(),4)
  recall = round(R.mean().item(),4)
  F1 = round(F1.mean().item(),4)
  return precision, recall, F1

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
