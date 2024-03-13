from bert_score import BERTScorer
from rouge_score import rouge_scorer
import numpy as np
from typing import List, Callable

class ModelEvaluator:
    # Class to evaluate models. Includes Rouge, Bleu and Bert score.
    # Input is a pair of lists of the generated summaries and the reference summaries.
    def __init__(self):
        self.BERTscorer = BERTScorer(model_type='bert-base-uncased')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def rouge_score(self, y_pred, y_true):
    
        """Rouge loss function to be applied element wise.

        Args:
            y_true (str): True value.
            y_pred (str): Predicted value.

        Returns:
            float: Loss value.
        """
        scorer = self.rouge_scorer
        scores = scorer.score(y_true, y_pred)

        precision = [scores['rouge1'].precision, scores['rouge2'].precision, scores['rougeL'].precision]
        recall = [scores['rouge1'].recall, scores['rouge2'].recall, scores['rougeL'].recall]
        f1_score = [scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure]

        return precision, recall, f1_score
    def BERTscore(self, y_pred, y_true):
        scorer = self.BERTscorer
        # scores = {"f1": [], "p": [], "r": []}
        # for i in range(0, len(self.y_pred), self.batch_size):
        #     pred_batches = self.y_pred[i:i+self.batch_size]
        #     true_batches = self.y_true[i:i+self.batch_size]
        #     P, R, F1 = scorer.score(true_batches, pred_batches)

        #     scores["f1"].append(F1.numpy())
        #     scores["p"].append(P.numpy())
        #     scores["r"].append(R.numpy())

        # scores_mean = {key: np.mean(value) for key, value in scores.items()}
        # scores_var = {key: np.var(value) for key, value in scores.items()}

        
        # return scores_mean, scores_var, np.array(scores)
        P, R, F1 = scorer.score([y_true], [y_pred])

        precision = round(P.mean().item(),4)
        recall = round(R.mean().item(),4)
        F1 = round(F1.mean().item(),4)
        return precision, recall, F1
    
    #https://github.com/Tiiiger/bert_score/tree/master/example 
    # dont know why we cant just use the score function from the library
    # why is bert rounded but not rouge.
    # is bert input lists?
    # why not using robert large? 


    def batchify(self, data: List, batch_size: int):
        """Split a list into batches."""
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def evaluation(self, y_true: List[str], y_pred: List[str], loss_func: Callable, batch_size: int = 32) -> List:
        """Apply chosen loss metric to each batch of elements.

        Args:
            y_true (List[str]): List of true values.
            y_pred (List[str]): List of predicted values.
            loss_func (Callable): Loss function to apply.
            batch_size (int, optional): Batch size. Defaults to 32.

        Returns:
            List: List of losses for each batch.
        """
        assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
        
        # Split data into batches
        y_true_batches = self.batchify(y_true, batch_size)
        y_pred_batches = self.batchify(y_pred, batch_size)
        
          # Apply loss function to each pair in each batch
        losses = []
        for y_true_batch, y_pred_batch in zip(y_true_batches, y_pred_batches):
            batch_losses = [loss_func(y_true_item, y_pred_item) for y_true_item, y_pred_item in zip(y_true_batch, y_pred_batch)]
            losses.extend(batch_losses)
        
        return losses