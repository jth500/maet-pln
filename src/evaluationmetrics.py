import evaluate
from bert_score import BERTScorer
import numpy as np
from typing import List, Callable
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    # Class to evaluate models. Includes Rouge, Bleu and Bert score.
    # Input is a pair of lists of the generated summaries and the reference summaries.
    def __init__(self):
        self.BERTscorer = BERTScorer(model_type='bert-base-uncased')
        self.rouge_scorer = evaluate.load('rouge')

    def rouge_score(self, y_pred, y_true):
    
        """Rouge loss function.

        Args:
            y_true (list): True value.
            y_pred (list): Predicted value.

        Returns:
            f1_score for rouge 1, 2 and : float
        """
        scorer = self.rouge_scorer
        scores = scorer.compute(references = y_true, predictions  =y_pred, use_stemmer=True, use_aggregator=False)
        return scores
    def BERTscore(self, y_pred, y_true):
        """BERT loss function.
:
        Args:
            y_true (list): True value.
            y_pred (list): Predicted value.

        Returns:
            F1: float.
        """
        scorer = self.BERTscorer    
        _, _, F1 = scorer.score(y_true, y_pred)
        return  F1
    
    def batchify(self, data: List, batch_size: int):
        """Split a list into batches."""
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def evaluation(self, y_true: List[str], y_pred: List[str], loss_func: Callable, batch_size: int = 100) -> List:
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
            batch_losses = [loss_func(y_true_batch, y_pred_batch)]
            losses.extend(batch_losses)
        
        # # Calculate mean and variance of scores
        if loss_func == self.rouge_score:
            rouge1_vals = [d['rouge1'] for d in losses]
            rouge2_vals = [d['rouge2'] for d in losses]
            rougeL_vals = [d['rougeL'] for d in losses]
            #rougeLsum_vals = [d['rougeLsum'][0] for d in losses]
            mean_rouge1 = np.mean(rouge1_vals)
            var_rouge1 = np.var(rouge1_vals)
            mean_rouge2 = np.mean(rouge2_vals)
            var_rouge2 = np.var(rouge2_vals)
            mean_rougeL = np.mean(rougeL_vals)
            var_rougeL = np.var(rougeL_vals)
            return rouge1_vals, rouge2_vals, rougeL_vals, mean_rouge1, var_rouge1, mean_rouge2, var_rouge2, mean_rougeL, var_rougeL

        else:
            mean_loss = np.mean(losses)
            var_loss = np.var(losses)
            return losses, mean_loss, var_loss
    def plot_scores(self, score_values, score_name):
        """
        Plots a histogram and boxplot for the given scores.

        Args:
        scores (list): The list of scores to plot.
        score_name (str): The name of the score (for the plot title).
        """
        # Plot histogram
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.hist(score_values, bins=20, color='blue', alpha=0.7)
        plt.title(f'Histogram of {score_name} Scores')

        # Plot boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(score_values, color='green')
        plt.title(f'Boxplot of {score_name} Scores')

        plt.tight_layout()
        plt.show()