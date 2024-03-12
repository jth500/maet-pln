'''
pip install transformers
pip install bert-score
'''

from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer
from rouge_score import rouge_scorer
import tqdm
import numpy as np

def BERTScore_batches(pred, true, batch_size = 30):

    scores = {}
    scores["f1"] = []
    scores["p"] = []
    scores["r"] = []

    scorer = BERTScorer(model_type='bert-base-uncased')

    for i in (range(0,len(pred), batch_size)):
        pred_batches = pred[i:i+batch_size]
        true_batches = true[i:i+batch_size]
        P, R, F1 = scorer.score(true_batches, pred_batches)

        scores["f1"].append(F1.numpy())
        scores["p"].append(P.numpy())
        scores["r"].append(R.numpy())

    scores_mean = {}
    scores_mean["f1"] = np.mean(scores["f1"])
    scores_mean["p"] = np.mean(scores["p"])
    scores_mean["r"] = np.mean(scores["r"])
    scores_var = {}
    scores_var["f1"] = np.var(scores["f1"])
    scores_var["p"] = np.var(scores["p"])
    scores_var["r"] = np.var(scores["r"])

    return scores_mean, scores_var, np.array(scores)

###rouge batch scorer
def rouge_batches(pred, true, batch_size = 30):

    scores = {}
    scores["f1"] = []
    scores["p"] = []
    scores["r"] = []

    #setting ROUGEL as our scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'],
                                  use_stemmer=True)

    #creating predicted and true input values pairs
    input_pairs = [(pred_input, true_input)
                      for pred_input, true_input in zip(pred, true)]

    #iterating over pairs in specified batch size
    for i in (range(0,len(input_pairs), batch_size)):

        #selecting our input batch
        input_pairs_batch = input_pairs[i:i+batch_size]

        #creating an empty array to store scores for each batch
        batch_p, batch_r, batch_f1 = [],[],[]

        #iterating over the pairs in the batch
        for pred_input, true_input in input_pairs_batch:

          #scoring each of the input pairs and adding to batch array
          score_dict = scorer.score(pred_input, true_input)

          F1 = score_dict['rougeL'].fmeasure
          P = score_dict['rougeL'].precision
          R = score_dict['rougeL'].recall

          batch_p.append(P)
          batch_r.append(R)
          batch_f1.append(F1)

        #adding all batch scores to score arrays
        scores["f1"].extend(batch_f1)
        scores["p"].extend(batch_p)
        scores["r"].extend(batch_r)

    #calculating mean and variance for each score
    f1_mean = np.mean(scores["f1"])
    p_mean = np.mean(scores["p"])
    r_mean = np.mean(scores["r"])

    f1_var = np.var(scores["f1"])
    p_var = np.var(scores["p"])
    r_var = np.var(scores["r"])

    scores_means = [f1_mean, p_mean, r_mean]
    scores_vars = [f1_var, p_var, r_var]

    return scores_means, scores_vars, np.array(scores)
    
