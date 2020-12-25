import json
import logging
from datetime import datetime
from pprint import pprint
from statistics import mean

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from simpletransformers.t5 import T5Model
from sklearn.metrics import accuracy_score, f1_score
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1

logging.basicConfig(level=logging.ERROR)


def f1(truths, preds):
    return mean([compute_f1(truth, pred) for truth, pred in zip(truths, preds)])


def exact(truths, preds):
    return mean([compute_exact(truth, pred) for truth, pred in zip(truths, preds)])


def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]


def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]


model_args = {
    "overwrite_output_dir": True,
    "max_seq_length": 256,
    "eval_batch_size": 32,
    "num_train_epochs": 1,
    "use_multiprocessing": True,
    "num_beams": None,
    "do_sample": True,
    "max_length": 50,
    "top_k": 120,
    "top_p": 0.95,
    "num_return_sequences": 5,
}

prefix = "paraphrasing"

# Load the trained model
model = T5Model("ceshine/t5-paraphrase-paws-msrp-opinosis", args=model_args)

# Load the evaluation data
df = pd.read_csv("paraphrase_data/val.tsv", sep="\t")
df.columns = ["input_text", "target_text"]
df.insert(0, "prefix", ['paraphrase']*len(df), True)

# Prepare the data for testing
to_predict = [
    prefix + ": " + str(input_text)
    for prefix, input_text in zip(df["prefix"].tolist(), df["input_text"].tolist())
]
truth = df["target_text"].tolist()
tasks = df["prefix"].tolist()

# Get the model predictions
preds = model.predict(to_predict)

# Saving the predictions if needed
with open(f"predictions/predictions_{datetime.now()}.txt", "w") as f:
    for i, text in enumerate(df["input_text"].tolist()):
        f.write(str(text) + "\n\n")

        f.write("Truth:\n")
        f.write(truth[i] + "\n\n")

        f.write("Prediction:\n")
        for pred in preds[i]:
            f.write(str(pred) + "\n")
        f.write(
            "________________________________________________________________________________\n"
        )
