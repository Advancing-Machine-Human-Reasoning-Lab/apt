"""
COMMAND LINE ARGUMENTS -
1. Model family
2. Model name (or path for a saved model)
3. Path to the directory where predictions should be written
4. Test Dataset (optional)
"""

import sys
import os
from numpy import argmax
import pandas as pd
from sklearn.metrics import accuracy_score
from simpletransformers.classification import ClassificationModel

if not os.path.exists(sys.argv[3]):
    os.makedirs(sys.argv[3])

model = ClassificationModel(
    sys.argv[1],
    sys.argv[2],
    num_labels=2,
    use_cuda=True,
    cuda_device=0,
    args={
        "n_gpu": 1,
        "op_dir": sys.argv[3],
        "reprocess_input_data": True,
    },
)
msrp = pd.DataFrame(columns=["text_a", "text_b", "labels"])
with open("/raid/datasets/msrp/msr_paraphrase_train.txt", "r") as f:
    lines = f.readlines()[1:]
    for i in range(len(lines)):
        l = lines[i].strip().split("\t")
        msrp.loc[i] = [l[3], l[4], int(l[0])]
print(msrp.shape)
print(msrp[msrp.labels > 0].shape)

result, model_outputs, wrong_predictions = model.eval_model(msrp, acc=accuracy_score)
print(result)
result_rev, model_outputs_rev, wrong_predictions_rev = model.eval_model(msrp.reindex(columns=["text_b", "text_a", "labels"]).rename(columns={"text_b":"text_a", "text_a":"text_b"}), acc=accuracy_score)
print(result_rev)

msrp = pd.DataFrame(columns=["text_a", "text_b", "labels"])
with open("/raid/datasets/msrp/msr_paraphrase_test.txt", "r") as f:
    lines = f.readlines()[1:]
    for i in range(len(lines)):
        l = lines[i].strip().split("\t")
        msrp.loc[i] = [l[3], l[4], int(l[0])]
print(msrp.shape)
print(msrp[msrp.labels > 0].shape)

result, model_outputs, wrong_predictions = model.eval_model(msrp, acc=accuracy_score)
print(result)
result_rev, model_outputs_rev, wrong_predictions_rev = model.eval_model(msrp.reindex(columns=["text_b", "text_a", "labels"]).rename(columns={"text_b":"text_a", "text_a":"text_b"}), acc=accuracy_score)
print(result_rev)
