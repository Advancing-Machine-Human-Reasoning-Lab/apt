"""
COMMAND LINE ARGUMENTS -
1. Model family
2. Model name (or path for a saved model)
3. Path to the directory where predictions should be written
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
    cuda_device=2,
    args={
        "n_gpu": 1,
        "op_dir": sys.argv[3],
        "reprocess_input_data": True,
    },
)
app = pd.DataFrame(columns=["text_a", "text_b", "labels"])
with open("sentences/final_checks", "r") as f:
    lines = f.readlines()[1:]
    for i in range(len(lines)):
        l = lines[i].strip().split("\t")
        app.loc[i] = [l[5], l[6], int(l[8])]
print(app.shape)
test = pd.read_csv("sentences/test", sep="\t")
print(test.shape)

result_app, model_outputs_app, wrong_predictions_app = model.eval_model(app, acc=accuracy_score)
print(result_app)
result_test, model_outputs_test, wrong_predictions_test = model.eval_model(test, acc=accuracy_score)
print(result_test)

with open(sys.argv[3] + "app-predictions", "w+") as f:
    f.write(str(result_app) + "\n\n")
    for i, row in app.iterrows():
        prediction = argmax(model_outputs_app[i])
        f.write("\t".join([row["text_a"], row["text_b"], str(row["labels"]), str(prediction)]) + "\n")

with open(sys.argv[3] + "test-predictions", "w+") as f:
    f.write(str(result_test) + "\n\n")
    for i, row in test.iterrows():
        prediction = argmax(model_outputs_test[i])
        f.write(
            "\t".join(
                [
                    str(row["text_a"]),
                    str(row["text_b"]),
                    str(row["labels"]),
                    str(prediction),
                ]
            )
            + "\n"
        )
