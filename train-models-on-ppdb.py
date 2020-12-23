"""
COMMAND LINE ARGUMENTS -
1. Dataset path
2. Model family
3. Model name (or path for a saved model)
4. Output dir
5. Additional dataset name(s) comma separated
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from simpletransformers.classification import ClassificationModel
from utils import read

if not os.path.exists(sys.argv[4]):
    os.makedirs(sys.argv[4])

dataset = sys.argv[1]
data_path = "/raid/datasets/" + dataset + "/"

if "twitter" in dataset:
    train_df = pd.read_csv(data_path + "train-preprocessed.txt", sep="\t", header=None)
    train = pd.DataFrame(
        {
            "text_a": train_df.iloc[:, 0],
            "text_b": train_df.iloc[:, 1],
            "labels": train_df.iloc[:, 2],
        }
    )
    test_df = pd.read_csv(data_path + "test-preprocessed.txt", sep="\t", header=None)
    test = pd.DataFrame(
        {
            "text_a": test_df.iloc[:, 0],
            "text_b": test_df.iloc[:, 1],
            "labels": test_df.iloc[:, 2],
        }
    )
else:
    train_df = pd.read_csv(data_path + "train-preprocessed.txt", sep=" ||| ", header=None)
    df = pd.DataFrame(
        {
            "text_a": train_df.iloc[:, 0],
            "text_b": train_df.iloc[:, 1],
            "labels": train_df.iloc[:, 2],
        }
    )
    train, test = train_test_split(df, test_size=0.1)

print(train.shape)
try:
    datasets = sys.argv[5].split(",")
    new_dataset = read(datasets[0], sep="\t", header=True)
    for dataset in datasets[1:]:
        new_dataset = new_dataset.append(read(dataset, sep="\t", header=True))
    print(new_dataset.shape)
    try:
        if sys.argv[6] == "balanced":
            new_dataset_mi = new_dataset[new_dataset.iloc[:, -1] == 0]
            new_dataset_nmi = new_dataset[new_dataset.iloc[:, -1] == 1]
            new_dataset = pd.concat(
                [
                    new_dataset_mi,
                    resample(
                        new_dataset_nmi,
                        replace=True,
                        n_samples=len(new_dataset_mi),  # to match minority class
                        random_state=0,  # reproducible results
                    ),
                ]
            )
    except:
        pass
    train = pd.concat([train, new_dataset])
except:
    pass
print(train.shape)

# shuffle
train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)

model = ClassificationModel(
    sys.argv[2],
    sys.argv[3],
    num_labels=2,
    use_cuda=True,
    cuda_device=2,
    args={
        "output_dir": sys.argv[4],
        "overwrite_output_dir": True,
        "fp16": True,  # uses apex
        "num_train_epochs": 5,
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "do_lower_case": False,
        "evaluate_during_training": True,
        "evaluate_during_verbose": True,
        "evaluate_during_training_steps": 10000,
        "n_gpu": 1,
        "reprocess_input_data": True,
    },
)

model.train_model(train, eval_df=test)
