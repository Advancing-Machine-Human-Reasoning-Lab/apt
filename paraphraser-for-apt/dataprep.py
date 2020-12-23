# https://towardsdatascience.com/paraphrase-any-question-with-t5-text-to-text-transfer-transformer-pretrained-model-and-cbb9e35f1555

import pandas as pd
from sklearn.model_selection import train_test_split

# preprocess paws_qqp
question_pairs = pd.concat([pd.read_csv("/raid/datasets/paws/paws_qqp/paws_qqp_processed/train.tsv", sep="\t"), pd.read_csv("/raid/datasets/paws/paws_qqp/paws_qqp_processed/dev_and_test.tsv", sep="\t")])
question_pairs_correct_paraphrased = question_pairs[question_pairs["label"] == 1]
question_pairs_correct_paraphrased.drop(["id", "label"], axis=1, inplace=True)
question_pairs_correct_paraphrased["sentence1"].str.decode("utf-8")
question_pairs_correct_paraphrased["sentence2"].str.decode("utf-8")

# preprocess paws_wiki
wiki_pairs = pd.concat([pd.read_csv("/raid/datasets/paws/paws_wiki/train.tsv", sep="\t"), pd.read_csv("/raid/datasets/paws/paws_wiki/dev.tsv", sep="\t"), pd.read_csv("/raid/datasets/paws/paws_wiki/test.tsv", sep="\t")])
wiki_pairs_correct_paraphrased = wiki_pairs[wiki_pairs["label"] == 1]
wiki_pairs_correct_paraphrased.drop(["id", "label"], axis=1, inplace=True)

train, test = train_test_split(pd.concat([question_pairs_correct_paraphrased, wiki_pairs_correct_paraphrased]), test_size=0.1)
train.to_csv("paraphrase_data/train.tsv", sep="\t", index=False)
test.to_csv("paraphrase_data/val.tsv", sep="\t", index=False)
