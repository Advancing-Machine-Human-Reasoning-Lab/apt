# https://towardsdatascience.com/paraphrase-any-question-with-t5-text-to-text-transfer-transformer-pretrained-model-and-cbb9e35f1555

import re
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# preprocess paws_wiki
wiki_pairs = pd.concat([pd.read_csv("/raid/datasets/paws/paws_wiki/train.tsv", sep="\t"), pd.read_csv("/raid/datasets/paws/paws_wiki/dev.tsv", sep="\t"), pd.read_csv("/raid/datasets/paws/paws_wiki/test.tsv", sep="\t")])
wiki_pairs_correct_paraphrased = wiki_pairs[wiki_pairs["label"] == 1]
wiki_pairs_correct_paraphrased.drop(["id", "label"], axis=1, inplace=True)
print(wiki_pairs_correct_paraphrased.shape)

# preprocess twitter_ppdb
twitter_pairs = pd.concat([pd.read_csv("/raid/datasets/twitter-ppdb/train-preprocessed.txt", sep="\t", header=None), pd.read_csv("/raid/datasets/twitter-ppdb/test-preprocessed.txt", sep="\t", header=None)])
twitter_pairs.columns = ["sentence1", "sentence2", "label"]
twitter_pairs_correct_paraphrased = twitter_pairs[twitter_pairs["label"] == 1]
twitter_pairs_correct_paraphrased.drop(["label"], axis=1, inplace=True)
print(twitter_pairs_correct_paraphrased.shape)

# preprocess ppdb
ppdb_pairs_correct_paraphrased, i = pd.DataFrame(columns=("sentence1", "sentence2")), 0
with open("/raid/datasets/ppdb/ppdb-2.0-xl-all", "r") as f:
    for line in tqdm(f.readlines()):
        line = [l.strip() for l in line.strip().split("|||")]
        if line[-1] == "Equivalence" and line[1].isalpha():
            ppdb_pairs_correct_paraphrased.loc[i] = [line[1], line[2]]
            i += 1

# combine all
train, test = train_test_split(pd.concat([wiki_pairs_correct_paraphrased, twitter_pairs_correct_paraphrased, ppdb_pairs_correct_paraphrased]), test_size=0.1)
# train, test = train_test_split(pd.concat([twitter_pairs_correct_paraphrased]), test_size=0.02)
train.to_csv("paraphrase_data/train.tsv", sep="\t", index=False)
test.to_csv("paraphrase_data/val.tsv", sep="\t", index=False)
print(train.shape, test.shape)
