import pandas as pd
from random import randint

sentences = {}
train = pd.DataFrame(columns=["text_a", "text_b", "labels"])
test = pd.DataFrame(columns=["text_a", "text_b", "labels"])
with open("sentences/final_checks", "r") as f:
    lines = f.readlines()[1:]
    for i in range(len(lines)):
        l = lines[i].strip().split("\t")
        sentences[l[5]] = sentences.get(l[5], randint(0, 3))
        if sentences[l[5]] == 3:
            test.loc[i] = [l[5], l[6], int(l[8])]
        else:
            train.loc[i] = [l[5], l[6], int(l[8])]
print(train.shape, test.shape)
train.to_csv("sentences/train", sep="\t", index=False)
test.to_csv("sentences/test", sep="\t", index=False)