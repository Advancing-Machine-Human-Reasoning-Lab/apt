import json
import logging
import torch
from datetime import datetime

import numpy as np
import pandas as pd
from transformers import T5ForConditionalGeneration,T5Tokenizer

from tqdm import tqdm

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

logging.basicConfig(level=logging.ERROR)
device = "cuda:0"

model_args = {
    "overwrite_output_dir": True,
    "max_seq_length": 256,
    "num_train_epochs": 1,
    "use_multiprocessing": True,
    "num_beams": None,
    "do_sample": True,
    "max_length": 50,
    "top_k": 120,
    "top_p": 0.95,
    "num_return_sequences": 5,
}

# Load the trained model
model = T5ForConditionalGeneration.from_pretrained('t5_paraphrase1/model2')
model = model.to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-base')

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
print(to_predict[:5])

preds = []
# Get the model predictions
for sentence in tqdm(to_predict):
    encoding = tokenizer.encode_plus(sentence,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=50,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=5
    )
    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)
    preds.append(final_outputs)

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


# import torch
# from transformers import T5ForConditionalGeneration, T5Tokenizer


# def set_seed(seed):
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# set_seed(42)

# model = T5ForConditionalGeneration.from_pretrained("t5_paraphrase1/model/")
# # model = T5ForConditionalGeneration.from_pretrained("t5_paraphrase_trial/epoch2/")
# # model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
# tokenizer = T5Tokenizer.from_pretrained("t5-base")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device ", device)
# model = model.to(device)

# sentence = "Manvi is not able to suggest a sentence for T5 paraphraser."
# # sentence = "What are the ingredients required to bake a perfect cake?"
# # sentence = "What is the best possible approach to learn aeronautical engineering?"
# # sentence = "Do apples taste better than oranges in general?"


# text = "paraphrase: " + sentence + " </s>"


# max_len = 256

# encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
# input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


# # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
# beam_outputs = model.generate(input_ids=input_ids, attention_mask=attention_masks, do_sample=True, max_length=256, top_k=120, top_p=0.95, early_stopping=True, num_return_sequences=10)


# print("\nOriginal Question ::")
# print(sentence)
# print("\n")
# print("Paraphrased Questions :: ")
# final_outputs = []
# for beam_output in beam_outputs:
#     sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#     if sent.lower() != sentence.lower() and sent not in final_outputs:
#         final_outputs.append(sent)

# for i, final_output in enumerate(final_outputs):
#     print("{}: {}".format(i, final_output))


# """
# 0: How does one get to know about data science?
# 1: Which is best online course for Data Science?
# 2: What kind of course should we take in data science?
# 3: What courses can I take to get started in data science?
# 4: What are some good books for data scientists?
# 5: Which course should I take to get started with data science?
# 6: What are the requirements for a start in data science?
# 7: What is the best course to start data science?
# 8: Who should I take to get started in data science?
# 9: What are the best courses on data science?
# """
