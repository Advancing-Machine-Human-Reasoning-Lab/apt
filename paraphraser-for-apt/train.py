import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bleurt.score import BleurtScorer
import logging
import torch
from tqdm import tqdm
from numpy import argmax
from math import exp

logging.basicConfig(level=logging.ERROR)

train_df = pd.read_csv("paraphrase_data/train.tsv", sep="\t")
train_df.columns = ["input_text", "target_text"]
train_df.insert(0, "prefix", ['paraphrase']*len(train_df), True)
eval_df = pd.read_csv("paraphrase_data/val.tsv", sep="\t")
eval_df.columns = ["input_text", "target_text"]
eval_df.insert(0, "prefix", ['paraphrase']*len(eval_df), True)

print(train_df.shape, eval_df.shape)

# Initialize model
model = T5Model(
    't5-base',
    args=T5Args(
        output_dir="outputs/",
        overwrite_output_dir=True,
        do_lower_case=False,
        train_batch_size=192,
        eval_batch_size=192,
        num_train_epochs=10,
        no_save=True,
        evaluate_generated_text=True,
        evaluate_during_training=True,
        evaluate_during_training_steps=len(train_df)/(192*2),
        evaluate_during_training_verbose=True,
        fp16=True,
        n_gpu=3,
        save_model_every_epoch=True,
    ),
    early_stopping=True,
    evaluate_generated_text=True,
    use_cuda=True,
    num_beams=10,
    num_return_sequences=5,
    preprocess_inputs=True,
    top_k=120,
    top_p=0.95,
    use_multiprocessed_decoding=True,
)


bleurt_scorer = BleurtScorer("/home/animesh/MIforSE/bleurt-score/bleurt/bleurt-base-128/")
mi_tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
mi_model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")  # predicts E, N, C


def get_mi_score(s1, s2):  # returns average of s1 and s2
    tokenized_input_seq_pair = mi_tokenizer.encode_plus(s1, s2, max_length=256, return_token_type_ids=True, truncation=True)
    input_ids = torch.Tensor(tokenized_input_seq_pair["input_ids"]).long().unsqueeze(0)
    token_type_ids = torch.Tensor(tokenized_input_seq_pair["token_type_ids"]).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair["attention_mask"]).long().unsqueeze(0)
    outputs = mi_model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=None,
    )
    predicted_probability_12 = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    tokenized_input_seq_pair = mi_tokenizer.encode_plus(s2, s1, max_length=256, return_token_type_ids=True, truncation=True)
    input_ids = torch.Tensor(tokenized_input_seq_pair["input_ids"]).long().unsqueeze(0)
    token_type_ids = torch.Tensor(tokenized_input_seq_pair["token_type_ids"]).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair["attention_mask"]).long().unsqueeze(0)
    outputs = mi_model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=None,
    )
    predicted_probability_21 = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    return int(argmax(predicted_probability_12) == 0 and argmax(predicted_probability_21) == 0)


def get_bleurt(s1, s2):
    return (bleurt_scorer.score([s1], [s2])[0] + bleurt_scorer.score([s1], [s2])[0]) / 2


def count_matches(labels, preds):
    return sum([get_mi_score(label, pred) / ((1 + exp(5 * get_bleurt(label, pred))) ** 2) for label, pred in tqdm(zip(labels, preds))])


# Train the model
model.train_model(train_data=train_df, eval_data=eval_df, show_running_loss=True, matches=count_matches)

# Evaluate the model
results = model.eval_model(eval_df, matches=count_matches)
print(results)

# Use the model for prediction
print(model.predict(["paraphrase: Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army."]))
