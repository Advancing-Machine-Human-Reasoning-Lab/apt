from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from numpy import argmax
from pandas import DataFrame

from simpletransformers.classification import ClassificationModel

hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
# hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"
tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
model = AutoModelForSequenceClassification.from_pretrained(
    hg_model_hub_name
)  # predicts E, N, C

# mi_scorer = ClassificationModel('roberta', '/home/animesh/MIScore-study/roberta_nli/', use_cuda=False, args = {'reprocess_input_data':True})


def get_mi_score(s1, s2):  # returns average of s1 and s2
    tokenized_input_seq_pair = tokenizer.encode_plus(
        s1, s2, max_length=256, return_token_type_ids=True, truncation=True
    )
    input_ids = torch.Tensor(tokenized_input_seq_pair["input_ids"]).long().unsqueeze(0)
    token_type_ids = (
        torch.Tensor(tokenized_input_seq_pair["token_type_ids"]).long().unsqueeze(0)
    )
    attention_mask = (
        torch.Tensor(tokenized_input_seq_pair["attention_mask"]).long().unsqueeze(0)
    )
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=None,
    )
    print(outputs)
    # outputs = model(input_ids, attention_mask=attention_mask, labels=None)
    predicted_probability_12 = torch.softmax(outputs[0], dim=1)[
        0
    ].tolist()  # batch_size only one
    tokenized_input_seq_pair = tokenizer.encode_plus(
        s2, s1, max_length=256, return_token_type_ids=True, truncation=True
    )
    input_ids = torch.Tensor(tokenized_input_seq_pair["input_ids"]).long().unsqueeze(0)
    token_type_ids = (
        torch.Tensor(tokenized_input_seq_pair["token_type_ids"]).long().unsqueeze(0)
    )
    attention_mask = (
        torch.Tensor(tokenized_input_seq_pair["attention_mask"]).long().unsqueeze(0)
    )
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=None,
    )
    print(outputs)
    # outputs = model(input_ids, attention_mask=attention_mask, labels=None)
    predicted_probability_21 = torch.softmax(outputs[0], dim=1)[
        0
    ].tolist()  # batch_size only one
    print(predicted_probability_12, predicted_probability_21)
    return (
        argmax(predicted_probability_12) == 0 and argmax(predicted_probability_21) == 0
    )

    # _, s1s2, __ = mi_scorer.eval_model(DataFrame({'text_a':s1, 'text_b':s2, 'labels':2}))
    # _, s2s1, __ = mi_scorer.eval_model(DataFrame({'text_a':s2, 'text_b':s1, 'labels':2}))
    # print(s1s2[0], s2s1[0], argmax(s1s2[0]), argmax(s2s1[0]))
    # return s1s2[0][2] > 0 and argmax(s1s2[0]) == 2 and s2s1[0][2] > 0 and argmax(s2s1[0]) == 2


print(get_mi_score("Please don't tell Navid.", "Animesh."))
