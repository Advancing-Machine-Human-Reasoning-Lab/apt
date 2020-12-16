"""
Referred from https://huggingface.co/ramsrigouthamg/t5_paraphraser

COMMAND LINE ARGUMENTS -
1. BLEURT threshold
2. Initial top_k = 120
3. Initial top_p = 0.95
4. Offset top_k = 10
5. Offset top_p = 0.05
(refer https://huggingface.co/blog/how-to-generate)
"""

import sys
from numpy import argmax
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from bleurt.score import BleurtScorer
from tqdm import tqdm

bleurt_threshold = float(sys.argv[1])
initial_top_k, initial_top_p, offset_top_k, offset_top_p = (
    int(sys.argv[2]),
    float(sys.argv[3]),
    int(sys.argv[4]),
    float(sys.argv[5]),
)

paraphrasing_model = T5ForConditionalGeneration.from_pretrained("ramsrigouthamg/t5_paraphraser").to("cuda:2")
paraphrasing_tokenizer = T5Tokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
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


def generate_paraphrases(sentence, top_k, top_p):
    text = "paraphrase: " + sentence + " </s>"
    encoding = paraphrasing_tokenizer.encode_plus(text, max_length=256, padding="max_length", return_tensors="pt")
    input_ids, attention_masks = (
        encoding["input_ids"].to("cuda:2"),
        encoding["attention_mask"].to("cuda:2"),
    )
    beam_outputs = paraphrasing_model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=top_k,
        top_p=top_p,
        early_stopping=True,
        num_return_sequences=10,
    )
    final_outputs = []
    for beam_output in beam_outputs:
        sent = paraphrasing_tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)
    return final_outputs


def write_paraphrases(input_file, output_file, position):  # position of sentence in the input tsv
    op = open(output_file, "w+")  # [quality, id1, id2, s1, s2]
    with open(input_file, "r") as f:
        for l in tqdm(f.readlines()[1:]):
            sentence, paraphrases, bad_sentences, top_k, top_p, c = (
                l.strip().split("\t")[position],
                [],
                set(),
                initial_top_k,
                initial_top_p,
                1,
            )
            for p in generate_paraphrases(sentence, top_k, top_p):
                if p not in bad_sentences:
                    if get_mi_score(sentence, p):
                        bleurt = get_bleurt(sentence, p)
                        if bleurt < bleurt_threshold:
                            paraphrases.append((p, bleurt))
                        else:
                            bad_sentences.add(p)
                    else:
                        bad_sentences.add(p)
            while not paraphrases and c <= 10:
                top_k += offset_top_k
                top_p -= offset_top_p
                for p in generate_paraphrases(sentence, top_k, top_p):
                    if p not in bad_sentences:
                        if get_mi_score(sentence, p):
                            bleurt = get_bleurt(sentence, p)
                            if bleurt < bleurt_threshold:
                                paraphrases.append((p, bleurt))
                            else:
                                bad_sentences.add(p)
                        else:
                            bad_sentences.add(p)
                c += 1
            for paraphrase, bleurt in paraphrases:
                op.write(sentence + "\t" + paraphrase + "\t" + str(bleurt) + "\n")


write_paraphrases("/raid/datasets/msrp/msr_paraphrase_train.txt", "nap/msrp1", 3)
write_paraphrases("/raid/datasets/msrp/msr_paraphrase_train.txt", "nap/msrp2", 4)
write_paraphrases("/home/animesh/MIforSE/czeng/czeng_test_engeng.txt", "nap/ppnmt1", 1)
write_paraphrases("/home/animesh/MIforSE/czeng/czeng_test_engeng.txt", "nap/ppnmt2", 2)