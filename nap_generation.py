"""
Referred from https://huggingface.co/ramsrigouthamg/t5_paraphraser

COMMAND LINE ARGUMENTS -
1. CUDA device number
2. Dataset to run on: 'msrp1', 'msrp2', 'ppnmt1', 'ppnmt2'
3. Number of parts
3. Portion
(refer https://huggingface.co/blog/how-to-generate)
"""

import sys
from numpy import argmax
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
from bleurt.score import BleurtScorer
from tqdm import tqdm

bleurt_threshold, initial_top_k, initial_top_p, offset_top_k, offset_top_p, device = 0.5, 120, 0.95, 20, 0.05, "cuda:" + sys.argv[1]
paraphrasing_model = T5ForConditionalGeneration.from_pretrained("paraphraser-for-apt/t5_paraphrase1/model2").to(device)
paraphrasing_tokenizer = T5Tokenizer.from_pretrained("t5-base")
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
    del tokenized_input_seq_pair, input_ids, token_type_ids, attention_mask, outputs

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
    
    del tokenized_input_seq_pair, input_ids, token_type_ids, attention_mask, outputs
    return int(argmax(predicted_probability_12) == 0 and argmax(predicted_probability_21) == 0)


def get_bleurt(s1, s2):
    return (bleurt_scorer.score([s1], [s2])[0] + bleurt_scorer.score([s1], [s2])[0]) / 2


def generate_paraphrases(sentence, top_k, top_p):
    text = "paraphrase: " + sentence + " </s>"
    encoding = paraphrasing_tokenizer.encode_plus(text, max_length=256, padding="max_length", return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
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


def write_paraphrases(input_file, apt_output_file, mi_output_file, nmi_output_file, position, startFrom=1):  # position of sentence in the input tsv
    n, i = int(sys.argv[3]), int(sys.argv[4])
    written_sentences = set()
    try:
        with open(apt_output_file + str(i), "r") as f:
            for l in f.readlines():
                written_sentences.add(l.strip().split("\t")[0])
    except:
        pass
    try:
        with open(mi_output_file + str(i), "r") as f:
            for l in f.readlines():
                written_sentences.add(l.strip().split("\t")[0])
    except:
        pass
    try:
        with open(nmi_output_file + str(i), "r") as f:
            for l in f.readlines():
                written_sentences.add(l.strip().split("\t")[0])
    except:
        pass
    apt = open(apt_output_file + str(i), "a+")
    mi = open(mi_output_file + str(i), "a+")
    nmi = open(nmi_output_file + str(i), "a+")
    with open(input_file, "r") as f:
        allLines = f.readlines()[startFrom:]
        for l in tqdm(allLines[i * len(allLines) // n : (i + 1) * len(allLines) // n]):
            sentence, bad_sentences, written, top_k, top_p, c = l.strip().split("\t")[position], set(), False, initial_top_k, initial_top_p, 1
            if sentence in written_sentences:
                continue
            for p in generate_paraphrases(sentence, top_k, top_p):
                if p not in bad_sentences:
                    bleurt, miscore = get_bleurt(sentence, p), get_mi_score(sentence, p)
                    if miscore:
                        if bleurt < bleurt_threshold:
                            apt.write(sentence + "\t" + p + "\t" + str(bleurt) + "\t" + str(miscore) + "\n")
                            written = True
                        else:
                            bad_sentences.add(p)
                            mi.write(sentence + "\t" + p + "\t" + str(bleurt) + "\t" + str(miscore) + "\n")
                    else:
                        bad_sentences.add(p)
                        nmi.write(sentence + "\t" + p + "\t" + str(bleurt) + "\t" + str(miscore) + "\n")
            while not written and c <= 5:
                top_k += offset_top_k
                top_p -= offset_top_p
                for p in generate_paraphrases(sentence, top_k, top_p):
                    if p not in bad_sentences:
                        bleurt, miscore = get_bleurt(sentence, p), get_mi_score(sentence, p)
                        if miscore:
                            if bleurt < bleurt_threshold:
                                apt.write(sentence + "\t" + p + "\t" + str(bleurt) + "\t" + str(miscore) + "\n")
                                written = True
                            else:
                                bad_sentences.add(p)
                                mi.write(sentence + "\t" + p + "\t" + str(bleurt) + "\t" + str(miscore) + "\n")
                        else:
                            bad_sentences.add(p)
                            nmi.write(sentence + "\t" + p + "\t" + str(bleurt) + "\t" + str(miscore) + "\n")
                c += 1
            del bad_sentences, written


if sys.argv[2] == "msrp1":  # [quality, id1, id2, s1, s2]
    write_paraphrases("/raid/datasets/msrp/msr_paraphrase_train.txt", "nap/msrp1/msrp1-apt", "nap/msrp1/msrp1-mi", "nap/msrp1/msrp1-nmi", 3)  # , startFrom=2668) # startFrom is the 0-based index of the line you want to start processing from
elif sys.argv[2] == "msrp2":
    write_paraphrases("/raid/datasets/msrp/msr_paraphrase_train.txt", "nap/msrp2-apt", "nap/msrp2-mi", "nap/msrp2-nmi", 4)
elif sys.argv[2] == "ppnmt1":  # [c1, s1, s2]
    write_paraphrases("/home/animesh/MIforSE/czeng/czeng_test_engeng.txt", "nap/ppnmt1/ppnmt1-apt", "nap/ppnmt1/ppnmt1-mi", "nap/ppnmt1/ppnmt1-nmi", 1)  # startFrom is the 0-based index of the line you want to start processing from
elif sys.argv[2] == "ppnmt2":
    write_paraphrases("/home/animesh/MIforSE/czeng/czeng_test_engeng.txt", "nap/ppnmt2-apt", "nap/ppnmt2-mi", "nap/ppnmt2-nmi", 2)
elif sys.argv[2] == "twitter-ppdb1":
    write_paraphrases("/raid/datasets/twitter-ppdb/unique.txt", "nap/twitterppdb1/twitterppdb1-apt", "nap/twitterppdb1/twitterppdb1-mi", "nap/twitterppdb1/twitterppdb1-nmi", 0)
else:
    print("!!! Wrong dataset name !!!")
