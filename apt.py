from random import randrange, choice
import string
from time import time
from math import exp

from bleurt.score import BleurtScorer
# from simpletransformers.classification import ClassificationModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pandas import DataFrame
from numpy import argmax

from flask import Flask, request, jsonify, session, render_template
from flask_cors import CORS, cross_origin
from flask_session import Session
from waitress import serve

bleurt_scorer = BleurtScorer('bleurt/bleurt/bleurt-base-128/')
hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name) # predicts E, N, C
# mi_scorer = ClassificationModel('roberta', 'roberta_nli/', use_cuda=False, args = {'reprocess_input_data':True})

def get_mi_score(s1, s2): # returns average of s1 and s2
    tokenized_input_seq_pair = tokenizer.encode_plus(s1, s2, max_length=256, return_token_type_ids=True, truncation=True)
    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)
    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=None)
    predicted_probability_12 = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    tokenized_input_seq_pair = tokenizer.encode_plus(s2, s1, max_length=256, return_token_type_ids=True, truncation=True)
    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)
    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=None)
    predicted_probability_21 = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    return argmax(predicted_probability_12) == 0 and argmax(predicted_probability_21) == 0
    
    # _, s1s2, __ = mi_scorer.eval_model(DataFrame({'text_a':s1, 'text_b':s2, 'labels':2}))
    # _, s2s1, __ = mi_scorer.eval_model(DataFrame({'text_a':s2, 'text_b':s1, 'labels':2}))
    # print(s1s2[0], s2s1[0], argmax(s1s2[0]), argmax(s2s1[0]))
    # return int(s1s2[0][2] > 0 and argmax(s1s2[0]) == 2 and s2s1[0][2] > 0 and argmax(s2s1[0]) == 2)

mrpc = [] # [quality, id1, id2, s1, s2]
with open('mrpc/msr_paraphrase_train.txt', 'r') as f:
    for l in f.readlines()[1:]:
        mrpc.append(l.split('\t'))

ppnmt = [] # [quality, id1, id2, s1, s2]
with open('ppnmt/czeng_test_engeng.txt', 'r') as f:
    for l in f.readlines()[1:]:
        ppnmt.append(l.split('\t'))

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'redis'
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["DEBUG"] = True
app.config['SECRET_KEY'] = b'\xb3\xb6\x02\x08E\\\xcb.\x13b(\x0f\xfb\x15\xcf\xc5'
Session(app)
    
letters_digits = string.ascii_uppercase + string.digits

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def init():
    session['token'] = ''.join((choice(letters_digits) for i in range(10)))
    session['final_amt'] = 0.0
    session['sentence'] = None
    print(session['token'])
    print(session['final_amt'])
    return start()

@app.route('/start', methods=['GET', 'POST'])
@cross_origin()
def start():
    print('in start')
    session['dataset'] = choice(['mrpc', 'ppnmt'])
    if session['dataset'] == 'mrpc':
        session['sentence_index'] = randrange(len(mrpc))
        session['sentence'] = str(mrpc[session['sentence_index']][3])
    else:
        session['sentence_index'] = randrange(len(ppnmt))
        session['sentence'] = str(ppnmt[session['sentence_index']][1])
    print(session['sentence'])
    print(session['token'])
    print(session['final_amt'])
    return render_template("form.html", data=session)

@app.route('/check', methods=['POST'])
@cross_origin()
def check_candidate():
    session['candidate'] = request.form.get('candidate').strip()
    print(session['token'])
    print("Candidate:", str(session['candidate']))
    if session['candidate'] == session['sentence']:
        session['dollars'] = 0
    else:
        bleurtscore = (bleurt_scorer.score([session['sentence']], [session['candidate']])[0] + bleurt_scorer.score([session['candidate']], [session['sentence']])[0]) / 2
        print("BLEURT:", str(bleurtscore))
        miscore = get_mi_score([session['sentence']], [session['candidate']])
        print("MI:", str(miscore))
        # session['dollars'] = round(max(0, (miscore - (1 / (1 + exp(-bleurtscore)))) / 2), 2)
        session['dollars'] = round(0.5 / ((1+exp(5*bleurtscore))**2), 2) if miscore else 0
        print("Dollars:", str(session['dollars']))
        with open('sentences/checks', 'a+') as f:
            f.write('\t'.join([session['token'], str(time()), session['dataset'], str(session['sentence_index']), session['sentence'], session['candidate'], str(bleurtscore), str(miscore), str(session['dollars'])]) + '\n')
    return dict(session)

@app.route('/submit', methods=['POST'])
@cross_origin()
def submit_candidate():
    candidate = request.form.get('candidate').strip()
    print(session['token'])
    print("Candidate:", str(candidate))
    print("Sentence:", str(session['sentence']))
    if session['candidate'] == session['sentence']:
        session['dollars'] = 0
    else:
        if session['dataset'] == 'mrpc':
            del mrpc[session['sentence_index']]
        else:
            del ppnmt[session['sentence_index']]
        bleurtscore = (bleurt_scorer.score([session['sentence']], [candidate])[0] + bleurt_scorer.score([candidate], [session['sentence']])[0]) / 2
        miscore = get_mi_score([session['sentence']], [candidate])
        # session['dollars'] = round(max(0, (miscore - (1 / (1 + exp(-bleurtscore)))) / 2), 2)
        session['dollars'] = round(0.5 / ((1+exp(5*bleurtscore))**2), 2) if miscore else 0
        with open('sentences/submits', 'a+') as f:
            f.write('\t'.join([session['token'], str(time()), session['dataset'], str(session['sentence_index']), session['sentence'], session['candidate'], str(bleurtscore), str(miscore), str(session['dollars'])]) + '\n')
        session['final_amt'] += session['dollars']
    if session['final_amt'] >= 10:
        return end()
    return start()

@app.route('/end', methods=['GET', 'POST'])
@cross_origin()
def end():
    print('in end')
    print(session['token'])
    print(session['final_amt'])
    if session['final_amt'] < 1:
        session['final_amt'] = 0
    with open('sentences/ends', 'a+') as f:
        f.write('\t'.join([session['token'], str(session['final_amt'])]) + '\n')
    return render_template("end.html", data=session)

# app.run(host='0.0.0.0')
serve(app, host='0.0.0.0', port=5000)