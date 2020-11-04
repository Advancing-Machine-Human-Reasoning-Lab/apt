import random
from time import time
from math import exp

from bleurt.score import BleurtScorer
from simpletransformers.classification import ClassificationModel
from pandas import DataFrame
from numpy import argmax

from flask import Flask, request, jsonify, session, render_template
from flask_cors import CORS, cross_origin

bleurt_scorer = BleurtScorer('bleurt/bleurt/bleurt-base-128/')
mi_scorer = ClassificationModel('roberta', 'roberta_nli/', use_cuda=False, args = {'reprocess_input_data':True})

def get_mi_score(s1, s2): # returns average of s1 and s2
    _, s1s2, __ = mi_scorer.eval_model(DataFrame({'text_a':s1, 'text_b':s2, 'labels':2}))
    _, s2s1, __ = mi_scorer.eval_model(DataFrame({'text_a':s2, 'text_b':s1, 'labels':2}))
    print(s1s2[0], s2s1[0], argmax(s1s2[0]), argmax(s2s1[0]))
    return (int(s1s2[0][2] > 0 and argmax(s1s2[0]) == 2) + int(s2s1[0][2] > 0 and argmax(s2s1[0]) == 2)) / 2

mrpc = [] # [quality, id1, id2, s1, s2]
with open('mrpc/msr_paraphrase_train.txt', 'r') as f:
    for l in f.readlines()[1:]:
        mrpc.append(l.split('\t'))

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["DEBUG"] = True

sentence = ''

@app.route('/form.html', methods=['GET', 'POST'])
@cross_origin()
def start():
    global sentence
    print('in start')
    sentence = str(random.choice(mrpc)[3])
    print(sentence)
    return sentence

@app.route('/check', methods=['POST'])
@cross_origin()
def check_candidate():
    candidate = request.form.get('candidate')
    print("Candidate:", str(candidate))
    bleurtscore = (bleurt_scorer.score([sentence], [candidate])[0] + bleurt_scorer.score([candidate], [sentence])[0]) / 2
    print("BLEURT:", str(bleurtscore))
    miscore = get_mi_score([sentence], [candidate])
    print("MI:", str(miscore))
    dollars = min(0.5, max(0, miscore - (1 / (1 + exp(-bleurtscore)))))
    print("Dollars:", str(dollars))
    with open('sentences/checks', 'a+') as f:
        f.write('\t'.join([str(time()), sentence, candidate, str(bleurtscore), str(miscore), str(dollars)]) + '\n')
    return {'candidate':candidate, 'bleurtscore':bleurtscore, 'miscore':miscore, 'dollars':dollars}

@app.route('/submit', methods=['POST'])
@cross_origin()
def submit_candidate():
    candidate = request.form.get('candidate')
    print("Candidate:", str(candidate))
    bleurtscore = (bleurt_scorer.score([sentence], [candidate])[0] + bleurt_scorer.score([candidate], [sentence])[0]) / 2
    miscore = get_mi_score([sentence], [candidate])
    dollars = min(0.5, max(0, miscore - (1 / (1 + exp(-bleurtscore)))))
    with open('sentences/submits', 'a+') as f:
        f.write('\t'.join([str(time()), sentence, candidate, str(bleurtscore), str(miscore), str(dollars)]) + '\n')
    return ''

app.run(host='0.0.0.0')