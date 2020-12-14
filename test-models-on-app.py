'''
COMMAND LINE ARGUMENTS -
1. Model family
2. Model name (or path for a saved model)
3. Path to the directory where predictions should be written
'''

import sys
import os
from numpy import argmax
import pandas as pd
from sklearn.metrics import accuracy_score
from simpletransformers.classification import ClassificationModel

if not os.path.exists(sys.argv[3]):
    os.makedirs(sys.argv[3])

model = ClassificationModel(
    sys.argv[1],
    sys.argv[2],
    num_labels=2,
    use_cuda=True,
    cuda_device=2,
    args={
        'n_gpu':1,
        'op_dir':sys.argv[3],
        'reprocess_input_data':True,
    }
)
test = pd.DataFrame(columns=['text_a', 'text_b', 'labels'])
with open('sentences/final_checks', 'r') as f:
    lines = f.readlines()[1:]
    for i in range(len(lines)):
        l = lines[i].strip().split('\t')
        test.loc[i] = [l[5], l[6], int(l[8])]
print(test.shape)

result, model_outputs, wrong_predictions = model.eval_model(test, acc=accuracy_score)
print(result)
correct, total = 0, 0
with open(sys.argv[3]+'predictions', 'w+') as f:
    f.write(str(result)+'\n\n')
    for i, row in test.iterrows():
        prediction = argmax(model_outputs[i])
        f.write('\t'.join([row['text_a'], row['text_b'], str(row['labels']), str(prediction)])+'\n')
        total += 1