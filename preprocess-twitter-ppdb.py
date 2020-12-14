from tqdm import tqdm

op = open('/raid/datasets/twitter-ppdb/train-preprocessed.txt', 'w+')
with open('/raid/datasets/twitter-ppdb/train.txt', 'r') as f:
    for l in tqdm(f.readlines()):
        l = l.strip().split('\t')
        if l[2][1] == '3':
            continue
        if l[2][1] < '3':
            op.write('\t'.join([l[0], l[1], '0'])+'\n')
        else:
            op.write('\t'.join([l[0], l[1], '1'])+'\n')

op = open('/raid/datasets/twitter-ppdb/test-preprocessed.txt', 'w+')
with open('/raid/datasets/twitter-ppdb/test.txt', 'r') as f:
    for l in tqdm(f.readlines()):
        l = l.strip().split('\t')
        if l[2][1] == '3':
            continue
        if l[2][1] < '3':
            op.write('\t'.join([l[0], l[1], '0'])+'\n')
        else:
            op.write('\t'.join([l[0], l[1], '1'])+'\n')