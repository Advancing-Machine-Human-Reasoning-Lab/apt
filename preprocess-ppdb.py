from tqdm import tqdm

op = open('/raid/datasets/ppdb/train-preprocessed.txt', 'w+')
with open('/raid/datasets/ppdb/ppdb-2.0-xxxl-all', 'r') as f:
    for l in tqdm(f.readlines()):
        l = l.strip().split(' ||| ')
        if l[-1] == 'Equivalence':
            op.write('\t'.join([l[0], l[1], '1'])+'\n')
        else:
            op.write('\t'.join([l[0], l[1], '0'])+'\n')
