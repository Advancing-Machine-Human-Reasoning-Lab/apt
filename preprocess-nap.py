from tqdm import tqdm
import pandas as pd

n = 2 # number of files (threads)
data_dir = 'nap/msrp1'
op = open(data_dir, "w+")
op1 = open(data_dir + "-apt", "w+")
op2 = open(data_dir + "-mi", "w+")
op3 = open(data_dir + "-nmi", "w+")
op.write('\t'.join(['sentence', 'paraphrase', 'mi'])+'\n')
op1.write('\t'.join(['sentence', 'paraphrase', 'mi'])+'\n')
op2.write('\t'.join(['sentence', 'paraphrase', 'mi'])+'\n')
op3.write('\t'.join(['sentence', 'paraphrase', 'mi'])+'\n')
unique_sentences = set()
for i in range(n):
    with open(data_dir + "-apt" + str(i), "r") as f:
        for l in tqdm(f.readlines()):
            l = l.strip().split("\t")
            op.write('\t'.join([l[0], l[1], l[3]])+'\n')
            op1.write('\t'.join([l[0], l[1], l[3]])+'\n')
            unique_sentences.add(l[0])
    with open(data_dir + "-mi" + str(i), "r") as f:
        for l in tqdm(f.readlines()):
            l = l.strip().split("\t")
            op.write('\t'.join([l[0], l[1], l[3]])+'\n')
            op2.write('\t'.join([l[0], l[1], l[3]])+'\n')
    with open(data_dir + "-nmi" + str(i), "r") as f:
        for l in tqdm(f.readlines()):
            l = l.strip().split("\t")
            op.write('\t'.join([l[0], l[1], l[3]])+'\n')
            op3.write('\t'.join([l[0], l[1], l[3]])+'\n')
print(len(unique_sentences))
