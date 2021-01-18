from tqdm import tqdm
import pandas as pd

n = 3  # number of files (threads)
data_dir = "nap/twitterppdb1/twitterppdb1"
# op = open(data_dir, "w+")
opg = open(data_dir + "-graph", "w+")
# op1 = open(data_dir + "-apt", "w+")
# op2 = open(data_dir + "-mi", "w+")
# op3 = open(data_dir + "-nmi", "w+")
# op.write("\t".join(["sentence", "paraphrase", "mi"]) + "\n")
opg.write("\t".join(["sentence", "paraphrase", "mi", "apt", "bleurt"]) + "\n")
# op1.write("\t".join(["sentence", "paraphrase", "mi"]) + "\n")
# op2.write("\t".join(["sentence", "paraphrase", "mi"]) + "\n")
# op3.write("\t".join(["sentence", "paraphrase", "mi"]) + "\n")
# apt, mi, nmi = set(), set(), set()
# apt_count, mi_count, nmi_count = 0, 0, 0
for i in range(n):
    with open(data_dir + "-apt" + str(i), "r") as f:
        for l in tqdm(f.readlines()[1:]):
            l = l.strip().split("\t")
            # op.write("\t".join([l[0], l[1], l[3]]) + "\n")
            # op1.write("\t".join([l[0], l[1], l[3]]) + "\n")
            opg.write("\t".join([l[0], l[1], '1', '1', l[2]]) + "\n")
            # apt_count += 1
            # apt.add(l[0])
    with open(data_dir + "-mi" + str(i), "r") as f:
        for l in tqdm(f.readlines()[1:]):
            l = l.strip().split("\t")
            # op.write("\t".join([l[0], l[1], l[3]]) + "\n")
            # op2.write("\t".join([l[0], l[1], l[3]]) + "\n")
            opg.write("\t".join([l[0], l[1], '1', '0', l[2]]) + "\n")
            # mi_count += 1
            # mi.add(l[0])
    with open(data_dir + "-nmi" + str(i), "r") as f:
        for l in tqdm(f.readlines()[1:]):
            l = l.strip().split("\t")
            # op.write("\t".join([l[0], l[1], l[3]]) + "\n")
            # op3.write("\t".join([l[0], l[1], l[3]]) + "\n")
            opg.write("\t".join([l[0], l[1], '0', '0', l[2]]) + "\n")
            # nmi_count += 1
            # nmi.add(l[0])
# print(apt_count, mi_count, nmi_count, apt_count + mi_count + nmi_count, len(apt), len(mi), len(nmi), len(apt.union(mi)), len(apt.union(mi).union(nmi)))
