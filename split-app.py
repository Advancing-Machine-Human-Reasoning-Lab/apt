# import pandas as pd
# from random import randint

# sentences = {}
# train = pd.DataFrame(columns=["text_a", "text_b", "labels"])
# test = pd.DataFrame(columns=["text_a", "text_b", "labels"])
# with open("sentences/final_checks", "r") as f:
#     lines = f.readlines()[1:]
#     for i in range(len(lines)):
#         l = lines[i].strip().split("\t")
#         sentences[l[5]] = sentences.get(l[5], randint(0, 3))
#         if sentences[l[5]] == 3:
#             test.loc[i] = [l[5], l[6], int(l[8])]
#         else:
#             train.loc[i] = [l[5], l[6], int(l[8])]
# print(train.shape, test.shape)
# train.to_csv("sentences/train", sep="\t", index=False)
# test.to_csv("sentences/test", sep="\t", index=False)

# COUNT MI AND NON-MI IN APP
# mi, nmi = 0, 0
# with open('app/train', 'r') as f:
#     for line in f.readlines()[1:]:
#         line = line.strip().split('\t')
#         mi += int(line[-1] == '1')
#         nmi += int(line[-1] == '0')
# print(mi, nmi)

# mi, nmi = 0, 0
# with open('app/test', 'r') as f:
#     for line in f.readlines()[1:]:
#         line = line.strip().split('\t')
#         mi += int(line[-1] == '1')
#         nmi += int(line[-1] == '0')
# print(mi, nmi)

# apt, mi, nmi = set(), set(), set()
# apt_count, mi_count, nmi_count = 0, 0, 0
# with open("app/final_checks", "r") as f:
#     for l in f.readlines()[1:]:
#         l = l.strip().split("\t")
#         apt_count += int(l[9] > "0.0")
#         mi_count += int(l[8] == "1")
#         nmi_count += int(l[8] == "0")
#         if l[5] in apt:
#             continue
#         if l[9] > "0.0":
#             apt.add(l[5])
#             mi.discard(l[5])
#             nmi.discard(l[5])
#             continue
#         if l[5] in mi:
#             continue
#         if l[8] == "1":
#             mi.add(l[5])
#             nmi.discard(l[5])
#             continue
#         nmi.add(l[5])
# print(apt_count, mi_count, nmi_count, len(apt), len(mi), len(nmi), len(apt.union(mi)), len(apt.union(mi).union(nmi)))

# data_dir = "app/final_checks"
# opg = open(data_dir + "-graph", "w+")
# opg.write("\t".join(["sentence", "paraphrase", "mi", "apt", "bleurt"]) + "\n")
# with open(data_dir, "r") as f:
#     for l in f.readlines()[1:]:
#         l = l.strip().split("\t")
#         if l[9] > "0.0":
#             opg.write("\t".join([l[5], l[6], '1', '1', l[7]]) + "\n")
#         elif l[8] == "1":
#             opg.write("\t".join([l[5], l[6], '1', '0', l[7]]) + "\n")
#         else:
#             opg.write("\t".join([l[5], l[6], '0', '0', l[7]]) + "\n")

msrp, ppnmt = 0, 0
mrpc = set()
with open("/raid/datasets/msrp/msr_paraphrase_train.txt", "r") as f:
    for l in f.readlines()[1:]:
        mrpc.add(l.split("\t")[3].strip())
s1 = set()
with open('app/test', 'r') as f:
    for line in f.readlines()[1:]:
        s1.add(line.strip().split('\t')[0].strip())
print(len(s1.intersection(mrpc)))
with open('app/train', 'r') as f:
    for line in f.readlines()[1:]:
        s1.add(line.strip().split('\t')[0].strip())
print(len(s1.intersection(mrpc)))