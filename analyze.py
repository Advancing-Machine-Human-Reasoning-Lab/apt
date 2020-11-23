from pprint import pprint

def get_checks(token):
    ret = []
    is_first = True
    start_time = 0
    with open('sentences/checks', 'r') as f:
        for l in f.readlines():
            l = l.split('\t')
            if l[0] == token:
                if is_first:
                    start_time = float(l[1])
                    is_first = False
                ret.append({'timestamp':float(l[1]) - start_time, 'time_taken':float(l[2]), 'dataset':l[3], 'index':int(l[4]), 'sentence':l[5], 'candidate':l[6], 'bleurt':float(l[7]), 'mi':int(l[8]),  'dollars':float(l[9])})
    return ret

def get_submits(token):
    ret = []
    is_first = True
    start_time = 0
    with open('sentences/submits', 'r') as f:
        for l in f.readlines():
            try:
                l = l.split('\t')
                if l[0] == token:
                    if is_first:
                        start_time = float(l[1])
                        is_first = False
                    ret.append({'timestamp':float(l[1]) - start_time, 'time_taken':float(l[2]), 'dataset':l[3], 'index':int(l[4]), 'sentence':l[5], 'candidate':l[6], 'bleurt':float(l[7]), 'mi':int(l[8]),  'dollars':float(l[9])})
            except:
                print(l)
    return ret

def print_sent_cand(d): # a list of dict of sentences and candidates
    for r in d:
        pprint({'sentence':r['sentence'], 'candidate':r['candidate'], 'bleurt':r['bleurt']})

# person = get_submits('BVZ2UNUUSH')
# print(len(person), person[-1]['timestamp']/60, sum(_['dollars'] for _ in person))
# print_sent_cand(person)

F = open('sentences/final_submits', 'w+')
F.write('\t'.join(['token', 'time', 'duration', 'dataset', 'index', 'sentence', 'candidate', 'BLEURT', 'MI', 'dollar']) + '\n')
with open('sentences/submits', 'r') as f:
    for l in f.readlines():
        l = l.split('\t')
        try:
            q = float(l[1])
            if l[0] != 'NFIPX9BDAS':
                for _ in range(len(l)):
                    if l[_] == 'mrpc':
                        l[_] = 'msrp'
                    if l[_] == 'True':
                        l[_] = '1'
                    if l[_] == 'False':
                        l[_] = '0'
                F.write('\t'.join(l))
        except:
            continue