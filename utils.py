import pandas as pd
from tqdm import tqdm

def read(file, sep='\t', header=True): # filename, dataframe to get output in, sep, header
    df = pd.DataFrame(columns=["text_a", "text_b", "labels"])
    with open(file, 'r') as f:
        lines = f.readlines()[int(header):]
        for i in tqdm(range(len(lines))):
            l = lines[i].strip().split(sep)
            df.loc[i] = [l[0], l[1], int(l[2])]
    return df