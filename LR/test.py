import pandas as pd

df = pd.read_csv(r"D:\data\samllMIND\MINDsmall_train\news.tsv", sep='\t')
print(df)
for i in df["nid"]:
    print(i)
    break
