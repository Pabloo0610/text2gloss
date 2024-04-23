import torch
import pandas as pd

with open('test_new2_800.txt', 'r') as file:
    lines = file.readlines()
df = pd.read_csv("../../data/20kdata/t2g_test_new2.csv")
with open('test_new2_gt.txt', 'w') as file:
    for ids,rows in df.iterrows():
        if ids>799:
            break
        gt = rows['text'].split('</s><s>Assistant: ')[1].rstrip('</s>')
        #print(gt)
        new_line = lines[ids].rstrip().replace('</s>','').replace('<s>','') + 'Gt:' + gt + '\n'
        file.writelines(new_line)