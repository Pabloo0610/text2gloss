import torch
import pandas as pd


#df = pd.read_csv("../../data/20kdata/t2g_test_w_prompt.csv")
with open('../../data/20kdata/yfs_gt.txt','r') as file:
    gt_lines = file.readlines()
with open('yfs_10000.txt', 'r') as file:
    lines = file.readlines()
print(len(lines))
print(len(gt_lines))
with open('yfs_10000_gt.txt', 'w') as file:
    for ids,gt_line in enumerate(gt_lines):
        #gt = rows['text'].split('</s><s>Assistant: ')[1]
        new_line = lines[ids].rstrip().replace('</s>','').replace('<s>','') + 'Gt:' + gt_line.lstrip('#')
        file.writelines(new_line)