import json
import torch
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util

def text_to_number(text):
    try:
        # 尝试将文本转换为整数
        return int(text)
    except ValueError:
        try:
            # 如果无法转换为整数,则尝试转换为浮点数
            return float(text)
        except ValueError:
            # 如果无法转换为数字,则返回原始文本
            return text

def remove_digits(text):
    text = text_to_number(text)
    if text.isdigit():
        return text
    else:
        new_text = re.sub(r'\d', '', text)
        if text!=new_text:
            print(text)
        return new_text
# model = SentenceTransformer('moka-ai/m3e-base', cache_folder='./weights/embedding') # 第一次需要下载， 可以指定下载位置，默认为用户cache目录
model = SentenceTransformer('../../clip/m3e-base')  # 加载本地模型

# 加载手语词典
with open("../../clip/glosses.json", "r") as f:
    sign_dictionary = json.load(f)

with open('filtered_all.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    
# 获取词典中的词和其对应的编码
dictionary_entries = [entry["Entry"] for entry in sign_dictionary]
#print(dictionary_entries)
dictionary_embeddings = model.encode(dictionary_entries)
# similarities = util.cos_sim(vectors,dictionary_embeddings)
# ind = similarities[0].argmax().item()

def replace_with_closest_sign(sentence):
    # 对句子进行tokenize
    sentence_tokens = sentence.split("#")
    sentence_tokens = [remove_digits(text) for text in sentence_tokens]
    # 对句子中的每个词进行编码
    sentence_embeddings = model.encode(sentence_tokens)
    
    # 计算句子中每个词与词典中词的余弦相似度
    similarities = util.cos_sim(sentence_embeddings,dictionary_embeddings)
    
    # 对每个词找到最相似的词典词
    replaced_tokens = []
    for i in range(len(sentence_tokens)):
        closest_index = similarities[i].argmax().item()
        replaced_tokens.append(dictionary_entries[closest_index])
    
    return "#".join(replaced_tokens)



output_file = open("replaced_all.txt","w")

for i in range(0, len(lines), 2):
    output_file.write(lines[i]+'\n')
    new_gloss = replace_with_closest_sign(lines[i+1])
    output_file.write(new_gloss + '\n')
# 示例用法
# sentence = "现在##权力#,坏 #完了1"
# replaced_sentence = replace_with_closest_sign(sentence)
# print(replaced_sentence)