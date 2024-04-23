import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer('moka-ai/m3e-base', cache_folder='./weights/embedding') # 第一次需要下载， 可以指定下载位置，默认为用户cache目录
model = SentenceTransformer('m3e-base')  # 加载本地模型

# 加载手语词典
with open("glosses.json", "r") as f:
    sign_dictionary = json.load(f)

# 获取词典中的词和其对应的编码
dictionary_entries = [entry["Entry"] for entry in sign_dictionary]
#print(dictionary_entries)
dictionary_embeddings = model.encode(dictionary_entries)
# similarities = util.cos_sim(vectors,dictionary_embeddings)
# ind = similarities[0].argmax().item()

def replace_with_closest_sign(sentence):
    # 对句子进行tokenize
    sentence_tokens = sentence.split("#")
    
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

# 示例用法
sentence = "佣金#问题#散户#遗忘"
replaced_sentence = replace_with_closest_sign(sentence)
print(replaced_sentence)