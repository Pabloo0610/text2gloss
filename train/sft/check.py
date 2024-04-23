import torch
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

#tokenizer = AutoTokenizer.from_pretrained("../../data/llama2/finetune_test/checkpoint-2200", use_fast=False)
tokenizer = AutoTokenizer.from_pretrained("../../data/Atom-7B", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("../../data/Atom-7B")

print(len(tokenizer))
v = tokenizer.get_vocab()
#with open('vocab_new.txt','w',encoding = 'utf-8') as f:
#    for (token,id) in v.items():
#        f.write(token)
#        f.write(" ")
#        f.write(str(id))
#        f.write('\n')
        
#text = "<s>Human: 生成手语动作：你会打手语吗？\n</s><s>Assistant: /399/1974/510/3745/2563\n</s>"


#
print(len(v))
#a = 0
#df = pd.read_csv("../../data/motiontest.csv")
#for ids,rows in df.iterrows():
#	a += 1
#	if a > 10:
#		break
#	input_text = rows['text']
#	encoding = tokenizer(input_text)
#	print(encoding)

	
#tokenized_text = tokenizer.tokenize(text)
#print(tokenized_text)

new_embeddings = model.get_input_embeddings()
print(new_embeddings)

output_dir = '../../data/llama2'
saved_path = os.path.join(output_dir, 'embedding_weight.bin')


# sd = model.state_dict()
embedding_name = 'model.embed_tokens.weight'
# weight_dict = {embedding_name: sd[embedding_name]}
# print(weight_dict)
# torch.save(weight_dict, os.path.join(output_dir, 'embedding_weight.bin'))

new_embedding_weight = torch.load(saved_path)
old_dict = model.state_dict()
new_dict = old_dict.copy()
new_dict[embedding_name] = new_embedding_weight[embedding_name]
print(new_embedding_weight)
model.load_state_dict(new_dict)

