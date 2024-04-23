import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# finetune_model_path='../../data/llama2/finetune_test_fullembedding1/checkpoint-2800'
# # 例如: base_model_name_or_path='meta-llama/Llama-2-7b-chat'
#
# embedding_name = 'model.embed_tokens.weight'
# new_embedding_name = 'base_model.model.model.embed_tokens.weight'
# base_model_name_or_path = '../../data/Atom-7B'
# saved_path = os.path.join(finetune_model_path, 'embedding_weight.bin')
tokenizer = AutoTokenizer.from_pretrained("../../data/Atom-7B", use_fast=False)
add_tokens = []
for cnt in range(0,5000):
       s = '/' + str(cnt)
       add_tokens.append(s)
print(add_tokens)
tokenizer.add_tokens(add_tokens)
#model.resize_token_embeddings(len(tokenizer))
tokenizer.save_pretrained("../../data/Atom-7B")
#new_embeddings = model.get_input_embeddings()
#print(new_embeddings)
#print(len(tokenizer))
# v = tokenizer.get_vocab()
# with open('vocab_Atom.txt','w',encoding = 'utf-8') as f:
#      for (token,id) in v.items():
#         f.write(token)
#         f.write(" ")
#         f.write(str(id))
#        f.write('\n')
print(tokenizer.get_vocab())
# model = AutoModelForCausalLM.from_pretrained("../../data/Atom-7B",device_map='cuda',torch_dtype=torch.float16, load_in_8bit=True)
# tokenizer = AutoTokenizer.from_pretrained("../../data/Atom-7B",use_fast=False)
# print(len(tokenizer))
# print(model.get_input_embeddings())
# #if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
# model.resize_token_embeddings(len(tokenizer))
# new_embedding_weight = torch.load(saved_path)
# old_dict = model.state_dict()
#     #print(old_dict)
# new_dict = old_dict.copy()
# new_dict[embedding_name] = new_embedding_weight[new_embedding_name]
# #print(new_dict)
# model.load_state_dict(new_dict)
# model.save_pretrained("../../data/Atom-7B")