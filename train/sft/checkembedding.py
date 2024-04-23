import torch
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

#tokenizer = AutoTokenizer.from_pretrained("../../data/llama2/finetune_test/checkpoint-2200", use_fast=False)
tokenizer = AutoTokenizer.from_pretrained("../../data/Atom-7B-Chat", use_fast=False)
#model = AutoModelForCausalLM.from_pretrained("../../data/Atom-7B")

input_ids = tokenizer("我#是#人类", return_tensors="pt",add_special_tokens=False)
print(input_ids)