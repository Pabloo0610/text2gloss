import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
finetune_model_path='../../data/g2t_model/t2g_w_prompt_r_32'
config = PeftConfig.from_pretrained(finetune_model_path)
#base_model_name_or_path='../../data/Llama2-Chinese-13b-Chat'
#base_model_name_or_path='../../data/Atom-7B-Chat'
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
device_map = "cuda:0" if torch.cuda.is_available() else "auto"
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True)
model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
model =model.eval()

  
  
input_ids = tokenizer(['''<s>Human: 你是一个优秀的手语翻译助手。你的任务是首先理解给定的汉语自然语言文本的含义, 然后将其翻译成对应的手语gloss序列, 遵循以下规则:
1.手语序列可以划分成不同的片段, 每个片段代表了一个有独立意义的词,这个词我们就叫做gloss。gloss序列是由许多gloss组成的,每两个gloss中间用"#"连接。
2.否定词通常要放在句子后面。例如 "你好！好久没看见你了。"翻译为"你#好#长#时间#看你#没有"。
3.手语的疑问词"什么,谁,哪里,多少,为什么"等一般都放在句子末尾。如果是"是不是,要不要,有没有"这类, 一般在句尾翻译为"是,要,有"。例如 "你做什么工作?"翻译为"你#工作#什么"; "你现在有空吗?"翻译为"现在#你#时间#有"。
4.通常情况下, 手语的形容词都需要放在名词后面表修饰。例如"兔子拿着一根长棍子。"翻译为"兔子#拿#棒子#长"。
5.数词修饰名词的时候需要把数词放在名词后标注。例如"我有四条狗。"翻译为"狗#四#我#有"。
6.时间名词需要放在句首。例如"你今天有空吗？"翻译为"今天#你#时间#有"。
7.做宾语的名词要提到动词之前,宾语和主语的位置需要看语料的侧重点判定,通常情况下被强调的成分需要位于句首（大部分情况下是宾语）。例如"女孩吃点心。"翻译为"点心#女孩#吃";"怎么没买件羽绒服？"翻译为"羽绒服#买#没有#为什么"。
8.手语中几乎所有的虚词都需要省略（着,了,过,的,地）;助词"的","地","得"：在视觉中没有特定的物化内容,手语不标注。例如"我的爸爸五十岁了,还在学开车呢。"翻译为"我#爸爸#岁#五十#现在#他#学#开车"。
9.动宾一体。手语中带动词的名词同时可以代替动词使用,比如"乒乓球"也可以指"打乒乓球",名词"篮球"也可以是"打篮球","自行车"也可以指的是"骑自行车"。打这一类手势的时候,动词是省略的。例如"打乒乓球"应直接翻译为"乒乓球",不需要"打"。
10.量词如"本","个","只"一般省略不标注,数词放在句子最后面。例如"我家有五口人。"翻译为"我#家#人#五"。
请翻译以下汉语文本: 银行卡的密码也是这个。 </s><s>Assistant: '''], return_tensors="pt",add_special_tokens=False).input_ids
if torch.cuda.is_available():
  input_ids = input_ids.to('cuda')
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":40,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)