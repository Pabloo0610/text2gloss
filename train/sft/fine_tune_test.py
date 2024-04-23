import torch
#torch.set_printoptions(profile="full")

import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
# 例如: finetune_model_path='FlagAlpha/Llama2-Chinese-7b-Chat-LoRA'
# 2600 / 5200
finetune_model_path='../../data/g2t_model/g2t_w_prompt_test_r_16_chat/checkpoint-5200'
config = PeftConfig.from_pretrained(finetune_model_path)
# 例如: base_model_name_or_path='meta-llama/Llama-2-7b-chat'

# embedding_name = 'model.embed_tokens.weight'
# new_embedding_name = 'base_model.model.model.embed_tokens.weight'
#base_model_name_or_path = '../../data/Atom-7B'
#saved_path = os.path.join(finetune_model_path, 'embedding_weight.bin')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,device_map='auto', load_in_8bit=True)
# if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
#     model.resize_token_embeddings(len(tokenizer))
#     new_embedding_weight = torch.load(saved_path)
#     old_dict = model.state_dict()
#     #print(old_dict)
#     new_dict = old_dict.copy()
#     new_dict[embedding_name] = new_embedding_weight[new_embedding_name]
#     print(new_dict)
#     model.load_state_dict(new_dict)
#     model.save_pretrained(base_model_name_or_path)

model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
model =model.eval()

# new_embedding = model.get_input_embeddings()

# print(type(new_embedding))
# #tensor = torch.LongTensor([2, 31999, 65100, 69808])

# tensors = new_embedding(tensor)
# tensors = tensors.to(device)
# print(tensors)

#print(model)
#new_sd = model.state_dict()
#embd = new_sd[new_embedding_name]
#print(embd.size())
#input_list = ['<s>我让他去银行查询存款情况。\n</s><s>','<s>超市为人们的生活提供了便利。\n</s><s>','<s>我们班的班长是我们学习的榜样。\n</s><s>','<s>他因为犯错而受到妈妈的处罚。\n</s><s>','<s>公司公布了五月份处罚名单，人真多。\n</s><s>']

#print(len(tokenizer))
input_ids = tokenizer(['''<s>Human:手语序列可以划分成不同的片段,每个片段代表了一个token——有独立意义的词,这个词我们就叫做gloss。
而汉语自然语言,我们称之为text。
你是一个手语翻译器。你的任务是将给定的中文gloss序列翻译为对应的中文text。
以下是一些可以帮助你翻译的gloss规则：
代词（例如""你""， ""我""， ""他""）通常会在句首出现。
在谈论某人或某事时，相应的名词或代词通常会先出现，然后再是一系列描述这个名词或代词的动词、形容词。
数字（表示数量、时间）通常会紧跟在相应要计数的名词后面。
形容词和副词通常位于它们修饰的名词或动词的后面。
疑问词（例如""什么"", ""哪""）在问句中通常会放在句尾。
在提出疑问的句子中，""吗""常常在句尾表示疑问。
在否定句中，否定词""不""通常会紧跟在被否定的动词后面。
特征动词（例如'是','在','做','有','去','想','会','叫', '工作', '看', '喝', '写', '炒', '买', '学'） 后接宾语时，全为'#'符号连接。如'你#名字#什么。' 翻译为 '你叫什么名字。'。
下列种类的词语(问题词（如'什么','哪里'），人称代词（'我'，'你'），时间词（'今天','明天','昨天'）,数词（'一','两','三'）,方位词（'这','那','哪'），问句词（'吗'）)与其他词直接接触时，用'#'符号连接。如'你叫什么名字。' 翻译为 '你#名字#什么。'。
当问句词（'吗'）出现在句尾时，语序由S(主语)V(动词)O(宾语)变为O(宾语)S(主语)V(动词)。如'手语#你#会。' 翻译为 '你会打手语吗。'。
动词'是'在否定句中，语序由S(主语)V(动词)O(宾语)变为S(主语)O(宾语)V(动词)。如'我#老师#不是#学生#是。' 翻译为 '我不是老师,我是学生。'。
当句子中同时出现两个动词时，动词'想'和'会'在其它动词前面，语序变为O(宾语)S(主语)动词。如'手语#我#想#学。'翻译为 '我想学手语。'。
当表示时间的词（如'昨天','明天'等）接在动词后面，手语中的词序需要调整。如'明天天气很好，不冷不热。' 翻译为 '明天#天气#好#冷#不#热#不。'。
当句子中出现形容词描述状态时，通常的词序是""主语+形容词+状态""，如'我#身体#不好#天气#热#吃#想#不。' 翻译为 '我身体不太好。天气太热了，不想吃饭。'。
中文中的提问句结构在手语中常常要重新排列，例如手语序列为'你#每天#点#多少#起床？'，对应着'你每天几点起床？'。
对于类似'他是我同学。'的句子，""是""并非删除，而是放在句尾，对应的手语序列为'他#我#同学#是'。
在表达方向和位置关系时，通常的语序是：主体 + 位置词/方向词 + 视野标志/目标，例如'这#杯子#颜色#粉#我#旁边#左。'可以翻译为'左边那个粉色的杯子是我的。'。
动词在一些情况下会被转换为名词，例如'我已经在这里工作两年多了'对应的手语序列为 '我#这#工作#年#2。'。
对于否定句中动词‘能’的使用，语序由S(主语)V(动词)O(宾语)变为S(主语)O(宾语)V(动词)。例如：'我#吃#小#能。' 翻译为 '我能少吃吗？'。
你要翻译的句子是：我#不懂#他#还1#他#伪-装#好？（疑惑）
</s><s>Assistant: 是我看不透他还是他伪装的好？"
"<s>Human:手语序列可以划分成不同的片段,每个片段代表了一个token——有独立意义的词,这个词我们就叫做gloss。
而汉语自然语言,我们称之为text。
你是一个手语翻译器。你的任务是将给定的中文gloss序列翻译为对应的中文text。
以下是一些可以帮助你翻译的gloss规则：
代词（例如""你""， ""我""， ""他""）通常会在句首出现。
在谈论某人或某事时，相应的名词或代词通常会先出现，然后再是一系列描述这个名词或代词的动词、形容词。
数字（表示数量、时间）通常会紧跟在相应要计数的名词后面。
形容词和副词通常位于它们修饰的名词或动词的后面。
疑问词（例如""什么"", ""哪""）在问句中通常会放在句尾。
在提出疑问的句子中，""吗""常常在句尾表示疑问。
在否定句中，否定词""不""通常会紧跟在被否定的动词后面。
特征动词（例如'是','在','做','有','去','想','会','叫', '工作', '看', '喝', '写', '炒', '买', '学'） 后接宾语时，全为'#'符号连接。如'你#名字#什么。' 翻译为 '你叫什么名字。'。
下列种类的词语(问题词（如'什么','哪里'），人称代词（'我'，'你'），时间词（'今天','明天','昨天'）,数词（'一','两','三'）,方位词（'这','那','哪'），问句词（'吗'）)与其他词直接接触时，用'#'符号连接。如'你叫什么名字。' 翻译为 '你#名字#什么。'。
当问句词（'吗'）出现在句尾时，语序由S(主语)V(动词)O(宾语)变为O(宾语)S(主语)V(动词)。如'手语#你#会。' 翻译为 '你会打手语吗。'。
动词'是'在否定句中，语序由S(主语)V(动词)O(宾语)变为S(主语)O(宾语)V(动词)。如'我#老师#不是#学生#是。' 翻译为 '我不是老师,我是学生。'。
当句子中同时出现两个动词时，动词'想'和'会'在其它动词前面，语序变为O(宾语)S(主语)动词。如'手语#我#想#学。'翻译为 '我想学手语。'。
当表示时间的词（如'昨天','明天'等）接在动词后面，手语中的词序需要调整。如'明天天气很好，不冷不热。' 翻译为 '明天#天气#好#冷#不#热#不。'。
当句子中出现形容词描述状态时，通常的词序是""主语+形容词+状态""，如'我#身体#不好#天气#热#吃#想#不。' 翻译为 '我身体不太好。天气太热了，不想吃饭。'。
中文中的提问句结构在手语中常常要重新排列，例如手语序列为'你#每天#点#多少#起床？'，对应着'你每天几点起床？'。
对于类似'他是我同学。'的句子，""是""并非删除，而是放在句尾，对应的手语序列为'他#我#同学#是'。
在表达方向和位置关系时，通常的语序是：主体 + 位置词/方向词 + 视野标志/目标，例如'这#杯子#颜色#粉#我#旁边#左。'可以翻译为'左边那个粉色的杯子是我的。'。
动词在一些情况下会被转换为名词，例如'我已经在这里工作两年多了'对应的手语序列为 '我#这#工作#年#2。'。
对于否定句中动词‘能’的使用，语序由S(主语)V(动词)O(宾语)变为S(主语)O(宾语)V(动词)。例如：'我#吃#小#能。' 翻译为 '我能少吃吗？'。
你要翻译的句子是：关-于#报-复#谣-言？（疑惑）
</s><s>Assistant: '''], return_tensors="pt", add_special_tokens=False).input_ids
# print(input_ids)
if torch.cuda.is_available():
  input_ids = input_ids.to('cuda')
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":10,
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


# df = pd.read_csv("../../data/gloss2text.csv_test_set.csv")
# for ids,rows in df.iterrows():
#    input_text = '<s>'+rows['text'].split('<s>')[1]+'<s>Assistant:'
#    gt_text = rows['text'].split('<s>')[2]
#    gloss = input_text.split('这是你需要翻译的中文gloss序列：')[1]
#    print("input:", gloss)
#    input_ids = tokenizer(input_text, return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')
#    generate_input = {
#        "input_ids":input_ids,
#        "max_new_tokens":512,
#        "do_sample":True,
#        "top_k":50,
#        "top_p":0.95,
#        "temperature":0.3,
#        "repetition_penalty":1.3,
#        "eos_token_id":tokenizer.eos_token_id,
#        "bos_token_id":tokenizer.bos_token_id,
#        "pad_token_id":tokenizer.pad_token_id
#    }
#    generate_ids  = model.generate(**generate_input)
#    text = tokenizer.decode(generate_ids[0])
#    #all = rows['text']
#    # print('text: '+input_text)
#    # print('model:'+text.split('</s>')[1].lstrip('<s>Assistant: '))
#    # print('ground truth:'+all.split('</s>')[1].lstrip('<s>Assistant: '))
#    print("output",text.split('Assistant')[1])
#    print("ground truth:",gt_text)