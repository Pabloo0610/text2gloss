import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

# 位置
llama_tokenizer_dir = "../data/Atom-7B" # 换成你自己模型的位置
chinese_sp_model_file ="gloss_vocab_test.model" # 刚才训练的模型

# 加载
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load(chinese_sp_model_file)
llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
chinese_spm = sp_pb2_model.ModelProto()
chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())


# 打印两个词表的大小和原llama的特殊token
print(len(llama_tokenizer),len(chinese_sp_model))
# print(llama_tokenizer.all_special_tokens)
# print(llama_tokenizer.all_special_ids)
# print(llama_tokenizer.special_tokens_map)

llama_spm_tokens_set=set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before:{len(llama_spm_tokens_set)}")
for p in chinese_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New model pieces: {len(llama_spm.pieces)}")

# 保存合并后的模型
output_sp_dir = 'merged_tokenizer_sp_test'
output_hf_dir = 'merged_tokenizer_hf_test'
os.makedirs(output_sp_dir,exist_ok=True)
with open(output_sp_dir+'/chinese_llama.model', 'wb') as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir+'/chinese_llama.model')

tokenizer.save_pretrained(output_hf_dir)
print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")