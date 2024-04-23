import sentencepiece as spm

sp_bpe = spm.SentencePieceProcessor() 
sp_bpe.load('gloss_vocab_test.model')

print(sp_bpe.encode_as_pieces('人们/拭目-以-待/,/事实/在/哪里（疑惑）/？'))