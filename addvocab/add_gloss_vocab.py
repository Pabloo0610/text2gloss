import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input = '../../data/20kdata/all.txt',
    input_format = 'text',
    model_prefix = 'gloss_vocab_test',
    model_type = 'bpe',
    vocab_size = 8000,
    character_coverage = 0.9995,
    num_threads = 4,
    split_digits = True,
    byte_fallback = True,
)