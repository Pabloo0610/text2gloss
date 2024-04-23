nohup spm_train --input './data/20kdata/vocab_gloss.txt' \
--input_format text \
--model_prefix gloss_vocab_test \
--model_type bpe \
--vocab_size 8000 \
--character_coverage 0.9995 \
--num_threads 4 \
--split_digits True \
--byte_fallback True \
--max_sentence_length 24000 > add_vocab_test.log &