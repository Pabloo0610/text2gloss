output_model=../../data/t2g_model/t2g_match_r_64
# 需要修改到自己的输入目录
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi

cp ./finetune.sh ${output_model}
deepspeed --include localhost:1,2,3 \finetune_clm_lora_new.py \
    --model_name_or_path ../../data/Atom-7B-Chat \
    --train_files ../../data/20kdata/t2g_train_match.csv \
    --validation_files  ../../data/20kdata/t2g_val_match.csv \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 800 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 15 \
    --warmup_steps 100 \
    --load_in_bits 4 \
    --lora_r 64 \
    --lora_alpha 128 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 400 \
    --eval_steps 200 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 4096 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --deepspeed ds_config_zero2.json \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log
    
# --tokenizer_name ../../addvocab/merged_tokenizer_hf_test\