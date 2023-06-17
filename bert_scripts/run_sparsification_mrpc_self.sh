#!/bin/bash
cd ..

# CUDA_VISIBLE_DEVICES=0 PORT=10000 
python run_sparsification_cls.py \
    --model_type sparsebert_cls \
    --teacher_model_path 'models_saved/bert-base-mrpc/' \
    --task_name mrpc \
    --data_type bert_cls \
    --max_length 128 \
    --per_device_eval_batch_size 32 

# for teacher_model_path, you only need to specific to datafolder
