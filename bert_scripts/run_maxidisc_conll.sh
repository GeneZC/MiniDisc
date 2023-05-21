#!/bin/bash

python run_maxidisc_ner.py \
    --model_type sparsebert_ner \
    --teacher_model_path path/to/sparsebert_base_conll \
    --task_name conll \
    --data_type bert_ner \
    --selection_metric f1 \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --weight_decay 1e-2 \
    --log_interval 100 \
    --num_train_epochs 10 \
    --num_patience_epochs 5 \
    --warmup_proportion 0.1 \
    --max_grad_norm 5.0 \
    --seed 776 \
    --do_train \
    --layer_or_sparsity_path ${1} \
    --model_suffix ${2}
