#!/bin/bash

python run_maxidisc_cls.py \
    --model_type ${1} \
    --teacher_model_path path/to/sparset5_3b_qqp \
    --task_name qqp \
    --data_type enct5_cls \
    --selection_metric acc_and_f1 \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 0.0001 \
    --weight_decay 1e-2 \
    --log_interval 3000 \
    --num_train_epochs 10 \
    --num_patience_epochs 5 \
    --warmup_proportion 0.1 \
    --max_grad_norm 5.0 \
    --seed 776 \
    --do_train \
    --do_infer \
    --layer_or_sparsity_path ${2} \
    --model_suffix ${3}
