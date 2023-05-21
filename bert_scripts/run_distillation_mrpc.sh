#!/bin/bash

python run_distillation_cls.py \
    --model_type ${1} \
    --teacher_model_path path/to/sparsebert_base_mrpc \
    --layer_or_sparsity ${2} \
    --task_name mrpc \
    --data_type bert_cls \
    --selection_metric acc_and_f1 \
    --max_length 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --weight_decay 1e-2 \
    --log_interval 10 \
    --num_train_epochs 10 \
    --num_patience_epochs 5 \
    --warmup_proportion 0.1 \
    --max_grad_norm 5.0 \
    --seed 776 \
    --do_train \
    --do_infer \
    --model_suffix ${3}
