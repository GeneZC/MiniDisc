#!/bin/bash

python run_sparsification_cls.py \
    --model_type sparsebert_cls \
    --teacher_model_path path/to/bert_base_qqp \
    --task_name qqp \
    --data_type bert_cls \
    --max_length 128 \
    --per_device_eval_batch_size 32
