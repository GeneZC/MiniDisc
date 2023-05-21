#!/bin/bash

python run_sparsification_cls.py \
    --model_type sparseenct5_cls \
    --teacher_model_path path/to/t5_3b_mrpc \
    --task_name mrpc \
    --data_type enct5_cls \
    --max_length 128 \
    --per_device_eval_batch_size 16
