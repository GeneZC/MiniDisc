#!/bin/bash

python run_sparsification_ner.py \
    --model_type sparsebert_ner \
    --teacher_model_path path/to/bert_base_conll \
    --task_name conll \
    --data_type bert_ner \
    --max_length 128 \
    --per_device_eval_batch_size 32
