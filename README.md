## MiniDisc

This repository contains code for preprint titled [MiniDisc: Minimal Distillation Schedule for Language Model Compression](https://arxiv.org/abs/2205.14570).

**************************** **Updates** ****************************

<!-- Thanks for your interest in our repo! -->

* 5/5/23: We released our paper. Check it out!
* Code is under preparation. Stay tuned.

## Quick Links

  - [Overview](#overview)
  - [Getting Started](#getting-started)
    - [Requirements](#requirements)
    - [GLUE Data](#glue-data)
    - [CoNLL Data](#conll-data)
    - [Training & Evaluation](#training&evaluation)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Overview

Recent studies have uncovered that language model distillation is less effective when facing a large capacity gap between the teacher and the student, and introduced teacher assistant-based distillation to bridge the gap. As a connection, the scale and the performance of the teacher assistant is of vital importance to bring the knowledge from the teacher to the student. However, existing teacher assistant-based methods require maximally many trials before scheduling an optimal teacher assistant. To this end, we propose a minimal distillation schedule (MiniDisc) for scheduling the optimal teacher assistant in minimally one trial. In particular, motivated by the finding that the performance of the student is positively correlated to the scale-performance tradeoff of the teacher assistant, MiniDisc is designed with a λ-tradeoff to measure the optimality of the teacher assistant without trial distillation to the student. MiniDisc then can schedule the optimal teacher assistant with the best λ-tradeoff in a sandwich framework. MiniDisc is evaluated with an extensive set of experiments on GLUE. Experimental results demonstrate the improved efficiency our MiniDisc compared to several state-of-the-art baselines. We further apply MiniDisc to a language model with billions of parameters and show its scalability.

## Getting Started

### Requirements

- PyTorch
- Numpy
- Transformers

### GLUE Data

Get GLUE data through the [link](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py) and put it to the corresponding directory. For example, MRPC dataset should be placed into `datasets/mrpc`.

### CoNLL Data

### Training & Evaluation

<!--
The training and evaluation are achieved in several scripts. We provide example scripts as follows.

**Finetuning**

We provide an example of finetuning `bert-base-uncased` on RTE in `scripts/run_finetuning_rte.sh`. We explain some important arguments in following:
* `--model_type`: Variant to use, should be `ft` in the case.
* `--model_path`: Pretrained language models to start with, should be `bert-base-uncased` in the case and can be others as you like.
* `--task_name`: Task to use, should be chosen from `rte`, `mrpc`, `stsb`, `sst2`, `qnli`, `qqp`, `mnli`, and `mnlimm`.
* `--data_type`: Input format to use, default to `combined`.

**Pruning**

We provide and example of pruning a finetuned checkpoint on RTE in `scripts/run_pruning_rte.sh`. The arguments should be self-contained.

**Distillation**

We provide an example of distilling a finetuned teacher to a layer-dropped or parameter-pruned student on RTE in `scripts/run_distillation_rte.sh`. We explain some important arguments in following:
* `--model_type`: Variant to use, should be `kd` in the case.
* `--teacher_model_path`: Teacher models to use, should be the path to the finetuned teacher checkpoint.
* `--student_model_path`: Student models to initialize, should be the path to the pruned/finetuned teacher checkpoint depending on the way you would like to initialize the student.
* `--student_sparsity`: Student sparsity, should be set if you would like to use parameter-pruned student, e.g., 70. Otherwise, this argument should be left blank.
* `--student_layer`: Student layer, should be set if you would like to use layer-dropped student, e.g., 4.

**Teacher Sparsification**

We provide an example of sparsfying the teacher based on the student on RTE in `scripts/run_sparsification_rte.sh`. We explain some important arguments in following:
* `--model_type`: Variant to use, should be `kd` in the case.
* `--teacher_model_path`: Teacher models to use, should be the path to the finetuned teacher checkpoint.
* `--student_model_path`: Student models to use, should be the path to the distilled student checkpoint.
* `--student_sparsity`: Student sparsity, should be set if you would like to use parameter-pruned student, e.g., 70. Otherwise, this argument should be left blank.
* `--student_layer`: Student layer, should be set if you would like to use layer-dropped student, e.g., 4.
* `--lam`: the knowledgeableness tradeoff term to keep a balance between expressiveness and student-friendliness.

**Rewinding**

We provide an example of rewinding the student on RTE in `scripts/run_rewinding_rte.sh`. We explain some important arguments in following:
* `--model_type`: Variant to use, should be `kd` in the case.
* `--teacher_model_path`: Teacher models to use, should be the path to the sparsified teacher checkpoint.
* `--student_model_path`: Student models to initialize, should be the path to the pruned/finetuned teacher checkpoint depending on the way you would like to initialize the student.
* `--student_sparsity`: Student sparsity, should be set if you would like to use parameter-pruned student, e.g., 70. Otherwise, this argument should be left blank.
* `--student_layer`: Student layer, should be set if you would like to use layer-dropped student, e.g., 4.
* `--lam`: the knowledgeableness tradeoff term to keep a balance between expressiveness and student-friendliness. Here, it is just used for folder names.
! -->

## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Chen (`czhang@bit.edu.cn`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use the code in your work:

```bibtex
@inproceedings{zhang2022minidisc,
   title={MiniDisc: Minimal Distillation Schedule for Language Model Compression},
   author={Chen Zhang, Yang Yang, Qifan Wang, Jiahao Liu, Jingang Wang, Yunsen Xian, Wei Wu, and Dawei Song},
   booktitle={Preprint},
   year={2022}
}
```

