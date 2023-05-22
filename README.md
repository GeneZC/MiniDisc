## MiniDisc

This repository contains code for paper titled [MiniDisc: Minimal Distillation Schedule for Language Model Compression](https://arxiv.org/abs/2205.14570).

**************************** **Updates** ****************************

<!-- Thanks for your interest in our repo! -->

* 5/5/23: We released our paper. Check it out!
* 21/5/23: We released our code. Check it out!

## Quick Links

  - [Overview](#overview)
  - [Getting Started](#getting-started)
    - [Requirements](#requirements)
    - [GLUE & CoNLL Data](#glue&conll-data)
    - [Distillation](#distillation)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Overview

Recent studies have uncovered that language model distillation is less effective when facing a large capacity gap between the teacher and the student, and introduced teacher assistant-based distillation to bridge the gap. As a connection, the scale and the performance of the teacher assistant is of vital importance to bring the knowledge from the teacher to the student. However, existing teacher assistant-based methods require maximally many trials before scheduling an optimal teacher assistant. To this end, we propose a minimal distillation schedule (MiniDisc) for scheduling the optimal teacher assistant in minimally one trial. In particular, motivated by the finding that the performance of the student is positively correlated to the scale-performance tradeoff of the teacher assistant, MiniDisc is designed with a λ-tradeoff to measure the optimality of the teacher assistant without trial distillation to the student. MiniDisc then can schedule the optimal teacher assistant with the best λ-tradeoff in a sandwich framework. MiniDisc is evaluated with an extensive set of experiments on GLUE. Experimental results demonstrate the improved efficiency our MiniDisc compared to several state-of-the-art baselines. We further apply MiniDisc to a language model with billions of parameters and show its scalability.

## Getting Started

### Requirements

- PyTorch
- Numpy
- Transformers

### GLUE & CoNLL Data

Download GLUE data through the [link](https://github.com/nyu-mll/jiant/blob/master/scripts/download_glue_data.py), and CoNLL data through another [link](https://www.clips.uantwerpen.be/conll2003/ner/) in exact CoNLL format. Put them to the corresponding directories. For example, MRPC dataset should be placed into `datasets/mrpc`.

:warning: The CoNLL data is not fully tested, but we want to include it for potential interests.

### Distillation

The distillation is achieved in several scripts. We provide example scripts in the followings.

**Finetuning**

We do not provide scripts of finetuning teacher models, but you can find ones in our previous work [StarK](https://github.com/GeneZC/StarK/blob/main/run_finetuning.py), along with finetuned [checkpoints](https://github.com/GeneZC/StarK/blob/main/README.md#training&evaluation). Otherwise, you can also use our code to realize the finetuning by ignoring the existence of teacher models, an example could be `bert_scripts/run_finetuning_conll.sh`.

**Sparsification**

We provide example scripts of sparsifying/pruning finetuned teacher models. The pruned models would be used to initialize the student models. For example, `bert_scripts/run_sparsification_mrpc.sh` is used to prune a teacher model finetuned on MRPC. We explain some key arguments in the following:
* `--model_type`: variant to use, should be `sparsebert_cls` for MRPC. Here, `cls` stands for classification that accords with GLUE, and `ner` stands for named entity recognition instead.
* `--teacher_model_path`: finetuned teacher models to sparsify, should be the path to finetuned checkpoint.
* `--task_name`: task data to use, should align with the data that teacher models are finetuned on and be `mrpc` for MRPC. 
* `--data_type`: task pipeline to use, should align with the `model_type`. For example, `bert_cls` should always be used if the `model_type` is `sparsebert_cls`.

**(Conventional) Distillation**

We provide example scripts of conventionally distilling finetuned teacher models to layer-dropped or parameter-sparsified student models. For example, `bert_scripts/run_distillation_mrpc.sh` is used to distill a teacher model finetuned on MRPC to a properly-initialized (either layer-dropped or parameter-sparsified) student model. We explain some key arguments in following:
* `--model_type`: similar to above.
* `--teacher_model_path`: similar to above.
* `--task_name`: similar to above.
* `--data_type`: similar to above.
* `--selection_metric`: the metric to guide the selection of the best model, should align with the task and be `acc_and_f1` for MRPC.
* `--layer_or_sparsity`: the way to initialize the student, could be a path to a pretrained checkpoint. For example, `4L` to indicate 4 layers should be preserved for the layer-dropped student, and `90S` to indicate 10% parameters should be preserved for the parameter-sparsified student (only when the teacher is prunable, i.e., has been sparsified before).

**MaxiDisc**

We provide example scripts of distilling finetuned teacher models via teacher assistants with maximal efforts. For example, For example, `bert_scripts/run_maxidisc_mrpc.sh` is used to distill a teacher model finetuned on MRPC to a properly-initialized (either layer-dropped or parameter-sparsified) student model via teacher assistants. And you should find the optimal teacher assiatant by many trials. We explain some important arguments in following:
* `--model_type`: similar to above.
* `--teacher_model_path`: similar to above.
* `--task_name`: similar to above.
* `--data_type`: similar to above.
* `--layer_or_sparsity_path`: similar to `layer_or_sparsity`, this gives a sequential distillation path. `6,4L` to indicate the 4-layer student distilled with a 6-layer teacher assistant, and `80,90S` to indicate the 90%-sparsity student distilled with a 80%-sparsity teacher assistant.

**MiniDisc**

We provide example scripts of distilling finetuned teacher models via teacher assistants with minimal efforts. For example, For example, `bert_scripts/run_minidisc_mrpc.sh` is used to distill a teacher model finetuned on MRPC to a properly-initialized (either layer-dropped or parameter-sparsified) student model via teacher assistants. And you should find the optimal teacher assiatant in only one trial. We explain some important arguments in following:
* `--model_type`: similar to above.
* `--teacher_model_path`: similar to above.
* `--task_name`: similar to above.
* `--data_type`: similar to above.
* `--target_iteration`: num of iterations, equals to num of inserted teacher assiatants plus 1, default to `2`, which is fairly enough as discussed in our paper.
* `--target_sparsity`: sparsity of the student, and MiniDisc only supports parameter-sparsified students.
* `--lam`: lambda to use, the value in lambda-tradeoff, default to `0.2`.

:warning: After experiments, we find that the optimal teacher assistants can hardly fall in sparsities smaller than 50%. So we directly truncate the number of teacher assitant candidates according to this obervation, leading to a further speedup in practice. However, we do think this heuristic may not fit for all cases (e.g., large language models) so we do not include it in the paper.

## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Chen (`czhang@bit.edu.cn`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use the code in your work:

```bibtex
@inproceedings{zhang2022minidisc,
   title={MiniDisc: Minimal Distillation Schedule for Language Model Compression},
   author={Zhang, Chen and Yang, Yang and Wang, Qifan and Liu, Jiahao and Wang, Jingang and Xian, Yunsen and Wu, Wei and Song, Dawei},
   booktitle={arXiv},
   year={2022}
}
```

