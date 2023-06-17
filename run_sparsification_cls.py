# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import transformers

from tqdm.auto import tqdm

from data import get_reader_class, get_pipeline_class, Dataset
from metrics import get_metric_fn
from models import get_model_class
from utils import add_kwargs_to_config

"""
Sparsification: making a transformer model to its sparsified version.
1) Reorder the heads and neurons for efficient sparsity indexing.
2) Add sparsity map to config for module-wise sparsity.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Sparsify a transformers model on a classification task.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Type of pretrained model, for indexing model class.",   
    )
    parser.add_argument( # We'd better download the model for ease of use.
        "--teacher_model_path",
        type=str,
        required=True,
        help="Path to pretrained model.",    
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The task to train on, for indexing data reader.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        help="Type of formatted data, for indexing data pipeline.",
    )
    parser.add_argument( # {cls}{text_a}这里的{text_b}看起来{mask}好。{sep}
        "--template",
        type=str,
        default="",
        help="Template for constructing the prompt.",
    )
    parser.add_argument( # {"-1": "不", "0": "较", "1": "很"}
        "--verbalizer",
        type=str,
        default="",
        help="Verbalizer for constructing the prompt.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets",
        help="Where to load a glue dataset.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs", 
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation loader.",
    )
    parser.add_argument("--use_cpu", action="store_false", help="Use CPU or not.") # why set to store-ture here, should be store false
    parser.add_argument("--model_suffix", type=str, default="none", help="Model suffix.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    args.output_dir = os.path.join(args.output_dir, f"{args.model_type}_{args.model_suffix}_{args.task_name}")
    os.makedirs(args.output_dir, exist_ok=True)
    args.data_dir = os.path.join(args.data_dir, args.task_name)

    device = torch.device("cpu") if args.use_cpu else torch.device("cuda")

    # Load metric functin and data reader.
    metric_fn = get_metric_fn(args.task_name)
    data_reader = get_reader_class(args.task_name)(args.data_dir)
    label_map, reverse_label_map, num_labels = data_reader.get_label_map()

    # Get classes which shall be used.
    tokenizer_class, config_class, model_class = get_model_class(args.model_type)
    pipeline_class = get_pipeline_class(args.data_type)

    # Sparsification.
    # Load pretrained tokenizer with necessary resizing.
    tokenizer = tokenizer_class.from_pretrained(args.teacher_model_path, use_fast=not args.use_slow_tokenizer)
    
    # Data pipeline.
    data_pipeline = pipeline_class(tokenizer, label_map, args.max_length)

    config = config_class.from_pretrained(args.teacher_model_path)
    add_kwargs_to_config(
        config,
        num_labels=num_labels,
    )
    model = model_class.from_pretrained(
        args.teacher_model_path,
        config=config,
    )
    model = model.to(device)

    dev_examples = data_reader.get_dev_examples()
    dev_examples = data_pipeline.build(dev_examples, template=args.template, verbalizer=args.verbalizer)

    dev_dataset = Dataset(dev_examples, shuffle=False)
    dev_loader = DataLoader(dev_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_pipeline.collate)

    # Sparsifying!
    print("***** Running sparsification (w. sanity check) *****")
    # Set student to sparsified student with dev set.
    if "enct5" in args.model_type:
        num_layers, num_heads, num_neurons = \
            config.num_layers, config.num_heads, config.d_ff
        head_importance = torch.zeros(num_layers + 2, num_heads).to(device)
        head_mask = torch.ones(num_layers + 2, num_heads).to(device)
        head_mask.requires_grad_(True)
        neuron_importance = torch.zeros(num_layers + 1, num_neurons).to(device)
        neuron_mask = torch.ones(num_layers + 1, num_neurons).to(device)
        neuron_mask.requires_grad_(True)
    else:
        num_layers, num_heads, num_neurons = \
            config.num_hidden_layers, config.num_attention_heads, config.intermediate_size
        head_importance = torch.zeros(num_layers, num_heads).to(device)
        head_mask = torch.ones(num_layers, num_heads).to(device)
        head_mask.requires_grad_(True)
        neuron_importance = torch.zeros(num_layers, num_neurons).to(device)
        neuron_mask = torch.ones(num_layers, num_neurons).to(device)
        neuron_mask.requires_grad_(True)
    
    # Compute importance.
    model.eval()
    for batch in dev_loader:
        batch = [v.to(device) for k, v in batch._asdict().items()]
        output = model(batch, head_mask=head_mask, neuron_mask=neuron_mask)
        if output.logit.shape[-1] == 1:
            loss = F.mse_loss(output.logit.squeeze(-1), output.label, reduction="mean")
        else:
            loss = F.cross_entropy(output.logit, output.label, reduction="mean")
        loss.backward()
        head_importance += head_mask.grad.abs().detach()
        neuron_importance += neuron_mask.grad.abs().detach()
        # Clear the gradients in case of potential overflow.
        head_mask.grad = None
        neuron_mask.grad = None
        model.zero_grad()
    
    norm_per_layer = torch.pow(torch.pow(head_importance, 2).sum(-1), 0.5)
    head_importance /= norm_per_layer.unsqueeze(-1) + 1e-17
    norm_per_layer = torch.pow(torch.pow(neuron_importance, 2).sum(-1), 0.5)
    neuron_importance /= norm_per_layer.unsqueeze(-1) + 1e-17
    # Reorder for efficient indexing with module-wise sparsity.
    if "enct5" in args.model_type:
        enc_model = getattr(model, "encoder", model)
        dec_model = getattr(model, "decoder", model)
    else:    
        base_model = getattr(model, model.base_model_prefix, model)
    head_importance, head_indices = torch.sort(head_importance, dim=1, descending=True)
    neuron_importance, neuron_indices = torch.sort(neuron_importance, dim=1, descending=True)
    head_indices = {layer_idx: indices for layer_idx, indices in enumerate(head_indices)}
    neuron_indices = {layer_idx: indices for layer_idx, indices in enumerate(neuron_indices)}
    if "enct5" in args.model_type:
        enc_head_indices = {layer_idx: head_indices[layer_idx] for layer_idx in range(num_layers)}
        enc_neuron_indices = {layer_idx: neuron_indices[layer_idx] for layer_idx in range(num_layers)}
        enc_model.reorder(enc_head_indices, enc_neuron_indices)
        dec_head_indices = {0: head_indices[num_layers]}
        dec_neuron_indices = {0: neuron_indices[num_layers]}
        dec_cross_head_indices = {0: head_indices[num_layers + 1]}
        dec_model.reorder(dec_head_indices, dec_neuron_indices, dec_cross_head_indices)
    else:
        base_model.reorder(head_indices, neuron_indices)

    # Compute module-wise sparsity from overall sparsity.
    if "enct5" in args.model_type:
        head_sort = [
            (layer_idx, head_importance[layer_idx, head_idx].item())
            for layer_idx in range(num_layers + 2)
            for head_idx in range(num_heads)
        ]
        head_sort = sorted(head_sort, key=lambda x: x[1])
        neuron_sort = [
            (layer_idx, neuron_importance[layer_idx, neuron_idx].item())
            for layer_idx in range(num_layers + 1)
            for neuron_idx in range(num_neurons)
        ]
        neuron_sort = sorted(neuron_sort, key=lambda x: x[1])
    else:
        head_sort = [
            (layer_idx, head_importance[layer_idx, head_idx].item())
            for layer_idx in range(num_layers)
            for head_idx in range(num_heads)
        ]
        head_sort = sorted(head_sort, key=lambda x: x[1])
        neuron_sort = [
            (layer_idx, neuron_importance[layer_idx, neuron_idx].item())
            for layer_idx in range(num_layers)
            for neuron_idx in range(num_neurons)
        ]
        neuron_sort = sorted(neuron_sort, key=lambda x: x[1])

    if "t5" in args.model_type: 
        num_total_heads = (num_layers + 2) * num_heads
        num_total_neurons = (num_layers + 1) * num_neurons
    else:
        num_total_heads = num_layers * num_heads
        num_total_neurons = num_layers * num_neurons
    sparsity_map = {str(s): {"head": {}, "neuron": {}} for s in range(0, 100, 10)}
    sparsity_map[str(85)] = {"head": {}, "neuron": {}}
    sparsity_map[str(95)] = {"head": {}, "neuron": {}}
    for sparsity in sparsity_map:
        heads_sparsified = head_sort[:round(float(sparsity) / 100 * num_total_heads)]
        for (layer_idx, _) in heads_sparsified:
            if str(layer_idx) not in sparsity_map[sparsity]["head"]:
                sparsity_map[sparsity]["head"][str(layer_idx)] = 0
            sparsity_map[sparsity]["head"][str(layer_idx)] += 1
        neurons_sparsified = neuron_sort[:round(float(sparsity) / 100 * num_total_neurons)]
        for (layer_idx, _) in neurons_sparsified:
            if str(layer_idx) not in sparsity_map[sparsity]["neuron"]:
                sparsity_map[sparsity]["neuron"][str(layer_idx)] = 0
            sparsity_map[sparsity]["neuron"][str(layer_idx)] += 1

    if "enct5" in args.model_type:
        enc_sparsity_map = {}
        dec_sparsity_map = {}
        for sparsity in sparsity_map:
            enc_sparsity_map[sparsity] = {"head": {}, "neuron": {}}
            dec_sparsity_map[sparsity] = {"head": {}, "neuron": {}, "cross_head": {}}
            for layer_idx in sparsity_map[sparsity]["head"]:
                if int(layer_idx) < num_layers:
                    enc_sparsity_map[sparsity]["head"][layer_idx] = sparsity_map[sparsity]["head"][layer_idx]
                elif int(layer_idx) == num_layers:
                    dec_sparsity_map[sparsity]["head"]["0"] = sparsity_map[sparsity]["head"][layer_idx]
                else:
                    dec_sparsity_map[sparsity]["cross_head"]["0"] = sparsity_map[sparsity]["head"][layer_idx]
            for layer_idx in sparsity_map[sparsity]["neuron"]:
                if int(layer_idx) < num_layers:
                    enc_sparsity_map[sparsity]["neuron"][layer_idx] = sparsity_map[sparsity]["neuron"][layer_idx]
                else:
                    dec_sparsity_map[sparsity]["neuron"]["0"] = sparsity_map[sparsity]["neuron"][layer_idx]
        sparsity_map = enc_sparsity_map

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch = [v.to(device) for k, v in batch._asdict().items()]
            output = model(batch)
            pred, label = output.prediction, output.label
            preds.extend(pred.cpu().numpy().tolist())
            labels.extend(label.cpu().numpy().tolist())

    dev_metric = metric_fn(preds, labels)

    print("***** Finalizing sparsification *****")
    print(f"  Verified dev metric = {dev_metric}")
    print("***** Adding sparsity & sparsity map to config *****")
    config.sparsity = "0"
    config.sparsity_map = sparsity_map
    if "enct5" in args.model_type:
        config.dec_sparsity_map = dec_sparsity_map
    print("***** Saving sparsified model *****")
    save_path = args.output_dir
    tokenizer.save_pretrained(save_path)
    config.save_pretrained(save_path)
    model.save_pretrained(save_path)
    

if __name__ == "__main__":
    main()
