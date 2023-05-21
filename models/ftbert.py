# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from modules.modeling_sparsebert import SparseBertModel

import collections


class FTBertCls(BertPreTrainedModel):
    Output = collections.namedtuple(
        "Output", 
        (
            "logit",
            "prediction", 
            "label",
        )
    )
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = SparseBertModel(config)
        self.cls = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.num_labels),
        )
        self.init_weights()

    def forward(self, inputs):
        text_indices, text_mask, text_segments, label = inputs

        hidden_states = self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments)[0]
        logit = self.cls(hidden_states[:, 0])

        if logit.shape[-1] == 1:
            prediction = logit.squeeze(-1)
        else:
            prediction = logit.argmax(-1)

        return FTBertCls.Output(
            logit=logit,
            prediction=prediction, 
            label=label,
        )

    @staticmethod
    def loss_fn(t_output, s_output):
        if s_output.logit.shape[-1] == 1:
            loss = F.mse_loss(s_output.logit.squeeze(-1), s_output.label, reduction="mean")
        else:
            loss = F.cross_entropy(s_output.logit, s_output.label, reduction="mean")
        return loss


class FTBertNer(BertPreTrainedModel):
    Output = collections.namedtuple(
        "Output", 
        (
            "logits",
            "predictions", 
            "labels",
        )
    )
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = SparseBertModel(config)
        self.cls = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.num_labels),
        )
        self.init_weights()

    def forward(self, inputs):
        text_indices, text_mask, text_segments, labels, label_mask = inputs

        hidden_states = self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments)[0]
        logits = self.cls(hidden_states)
        
        logit_size = logits.shape[-1]
        mask = label_mask.unsqueeze(-1).expand_as(logits)
        logits = torch.masked_select(logits, mask)
        logits = logits.reshape(-1, logit_size)
        mask = label_mask
        labels = torch.masked_select(labels, mask)

        predictions = logits.argmax(-1)

        return FTBertNer.Output(
            logits=logits,
            predictions=predictions, 
            labels=labels,
        )

    @staticmethod
    def loss_fn(t_output, s_output):
        loss = F.cross_entropy(s_output.logits, s_output.labels, reduction="mean")
        return loss

        
