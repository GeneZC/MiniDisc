# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from modules.modeling_sparsebert import SparseBertModel

import collections


def soft_cross_entropy(input, target, reduction="mean"):
    s_likelihood = F.log_softmax(input, dim=-1)
    t_probability = F.softmax(target, dim=-1)
    cross_entropy = -torch.sum(t_probability * s_likelihood, dim=-1)
    if reduction == "mean":
        cross_entropy = cross_entropy.mean()
    else:
        pass
    return cross_entropy


class PKDBertCls(BertPreTrainedModel):
    Output = collections.namedtuple(
        "Output", 
        (
            "hidden_states",
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
        #self.layer_skip = config.layer_skip
        #self.layer_num = config.num_hidden_layers

    def forward(self, inputs):
        text_indices, text_mask, text_segments, label = inputs

        all_hidden_states = \
            self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments, output_hidden_states=True)[2]

        #hidden_states = [all_hidden_states[i] for i in list(range(0, self.layer_num + 1, self.layer_skip))[1:]] # 6, 12
        hidden_states = [all_hidden_states[-1]]

        # batch_size x num_selected_layers x seq_length x hidden_size
        hidden_states = torch.stack(hidden_states, dim=1)
        logit = self.cls(all_hidden_states[-1][:, 0])

        hidden_size = hidden_states.shape[-1]
        mask = text_mask.unsqueeze(1).unsqueeze(-1).expand_as(hidden_states)
        hidden_states = torch.masked_select(hidden_states, mask)
        hidden_states = hidden_states.reshape(-1, hidden_size)

        if logit.shape[-1] == 1:
            prediction = logit.squeeze(-1)
        else:
            prediction = logit.argmax(-1)

        return PKDBertCls.Output(
            hidden_states=hidden_states,
            logit=logit,
            prediction=prediction,
            label=label,
        )

    @staticmethod
    def loss_fn(t_output, s_output, temperature=1.0):
        loss = F.mse_loss(F.normalize(s_output.hidden_states, p=2, dim=-1), F.normalize(t_output.hidden_states.detach(), p=2, dim=-1), reduction="mean")
        if s_output.logit.shape[-1] == 1:
            loss += F.mse_loss(s_output.logit.squeeze(-1), t_output.logit.squeeze(-1).detach(), reduction="mean")
        else:
            loss += (temperature ** 2) * soft_cross_entropy(s_output.logit / temperature, t_output.logit.detach() / temperature, reduction="mean")
        if s_output.logit.shape[-1] == 1:
            loss += F.mse_loss(s_output.logit.squeeze(-1), s_output.label, reduction="mean")
        else:
            loss += F.cross_entropy(s_output.logit, s_output.label, reduction="mean")
        loss = loss / 3.0
        return loss

        
