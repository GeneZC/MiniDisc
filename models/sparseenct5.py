# -*- coding: utf-8 -*-

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Stack
from modules.modeling_sparset5 import SparseT5Stack

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


class SparseEncT5Cls(T5PreTrainedModel):
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

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.shared.weight.data.normal_(mean=0.0, std=config.initializer_factor * 1.0)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = SparseT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = 1
        self.decoder = SparseT5Stack(decoder_config)

        self.prompt = nn.Parameter(torch.empty(1, config.d_model))
        self.prompt.data.normal_(mean=0.0, std=config.initializer_factor * 1.0)
        self.cls = nn.Linear(config.d_model, config.num_labels)
        self.cls.weight.data.normal_(mean=0.0, std=config.initializer_factor * ((config.d_model) ** -0.5))
        self.cls.bias.data.zero_()

        self.init_weights()
        self.layer_num = config.num_hidden_layers
        self.layer_skip = int(self.layer_num / 6)

    def forward(self, inputs, head_mask=None, neuron_mask=None):
        text_indices, text_mask, label = inputs

        batch_size = text_indices.shape[0]
        if neuron_mask is None:
            all_enc_hidden_states = self.encoder(text_indices, attention_mask=text_mask, output_hidden_states=True)[1]
            dec_hidden_states = self.decoder(inputs_embeds=self.prompt.unsqueeze(0).expand(batch_size, -1, -1), encoder_hidden_states=all_enc_hidden_states[-1], encoder_attention_mask=text_mask)[0]
        else:
            all_enc_hidden_states = self.encoder(text_indices, attention_mask=text_mask, head_mask=head_mask[:self.layer_num], neuron_mask=neuron_mask[:self.layer_num], output_hidden_states=True)[1]
            dec_hidden_states = self.decoder(inputs_embeds=self.prompt.unsqueeze(0).expand(batch_size, -1, -1), encoder_hidden_states=all_enc_hidden_states[-1], encoder_attention_mask=text_mask, head_mask=head_mask[self.layer_num].unsqueeze(0), neuron_mask=neuron_mask[self.layer_num].unsqueeze(0), cross_attn_head_mask=head_mask[self.layer_num+1].unsqueeze(0))[0] 

        hidden_states = [all_enc_hidden_states[i] for i in list(range(0, self.layer_num + 1, self.layer_skip))] # 0, 6, 12
        # batch_size x num_selected_layers x seq_length x hidden_size
        hidden_states = torch.stack(hidden_states, dim=1)
        logit = self.cls1(dec_hidden_states[:, 0])

        hidden_size = hidden_states.shape[-1]
        mask = text_mask.unsqueeze(1).unsqueeze(-1).expand_as(hidden_states)
        hidden_states = torch.masked_select(hidden_states, mask)
        hidden_states = hidden_states.reshape(-1, hidden_size)

        if logit.shape[-1] == 1:
            prediction = logit.squeeze(-1)
        else:
            prediction = logit.argmax(-1)

        return SparseEncT5Cls.Output(
            hidden_states=hidden_states,
            logit=logit,
            prediction=prediction,
            label=label,
        )

    @staticmethod
    def loss_fn(t_output, s_output, temperature=1.0):
        loss = F.mse_loss(s_output.hidden_states, t_output.hidden_states.detach(), reduction="mean")
        if s_output.logit.shape[-1] == 1:
            loss += F.mse_loss(s_output.logit.squeeze(-1), t_output.logit.squeeze(-1).detach(), reduction="mean")
        else:
            loss += (temperature ** 2) * soft_cross_entropy(s_output.logit / temperature, t_output.logit.detach() / temperature, reduction="mean")
        #if s_output.logit.shape[-1] == 1:
        #    loss2 = F.mse_loss(s_output.logit.squeeze(-1), s_output.label, reduction="mean")
        #else:
        #    loss2 = F.cross_entropy(s_output.logit, s_output.label, reduction="mean")
        loss = loss / 2.0
        return loss  


class SparseEncT5Ner(T5PreTrainedModel):
    Output = collections.namedtuple(
        "Output",
        (   
            "hidden_states",
            "logits",
            "predictions",
            "labels",
        )
    )

    def __init__(self, config):
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.shared.weight.data.normal_(mean=0.0, std=config.initializer_factor * 1.0)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = SparseT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = 1
        self.decoder = SparseT5Stack(decoder_config)

        self.prompt = nn.Parameter(torch.empty(config.n_positions, config.d_model))
        self.prompt.data.normal_(mean=0.0, std=config.initializer_factor * 1.0)
        self.cls = nn.Linear(config.d_model, config.num_labels)
        self.cls.weight.data.normal_(mean=0.0, std=config.initializer_factor * ((config.d_model) ** -0.5))
        self.cls.bias.data.zero_()

        self.init_weights()
        self.layer_num = config.num_hidden_layers
        self.layer_skip = int(self.layer_num / 6)

    def forward(self, inputs, head_mask=None, neuron_mask=None):
        text_indices, text_mask, labels, label_mask = inputs

        batch_size = text_indices.shape[0]
        label_length = labels.shape[1]
        if neuron_mask is None:
            all_enc_hidden_states = self.encoder(text_indices, attention_mask=text_mask, output_hidden_states=True)[1]
            dec_hidden_states = self.decoder(inputs_embeds=self.prompt[:label_length].unsqueeze(0).expand(batch_size, -1, -1), encoder_hidden_states=all_enc_hidden_states[-1], encoder_attention_mask=text_mask)[0]
        else:
            all_enc_hidden_states = self.encoder(text_indices, attention_mask=text_mask, head_mask=head_mask[:self.layer_num], neuron_mask=neuron_mask[:self.layer_num], output_hidden_states=True)[1]
            dec_hidden_states = self.decoder(inputs_embeds=self.prompt[:label_length].unsqueeze(0).expand(batch_size, -1, -1), encoder_hidden_states=all_enc_hidden_states[-1], encoder_attention_mask=text_mask, head_mask=head_mask[self.layer_num].unsqueeze(0), neuron_mask=neuron_mask[self.layer_num].unsqueeze(0), cross_attn_head_mask=head_mask[self.layer_num+1].unsqueeze(0))[0] 

        hidden_states = [all_enc_hidden_states[i] for i in list(range(0, self.layer_num + 1, self.layer_skip))] # 0, 6, 12
        # batch_size x num_selected_layers x seq_length x hidden_size
        hidden_states = torch.stack(hidden_states, dim=1)
        logits = self.cls1(dec_hidden_states)

        logit_size = logits.shape[-1]
        mask = label_mask.unsqueeze(-1).expand_as(logits)
        logits = torch.masked_select(logits, mask)
        logits = logits.reshape(-1, logit_size)
        mask = label_mask
        labels = torch.masked_select(labels, mask)
        hidden_size = hidden_states.shape[-1]
        mask = text_mask.unsqueeze(1).unsqueeze(-1).expand_as(hidden_states)
        hidden_states = torch.masked_select(hidden_states, mask)
        hidden_states = hidden_states.reshape(-1, hidden_size)

        predictions = logits.argmax(-1)

        return SparseEncT5Ner.Output(
            hidden_states=hidden_states,
            logits=logits,
            predictions=predictions,
            labels=labels,
        )

    @staticmethod
    def loss_fn(t_output, s_output, temperature=1.0):
        loss = F.mse_loss(s_output.hidden_states, t_output.hidden_states.detach(), reduction="mean")
        loss += (temperature ** 2) * soft_cross_entropy(s_output.logits / temperature, t_output.logits.detach() / temperature, reduction="mean")
        # loss += F.cross_entropy(s_output.logits, s_output.labels, reduction="mean")
        loss = loss / 2.0
        return loss  
