# -*- coding: utf-8 -*-

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Stack
from modules.modeling_sparset5 import SparseT5Stack

import collections


class FTEncT5Cls(T5PreTrainedModel):
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

        self.layer_num = config.num_layers

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def forward(self, inputs):
        text_indices, text_mask, label = inputs

        batch_size = text_indices.shape[0]
        enc_hidden_states = self.encoder(text_indices, attention_mask=text_mask)[0]
        dec_hidden_states = self.decoder(inputs_embeds=self.prompt.unsqueeze(0).expand(batch_size, -1, -1), encoder_hidden_states=enc_hidden_states, encoder_attention_mask=text_mask)[0]
        logit = self.cls(dec_hidden_states[:, 0])
        
        if logit.shape[-1] == 1:
            prediction = logit.squeeze(-1)
        else:
            prediction = logit.argmax(-1)

        return FTEncT5Cls.Output(
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

        
class FTEncT5Ner(T5PreTrainedModel):
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

        self.layer_num = config.num_layers

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def forward(self, inputs):
        text_indices, text_mask, labels, label_mask = inputs

        batch_size = text_indices.shape[0]
        label_length = labels.shape[1]
        enc_hidden_states = self.encoder(text_indices, attention_mask=text_mask)[0]
        dec_hidden_states = self.decoder(inputs_embeds=self.prompt[:label_length].unsqueeze(0).expand(batch_size, -1, -1), encoder_hidden_states=enc_hidden_states, encoder_attention_mask=text_mask)[0]
        logits = self.cls(dec_hidden_states)

        logit_size = logits.shape[-1]
        mask = label_mask.unsqueeze(-1).expand_as(logits)
        logits = torch.masked_select(logits, mask)
        logits = logits.reshape(-1, logit_size)
        mask = label_mask
        labels = torch.masked_select(labels, mask)
        
        predictions = logit.argmax(-1)

        return FTEncT5Ner.Output(
            logits=logits,
            predictions=predictions, 
            labels=labels,
        )

    @staticmethod
    def loss_fn(t_output, s_output):
        loss = F.cross_entropy(s_output.logits, s_output.labels, reduction="mean")
        return loss