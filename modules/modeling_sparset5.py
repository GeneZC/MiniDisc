# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch SparseT5 model. """


import copy
import math
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from transformers.activations import ACT2FN
from transformers.file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
)
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.modeling_t5 import load_tf_weights_in_t5, T5Config, T5PreTrainedModel


logger = logging.get_logger(__name__)


def round_to_multiple_of_eight(input_size):
    return round(input_size * 1.0 / 8) * 8


class SparseEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(SparseEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embedding_dim_origin = embedding_dim
        self.embedding_dim_sparsified = embedding_dim

    def forward(self, input):
        us = torch.matmul(self.u[:, :self.embedding_dim_sparsified], torch.diag(self.s[:self.embedding_dim_sparsified]))
        vh = self.vh[:self.embedding_dim_sparsified]
        return F.linear(
                F.embedding(
                input, us, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse),
                vh.t(),
                False
            )

    def reorder(self, indices):
        indices = indices.to(self.weight.device)
        weight = self.weight.index_select(1, indices).clone().detach()
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True

    def sparsify(self, num_elements):
        self.embedding_dim_sparsified = self.embedding_dim_origin - num_elements

    def densify(self):
        self.embedding_dim = self.embedding_dim_sparsified
        weight = self.weight[:, :self.embedding_dim_sparsified].clone().detach()
        self.weight = nn.Parameter(torch.empty_like(weight))
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, element_size, bias=True, sparse_dim=0):
        super(SparseLinear, self).__init__(in_features, out_features, bias=bias)
        self.in_features_origin = in_features
        self.out_features_origin = out_features
        self.in_features_sparsified = in_features
        self.out_features_sparsified = out_features
        self.element_size = element_size
        self.sparse_dim = sparse_dim

    def forward(self, input):
        weight = self.weight[:self.out_features_sparsified, :self.in_features_sparsified]
        if self.bias is not None:
            bias = self.bias[:self.out_features_sparsified]
        else:
            bias = self.bias
        return F.linear(input, weight, bias)

    def reorder(self, indices):
        indices = indices.to(self.weight.device)
        weight = self.weight.index_select(1 - self.sparse_dim, indices).clone().detach()
        if self.bias is not None:
            if self.sparse_dim == 0:
                bias = self.bias.clone().detach()
            else:
                bias = self.bias[indices].clone().detach()
        #self.weight = nn.Parameter(torch.empty_like(weight))
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True
        if self.bias is not None:
            #self.bias = nn.Parameter(torch.empty_like(bias))
            self.bias.requires_grad = False
            self.bias.copy_(bias.contiguous())
            self.bias.requires_grad = True

    def sparsify(self, num_elements):
        if self.sparse_dim == 0:
            self.in_features_sparsified = self.in_features_origin - round_to_multiple_of_eight(num_elements * self.element_size)
        if self.sparse_dim == 1:
            self.out_features_sparsified = self.out_features_origin - round_to_multiple_of_eight(num_elements * self.element_size)

    def densify(self):
        self.in_features = self.in_features_sparsified
        self.out_features = self.out_features_sparsified
        weight = self.weight[:self.out_features_sparsified, :self.in_features_sparsified].clone().detach()
        if self.bias is not None:
            bias = self.bias[:self.out_features_sparsified].clone().detach()
        self.weight = nn.Parameter(torch.empty_like(weight))
        self.weight.requires_grad = False
        self.weight.copy_(weight.contiguous())
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias = nn.Parameter(torch.empty_like(bias))
            self.bias.requires_grad = False
            self.bias.copy_(bias.contiguous())
            self.bias.requires_grad = True


class SparseT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the SparseT5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states


class SparseT5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        # self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.wi = SparseLinear(config.d_model, config.d_ff, 1, bias=False, sparse_dim=1)
        self.wo = SparseLinear(config.d_ff, config.d_model, 1, bias=False, sparse_dim=0)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, layer_neuron_mask=None):
        if self.wo.in_features_sparsified >= 8:
            hidden_states = self.wi(hidden_states)
            hidden_states = F.relu(hidden_states)
            if layer_neuron_mask is not None:
                hidden_states = hidden_states * layer_neuron_mask
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.wo(hidden_states)
        else:
            hidden_states = torch.zeros_like(hidden_states)
        return hidden_states


class SparseT5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.wi_0 = SparseLinear(config.d_model, config.d_ff, 1, bias=False, sparse_dim=1)
        self.wi_1 = SparseLinear(config.d_model, config.d_ff, 1, bias=False, sparse_dim=1)
        self.wo = SparseLinear(config.d_ff, config.d_model, 1, bias=False, sparse_dim=0)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states, layer_neuron_mask=None):
        if self.wo.in_features_sparsified >= 8:
            hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
            hidden_linear = self.wi_1(hidden_states)
            hidden_states = hidden_gelu * hidden_linear
            if layer_neuron_mask is not None:
                hidden_states = hidden_states * layer_neuron_mask
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.wo(hidden_states)
        else:
            hidden_states = torch.zeros_like(hidden_states)
        return hidden_states


class SparseT5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = SparseT5DenseReluDense(config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = SparseT5DenseGatedGeluDense(config)
        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )

        self.layer_norm = SparseT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    # def forward(self, hidden_states, layer_neuron_mask=None):
    #     forwarded_states = self.layer_norm(hidden_states)
    #     forwarded_states = self.DenseReluDense(forwarded_states, layer_neuron_mask=layer_neuron_mask)
    #     hidden_states = hidden_states + self.dropout(forwarded_states)
    #     return hidden_states

    def _forward(self, hidden_states, layer_neuron_mask=None):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states, layer_neuron_mask=layer_neuron_mask)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

    def forward(self, hidden_states, layer_neuron_mask=None):
        # many t5/mt5 models are trained in bfloat16 and don't do well under mixed precision (fp16).
        # It appears that it's enough to disable autocast for this FF layer to avoid inf/nan
        # problems for the whole model
        if torch.is_autocast_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(hidden_states, layer_neuron_mask=layer_neuron_mask)
        else:
            return self._forward(hidden_states, layer_neuron_mask=layer_neuron_mask)


class SparseIndex(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size_origin = size
        self.size_sparsified = size
        self.indices = nn.Parameter(torch.arange(size).float()) # There should be no gradients for the indices.

    def forward(self,):
        return self.indices[:self.size_sparsified].long()

    def reorder(self, indices):
        indices = indices.to(self.indices.device)
        self.indices.requires_grad = False
        self.indices.copy_(indices.contiguous())
        self.indices.requires_grad = True

    def sparsify(self, num_elements):
        self.size_sparsified = self.size_origin - num_elements

    def densify(self):
        indices = self.indices[:self.size_sparsified].clone().detach()
        self.indices = nn.Parameter(torch.empty_like(indices))
        self.indices.requires_grad = False
        self.indices.copy_(indices.contiguous())
        self.indices.requires_grad = True


class SparseT5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.n_heads_sparsified = self.n_heads
        self.inner_dim_sparsified = self.inner_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        # self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        self.q = SparseLinear(self.d_model, self.inner_dim, self.key_value_proj_dim, bias=False, sparse_dim=1)
        self.k = SparseLinear(self.d_model, self.inner_dim, self.key_value_proj_dim, bias=False, sparse_dim=1)
        self.v = SparseLinear(self.d_model, self.inner_dim, self.key_value_proj_dim, bias=False, sparse_dim=1)
        self.o = SparseLinear(self.inner_dim, self.d_model, self.key_value_proj_dim, bias=False, sparse_dim=0)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
            # self.relative_attention_bias = SparseEmbedding(self.relative_attention_num_buckets, self.n_heads)
        
        self.head_indices = SparseIndex(self.n_heads)
        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value_state=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value_state[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value_state is not None:
            assert (
                len(past_key_value_state) == 2
            ), f"past_key_value_state should have 2 past states: keys and values. Got { len(past_key_value_state)} past states"
            real_seq_length += past_key_value_state[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=hidden_states.device, dtype=hidden_states.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value_state is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        self.inner_dim_sparsified = self.o.in_features_sparsified
        self.n_heads_sparsified = int(self.inner_dim_sparsified / self.key_value_proj_dim)

        if self.o.in_features_sparsified >= 8:
            def shape(states):
                """projection"""
                return states.view(batch_size, -1, self.n_heads_sparsified, self.key_value_proj_dim).transpose(1, 2)

            def unshape(states):
                """reshape"""
                return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim_sparsified)

            def project(hidden_states, proj_layer, key_value_states, past_key_value_state):
                """projects hidden states correctly to key/query states"""
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(hidden_states))
                elif past_key_value_state is None:
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))

                if past_key_value_state is not None:
                    if key_value_states is None:
                        # self-attn
                        # (batch_size, n_heads, key_length, dim_per_head)
                        hidden_states = torch.cat([past_key_value_state, hidden_states], dim=2)
                    else:
                        # cross-attn
                        hidden_states = past_key_value_state
                return hidden_states

            # get query states
            query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

            # get key/value states
            key_states = project(
                hidden_states, self.k, key_value_states, past_key_value_state[0] if past_key_value_state is not None else None
            )
            value_states = project(
                hidden_states, self.v, key_value_states, past_key_value_state[1] if past_key_value_state is not None else None
            )

            # compute scores
            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

            scores += position_bias.index_select(1, self.head_indices())
            attn_weights = F.softmax(scores.float(), dim=-1).type_as(
                scores
            )  # (batch_size, n_heads, seq_length, key_length)
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
            )  # (batch_size, n_heads, seq_length, key_length)

            attn_output = torch.matmul(attn_weights, value_states)
            # Mask heads if we want to
            if layer_head_mask is not None:
                attn_output = attn_output * layer_head_mask
            attn_output = unshape(attn_output)  # (batch_size, seq_length, dim)
            attn_output = self.o(attn_output)
        else:
            key_states, value_states = None, None
            attn_weights = None
            attn_output = torch.zeros_like(hidden_states)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class SparseT5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.has_relative_attention_bias = has_relative_attention_bias
        self.SelfAttention = SparseT5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = SparseT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value_state=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class SparseT5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = SparseT5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = SparseT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value_state=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class SparseT5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(SparseT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(SparseT5LayerCrossAttention(config))

        self.layer.append(SparseT5LayerFF(config))

    def reorder_heads(self, indices):
        self.layer[0].SelfAttention.head_indices.reorder(indices)
        n, h = self.layer[0].SelfAttention.n_heads, self.layer[0].SelfAttention.key_value_proj_dim
        indices = torch.arange(n * h).reshape(n, h)[indices].reshape(-1).contiguous().long()
        self.layer[0].SelfAttention.q.reorder(indices)
        self.layer[0].SelfAttention.k.reorder(indices)
        self.layer[0].SelfAttention.v.reorder(indices)
        self.layer[0].SelfAttention.o.reorder(indices)

    def reorder_neurons(self, indices):
        if hasattr(self.layer[-1].DenseReluDense, "wi"):
            self.layer[-1].DenseReluDense.wi.reorder(indices)
        else:
            self.layer[-1].DenseReluDense.wi_0.reorder(indices)
            self.layer[-1].DenseReluDense.wi_1.reorder(indices)
        self.layer[-1].DenseReluDense.wo.reorder(indices)

    def reorder_cross_heads(self, indices):
        assert self.is_decoder, "`reorder_cross_heads` can only be applied to decoder."
        self.layer[1].EncDecAttention.head_indices.reorder(indices)
        n, h = self.layer[1].EncDecAttention.n_heads, self.layer[1].EncDecAttention.key_value_proj_dim
        indices = torch.arange(n * h).reshape(n, h)[indices].reshape(-1).contiguous().long()
        self.layer[1].EncDecAttention.q.reorder(indices)
        self.layer[1].EncDecAttention.k.reorder(indices)
        self.layer[1].EncDecAttention.v.reorder(indices)
        self.layer[1].EncDecAttention.o.reorder(indices)

    def sparsify_heads(self, num_elements):
        self.layer[0].SelfAttention.head_indices.sparsify(num_elements)
        self.layer[0].SelfAttention.q.sparsify(num_elements)
        self.layer[0].SelfAttention.k.sparsify(num_elements)
        self.layer[0].SelfAttention.v.sparsify(num_elements)
        self.layer[0].SelfAttention.o.sparsify(num_elements)

    def sparsify_neurons(self, num_elements):
        if hasattr(self.layer[-1].DenseReluDense, "wi"):
            self.layer[-1].DenseReluDense.wi.sparsify(num_elements)
        else:
            self.layer[-1].DenseReluDense.wi_0.sparsify(num_elements)
            self.layer[-1].DenseReluDense.wi_1.sparsify(num_elements)
        self.layer[-1].DenseReluDense.wo.sparsify(num_elements)

    def sparsify_cross_heads(self, num_elements):
        assert self.is_decoder, "`sparsify_cross_heads` can only be applied to decoder."
        self.layer[1].EncDecAttention.head_indices.sparsify(num_elements)
        self.layer[1].EncDecAttention.q.sparsify(num_elements)
        self.layer[1].EncDecAttention.k.sparsify(num_elements)
        self.layer[1].EncDecAttention.v.sparsify(num_elements)
        self.layer[1].EncDecAttention.o.sparsify(num_elements)
        
    def densify(self):
        self.layer[0].SelfAttention.head_indices.densify()
        self.layer[0].SelfAttention.q.densify()
        self.layer[0].SelfAttention.k.densify()
        self.layer[0].SelfAttention.v.densify()
        self.layer[0].SelfAttention.o.densify()
        if hasattr(self.layer[-1].DenseReluDense, "wi"):
            self.layer[-1].DenseReluDense.wi.densify()
        else:
            self.layer[-1].DenseReluDense.wi_0.densify()
            self.layer[-1].DenseReluDense.wi_1.densify()
        self.layer[-1].DenseReluDense.wo.densify()
        if self.is_decoder:
            self.layer[1].EncDecAttention.head_indices.densify()
            self.layer[1].EncDecAttention.q.densify()
            self.layer[1].EncDecAttention.k.densify()
            self.layer[1].EncDecAttention.v.densify()
            self.layer[1].EncDecAttention.o.densify()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        layer_neuron_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value_state=None,
        use_cache=False,
        output_attentions=False,
    ):
        # HACK: weight and bias will be set to nearly 0 elements when the sparsity is nearly 0.
        # This manner is somehow fine for both sparsification and densification ; ).
        # However, we do some tricks here to avoid carrying out actual computation here,
        # which will result in errors.

        if past_key_value_state is not None:
            assert self.is_decoder, "Only decoder can use `past_key_value_states`"
            expected_num_past_key_value_states = 2 if encoder_hidden_states is None else 4

            if len(past_key_value_state) != expected_num_past_key_value_states:
                raise ValueError(
                    f"There should be {expected_num_past_key_value_states} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_value_states == 4 else ''}"
                    f"Got {len(past_key_value_state)} past key / value states"
                )

            self_attn_past_key_value_state = past_key_value_state[:2]
            cross_attn_past_key_value_state = past_key_value_state[2:]
        else:
            self_attn_past_key_value_state, cross_attn_past_key_value_state = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value_state=self_attn_past_key_value_state,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value_state=cross_attn_past_key_value_state,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, layer_neuron_mask=layer_neuron_mask)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class SparseT5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [SparseT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = SparseT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        if not hasattr(self.config, "sparsity"):
            self.config.sparsity = "0"
        if self.is_decoder:
            if not hasattr(self.config, "dec_sparsity_map"):
                self.config.sparsity_map = {"0": {"head": {}, "neuron": {}, "cross_head": {}}}
            else:
                self.config.sparsity_map = self.config.dec_sparsity_map
        else:
            if not hasattr(self.config, "sparsity_map"):
                self.config.sparsity_map = {"0": {"head": {}, "neuron": {}}}
        self.sparsify(self.config.sparsity)
        self.densify()

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        neuron_mask=None,
        cross_attn_head_mask=None,
        past_key_value_states=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_value_states[0][0].shape[2] + seq_length if past_key_value_states is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_value_states with `None` if past does not exist
        if past_key_value_states is None:
            past_key_value_states = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head and neuron mask if needed
        # 1.0 in mask indicate we keep the head or neuron neuron
        # input head_mask has shape [num_hidden_layers x num_heads]
        # input neuron_mask has shape [num_hidden_layers x intermediate_size]
        # and head_mask is converted to shape [num_hidden_layers x batch_size (*1) x num_heads x seq_length (*1) x head_size (*1)]
        # similarly neuron_mask is converted to shape [num_hidden_layers x batch_size (*1) x seq_length (*1) x intermediate_size]
        if head_mask is not None:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).to(self.dtype)
        else:
            head_mask = [None] * self.config.num_layers
        if neuron_mask is not None:
            neuron_mask = neuron_mask.unsqueeze(1).unsqueeze(1).to(self.dtype)
        else:
            neuron_mask = [None] * self.config.num_layers
        if cross_attn_head_mask is not None:
            cross_attn_head_mask = cross_attn_head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).to(self.dtype)
        else:
            cross_attn_head_mask = [None] * self.config.num_layers

        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value_state) in enumerate(zip(self.block, past_key_value_states)):
            layer_head_mask = head_mask[i]
            layer_neuron_mask = neuron_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    layer_neuron_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value_state is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    layer_neuron_mask=layer_neuron_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value_state=past_key_value_state,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states, (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                present_key_value_states,
                all_hidden_states,
                all_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )

    def reorder(self, head_indices, neuron_indices, cross_head_indices=None):
        for layer_idx, indices in head_indices.items():
            self.block[layer_idx].reorder_heads(indices)
        for layer_idx, indices in neuron_indices.items():
            self.block[layer_idx].reorder_neurons(indices)
        if self.is_decoder:
            assert cross_head_indices is not None, "`cross_head_indices` should be set for decoder."
            for layer_idx, indices in cross_head_indices.items():
                self.block[layer_idx].reorder_cross_heads(indices)

    def sparsify(self, sparsity):
        assert sparsity in self.config.sparsity_map, f"Sparsity {sparsity} is not in the sparsity map {self.config.sparsity_map}."
        head_map, neuron_map = self.config.sparsity_map[sparsity]["head"], self.config.sparsity_map[sparsity]["neuron"]
        if self.is_decoder:
            cross_head_map = self.config.sparsity_map[sparsity].get("cross_head", None)
            assert cross_head_map is not None, "`cross_head_map` should be available for decoder."
        for layer_idx in range(self.config.num_layers):
            self.block[layer_idx].sparsify_heads(head_map.get(str(layer_idx), 0))
            self.block[layer_idx].sparsify_neurons(neuron_map.get(str(layer_idx), 0))
            if self.is_decoder:
                self.block[layer_idx].sparsify_cross_heads(cross_head_map.get(str(layer_idx), 0))

    def densify(self):
        for layer_idx in range(self.config.num_layers):
            self.block[layer_idx].densify()

    def count_params(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return round(num_params / 10 ** 6, 2)