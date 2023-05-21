# -*- coding: utf-8 -*-

import os
import csv
import json
import collections

import torch

from utils import Logger


logger = Logger()


class DataPipeline:
    def __init__(self, tokenizer, label_map, max_length=None):
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    @staticmethod
    def _truncate_pair(text_a_tokens, text_b_tokens, max_length):
        """Truncate a pair input in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(text_a_tokens) + len(text_b_tokens)
            if total_length <= max_length:
                break
            if len(text_a_tokens) > len(text_b_tokens):
                text_a_tokens.pop()
            else:
                text_b_tokens.pop()

    def build(self, examples, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def _pad(indices, max_length, pad_idx):
        """Pad a sequence to the maximum length."""
        pad_length = max_length - len(indices)
        return indices + [pad_idx] * pad_length

    def collate(self, batch):
        raise NotImplementedError()


# For Bert.
class BertClsDataPipeline(DataPipeline):
    Example = collections.namedtuple(
        "Example", 
        (
            "text_indices",
            "text_segments",
            "label",
        )
    )
    Batch = collections.namedtuple(
        "Batch", 
        (
            "text_indices", 
            "text_mask",
            "text_segments", 
            "label",
        )
    )

    def __init__(self, tokenizer, label_map, max_length=None):
        super().__init__(tokenizer, label_map, max_length)

    def build(self, examples, **kwargs):
        builded_examples = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Converting example %d of %d" % (ex_index, len(examples)))

            label = self.label_map(example.label)
            text_a_tokens = self.tokenizer.tokenize(example.text_a)
            text_b_tokens = None
            if example.text_b:
                text_b_tokens = self.tokenizer.tokenize(example.text_b)
                # Account for [CLS], [SEP], [SEP] with "- 3" for combined input.
                self._truncate_pair(text_a_tokens, text_b_tokens, self.max_length - 3)
                text_tokens = [self.tokenizer.cls_token] + text_a_tokens + [self.tokenizer.sep_token]
                text_segments = [0] * len(text_tokens)
                text_tokens += text_b_tokens + [self.tokenizer.sep_token]
                text_segments += [1] * (len(text_b_tokens) + 1)
                text_indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

                if ex_index < 5:
                    logger.info("*** Example ***")
                    logger.info("uid: %s" % (example.uid))
                    logger.info("text_tokens: %s" % " ".join([str(x) for x in text_tokens]))
                    logger.info("text_indices: %s" % " ".join([str(x) for x in text_indices]))
                    logger.info("text_segments: %s" % " ".join([str(x) for x in text_segments]))
                    logger.info("label: %s (id = %d)" % (example.label, label))

                builded_examples.append(
                    BertClsDataPipeline.Example(
                        text_indices=text_indices,
                        text_segments=text_segments,
                        label=label,
                    )
                )
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(text_a_tokens) > self.max_length - 2:
                    text_a_tokens = text_a_tokens[:(self.max_length - 2)]
                text_tokens = [self.tokenizer.cls_token] + text_a_tokens + [self.tokenizer.sep_token]
                text_segments = [0] * len(text_tokens)
                text_indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

                if ex_index < 5:
                    logger.info("*** Example ***")
                    logger.info("uid: %s" % (example.uid))
                    logger.info("text_tokens: %s" % " ".join([str(x) for x in text_tokens]))
                    logger.info("text_indices: %s" % " ".join([str(x) for x in text_indices]))
                    logger.info("text_segments: %s" % " ".join([str(x) for x in text_segments]))
                    logger.info("label: %s (id = %d)" % (example.label, label))

                builded_examples.append(
                    BertClsDataPipeline.Example(
                        text_indices=text_indices,
                        text_segments=text_segments,
                        label=label,
                    )
                )
        return builded_examples

    def collate(self, batch):
        if self.max_length is None:
            max_length = max([len(example.text_indices) for example in batch])
        else:
            max_length = self.max_length
        
        batch_text_indices = []
        batch_text_mask = []
        batch_text_segments = []
        batch_label = []
        for example in batch:
            text_mask = [1] * len(example.text_indices)
            
            batch_text_indices.append(self._pad(example.text_indices, max_length, self.tokenizer.pad_token_id))
            batch_text_mask.append(self._pad(text_mask, max_length, 0))
            batch_text_segments.append(self._pad(example.text_segments, max_length, 0))
            batch_label.append(example.label)
        return BertClsDataPipeline.Batch(
            text_indices=torch.tensor(batch_text_indices, dtype=torch.long),
            text_mask=torch.tensor(batch_text_mask, dtype=torch.bool),
            text_segments=torch.tensor(batch_text_segments, dtype=torch.long),
            label=torch.tensor(batch_label),
        )


class BertNerDataPipeline(DataPipeline):
    Example = collections.namedtuple(
        "Example", 
        (
            "text_indices",
            "text_segments",
            "labels",
            "label_mask",
        )
    )
    Batch = collections.namedtuple(
        "Batch", 
        (
            "text_indices", 
            "text_mask",
            "text_segments", 
            "labels",
            "label_mask",
        )
    )

    def __init__(self, tokenizer, label_map, max_length=None):
        super().__init__(tokenizer, label_map, max_length)

    def build(self, examples, **kwargs):
        builded_examples = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Converting example %d of %d" % (ex_index, len(examples)))

            text_tokens, labels, label_mask = [], [], []
            for token, label in zip(example.tokens, example.labels):
                token_pieces = self.tokenizer.tokenize(token)
                text_tokens.extend(token_pieces)
                try:
                    tg, tp = label.split("-")
                except:
                    tg, tp = label, None
                if tg == "B":
                    label_pieces = ["B-"+tp] + ["I-"+tp] * (len(token_pieces)-1)
                elif tg == "I":
                    label_pieces = ["I-"+tp] * len(token_pieces)
                else:
                    label_pieces = ["O"] * len(token_pieces)
                labels.extend(label_pieces)
                label_mask.extend([1] + [0] * (len(token_pieces)-1))
            labels = [self.label_map(label) for label in labels]
            
            # Account for [CLS] and [SEP] with "- 2"
            if len(text_tokens) > self.max_length - 2:
                text_tokens = text_tokens[:(self.max_length - 2)]
                labels = labels[:(self.max_length - 2)]
                label_mask = label_mask[:(self.max_length - 2)]
            text_tokens = [self.tokenizer.cls_token] + text_tokens + [self.tokenizer.sep_token]
            labels = [0] + labels + [0]
            label_mask = [0] + label_mask + [0]
            text_segments = [0] * len(text_tokens)
            text_indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("uid: %s" % (example.uid))
                logger.info("text_tokens: %s" % " ".join([str(x) for x in text_tokens]))
                logger.info("text_indices: %s" % " ".join([str(x) for x in text_indices]))
                logger.info("text_segments: %s" % " ".join([str(x) for x in text_segments]))
                logger.info("labels: %s" % " ".join([str(x) for x in labels]))

            builded_examples.append(
                BertNerDataPipeline.Example(
                    text_indices=text_indices,
                    text_segments=text_segments,
                    labels=labels,
                    label_mask=label_mask,
                )
            )
        return builded_examples

    def collate(self, batch):
        if self.max_length is None:
            max_length = max([len(example.text_indices) for example in batch])
        else:
            max_length = self.max_length
        
        batch_text_indices = []
        batch_text_mask = []
        batch_text_segments = []
        batch_labels = []
        batch_label_mask = []
        for example in batch:
            text_mask = [1] * len(example.text_indices)
            
            batch_text_indices.append(self._pad(example.text_indices, max_length, self.tokenizer.pad_token_id))
            batch_text_mask.append(self._pad(text_mask, max_length, 0))
            batch_text_segments.append(self._pad(example.text_segments, max_length, 0))
            batch_labels.append(self._pad(example.labels, max_length, 0)) # 'O' = 0
            batch_label_mask.append(self._pad(example.label_mask, max_length, 0))
        return BertNerDataPipeline.Batch(
            text_indices=torch.tensor(batch_text_indices, dtype=torch.long),
            text_mask=torch.tensor(batch_text_mask, dtype=torch.bool),
            text_segments=torch.tensor(batch_text_segments, dtype=torch.long),
            labels=torch.tensor(batch_labels, dtype=torch.long),
            label_mask=torch.tensor(batch_label_mask, dtype=torch.bool),
        )


class BertPptDataPipeline(DataPipeline):
    Example = collections.namedtuple(
        "Example", 
        (
            "text_indices", 
            "text_segments",
            "mask_position",
            "verbalizer_indices",
            "verbalizer_mask",
            "label",
        )
    )
    Batch = collections.namedtuple(
        "Batch", 
        (
            "text_indices", 
            "text_mask",
            "text_segments", 
            "mask_position",
            "verbalizer_indices",
            "verbalizer_mask",
            "label",
        )
    )

    def __init__(self, tokenizer, label_map, max_length=None):
        super().__init__(tokenizer, label_map, max_length)

    @staticmethod
    def parse_template(template, tokenizer):
        """
            {cls}{text_a}这里的{text_b}看起来{mask}好。{sep}
            => [cls_token, text_a, 这, 里, 的, text_b, 看, 起, 来, mask_token, 好, 。, sep_token]
            {cls}{p0}{text_a}{p1}{p2}{p3]{text_b}{p4}{p5}{p6}{mask}{p7}{sep}
            => [cls_token, p0_token, text_a, p1_token, p2_token, p3_token, text_b, p4_token, p5_token, p6_token, mask_token, p7_token, sep_token]
        """
        template_tokens = []
        insert_positions = []
        is_container = False
        pattern = ""
        for c in template:
            if c == "{":
                if pattern:
                    template_tokens.extend(tokenizer.tokenize(pattern))
                pattern = ""
                is_container = True
            elif c == "}":
                if pattern == "cls":
                    template_tokens.append(tokenizer.cls_token)
                elif pattern == "sep":
                    template_tokens.append(tokenizer.sep_token)
                elif pattern == "mask":
                    template_tokens.append(tokenizer.mask_token)
                elif pattern == "text_a":
                    insert_positions.append(len(template_tokens))
                    template_tokens.append("text_a")
                elif pattern == "text_b":
                    insert_positions.append(len(template_tokens))
                    template_tokens.append("text_b")
                elif pattern.startswith("p"): # pseudo token
                    template_tokens.append(f"[{pattern.upper()}]")
                else:
                    raise ValueError(f"Unkonwn recognized pattern {temp}.")
                pattern = ""
                is_container = False
            else:
                pattern += c
        return template_tokens, insert_positions

    @staticmethod
    def parse_verbalizer(verbalizer, tokenizer):
        """
            {"-1": "不", "0": "较", "1": "很"}
            => {"-1": ["不"], "0": ["较"], "1": ["很"]}
            {"-1": "不好", "0": "还可以", "1": "不错"}
            => {"-1": ["不", "好", "[PAD]"], "0": ["还", "可", "以"], "1": ["不", "错", "[PAD]"]}
            or a path to json-like file
        """
        if os.path.exists(verbalizer):
            verbalizer = json.load(open(verbalizer, "r", encoding="utf-8"))
        else:
            try:
                verbalizer = eval(verbalizer)
            except:
                raise ValueError(f"The provided verbalizer {verbalizer} is invalid.")
        verbalizer = {k: tokenizer.tokenize(verbalizer[k]) for k in verbalizer}
        max_verbalizer_length = max([len(verbalizer[k]) for k in verbalizer])
        verbalizer_tokens = [verbalizer[k] + [tokenizer.pad_token] * (max_verbalizer_length - len(verbalizer[k])) for k in verbalizer] 
        verbalizer_indices = [tokenizer.convert_tokens_to_ids(vt) for vt in verbalizer_tokens] 
        verbailzer_mask = [[1] * len(verbalizer[k]) + [0] * (max_verbalizer_length - len(verbalizer[k])) for k in verbalizer]
        return verbalizer_tokens, verbalizer_indices, verbailzer_mask

    def build(self, examples, **kwargs):
        template = kwargs.get("template", "")
        verbalizer = kwargs.get("verbalizer", "")
        if not template or not verbalizer:
            raise ValueError("Either template or verbalizer is not offered for prompting.")
        template_tokens, insert_positions = self.parse_template(template, self.tokenizer)
        template_length = len(template_tokens)
        verbalizer_tokens, verbalizer_indices, verbalizer_mask = self.parse_verbalizer(verbalizer, self.tokenizer)
        builded_examples = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Converting example %d of %d" % (ex_index, len(examples)))
            
            label = self.label_map(example.label)
            text_a_tokens = self.tokenizer.tokenize(example.text_a)
            if example.text_b:
                assert len(insert_positions) == 2, "Example.text_b is given but not in the template."
                text_b_tokens = self.tokenizer.tokenize(example.text_b)
                self._truncate_pair(text_a_tokens, text_b_tokens, self.max_length - template_length + 2)
                text_tokens = template_tokens[:insert_positions[0]] + text_a_tokens \
                    + template_tokens[insert_positions[0] + 1: insert_positions[1]] \
                    + text_b_tokens + template_tokens[insert_positions[1] + 1:]
            else:
                assert len(insert_positions) == 1, "Example.text_b is not given but in the template."
                if len(text_a_tokens) > self.max_length - template_length + 1:
                    text_a_tokens = text_a_tokens[:(self.max_length - template_length + 1)]
                text_tokens = template_tokens[:insert_positions[0]] + text_a_tokens \
                    + template_tokens[insert_positions[0] + 1:]
            text_segments = [0] * len(text_tokens)
            mask_position = [text_tokens.index(self.tokenizer.mask_token)]
            assert mask_position[0] < self.max_length, "It seems the truncatenation does not work."
            text_indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("uid: %s" % (example.uid))
                logger.info("text_tokens: %s" % " ".join([str(x) for x in text_tokens]))
                logger.info("text_indices: %s" % " ".join([str(x) for x in text_indices]))
                logger.info("text_segments: %s" % " ".join([str(x) for x in text_segments]))
                logger.info("mask_position: %s" % " ".join([str(x) for x in mask_position]))
                logger.info("verbalizer_tokens: %s" % " ".join([str(x) for x in verbalizer_tokens]))
                logger.info("verbalizer_indices: %s" % " ".join([str(x) for x in verbalizer_indices]))
                logger.info("verbalizer_mask: %s" % " ".join([str(x) for x in verbalizer_mask]))
                logger.info("label: %s (id = %d)" % (example.label, label))

            builded_examples.append(
                BertPptDataPipeline.Example(
                    text_indices=text_indices,
                    text_segments=text_segments,
                    mask_position=mask_position,
                    verbalizer_indices=verbalizer_indices,
                    verbalizer_mask=verbalizer_mask,
                    label=label,
                )
            )
        return builded_examples

    def collate(self, batch):
        if self.max_length is None:
            max_length = max([len(example.text_indices) for example in batch])
        else:
            max_length = self.max_length
        
        batch_text_indices = []
        batch_text_mask = []
        batch_text_segments = []
        batch_mask_position = []
        batch_verbalizer_indices = []
        batch_verbalizer_mask = []
        batch_label = []
        for example in batch:
            text_mask = [1] * len(example.text_indices)

            batch_text_indices.append(self._pad(example.text_indices, max_length, self.tokenizer.pad_token_id))
            batch_text_mask.append(self._pad(text_mask, max_length, 0))
            batch_text_segments.append(self._pad(example.text_segments, max_length, 0))
            batch_mask_position.append(example.mask_position)
            batch_verbalizer_indices.append(example.verbalizer_indices)
            batch_verbalizer_mask.append(example.verbalizer_mask)
            batch_label.append(example.label)
        return BertPptDataPipeline.Batch(
            text_indices=torch.tensor(batch_text_indices, dtype=torch.long),
            text_mask=torch.tensor(batch_text_mask, dtype=torch.bool),
            text_segments=torch.tensor(batch_text_segments, dtype=torch.long),
            mask_position=torch.tensor(batch_mask_position, dtype=torch.long),
            verbalizer_indices=torch.tensor(batch_verbalizer_indices, dtype=torch.long),
            verbalizer_mask=torch.tensor(batch_verbalizer_mask, dtype=torch.bool),
            label=torch.tensor(batch_label),
        )


# For EncT5.
class EncT5ClsDataPipeline(DataPipeline):
    Example = collections.namedtuple(
        "Example", 
        (
            "text_indices",
            "label",
        )
    )
    Batch = collections.namedtuple(
        "Batch", 
        (
            "text_indices", 
            "text_mask", 
            "label",
        )
    )

    def __init__(self, tokenizer, label_map, max_length=None):
        super().__init__(tokenizer, label_map, max_length)

    def build(self, examples, **kwargs):
        builded_examples = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Converting example %d of %d" % (ex_index, len(examples)))

            label = self.label_map(example.label)
            text_a_tokens = self.tokenizer.tokenize(example.text_a)
            text_b_tokens = None
            if example.text_b:
                text_b_tokens = self.tokenizer.tokenize(example.text_b)
                # Account for [EOS] with "- 2" for combined input.
                self._truncate_pair(text_a_tokens, text_b_tokens, self.max_length - 2)
                text_tokens = text_a_tokens + [self.tokenizer.eos_token]
                text_tokens +=  text_b_tokens + [self.tokenizer.eos_token]
                text_indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

                if ex_index < 5:
                    logger.info("*** Example ***")
                    logger.info("uid: %s" % (example.uid))
                    logger.info("text_tokens: %s" % " ".join([str(x) for x in text_tokens]))
                    logger.info("text_indices: %s" % " ".join([str(x) for x in text_indices]))
                    logger.info("label: %s (id = %d)" % (example.label, label))

                builded_examples.append(
                    EncT5ClsDataPipeline.Example(
                        text_indices=text_indices,
                        label=label,
                    )
                )
            else:
                # Account for [EOS] with "- 1"
                if len(text_a_tokens) > self.max_length - 1:
                    text_a_tokens = text_a_tokens[:(self.max_length - 1)]
                text_tokens = text_a_tokens + [self.tokenizer.eos_token]
                text_indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

                if ex_index < 5:
                    logger.info("*** Example ***")
                    logger.info("uid: %s" % (example.uid))
                    logger.info("text_tokens: %s" % " ".join([str(x) for x in text_tokens]))
                    logger.info("text_indices: %s" % " ".join([str(x) for x in text_indices]))
                    logger.info("label: %s (id = %d)" % (example.label, label))

                builded_examples.append(
                    EncT5ClsDataPipeline.Example(
                        text_indices=text_indices,
                        label=label,
                    )
                )
        return builded_examples

    def collate(self, batch):
        if self.max_length is None:
            max_length = max([len(example.text_indices) for example in batch])
        else:
            max_length = self.max_length
        
        batch_text_indices = []
        batch_text_mask = []
        batch_label = []
        for example in batch:
            text_mask = [1] * len(example.text_indices)
            
            batch_text_indices.append(self._pad(example.text_indices, max_length, self.tokenizer.pad_token_id))
            batch_text_mask.append(self._pad(text_mask, max_length, 0))
            batch_label.append(example.label)
        return EncT5ClsDataPipeline.Batch(
            text_indices=torch.tensor(batch_text_indices, dtype=torch.long),
            text_mask=torch.tensor(batch_text_mask, dtype=torch.bool),
            label=torch.tensor(batch_label),
        )


class EncT5NerDataPipeline(DataPipeline):
    Example = collections.namedtuple(
        "Example", 
        (
            "text_indices",
            "labels",
            "label_mask",
        )
    )
    Batch = collections.namedtuple(
        "Batch", 
        (
            "text_indices", 
            "text_mask", 
            "labels",
            "label_mask",
        )
    )

    def __init__(self, tokenizer, label_map, max_length=None):
        super().__init__(tokenizer, label_map, max_length)

    def build(self, examples, **kwargs):
        builded_examples = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Converting example %d of %d" % (ex_index, len(examples)))

            text_tokens, labels, label_mask = [], [], []
            for token, label in zip(example.tokens, example.labels):
                token_pieces = self.tokenizer.tokenize(token)
                text_tokens.extend(token_pieces)
                try:
                    tg, tp = label.split("-")
                except:
                    tg, tp = label, None
                if tg == "B":
                    label_pieces = ["B-"+tp] + ["I-"+tp] * (len(token_pieces)-1)
                elif tg == "I":
                    label_pieces = ["I-"+tp] * len(token_pieces)
                else:
                    label_pieces = ["O"] * len(token_pieces)
                labels.extend(label_pieces)
                label_mask.extend([1] + [0] * (len(token_pieces)-1))
            labels = [self.label_map(label) for label in labels]
            
            # Account for [EOS] with "- 1"
            if len(text_tokens) > self.max_length - 1:
                text_tokens = text_tokens[:(self.max_length - 1)]
                labels = labels[:(self.max_length - 1)]
                label_mask = label_mask[:(self.max_length - 1)]
            text_tokens = text_tokens + [self.tokenizer.eos_token]
            labels = labels + [0]
            label_mask = label_mask + [0]
            text_indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("uid: %s" % (example.uid))
                logger.info("text_tokens: %s" % " ".join([str(x) for x in text_tokens]))
                logger.info("text_indices: %s" % " ".join([str(x) for x in text_indices]))
                logger.info("labels: %s" % " ".join([str(x) for x in labels]))

            builded_examples.append(
                EncT5NerDataPipeline.Example(
                    text_indices=text_indices,
                    labels=labels,
                    label_mask=label_mask,
                )
            )
        return builded_examples

    def collate(self, batch):
        if self.max_length is None:
            max_length = max([len(example.text_indices) for example in batch])
        else:
            max_length = self.max_length
        
        batch_text_indices = []
        batch_text_mask = []
        batch_labels = []
        batch_label_mask = []
        for example in batch:
            text_mask = [1] * len(example.text_indices)
            
            batch_text_indices.append(self._pad(example.text_indices, max_length, self.tokenizer.pad_token_id))
            batch_text_mask.append(self._pad(text_mask, max_length, 0))
            batch_labels.append(self._pad(example.labels, max_length, 0)) # 'O' = 0
            batch_label_mask.append(self._pad(example.label_mask, max_length, 0))
        return EncT5NerDataPipeline.Batch(
            text_indices=torch.tensor(batch_text_indices, dtype=torch.long),
            text_mask=torch.tensor(batch_text_mask, dtype=torch.bool),
            labels=torch.tensor(batch_labels, dtype=torch.long),
            label_mask=torch.tensor(batch_label_mask, dtype=torch.bool),
        )