# -*- coding: utf-8 -*-

"""
DataReader -> reading RawData from tsv, json, etc., in general form of Examples. [task-specific]
DataPipeline -> converting Examples to Examples of specific forms, collating Examples as Batches. [model-specific]
"""


from data.readers import (
    CoLAReader,
    SST2Reader,
    MRPCReader,
    STSBReader,
    QQPReader,
    MNLIReader,
    MNLIMMReader,
    QNLIReader,
    RTEReader,
    WNLIReader,
    CoNLLReader,
)
from data.pipelines import (
    BertClsDataPipeline,
    BertNerDataPipeline,
    BertPptDataPipeline,
    EncT5ClsDataPipeline,
    EncT5NerDataPipeline,
)


READER_CLASS = {
    "cola": CoLAReader,
    "sst2": SST2Reader,
    "mrpc": MRPCReader,
    "stsb": STSBReader,
    "qqp": QQPReader,
    "mnli": MNLIReader,
    "mnlimm": MNLIMMReader,
    "qnli": QNLIReader,
    "rte": RTEReader,
    "wnli": WNLIReader,
    "conll": CoNLLReader,
}

def get_reader_class(task_name):
    return READER_CLASS[task_name]


PIPELINE_CLASS = {
    "bert_cls": BertClsDataPipeline,
    "bert_ner": BertNerDataPipeline,
    "bert_ppt": BertPptDataPipeline,
    "enct5_cls": EncT5ClsDataPipeline,
    "enct5_ner": EncT5NerDataPipeline,
}

def get_pipeline_class(data_type):
    return PIPELINE_CLASS[data_type]


import torch
from torch.utils.data import IterableDataset


class Dataset(IterableDataset):
    def __init__(self, data, shuffle=True):
        super().__init__()
        self.data = data
        self.shuffle = shuffle
        self.num_examples = len(self.data)

    def __len__(self):
        return self.num_examples

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            for idx in torch.randperm(self.num_examples, generator=generator).tolist():
                yield self.data[idx]
        else:
            for idx in range(self.num_examples):
                yield self.data[idx]

class DistributedDataset(IterableDataset):
    def __init__(self, data, num_replicas=None, rank=None, shuffle=True):
        super().__init__()
        self.data = data
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        # Do ceiling to make the data evenly divisible among devices.
        self.num_examples = math.ceil(len(self.data) / self.num_replicas)
        self.total_num_examples = self.num_examples * self.num_replicas

    def __len__(self):
        return self.num_examples

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            indices = torch.randperm(self.num_examples, generator=generator).tolist()
        else:
            indices = list(range(self.num_examples))

        num_padding_examples = self.total_num_examples - len(indices)
        # Is the logic necessary?
        if num_padding_examples <= len(indices):
            indices += indices[:num_padding_examples]
        else:
            indices += (indices * math.ceil(num_padding_examples / len(indices)))[:num_padding_examples]

        assert len(indices) == self.num_total_examples

        # Subsample.
        indices = indices[self.rank:self.num_total_examples:self.num_replicas]
        assert len(indices) == self.num_examples

        for idx in indices:
            yield self.data[idx]
