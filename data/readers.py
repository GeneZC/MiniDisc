# -*- coding: utf-8 -*-

import os
import csv
import collections


ClsExample = collections.namedtuple(
    "ClsExample", 
    (
        "uid", 
        "text_a", 
        "text_b", 
        "label",
    )
)


class DataReader:
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self):
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "train.tsv")), 
            "train",
        )

    def get_dev_examples(self):
        return self._create_examples(
        self._read_tsv(os.path.join(self.data_dir, "dev.tsv")), 
            "dev",
        )

    def get_test_examples(self):
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "test.tsv")), 
            "test",
        )

    @staticmethod
    def _read_tsv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @staticmethod
    def get_label_map():
        """Gets the label map for this data set."""
        raise NotImplementedError()

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples."""
        raise NotImplementedError()


class CoLAReader(DataReader):
    """Reader for the CoLA data set (GLUE version)."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {"0": 0, "1": 1}
        r_d = {v:k for k,v in d.items()}
        return lambda x:d[x], lambda x:r_d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            uid = "%s-%s" % (set_type, i-1)
            if set_type == "test":
                text_a = line[1]
                label = "0"
            else:
                text_a = line[3]
                label = line[1]
            examples.append(
                ClsExample(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=None, 
                    label=label
                )
            )
        return examples

class SST2Reader(DataReader):
    """Reader for the SST-2 data set (GLUE version)."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {"0": 0, "1": 1}
        r_d = {v:k for k,v in d.items()}
        return lambda x:d[x], lambda x:r_d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = line[1]
                label = "0"
            else:
                text_a = line[0]
                label = line[1]
            examples.append(
                ClsExample(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=None, 
                    label=label
                )
            )
        return examples

class MRPCReader(DataReader):
    """Reader for the MRPC data set (GLUE version)."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {"0": 0, "1": 1}
        r_d = {v:k for k,v in d.items()}
        return lambda x:d[x], lambda x:r_d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            if set_type == "test":
                label = "0"
            else:
                label = line[0]
            examples.append(
                ClsExample(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=text_b, 
                    label=label
                )
            )
        return examples

class STSBReader(DataReader):
    """Reader for the STS-B data set (GLUE version)."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        return lambda x:float(x), lambda x:str(x), 1

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            if set_type == "test":
                label = "0"
            else:
                label = line[-1]
            examples.append(
                ClsExample(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=text_b, 
                    label=label
                )
            )
        return examples

class QQPReader(DataReader):
    """Reader for the QQP data set (GLUE version)."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {"0": 0, "1": 1}
        r_d = {v:k for k,v in d.items()}
        return lambda x:d[x], lambda x:r_d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, line[0])
            if set_type == "test":
                text_a = line[1]
                text_b = line[2]
                label = "0"
            else:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            examples.append(
                ClsExample(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=text_b, 
                    label=label
                )
            )
        return examples

class MNLIReader(DataReader):
    """Reader for the MultiNLI data set (GLUE version)."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def get_dev_examples(self):
        return self._create_examples(
                self._read_tsv(os.path.join(self.data_dir, "dev_matched.tsv")),
                "dev")

    def get_test_examples(self):
        return self._create_examples(
                self._read_tsv(os.path.join(self.data_dir, "test_matched.tsv")), "test")

    @staticmethod
    def get_label_map():
        d = {"contradiction": 0, "entailment": 2, "neutral": 1}
        r_d = {v:k for k,v in d.items()}
        return lambda x:d[x], lambda x:r_d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            if set_type == "test":
                label = "contradiction"
            else:
                label = line[-1]
            examples.append(
                ClsExample(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=text_b, 
                    label=label
                )
            )
        return examples

class MNLIMMReader(MNLIReader):
    """Reader for the MultiNLI Mismatched data set (GLUE version)."""
    def get_dev_examples(self):
        return self._create_examples(
                self._read_tsv(os.path.join(self.data_dir, "dev_mismatched.tsv")),
                "dev")
    
    def get_test_examples(self):
        return self._create_examples(
                self._read_tsv(os.path.join(self.data_dir, "test_mismatched.tsv")), "test")

class QNLIReader(DataReader):
    """Reader for the QNLI data set (GLUE version)."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {"entailment": 1, "not_entailment": 0}
        r_d = {v:k for k,v in d.items()}
        return lambda x:d[x], lambda x:r_d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if set_type == "test":
                label = "not_entailment"
            else:
                label = line[-1]
            examples.append(
                ClsExample(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=text_b, 
                    label=label
                )
            )
        return examples


class RTEReader(DataReader):
    """Reader for the RTE data set (GLUE version)."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {"entailment": 1, "not_entailment": 0}
        r_d = {v:k for k,v in d.items()}
        return lambda x:d[x], lambda x:r_d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if set_type == "test":
                label = "not_entailment"
            else:
                label = line[-1]
            examples.append(
                ClsExample(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=text_b, 
                    label=label
                )
            )
        return examples


class WNLIReader(DataReader):
    """Reader for the WNLI data set (GLUE version)."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {"0": 0, "1": 1}
        r_d = {v:k for k,v in d.items()}
        return lambda x:d[x], lambda x:r_d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            if set_type == "test":
                label = "0"
            else:
                label = line[-1]
            examples.append(
                ClsExample(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=text_b, 
                    label=label
                )
            )
        return examples


NerExample = collections.namedtuple(
    "NerExample", 
    (
        "uid", 
        "tokens",
        "labels",
    )
)


class CoNLLReader:
    """Reader for CoNLL data sets."""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self):
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, "train.txt")), 
            "train",
        )

    def get_dev_examples(self):
        return self._create_examples(
        self._read_txt(os.path.join(self.data_dir, "dev.txt")), 
            "dev",
        )

    def get_test_examples(self):
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, "test.txt")), 
            "test",
        )

    @staticmethod
    def _read_txt(input_file):
        """Reads a txt file in conll format."""
        with open(input_file, "r", encoding="utf-8") as f:
            #reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    @staticmethod
    def get_label_map():
        """Gets the label map for this data set."""
        d = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8}
        r_d = {v:k for k,v in d.items()}
        return lambda x:d[x], lambda x:r_d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples."""
        examples = []
        tokens, labels = [], []
        for line in lines:
            if line.startswith("-DOCSTART-") or line == "":
                #tokens, labels = [], []
                if tokens:
                    examples.append(
                        NerExample(
                            uid=f"{set_type}-{len(examples)}",
                            tokens=tokens,
                            labels=labels,
                        )
                    )
                tokens, labels = [], []
                continue

            token, _, _, label = line.split(" ")
            tokens.append(token)
            labels.append(label)

        if tokens:
            examples.append(
                NerExample(
                    uid=f"{set_type}-{len(examples)}",
                    tokens=tokens,
                    labels=labels,
                )
            )


        return examples
