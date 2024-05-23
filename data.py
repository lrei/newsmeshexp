import os
import time
import json
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union
from filelock import FileLock

import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import logging, DataProcessor, PreTrainedTokenizer


logger = logging.get_logger(__name__)


@dataclass
class InputExampleML:
    """A single training/test example for simple sequence classification.

    Args:
    ----
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) List of strings. The labels of the example.
            This should be specified for train and dev examples, but not for
            test examples.
    """

    guid: str
    text_a: str
    label: Optional[List[str]] = None

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeaturesML:
    """A single set of features of data.

    Property names are the same names as the corresponding inputs to a model.

    Args:
    ----
        input_ids: Indices of input sequence tokens in the vocabulary.
            attention_mask: Mask to avoid performing attention on padding token
            indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for
            MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and
            second portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[List[int]] = None

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass
class MESHDataTrainingArguments:
    """Data options for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after "
            "tokenization. Sequences longer than this will be truncated, "
            "sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    limit_length: int = field(
        default=-1, metadata={"help": "Limit number of examples."},
    )


class Split(Enum):
    """Dataset split."""

    train = "train"
    dev = "dev"
    test = "test"


class MESHProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        self._load_labels()

    def _load_labels(self):
        """Load labels from train.tsv."""
        self.le = MultiLabelBinarizer()
        all_labels = []
        with open(os.path.join(self.data_dir, "labels.txt")) as finp:
            for line in finp:
                all_labels.append(line.strip())
        self.le.fit([all_labels])

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        raise NotImplementedError

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_labels(self):
        """See base class."""
        return self.le.classes_

    def _create_examples(self, lines, set_type):
        """Create examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            labels = None if set_type == "test" else line[-1]
            if labels is not None:
                labels = labels.split()
                labels = [x for x in labels if x]
                labels = [x for x in labels if x in self.get_labels()]
            examples.append(
                InputExampleML(guid=guid, text_a=text_a, label=labels)
            )
        return examples


class MESHDataset(Dataset):
    """MESH dataset class."""

    args: MESHDataTrainingArguments
    output_mode: str
    features: List[InputFeaturesML]

    def __init__(
        self,
        args: MESHDataTrainingArguments,
        processor: MESHProcessor,
        tokenizer: PreTrainedTokenizer,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        limit_length = args.limit_length
        self.processor = processor
        self.le = self.processor.le
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
            ),
        )
        label_list = self.processor.get_labels()
        self.label_list = label_list

        # Make sure only the first process in distributed training processes
        # the dataset, and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if (
                os.path.exists(cached_features_file)
                and not args.overwrite_cache
            ):
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]",
                    time.time() - start,
                )
            else:
                logger.info(
                    f"Creating features from dataset file at {args.data_dir}"
                )

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                self.features = self.convert_examples_to_features(
                    examples, tokenizer, max_length=args.max_seq_length,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]",
                    cached_features_file,
                    time.time() - start,
                )
            if limit_length > 0:
                logger.info(f"Limiting examples to {limit_length}")
                self.features = self.features[:limit_length]

    def __len__(self):
        """Return number of examples in dataset."""
        return len(self.features)

    def __getitem__(self, i) -> InputFeaturesML:
        """Get i-th example."""
        return self.features[i]

    def get_labels(self):
        """Return list of labels."""
        return self.label_list

    def convert_examples_to_features(
        self,
        examples: List[InputExampleML],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
    ):
        """Convert list of InputExampleML to list of InputFeaturesML."""
        if max_length is None:
            max_length = tokenizer.max_len

        labels = []
        for ex in examples:
            ex_labels = self.processor.le.transform([ex.label])
            labels.append(ex_labels)

        batch_encoding = tokenizer(
            [(example.text_a, None) for example in examples],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            label = labels[i].tolist()[0]
            # assert len(label) == len(self.processor.get_labels())
            feature = InputFeaturesML(**inputs, label=label)
            features.append(feature)

        for i, example in enumerate(examples[:5]):
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features[i])

        return features
