#!/usr/bin/env python
# coding=utf-8
"""Finetuning."""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from sklearn.metrics import f1_score
from transformers import (
    XLNetConfig,
    XLNetTokenizer,
    EvalPrediction,
)
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from modeling import XLNetForSequenceClassificationML
from data import MESHProcessor, MESHDataset, MESHDataTrainingArguments


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer."""

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store downloaded pretrained models"
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, MESHDataTrainingArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a
        # json file, let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists "
            "and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if training_args.local_rank in [-1, 0]
        else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s,"
        "16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    processor = MESHProcessor(data_args.data_dir)
    num_labels = len(processor.get_labels())
    logger.info(f"Num Labels: {num_labels}")

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.

    config = XLNetConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        mem_len=1024,
        num_labels=num_labels,
    )
    tokenizer = XLNetTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir,
    )
    model = XLNetForSequenceClassificationML.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = None
    if training_args.do_train:
        train_dataset = MESHDataset(
            args=data_args,
            processor=processor,
            tokenizer=tokenizer,
            mode="train",
            cache_dir=model_args.cache_dir,
        )
    eval_dataset = (
        MESHDataset(
            args=data_args,
            processor=processor,
            tokenizer=tokenizer,
            mode="dev",
            cache_dir=model_args.cache_dir,
        )
        if training_args.do_eval
        else None
    )
    """
    test_dataset = (
        MESHDataset(
            data_args,
            tokenizer=tokenizer,
            mode="test",
            le=train_dataset.le,
            cache_dir=model_args.cache_dir,
        )
        if training_args.do_predict
        else None
    )
    """

    def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            print(type(p.predictions))
            print(len(p.predictions))

            print(type(p.predictions[0]))
            print(p.predictions[0].shape)

            y_pred = p.predictions
            y_pred = torch.tensor(y_pred)

            y_pred = torch.sigmoid(y_pred).numpy()
            y_pred = [np.where(y_el < 0.5, 0, 1) for y_el in y_pred]
            y_true = p.label_ids
            fs = f1_score(y_true, y_pred, average="micro")
            return {"f1": fs}

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(
            training_args.output_dir, "eval_results.txt",
        )
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        eval_results.update(eval_result)
    """
    if training_args.do_predict:
        logging.info("*** Test ***")
        # @TODO fix this for multilabel

        predictions = trainer.predict(test_dataset=test_dataset).predictions
        y_pred = torch.sigmoid(predictions).numpy()
        y_pred = [np.where(y_el < 0.5, 0, 1) for y_el in y_pred]

        output_test_file = os.path.join(
            training_args.output_dir, "test_results.txt",
        )
        if trainer.is_world_master():
            with open(output_test_file, "w") as writer:
                logger.info(
                    "***** Test results {} *****".format(
                        test_dataset.args.task_name
                    )
                )
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = test_dataset.get_labels()[item]
                    writer.write("%d\t%s\n" % (index, item))
    """
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
