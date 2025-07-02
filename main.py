import argparse

import pytorch_lightning as pl

from gtm.config import read_config
from gtm.pipeline import (

    run_pretrain_pipeline,
    run_finetune_pipeline,
    run_finetune_inference_pipeline,

)


def run(args: argparse.Namespace):
    task_config = read_config(args.task_config_path)
    pl.seed_everything(task_config["seed"])

    if args.task_type == "pretrain":
        run_pretrain_pipeline(task_config)
    elif args.task_type == "finetune":
        run_finetune_pipeline(task_config)
    elif args.task_type == "finetune_inference":
        run_finetune_inference_pipeline(task_config)
    else:
        raise ValueError("Task type is not supported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_type", type=str, required=True)
    parser.add_argument("-c", "--task_config_path", type=str, required=True)
    args = parser.parse_args()
    run(args)
