"""Script for training the premise retriever.
"""
import os
from loguru import logger
from pytorch_lightning.cli import LightningCLI

from retrieval.model import PremiseRetriever
from retrieval.datamodule import RetrievalDataModule

# Training the Premise Retriever
# 训练前提检索

# python retrieval/main.py fit --config retrieval/confs/cli_lean3_random.yaml
# Train the retriever on the `random` split of LeanDojo Benchmark.

# 训练脚本保存这超参数和模型的
# The training script saves hyperparameters, model checkpoints, and other information to ./lightning_logs/EXP_ID/,
# where EXP_ID is an arbitrary experiment ID that will be printed by the training script.


# Retrieving Premises for All Proof States
# 在模型训练后，针对证明状态搜寻前提

# python retrieval/main.py predict --config retrieval/confs/cli_lean3_random.yaml --ckpt_path PATH_TO_RETRIEVER_CHECKPOINT

# 被搜寻的前提保存在 ./lightning_logs/EXP_ID'/predictions.pickle.

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_seq_len", "model.max_seq_len")


def main() -> None:
    logger.info(f"PID: {os.getpid()}")
    cli = CLI(PremiseRetriever, RetrievalDataModule)
    logger.info("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
