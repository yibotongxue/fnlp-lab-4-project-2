#! /bin/bash

python -m src.finetune.trainer.full \
    --config=./configs/finetune/full.yaml \
    --model-name-or-path=thunlp/Lawformer \
    --train-data-path ./data/train_flattened.jsonl \
    --test-data-path ./data/train_flattened.jsonl \
    --model-max-length 512 \
    --enable-wandb
