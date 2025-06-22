#! /bin/bash

python -m src.finetune.trainer.normal \
    --config=./configs/finetune/charge.yaml \
    --model-name-or-path=thunlp/Lawformer \
    --train-data-path ./data/train_flattened.jsonl \
    --test-data-path ./data/train_flattened.jsonl \
    --model-max-length 512 \
    --is-charge \
    --enable-wandb \
    --extra-model-kwargs num_labels=321
