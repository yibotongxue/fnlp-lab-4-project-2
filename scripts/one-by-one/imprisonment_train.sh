#! /bin/bash

python -m src.finetune.trainer.normal \
    --config=./configs/finetune/imprisonment.yaml \
    --model-name-or-path=thunlp/Lawformer \
    --train-data-path ./data/train_flattened.jsonl \
    --test-data-path ./data/train_flattened.jsonl \
    --model-max-length 512
