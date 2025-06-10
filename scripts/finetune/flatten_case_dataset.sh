#! /bin/bash

python scripts/finetune/flatten_case_dataset.py \
    --input data/train.jsonl \
    --charge data/charges.json \
    --output data/train_flattened.jsonl \
    --overwrite
