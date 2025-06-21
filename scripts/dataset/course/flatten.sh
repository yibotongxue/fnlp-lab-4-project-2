#! /bin/bash

python scripts/dataset/course/flatten.py \
    --input data/train.jsonl \
    --charge data/charges.json \
    --output data/train_flattened.jsonl \
    --overwrite \
    --is-train

python scripts/dataset/course/flatten.py  \
    --input data/test.jsonl \
    --charge data/charges.json \
    --output data/test_flattened.jsonl \
    --overwrite
