#! /bin/bash

python ./scripts/dataset/cail/filter_charge.py \
    --cail-accu-file ./data/cail/accu.txt \
    --course-charge-file ./data/charges.json \
    --output ./data/cail/charges.json

python ./scripts/dataset/cail/transform.py \
    --input=./data/cail/data_train.jsonl \
    --output=./data/cail/transformed_train.jsonl \
    --charge ./data/cail/charges.json \
    --is-train \
    --overwrite

python ./scripts/dataset/cail/transform.py \
    --input=./data/cail/data_valid.jsonl \
    --output=./data/cail/transformed_valid.jsonl \
    --charge ./data/cail/charges.json \
    --is-train \
    --overwrite

python ./scripts/dataset/cail/transform.py \
    --input=./data/cail/data_test.jsonl \
    --output=./data/cail/transformed_test.jsonl \
    --charge ./data/cail/charges.json \
    --is-train \
    --overwrite
