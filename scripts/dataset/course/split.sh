#! /bin/bash

export PYTHONPATH=$PWD

python ./scripts/dataset/course/split.py \
    --input-file ./data/train.jsonl \
    --train-file ./data/course/train.jsonl \
    --eval-file ./data/course/eval.jsonl \
    --eval-size 500
