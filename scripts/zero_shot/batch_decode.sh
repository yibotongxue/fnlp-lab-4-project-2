#!/bin/bash

python scripts/zero_shot/batch_decode.py \
    --input-file ./output/zero_shot/result.jsonl \
    --output-dir ./output/zero_shot \
    --split train \
    --start-index 0 \
    --train-size 100
