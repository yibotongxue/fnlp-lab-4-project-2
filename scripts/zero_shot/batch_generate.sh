#!/bin/bash

python scripts/zero_shot/batch_generate.py \
    --output-file ./output/zero_shot/batch_generate.jsonl \
    --model-name qwen-max \
    --split train \
    --start-index 0 \
    --train-size 100
