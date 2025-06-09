#!/bin/bash

python main.py \
    --output-dir ./output/zero_shot \
    --split train \
    --model-name qwen-max \
    --start-index 0 \
    --train-size 100
