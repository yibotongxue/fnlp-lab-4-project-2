#!/bin/bash

python -m src.predictor.zero_shot \
    --output-dir ./output/zero_shot \
    --split train \
    --model-name qwen-max \
    --start-index 0 \
    --train-size 100
