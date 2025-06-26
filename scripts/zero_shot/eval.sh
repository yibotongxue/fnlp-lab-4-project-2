#!/bin/bash

python -m src.predictor.zero_shot \
    --output-dir ./output/zero_shot/test \
    --split test \
    --model-name qwen-max \
    --start-index 0 \
    --train-size 1000
