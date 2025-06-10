#!/bin/bash

python -m src.predictor.zero_shot \
    --output-dir ./output/zero_shot \
    --split test \
    --model-name qwen-max \
    --start-index 0
