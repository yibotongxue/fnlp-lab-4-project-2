#! /bin/bash

python -m src.predictor.lawformer \
    --model-path ./checkpoints/finetune/multi_task \
    --base-model-name thunlp/Lawformer \
    --split train \
    --output-dir ./output/finetune/multi_task
