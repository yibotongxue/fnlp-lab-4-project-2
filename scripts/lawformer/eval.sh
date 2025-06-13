#! /bin/bash

python -m src.predictor.lawformer \
    --model-path ./checkpoints/finetune/full \
    --base-model-name thunlp/Lawformer \
    --split train \
    --output-dir ./output/finetune/full
