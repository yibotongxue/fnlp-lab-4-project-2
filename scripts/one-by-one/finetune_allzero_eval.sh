#! /bin/bash

python -m src.predictor.one_by_one \
    --charge-predictor lawformer \
    --imprisonment-predictor all_zero \
    --charge-model-path ./checkpoints/finetune/charge-3epoch-checkpoint \
    --charge-file ./data/charges.json \
    --device cuda \
    --split train \
    --output-dir ./output/one_by_one/finetune_allzero_eval \
    --start-index 0 \
    --train-size 1000
