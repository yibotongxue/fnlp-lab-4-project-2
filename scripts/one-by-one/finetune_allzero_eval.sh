#! /bin/bash

python -m src.predictor.one_by_one \
    --charge-predictor lawformer \
    --imprisonment-predictor all_zero \
    --charge-model-path ./checkpoints/finetune/charge-checkpoint \
    --charge-file ./data/charges.json \
    --device cuda \
    --split test \
    --output-dir ./output/one_by_one/finetune_allzero_eval/test \
    --start-index 0 \
    --train-size 1000
