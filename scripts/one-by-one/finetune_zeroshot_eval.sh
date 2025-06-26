#! /bin/bash

python -m src.predictor.one_by_one \
    --charge-predictor lawformer \
    --imprisonment-predictor zero_shot \
    --charge-model-path ./checkpoints/finetune/charge-checkpoint \
    --charge-file ./data/charges.json \
    --imprisonment-llm qwen-max \
    --device cuda \
    --split test \
    --output-dir ./output/one_by_one/finetune_zeroshot_eval/test \
    --start-index 0 \
    --train-size 1000
