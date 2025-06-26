#! /bin/bash

python -m src.predictor.one_by_one \
    --charge-predictor lawformer \
    --imprisonment-predictor lawformer \
    --charge-model-path ./checkpoints/finetune/charge-checkpoint \
    --charge-file ./data/charges.json \
    --imprisonment-model-path ./checkpoints/finetune/imprisonment-checkpoint \
    --imprisonment-num 12 \
    --imprisonment-mapper-config ./configs/predictor/imprisonment_mapper.yaml \
    --device cuda \
    --split test \
    --output-dir ./output/one_by_one/finetune_finetune_eval/test \
    --start-index 0 \
    --train-size 1000
