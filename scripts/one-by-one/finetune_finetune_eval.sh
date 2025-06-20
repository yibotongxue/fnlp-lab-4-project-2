#! /bin/bash

python -m src.predictor.one_by_one \
    --charge-predictor lawformer \
    --imprisonment-predictor lawformer \
    --charge-model-path ./checkpoints/finetune/charge-6epoch-checkpoint \
    --charge-base-model-name thunlp/Lawformer \
    --charge-file ./data/charges.json \
    --imprisonment-model-path ./checkpoints/finetune/imprisonment-3epoch-checkpoint \
    --imprisonment-base-model-name thunlp/Lawformer \
    --imprisonment-num 12 \
    --imprisonment-mapper-config ./configs/predictor/imprisonment_mapper.yaml \
    --device cuda \
    --split test \
    --output-dir ./output/one_by_one/finetune_finetune_eval/test \
    --start-index 0 \
    --train-size 1000
