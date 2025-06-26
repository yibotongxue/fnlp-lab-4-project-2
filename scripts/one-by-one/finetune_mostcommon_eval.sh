#! /bin/bash

python -m src.predictor.one_by_one \
    --charge-predictor lawformer \
    --imprisonment-predictor most_common \
    --charge-model-path ./checkpoints/finetune/charge-checkpoint \
    --charge-file ./data/charges.json \
    --charge-imprisonment-dict-path ./data/charge_imprisonment_dict.json \
    --device cuda \
    --split test \
    --output-dir ./output/one_by_one/finetune_mostcommon_eval/test \
    --start-index 9000 \
    --train-size 1000
