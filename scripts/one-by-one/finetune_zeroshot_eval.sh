#! /bin/bash

python -m src.predictor.one_by_one \
    --charge-predictor lawformer \
    --imprisonment-predictor zeroshot \
    --charge-model-path ./checkpoints/finetune/charge \
    --charge-base-model-name thunlp/Lawformer \
    --charge-file ./data/charges.json \
    --imprisonment-llm qwen-max \
    --device cuda \
    --split train \
    --output-dir ./output/one_by_one/finetune_zeroshot_eval \
    --start-index 0 \
    --train-size 1000
