#! /bin/bash

python -m src.predictor.one_by_one \
    --charge-predictor refine \
    --imprisonment-predictor all_zero \
    --candidate-cnt 3 \
    --multiple-predictor-type lawformer \
    --multiple-device cuda \
    --multiple-charge-model-path ./checkpoints/finetune/charge-6epoch-checkpoint \
    --multiple-charge-base-model-name thunlp/Lawformer \
    --refiner-type zero_shot \
    --refiner-llm qwen-max \
    --charge-file ./data/charges.json \
    --split test \
    --output-dir ./output/one_by_one/refine_allzero_eval/test \
    --start-index 41 \
    --train-size 59
