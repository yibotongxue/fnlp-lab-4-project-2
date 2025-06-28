#! /bin/bash

CHARGE_MODEL_PATH="./checkpoints/finetune/charge-checkpoint" # 罪名预测模型的路径
CHARGE_FILE="./data/charges.json" # charges.json 文件的路径
IMPRISONMENT_MODEL_PATH="./checkpoints/finetune/imprisonment-checkpoint" # 刑期预测模型的路径
DATA_DIR="./data" # 数据集路径，目录下应该有train.jsonl和test.jsonl文件
OUTPUT_DIR="./output/one_by_one/finetune_finetune_eval/test" # 输出目录

python -m src.predictor.one_by_one \
    --charge-predictor lawformer \
    --imprisonment-predictor lawformer \
    --charge-model-path ${CHARGE_MODEL_PATH} \
    --charge-file ${CHARGE_FILE} \
    --imprisonment-model-path ${IMPRISONMENT_MODEL_PATH} \
    --imprisonment-num 12 \
    --imprisonment-mapper-config ./configs/predictor/imprisonment_mapper.yaml \
    --device cuda \
    --data-dir ${DATA_DIR} \
    --split test \
    --output-dir ${OUTPUT_DIR} \
    --start-index 0 \
    --train-size 1000
