#! /bin/bash

MODEL_NAME_OR_PATH="thunlp/Lawformer"
TRAIN_DATASET_NAME_OR_PATH="./data/course/train.jsonl"
TRAIN_DATASET_TEMPLATE="Course"
EVAL_DATASET_NAME_OR_PATH="./data/course/eval.jsonl"
EVAL_DATASET_TEMPLATE="Course"

python -m src.finetune.trainer.multi_task \
    --config-file-path ./configs/finetune/multi_task.yaml \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_dataset_name_or_path ${TRAIN_DATASET_NAME_OR_PATH} \
    --train_dataset_template ${TRAIN_DATASET_TEMPLATE} \
    --eval_dataset_name_or_path ${EVAL_DATASET_NAME_OR_PATH} \
    --eval_dataset_template ${EVAL_DATASET_TEMPLATE}
