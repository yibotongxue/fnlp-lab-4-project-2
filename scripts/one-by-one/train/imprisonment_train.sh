#! /bin/bash

MODEL_NAME_OR_PATH="thunlp/Lawformer"
TRAIN_DATASET_NAME_OR_PATH="./data/course/train.jsonl"
TRAIN_DATASET_TEMPLATE="Course"
EVAL_DATASET_NAME_OR_PATH="./data/course/eval.jsonl"
EVAL_DATASET_TEMPLATE="Course"
OUTPUT_DIR="./output/finetune/imprisonment/"

python -m src.finetune.trainer.normal \
    --config-file-path ./configs/finetune/imprisonment.yaml \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_dataset_name_or_path ${TRAIN_DATASET_NAME_OR_PATH} \
    --train_dataset_template ${TRAIN_DATASET_TEMPLATE} \
    --eval_dataset_name_or_path ${EVAL_DATASET_NAME_OR_PATH} \
    --eval_dataset_template ${EVAL_DATASET_TEMPLATE} \
    --output_dir ${OUTPUT_DIR}
