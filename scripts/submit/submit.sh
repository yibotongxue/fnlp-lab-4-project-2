#! /bin/bash

output_dir="./output/one_by_one/finetune_finetune_eval/test"

python scripts/submit/submit.py \
    --output-dir ${output_dir}

mv ${output_dir}/*.csv .
