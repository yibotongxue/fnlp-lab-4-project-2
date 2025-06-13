#! /bin/bash

output_dir="./output/finetune/one-by-one/charge-only/test"

python scripts/submit/submit.py \
    --output-dir ${output_dir}

mv ${output_dir}/*.csv .
