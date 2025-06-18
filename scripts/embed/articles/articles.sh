#! /bin/bash

python ./scripts/embed/articles/batch_file_generator.py \
    --articles-file ./data/articles.json \
    --output-file ./articles_batch_temp.jsonl \
    --model-name text-embedding-v3

python ./scripts/embed/batch_embed_generator.py \
    --input-file ./articles_batch_temp.jsonl \
    --result-file ./articles_embed_result.jsonl \
    --error-file ./error.jsonl

python ./scripts/embed/articles/match_text_embed.py \
    --batch-file ./articles_batch_temp.jsonl \
    --result-file ./articles_embed_result.jsonl \
    --output-file ./data/articles_embed.jsonl

rm ./articles_batch_temp.jsonl ./articles_embed_result.jsonl
