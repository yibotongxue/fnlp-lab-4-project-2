#! /bin/bash

python -m src.predictor.rag \
    --articles-file ./data/articles.json \
    --embed-file ./data/articles_embed.jsonl \
    --retriever-type numpy \
    --output-dir ./output/rag/test \
    --split test \
    --model-name qwen-max \
    --embedding-model text-embedding-v3 \
    --start-index 0 \
    --train-size 1000
