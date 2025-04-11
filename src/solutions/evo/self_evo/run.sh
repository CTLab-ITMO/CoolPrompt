#!/bin/bash

#--model_name AnatoliiPotapov/T-lite-instruct-0.1 \
# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
for dataset in causal_judgement date_understanding disambiguation_qa formal_fallacies geometric_shapes hyperbaton
do
python3 run.py \
    --model_name AnatoliiPotapov/T-lite-instruct-0.1 \
    --dataset $dataset \
    --task cls \
    --metric f1 \
    --population_size 10 \
    --batch_size 128 \
    --num_epochs 10 \
    --use_cache True \
    --output_path ./outputs/re/cls/$dataset
done