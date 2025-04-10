#!/bin/bash

python3 run.py \
    --model_name AnatoliiPotapov/T-lite-instruct-0.1 \
    --dataset boolean_expressions \
    --task cls \
    --metric f1 \
    --population_num 10 \
    --num_epochs 10 \
    --history_size 3 \
    --use_cache True \
    --output_path ./outputs/v2/cls/boolean_expressions