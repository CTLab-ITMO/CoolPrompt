#!/bin/bash

#--model_name AnatoliiPotapov/T-lite-instruct-0.1 \
# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
for dataset in sst-2
do
python3 run.py \
    --model_name AnatoliiPotapov/T-lite-instruct-0.1 \
    --dataset $dataset \
    --task cls \
    --metric f1 \
    --teams 4 \
    --players_per_team 3 \
    --batch_size 128 \
    --seasons 10 \
    --use_cache True \
    --output_path ./outputs/cls/$dataset
done


# for dataset in dyck_languages gsm8k math multistep_arithmetic_two object_counting samsum word_sorting
# do
# python3 run.py \
#     --model_name AnatoliiPotapov/T-lite-instruct-0.1 \
#     --dataset $dataset \
#     --task gen \
#     --metric meteor \
#     --population_size 10 \
#     --batch_size 128 \
#     --num_epochs 10 \
#     --use_cache True \
#     --output_path ./outputs/re/gen/$dataset
# done
