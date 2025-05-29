#!/bin/bash

#--model_name AnatoliiPotapov/T-lite-instruct-0.1 \
#deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

VERSION=v4

# for dataset in causal_judgement date_understanding disambiguation_qa formal_fallacies geometric_shapes hyperbaton logical_deduction_five_objects logical_deduction_seven_objects logical_deduction_three_objects medqa mnli movie_recommendation mr navigate openbookqa penguins_in_a_table qnli reasoning_about_colored_objects ruin_names salient_translation_error_detection snarks sports_understanding sst-2 task021 task050 task069 temporal_sequences tracking_shuffled_objects_five_objects tracking_shuffled_objects_seven_objects tracking_shuffled_objects_three_objects trec web_of_lies yahoo
# do
# python3 run.py \
#     --model_name AnatoliiPotapov/T-lite-instruct-0.1 \
#     --dataset $dataset \
#     --task cls \
#     --metric f1 \
#     --teams 4 \
#     --players_per_team 3 \
#     --batch_size 128 \
#     --seasons 10 \
#     --use_cache True \
#     --output_path ./outputs_v4/cls/$dataset
# done


for dataset in dyck_languages gsm8k math multistep_arithmetic_two object_counting samsum word_sorting
do
python3 run.py \
    --version $VERSION \
    --model_name AnatoliiPotapov/T-lite-instruct-0.1 \
    --dataset $dataset \
    --task gen \
    --metric meteor \
    --teams 4 \
    --players_per_team 3 \
    --batch_size 128 \
    --seasons 10 \
    --use_cache True \
    --output_path ./outputs_$VERSION/gen/$dataset
done
