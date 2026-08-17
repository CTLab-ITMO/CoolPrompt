#!/usr/bin/env bash

set -e

MODEL="gpt-4o-mini"
BASE_URL="https://openrouter.ai/api/v1"
TEMPERATURE="0.7"
N_ITERATIONS=20
TRAIN_BATCH_SIZE=50
TRAIN_POOL_SIZE=300
TEST_SIZE=300
OUTPUT_ROOT="method_evaluation_outputs/all_runs"

# The original method_comparison experiment used seed 42 for all three runs.
RUN_SEEDS=(42 42 42)

METHODS=(
    hyper_light
    hyper_light_playbook
    hyper_light_pea_playbook
    hyper_light_playbook_iterative
    hyper_light_pea_playbook_iterative
)

for run_index in "${!RUN_SEEDS[@]}"; do
    run_number=$((run_index + 1))
    seed="${RUN_SEEDS[run_index]}"

    echo "========== run ${run_number}/${#RUN_SEEDS[@]}, seed=${seed} =========="

    for method in "${METHODS[@]}"; do
        output_dir="${OUTPUT_ROOT}/${method}/run_${run_number}"

        if [[ "${method}" == *_iterative ]]; then
            dataset_configuration="${TRAIN_POOL_SIZE}/-/${TEST_SIZE}"

            echo "Running ${method}, run=${run_number}, seed=${seed}"
            python coolprompt/method_evaluation/evaluate_coolprompt_bench.py \
                --method "${method}" \
                --model "${MODEL}" \
                --base-url "${BASE_URL}" \
                --temperature "${TEMPERATURE}" \
                --dataset-configuration "${dataset_configuration}" \
                --n-iterations "${N_ITERATIONS}" \
                --train-batch-size "${TRAIN_BATCH_SIZE}" \
                --train-pool-size "${TRAIN_POOL_SIZE}" \
                --seed "${seed}" \
                --output-dir "${output_dir}"
        else
            dataset_configuration="-/-/${TEST_SIZE}"

            echo "Running ${method}, run=${run_number}, seed=${seed}"
            python coolprompt/method_evaluation/evaluate_coolprompt_bench.py \
                --method "${method}" \
                --model "${MODEL}" \
                --base-url "${BASE_URL}" \
                --temperature "${TEMPERATURE}" \
                --dataset-configuration "${dataset_configuration}" \
                --seed "${seed}" \
                --output-dir "${output_dir}"
        fi
    done
done

echo "Running mrpea_original, 3 runs, seed=42"
python scripts/run_mrpea_benchmark.py \
    --model "${MODEL}" \
    --base-url "${BASE_URL}" \
    --temperature "${TEMPERATURE}" \
    --max-tokens unlimited \
    --dataset-configuration "${TRAIN_POOL_SIZE}/-/${TEST_SIZE}" \
    --runs 3 \
    --n-iterations "${N_ITERATIONS}" \
    --seed 42 \
    --output-dir "${OUTPUT_ROOT}/mrpea_original"

echo "All benchmark runs completed. Results are in ${OUTPUT_ROOT}."
