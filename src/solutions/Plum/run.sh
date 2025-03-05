
  python ./main.py \
    --data-dir "./data/" \
    --algorithm "tabu" \
    --mode "Instruction Only" \
    --train-seed 0 \
    --num-compose 1 \
    --num-candidates 8 \
    --backbone "tlite" \
    --num-iter 3 \
    --patience 5 \
    --write-preds \
    --meta-dir "./logs/" \
    --meta-name "TS_batchsize_4_all_edits_l_1_m_8_n_20@task_001_agnostic_trainseed_0_seed_42_rho_7.txt" \
    --print-orig \
    --agnostic \
    --key-id 0 \
    --batch-size 4 \
    --tournament-selection 4 \
    --project-name 'ts-prompt' \
    --checkpoint-freq 10 \
    --output-dir "./output/" # dir to save cheskpoints

    # add the following argument to resume the searching from the chechpoint
    # --resume /home/szdiao/bbt/ours/grips_heuristicalgs/output/checkpoints/task0_step19.pickle" 

    # add the following arguments to test the performance of the loaded model
    # --model-dir /home/szdiao/bbt/ours/grips_heuristicalgs/output/checkpoints/task0_step19.pickle 
    # --eval-only