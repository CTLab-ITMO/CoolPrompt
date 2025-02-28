
  python ./main.py \
    --data-dir "./data/" \
    --algorithm "hc" \
    --mode "Instruction Only" \
    --train-seed 0 \
    --num-compose 1 \
    --num-candidates 10 \
    --backbone "tlite" \
    --num-iter 50 \
    --patience 7 \
    --write-preds \
    --meta-dir "./logs/" \
    --meta-name "HC_batchsize_20_all_edits_l_1_m_10_n_50@task_001_agnostic_trainseed_0_seed_42_rho_7.txt" \
    --print-orig \
    --agnostic \
    --key-id 0 \
    --batch-size 20 \
    --tournament-selection 5 \
    --project-name 'hc-prompt' \
    --checkpoint-freq 10 \
    --output-dir "./output/" # dir to save cheskpoints

    # add the following argument to resume the searching from the chechpoint
    # --resume /home/szdiao/bbt/ours/grips_heuristicalgs/output/checkpoints/task0_step19.pickle" 

    # add the following arguments to test the performance of the loaded model
    # --model-dir /home/szdiao/bbt/ours/grips_heuristicalgs/output/checkpoints/task0_step19.pickle 
    # --eval-only