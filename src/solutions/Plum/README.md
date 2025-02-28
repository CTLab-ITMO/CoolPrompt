<p align="center" width="60%">
<img src="assets/logo.png" alt="Plum" style="width: 60%; min-width: 200px; display: block; margin: auto; background-color: transparent;">
</p>

# Plum: Prompt Learning using Metaheuristic
Toolkits for discrete, black-box prompt learning based on metaheuristic algorithms.

## Installation
Simple run the following command to create virtual environment and install all dependencies automatically.

`source install.sh`

## Dataset
Our experiments are conducted on eight tasks from Natural-Instructions datasets v2 release: task 019, task 021, task 022, task 050, task 069, task 137, task 139, task 195. 
The datasets can be downloaded from [natural-instruction](https://github.com/allenai/natural-instructions) via:

```
git clone https://github.com/allenai/natural-instructions
```

Then, you can put it under `data` path and set your data path with the following parameters:
* `--data-dir ./data/natural-instructions-2.6/tasks/`

## Setting your Open AI API key
Put your Open AI API key in `API_KEY_LIST` in 'utils/nat_inst_gpt3.py'.

## Quick Start (For reproducing our results)

You can run text-babbage-001 + Genetic Algorithm with `run.sh`:

```bash
bash run.sh
```

## Setting backbone
Our code supports both GPT-2 and GPT-3 backbones via the following parameters:
* `--backbone`: 
  * `gpt2`: GPT2-XL (default)
  * `gpt3`: text-babbage-001 (default)

To use other GPT3 models from the API, please change by the argument `--model_name`. 
* `--model_name`: text-babbage-001 (default), text-ada-001, text-curie-001, text-davinci-001, text-davinci-002, text-davinci-003


## Running search
We implement several heuristic algorithms to search the optimal discrete prompt for downstream tasks in a black-box way, including Hill Climbing, Simulated Annealing, Genetic Algorithm, Tabu Search and Harmony Search.

### Using Hill Climbing and Simulated Annealing algorithm
These two strategies are based on [GrIPS](https://arxiv.org/abs/2203.07281), and we fixed some bugs of their [open-sourced code](https://github.com/archiki/GrIPS).

For `search with Hill Climbing`,

```
python main.py \ 
  --algorithm "hc" \
  --mode "Instruction Only" \
  --task-idx 0 \
  --train-seed 0 \
  --num-compose 1 \
  --num-candidates 10 \
  --num-iter 50 \
  --patience 7 \
  --write-preds \
  --meta-dir "[your_output_dir]" \
  --meta-name "HC_bs_20all_edits_l_1_m_10_n_50@task_0_agnostic_trainseed_0_dataseed_42.txt" \
  --print-orig \
  --agnostic \
  --batch-size 20 \
  --data-dir "[your_data_path]"  \
  --project-name 'Plum' \
  --checkpoint-freq 10 \
  --output-dir "[you_ckpt_dir]" 
```

For `search with Simulated Annealing`,


```
python main.py \ 
  --algorithm "hc" \
  --mode "Instruction Only" \
  --task-idx 0 \
  --train-seed 0 \
  --num-compose 1 \
  --num-candidates 10 \
  --num-iter 50 \
  --patience 7 \
  --write-preds \
  --meta-dir "[your_output_dir]" \
  --meta-name "SA_bs_20all_edits_l_1_m_10_n_50@task_0_agnostic_trainseed_0_dataseed_42.txt" \
  --print-orig \
  --agnostic \
  --batch-size 20 \
  --data-dir "[your_data_path]"  \
  --project-name 'Plum' \
  --checkpoint-freq 10 \
  --output-dir "[your_ckpt_dir]" \
  --simulated-anneal
```
### Using Genetic Algorithm

We propose a search strategy with the Genetic Algorithm (no crossover) for the optimization of discrete prompts. 

For `search with Genetic Algorithm`,

```
python main.py \ 
  --algorithm "ga" \
  --mode "Instruction Only" \
  --task-idx 0 \
  --train-seed 0 \
  --num-compose 1 \
  --num-candidates 10 \
  --num-iter 50 \
  --patience 7 \
  --write-preds \
  --meta-dir "[your_output_dir]" \
  --meta-name "GA_M_bs_20_all_edits_l_1_m_10_n_50@task_0_agnostic_trainseed_0_dataseed_42_rho_7.txt" \
  --print-orig \
  --agnostic \
  --batch-size 20 \
  --tournament-selection 5 \
  --data-dir "[your_data_path]"  \
  --project-name 'Plum' \
  --checkpoint-freq 10 \
  --output-dir "[your_ckpt_dir]" 
```
The `search with Genetic Algorithm` also can be combined with `Simulated Annealing`,

```
python main.py \
  --algorithm "ga" \
  --mode "Instruction Only" \
  --task-idx 0 \
  --train-seed 0 \
  --num-compose 1 \
  --num-candidates 10 \
  --num-iter 50 \
  --patience 7 \
  --write-preds \
  --meta-dir "[your_output_dir]" \
  --meta-name "GA_M_bs_20_all_edits_l_1_m_10_n_50@task_0_agnostic_trainseed_0_dataseed_42_rho_7.txt" \
  --print-orig \
  --agnostic \
  --batch-size 20 \
  --tournament-selection 5 \
  --data-dir "[your_data_path]"  \
  --project-name 'Plum' \
  --checkpoint-freq 10 \
  --output-dir "[your_ckpt_dir]" \
  --simulated-anneal
```

### Using Tabu Search

We propose a search strategy with the Tabu Search for the optimization of discrete prompts. 

For `search with Tabu Search`,

```
python main.py \ 
  --algorithm "tabu" \
  --mode "Instruction Only" \
  --task-idx 0 \
  --train-seed 0 \
  --num-compose 1 \
  --num-candidates 10 \
  --num-iter 50 \
  --patience 7 \
  --write-preds \
  --meta-dir "[your_output_dir]" \
  --meta-name "Tabu_bs_20_all_edits_l_1_m_10_n_50@task_0_agnostic_trainseed_0_dataseed_42_rho_7.txt" \
  --print-orig \
  --agnostic \
  --batch-size 20 \
  --tournament-selection 5 \
  --data-dir "[your_data_path]"  \
  --project-name 'Plum' \
  --checkpoint-freq 10 \
  --output-dir "[your_ckpt_dir]"
```

### Using Harmony Search

We propose a search strategy with the Harmony Search for the optimization of discrete prompts. 

For `search with Harmony Search`,

```
python main.py \ 
  --algorithm "hs" \
  --mode "Instruction Only" \
  --task-idx 0 \
  --train-seed 0 \
  --num-compose 1 \
  --num-candidates 10 \
  --num-iter 50 \
  --patience 7 \
  --write-preds \
  --meta-dir "[your_output_dir]" \
  --meta-name "HS_bs_20_all_edits_l_1_m_10_n_50@task_0_agnostic_trainseed_0_dataseed_42_rho_7.txt" \
  --print-orig \
  --agnostic \
  --batch-size 20 \
  --tournament-selection 5 \
  --data-dir "[your_data_path]"  \
  --project-name 'Plum' \
  --checkpoint-freq 10 \
  --output-dir "[your_ckpt_dir]"
```

## Contact
For help or issues using this package, please submit a GitHub issue.

For personal communication related to this package, please contact Shizhe Diao (sdiaoaa@connect.ust.hk) and Rui Pan (rpan@connect.ust.hk).

## Citation
We are more than happy if this code is helpful to your work. 
If you use our code or extend our work, please consider citing our paper:

```bibtex

@article{Plum-bbpl,
  title         = {Plum: Prompt Learning using Metaheuristic},
  author        = {Rui Pan, Shuo Xing, Shizhe Diao, Xiang Liu, Kashun Shum, Jipeng Zhang, and Tong Zhang},
  year          = {2023},
  eprint        = {2311.08364},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
}
```
