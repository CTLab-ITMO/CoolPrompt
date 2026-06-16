import argparse
import os


DATASET_CHOICES = [
    "squad_v2",
    "gsm8k",
    "common_gen",
    "xsum",
    "tweeteval",
    "mediqa",
    "code_to_text",
    "concode",
]

METRIC_CHOICES = ["bert_score", "exact_match", "f1_mera"]

# Per-dataset default metric used when --metric is omitted.
DEFAULT_METRICS = {
    "squad_v2":     "bert_score",
    "gsm8k":        "exact_match",
    "common_gen":   "bert_score",
    "xsum":         "bert_score",
    "tweeteval":    "f1_mera",
    "mediqa":       "bert_score",
    "code_to_text": "bert_score",
    "concode":      "bert_score",
}


def parse_args():
    parser = argparse.ArgumentParser(description='EvoPrompt training args.')

    # Dataset / task selection
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=DATASET_CHOICES,
        help='which generic dataset under datasets/data/ to optimize prompts for',
    )
    parser.add_argument('--task', type=str, default=None,
                        help='legacy BBH task name (kept for back-compat). If --dataset is set, --task is ignored.')
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batchsize in decoding. Left padding in default')
    parser.add_argument('--max-new-tokens', type=int, default=128,
                        help='max new tokens to generate by the model')
    parser.add_argument('--output', type=str, default=None,
                        help='output directory for logs and results')

    parser.add_argument('--sample_num', type=int, default=100,
                        help='number of dev samples used to score prompts during evolution')
    parser.add_argument('--test_sample_num', type=int, default=200,
                        help='number of validation samples used to evaluate the final population')

    # Optimisation metric. If omitted, the per-dataset default is used:
    #   gsm8k     -> exact_match
    #   tweeteval -> f1_mera
    #   all other -> bert_score
    parser.add_argument('--metric', type=str, default=None,
                        choices=METRIC_CHOICES,
                        help='metric used to score candidate prompts during '
                             'evolution. Defaults to the per-dataset default '
                             '(see DEFAULT_METRICS).')

    # EvoPrompt args
    parser.add_argument('--budget', type=int, default=10)
    parser.add_argument('--popsize', type=int, default=10)
    parser.add_argument('--evo_mode', type=str, default='de',
                        help='mode of the evolution', choices=['de', 'ape', 'ga'])
    parser.add_argument('--donor_random', action='store_true')
    parser.add_argument('--sel_mode', type=str, choices=["wheel", "random", "tour"],
                        default="wheel",
                        help='selection strategy for parents, only used for GA')

    # LLM args (gpt-5-nano via langchain_openai by default)
    parser.add_argument('--model', type=str, default='gpt-5-nano',
                        help='OpenAI chat model used through langchain_openai.ChatOpenAI')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='sampling temperature for ChatOpenAI')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key. Falls back to OPENAI_API_KEY env var when omitted.')
    parser.add_argument('--openai_base_url', type=str, default=None,
                        help='Optional base URL override for OpenAI-compatible endpoints.')
    parser.add_argument('--llm_type', type=str, default='turbo',
                        help='Internal routing flag. Use "turbo" (default) for chat models like gpt-5-nano.')

    # Prompt initialization
    parser.add_argument('--initial', type=str, default='cot',
                        choices=['cot', 'desc', 'all', 'ckpt'],
                        help='style of the prompt: cot (task agnostic), desc (task specific), all, or ckpt')
    parser.add_argument('--initial_mode', type=str, default='topk',
                        choices=['topk', 'bottomk', 'randomk', 'para_topk', 'para_bottomk', 'para_randomk'])
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--para_mode', type=str, default=None)
    parser.add_argument('--template', type=str, default='v1')
    parser.add_argument('--client', action='store_true')

    parser.add_argument('--cot_cache_path', type=str, default=None,
                        help='optional path to cache initial-population scores (cot). '
                             'Defaults to <output>/cot_cache.json')
    parser.add_argument('--desc_cache_path', type=str, default=None,
                        help='optional path to cache initial-population scores (desc/ape).')
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--setting', type=str, default='default')
    parser.add_argument('--ga_mode', type=str, default="topk",
                        help="update strategy for GA")
    parser.add_argument('--ckpt_pop', type=str, default=None)
    parser.add_argument('--demon', type=int, default=0,
                        help='few-shot or zero-shot', choices=[0, 1])
    parser.add_argument('--content', type=str, default='',
                        help='content of the prompt, used when testing single prompt')

    # Output of optimization history (JSON)
    parser.add_argument('--results_json', type=str, default=None,
                        help='path to the JSON file with the full optimization log '
                             '(initial prompts, every evolution step, final best prompt). '
                             'Defaults to <output>/optimization_log.json')

    args = parser.parse_args()

    # Sensible defaults / fallbacks
    if args.dataset is None and args.task is None:
        parser.error("Either --dataset (preferred) or --task (legacy BBH) must be provided.")

    if args.output is None:
        base = args.dataset or args.task
        args.output = os.path.join("outputs", f"{base}_{args.evo_mode}_seed{args.seed}")
    os.makedirs(args.output, exist_ok=True)

    if args.results_json is None:
        args.results_json = os.path.join(args.output, "optimization_log.json")

    if args.cot_cache_path is None:
        args.cot_cache_path = os.path.join(args.output, "cot_cache.json")
    if args.desc_cache_path is None:
        args.desc_cache_path = os.path.join(args.output, "desc_cache.json")

    # Metric resolution: CLI flag > per-dataset default > bert_score.
    if args.metric is None:
        key = args.dataset or args.task
        args.metric = DEFAULT_METRICS.get(key, "bert_score")

    # API key resolution: CLI flag > env var.
    if not args.openai_api_key:
        args.openai_api_key = os.environ.get("OPENAI_API_KEY", None)

    return args
