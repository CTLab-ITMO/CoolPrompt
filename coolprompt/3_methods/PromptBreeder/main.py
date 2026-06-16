"""PromptBreeder entry point.

Runs the genetic prompt-optimization loop over one of the supported
datasets in ``pb/data/`` using an OpenAI chat model (default:
``gpt-5-nano`` with temperature 0.7). The full run -- initial prompts,
per-generation snapshots, and the final / best prompt -- is persisted as
a single JSON document under ``runs/``.
"""

from pb import create_population, init_run, run_for_n, configure_dataset
from pb.mutation_prompts import mutation_prompts
from pb.thinking_styles import thinking_styles
from pb.llm import OpenAIClient
from pb.logging_utils import RunLogger
from pb import datasets
from pb.metrics import SUPPORTED_METRICS, default_metric_for

import os
import logging
import argparse
from getpass import getpass

from dotenv import load_dotenv
from rich import print

load_dotenv()  # load environment variables

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description='Run the PromptBreeder Algorithm. Number of units is mp * ts.'
)
parser.add_argument('-mp', '--num_mutation_prompts', default=2, type=int)
parser.add_argument('-ts', '--num_thinking_styles', default=4, type=int)
parser.add_argument('-e', '--num_evals', default=10, type=int)
parser.add_argument('-n', '--simulations', default=10, type=int)
parser.add_argument(
    '-d', '--dataset',
    default='gsm8k',
    choices=datasets.SUPPORTED_DATASETS,
    help='Which dataset under pb/data/ to optimize prompts against.',
)
parser.add_argument(
    '-p', '--problem',
    default=None,
    help='Problem description / starting task prompt. If omitted, a '
         'sensible default for the chosen dataset is used.',
)
parser.add_argument(
    '-m', '--model',
    default='openai/gpt-5-nano',
    help='OpenAI chat model name (default: gpt-5-nano).',
)
parser.add_argument(
    '-t', '--temperature',
    default=1.0,
    type=float,
    help='Sampling temperature for prompt generation (default: 0.7).',
)
parser.add_argument(
    '-M', '--metric',
    default=None,
    choices=SUPPORTED_METRICS,
    help='Fitness metric. If omitted, a sensible default for the chosen '
         'dataset is used (gsm8k -> exact_match, tweeteval -> f1_mera, '
         'others -> bert_score).',
)

args = parser.parse_args()

total_evaluations = args.num_mutation_prompts * args.num_thinking_styles * args.num_evals

# --- OpenAI API key -----------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = getpass("Enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key

# --- LLM client ---------------------------------------------------------------
co = OpenAIClient(
    model=args.model,
    api_key=api_key,
    num_workers=max(1, total_evaluations),
    max_retries=5,
    timeout=30,
    temperature=args.temperature,
)

# --- Dataset configuration ----------------------------------------------------
dataset_name = args.dataset
metric_name = args.metric or default_metric_for(dataset_name)
problem_description = args.problem or datasets.default_problem_description(dataset_name)
eval_examples = datasets.load_examples(dataset_name, split='train')
configure_dataset(dataset_name, eval_examples, metric=metric_name)

logger.info(f"Dataset: {dataset_name} ({len(eval_examples)} train examples)")
logger.info(f"Metric:  {metric_name}")
logger.info(f"Model:   {args.model} (temperature={args.temperature})")
logger.info(f"Problem: {problem_description}")

# --- Population --------------------------------------------------------------
tp_set = mutation_prompts[:args.num_mutation_prompts]
mutator_set = thinking_styles[:args.num_thinking_styles]

logger.info('Creating the population...')
p = create_population(
    tp_set=tp_set,
    mutator_set=mutator_set,
    problem_description=problem_description,
)

# --- Run logger ---------------------------------------------------------------
run_logger = RunLogger(metadata={
    'model': args.model,
    'temperature': args.temperature,
    'dataset': dataset_name,
    'metric': metric_name,
    'problem_description': problem_description,
    'num_mutation_prompts': args.num_mutation_prompts,
    'num_thinking_styles': args.num_thinking_styles,
    'num_evals': args.num_evals,
    'num_generations': args.simulations,
    'population_size': p.size,
})

# --- Initial prompts ----------------------------------------------------------
logger.info('Generating the initial prompts...')
init_run(p, co, args.num_evals)
run_logger.log_initial(p)

# --- Genetic algorithm --------------------------------------------------------
logger.info('Starting the genetic algorithm...')
run_for_n(
    n=args.simulations,
    population=p,
    model=co,
    num_evals=args.num_evals,
    on_generation=run_logger.log_generation,
)

# --- Persist ------------------------------------------------------------------
run_logger.log_final(p)
saved_path = run_logger.save()

print("%" * 80)
print("done processing! final gen:")
print(p.units)
print(f"Run log saved to: {saved_path}")
