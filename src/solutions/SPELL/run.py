import os
import sys


project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))
sys.path.append(project_root)


from src.solutions.SPELL.args import parse_args
from src.solutions.SPELL.evoluter import SPELLEvoluter
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai.chat_models import ChatOpenAI
from src.utils.load_dataset_coolprompt import load_dataset
from sklearn.model_selection import train_test_split
from coolprompt.evaluator import Evaluator, validate_and_create_metric
from coolprompt.utils.var_validation import validate_task


def run(args):
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=2,  # 1 запрос в секунду
        check_every_n_seconds=0.1,  # проверять каждые 100ms
        max_bucket_size=10  # максимальный размер буфера
    )

    my_model = ChatOpenAI(
        model="gpt-4o-mini",
        # model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=4000,
        timeout=60,
        max_retries=2,
        rate_limiter=rate_limiter
    )
    dataset, target = load_dataset(args.dataset_name, size=40)
    train_data, val_data, train_targets, val_targets = train_test_split(
        dataset, target, test_size=0.5
    )

    task = validate_task(args.task)
    metric = validate_and_create_metric(task, args.metric)
    evaluator = Evaluator(my_model, task, metric)

    evoluter = SPELLEvoluter(
        model=my_model,
        train_dataset=train_data,
        train_target=train_targets,
        validation_dataset=val_data,
        validation_target=val_targets,
        problem_description=args.problem_description,
        evaluator=evaluator,
        population_size=args.population_size,
        num_epochs=args.num_epochs,
        output_path=args.output_path,
        use_cache=args.use_cache,
    )
    evoluter.evolution()


if __name__ == "__main__":
    args = parse_args()
    run(args)
