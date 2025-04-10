import os
import sys


project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))
sys.path.append(project_root)


from src.solutions.evo.self_evo.args import parse_args
from src.solutions.evo.self_evo.evoluter import SelfEvoluter
from src.evaluation.evaluator import (
    TextClassificationEvaluator,
    GenerationEvaluator
)


def run(args):
    task2evaluator = {
        "cls": TextClassificationEvaluator,
        "gen": GenerationEvaluator,
    }
    evaluator = task2evaluator[args.task]()
    evoluter = SelfEvoluter(
        model_name=args.model_name,
        dataset=args.dataset,
        evaluator=evaluator,
        metric=args.metric,
        population_num=args.population_num,
        num_epochs=args.num_epochs,
        output_path=args.output_path,
        use_cache=args.use_cache,
        history_size=args.history_size
    )
    evoluter.evolution()


if __name__ == "__main__":
    args = parse_args()
    run(args)
