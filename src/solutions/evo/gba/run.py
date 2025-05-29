import os
import sys


project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))
sys.path.append(project_root)


from src.solutions.evo.gba.args import parse_args
from src.solutions.evo.gba.evoluter import GBAEvoluter
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
    evoluter = GBAEvoluter(
        model_name=args.model_name,
        dataset=args.dataset,
        evaluator=evaluator,
        metric=args.metric,
        task=args.task,
        teams=args.teams,
        players_per_team=args.players_per_team,
        num_seasons=args.seasons,
        output_path=args.output_path,
        use_cache=args.use_cache,
        batch_size=args.batch_size,
        version=args.version
    )
    evoluter.evolution()


if __name__ == "__main__":
    args = parse_args()
    run(args)
