import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # prompt args
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str, choices=["cls", "gen"])
    parser.add_argument(
        "--metric",
        type=str,
        choices=["f1", "accuracy", "bleu", "rouge", "meteor"]
    )
    parser.add_argument("--teams", type=int, default=5)
    parser.add_argument("--players_per_team", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seasons", type=int, default=10)
    parser.add_argument("--use_cache", type=bool, default=True)
    parser.add_argument("--output_path", type=str, default="./outputs")

    args = parser.parse_args()
    return args
