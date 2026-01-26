import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # prompt args
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument(
        "--metric",
        type=str,
    )
    parser.add_argument("--problem_description", type=str)
    parser.add_argument("--population_size", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--use_cache", type=bool, default=False)
    parser.add_argument("--output_path", type=str, default="./outputs")

    args = parser.parse_args()
    return args
