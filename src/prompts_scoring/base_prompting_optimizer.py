import argparse
import json


def load_prompts(input_file="basic_prompts.json"):
    with open(input_file) as f:
        return json.load(f)


def run_base_prompts(prompts, output_file_path):
    with open(output_file_path, "a") as f:
        for task, prompt in prompts.items():
            f.write(
                "".join([json.dumps({"task": task, "prompt": prompt}), "\n"])
            )
            f.flush()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process prompts and save to "
            "file as {'task': task, 'prompt': prompt} lines."
        )
    )
    parser.add_argument(
        "--method",
        required=True,
        type=str,
        help="Method of prompting. Must be one of 'basic'",
    )
    parser.add_argument(
        "--output-file-path", required=True, help="Path to the output file"
    )
    parser.add_argument(
        "--input-file-path",
        default="basic_prompts.json",
        help="Path to the input JSON file (default: basic_prompts.json)",
    )
    args = parser.parse_args()
    prompts = load_prompts(args.input_file_path)
    match args.method:
        case "basic":
            run_base_prompts(prompts, args.output_file_path)


if __name__ == "__main__":
    main()
