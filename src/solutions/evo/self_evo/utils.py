from pathlib import Path
import yaml


def parse_output(
    model_output: str,
    bracket: str = '<prompt>'
) -> str:
    closing_bracket = bracket[0] + '/' + bracket[1:]
    parts = model_output.split(bracket)
    if len(parts) > 1:
        prompt = parts[-1].split(closing_bracket)[0]
        prompt = prompt.strip()
        return prompt
    else:
        if (
            model_output.startswith("\"")
            and model_output.endswith("\"")
        ):
            model_output = model_output[1:-1]
        return model_output


def append_to_yaml(new_data: dict, filename: str = "./best_prompts.yaml"):
    file = Path(filename)
    existing_data = {}

    if file.exists():
        with open(file, 'r') as f:
            try:
                existing_data = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                existing_data = {}

    existing_data.update(new_data)

    with open(file, 'w') as f:
        yaml.safe_dump(
            existing_data,
            f,
            default_flow_style=False,
            sort_keys=True
        )
