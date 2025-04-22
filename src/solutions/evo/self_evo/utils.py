from pathlib import Path
import yaml


def parse_output(
    model_output: str,
    bracket: str = '<prompt>'
) -> str:
    """Extracts the data from model output.
    It is expected that the data will be bracketed in HTML-like brackets.
    The last brackets entry will be considered as the answer.
    Will return the whole model outputs if couldn't find any brackets entry.

    Args:
        model_output (str): string model output.
        bracket (str, optional): HTML-like bracket to find.
            This function only needs opening bracket,
            as the closing one can be inferred from it.
            Defaults to '<prompt>'.

    Returns:
        str: the extracted result.
    """
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


def append_to_yaml(new_data: dict, filepath: str):
    """Append provided data to yaml file on given filepath.
    This function will not rewrite or delete any other data in file.

    Args:
        new_data (dict): dictionary data to be saved.
        filepath (str): string path to file.
    """
    file = Path(filepath)
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
