from typing import Dict, Any
import yaml

from coolprompt.method_evaluation.methods import (
    ReflectivePromptMethod
)


def evaluate_method(
    config: str | Dict[str, Any],
    start_prompt: str,
    output_file_path: str = "./method_evaluation_output.yaml",
    saving_model_answers: bool = False
) -> None:

    if isinstance(config, str):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)

    match config['method']['name']:
        case "reflectiveprompt":
            autoprompting_method = ReflectivePromptMethod(config)
        case _:
            raise ValueError(
                f"Unsupported method name: {config['method']['name']}"
            )

    autoprompting_method.run(
        start_prompt,
        saving_model_answers=saving_model_answers
    )

    with open(output_file_path, 'w') as file:
        yaml.safe_dump(
            {
                'dataset': config['dataset']['name'],
                'configuration': config['dataset']['configuration'],
                'start_prompt': start_prompt,
                'final_prompt': autoprompting_method.final_prompt,
                'val_score': autoprompting_method.final_val_score,
                'test_score': autoprompting_method.final_test_score
            },
            file
        )
